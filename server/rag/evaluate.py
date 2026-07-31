"""Retrieval and answer evaluation.

Deliberately split into two tiers:

* **Deterministic retrieval metrics** (Recall@k, MRR, nDCG) computed from gold
  page numbers. No LLM, no judge, no variance - these are the numbers to tune
  chunking, embeddings and reranking against, and they are reproducible.
* **Answer-level checks** that need the LLM (does it cite? does it abstain when
  it should?), reported separately because they are slower and noisier.

`export_jsonl` writes {question, answer, contexts, ...} so a heavier scorer such
as RAGAS can run out-of-band, in its own environment, without imposing its
dependency pins on this project.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..settings import Settings, get_settings
from ..util.logging import get_logger
from .retrieve import retrieve

log = get_logger(__name__)


@dataclass
class EvalCase:
    question: str
    pages: list[int] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)
    should_abstain: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalCase:
        """Build a case from one hand-written YAML entry.

        The eval files are written by hand, so the singular and scalar spellings are
        accepted too: `page: 12` for `pages`, and a bare string for answer_contains.
        """
        pages = raw.get("pages") or raw.get("page") or []
        if isinstance(pages, int):
            pages = [pages]
        contains = raw.get("answer_contains") or []
        if isinstance(contains, str):
            contains = [contains]
        return cls(
            question=str(raw["question"]),
            pages=[int(p) for p in pages],
            answer_contains=[str(c) for c in contains],
            should_abstain=bool(raw.get("should_abstain", False)),
            tags=[str(t) for t in (raw.get("tags") or [])],
        )


@dataclass
class RetrievalScore:
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    hits: int = 0
    total: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "recall": round(self.recall_at_k, 3),
            "mrr": round(self.mrr, 3),
            "ndcg": round(self.ndcg, 3),
            "found": f"{self.hits}/{self.total}",
        }


def load_cases(path: Path) -> list[EvalCase]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if isinstance(raw, dict):
        raw = raw.get("questions", [])
    return [EvalCase.from_dict(item) for item in raw]


def score_retrieval(
    cases: list[EvalCase],
    cfg: Settings | None = None,
    *,
    k: int = 8,
    dense_only: bool = False,
    fts_only: bool = False,
    rerank: bool = True,
) -> tuple[RetrievalScore, list[dict[str, Any]]]:
    """Do the gold pages appear in the top-k retrieved chunks?

    Cases with no gold pages - the abstain cases - are skipped rather than scored
    zero, and every mean is taken over the graded subset only, so `total` is not
    len(cases).
    """
    cfg = cfg or get_settings()
    graded = [c for c in cases if c.pages]
    detail: list[dict[str, Any]] = []
    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    hits = 0

    for case in graded:
        result = retrieve(
            case.question,
            cfg,
            rerank=rerank,
            dense_only=dense_only,
            fts_only=fts_only,
            top_n=k,
        )
        ranked = result.sources[:k]
        gold = set(case.pages)

        first_rank: dict[int, int] = {}
        for rank, source in enumerate(ranked, start=1):
            for page in {int(p) for p in (source.page_span or [source.page_no])} & gold:
                first_rank.setdefault(page, rank)
        found_ranks = sorted(first_rank.values())
        found_pages = set(first_rank)

        recall = len(found_pages) / len(gold) if gold else 0.0
        rr = 1.0 / found_ranks[0] if found_ranks else 0.0
        dcg = sum(1.0 / math.log2(rank + 1) for rank in found_ranks)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
        ndcg = dcg / ideal if ideal else 0.0

        recalls.append(recall)
        rrs.append(rr)
        ndcgs.append(ndcg)
        if found_ranks:
            hits += 1

        detail.append(
            {
                "question": case.question,
                "gold_pages": sorted(gold),
                "retrieved_pages": [s.page_no for s in ranked],
                "first_hit_rank": found_ranks[0] if found_ranks else None,
                "recall": round(recall, 3),
                "top_score": round(result.top_score, 4),
                "tags": case.tags,
            }
        )

    n = max(1, len(graded))
    return (
        RetrievalScore(
            recall_at_k=sum(recalls) / n,
            mrr=sum(rrs) / n,
            ndcg=sum(ndcgs) / n,
            hits=hits,
            total=len(graded),
        ),
        detail,
    )


def ablate(
    cases: list[EvalCase],
    cfg: Settings | None = None,
    *,
    k: int = 8,
) -> dict[str, RetrievalScore]:
    """Compare retrieval strategies so tuning is measured, not assumed."""
    cfg = cfg or get_settings()
    variants = {
        "dense only": {"dense_only": True, "rerank": False},
        "bm25 only": {"fts_only": True, "rerank": False},
        "hybrid": {"rerank": False},
        "hybrid + rerank": {},
    }
    out: dict[str, RetrievalScore] = {}
    for name, kwargs in variants.items():
        score, _ = score_retrieval(cases, cfg, k=k, **kwargs)  # type: ignore[arg-type]
        out[name] = score
        log.info("ablation", variant=name, **score.as_row())
    return out


def score_answers(
    cases: list[EvalCase],
    cfg: Settings | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """End-to-end answer checks. Needs a working LLM, so it is slower.

    The grounded rate covers only the cases the verifier actually ran on, and is
    reported as null rather than zero when it ran on none - in fast mode nothing
    is verified, and a zero there would read as every answer being ungrounded.
    """
    from ..graphs.query_graph import answer_question

    cfg = cfg or get_settings()
    rows: list[dict[str, Any]] = []
    cited = grounded = abstain_ok = contains_ok = 0
    contains_total = verified_total = 0

    for case in cases:
        state = answer_question(case.question, cfg=cfg)
        answer = str(state.get("answer") or "")
        citations = state.get("citations") or []
        abstained = bool(state.get("abstained"))

        if citations:
            cited += 1
        if state.get("grounded") is not None:
            verified_total += 1
            if state["grounded"]:
                grounded += 1
        if case.should_abstain == abstained:
            abstain_ok += 1

        hit = None
        if case.answer_contains:
            contains_total += 1
            hit = all(token.lower() in answer.lower() for token in case.answer_contains)
            if hit:
                contains_ok += 1

        rows.append(
            {
                "question": case.question,
                "abstained": abstained,
                "expected_abstain": case.should_abstain,
                "contains_expected": hit,
                "n_citations": len(citations),
                "cited_pages": [c["page_label"] for c in citations],
                "answer": answer[:300],
            }
        )

    n = max(1, len(cases))
    summary = {
        "cases": len(cases),
        "with_citations": round(cited / n, 3),
        "grounded": (round(grounded / verified_total, 3) if verified_total else None),
        "abstain_correct": round(abstain_ok / n, 3),
        "contains_expected": (
            round(contains_ok / contains_total, 3) if contains_total else None
        ),
    }
    return summary, rows


def export_jsonl(
    cases: list[EvalCase],
    path: Path,
    cfg: Settings | None = None,
) -> int:
    """Write {question, answer, contexts, ...} for an external scorer."""
    from ..graphs.query_graph import answer_question

    cfg = cfg or get_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            state = answer_question(case.question, cfg=cfg)
            fh.write(
                json.dumps(
                    {
                        "question": case.question,
                        "answer": state.get("answer", ""),
                        "contexts": [d["text"] for d in (state.get("documents") or [])],
                        "gold_pages": case.pages,
                        "cited_pages": [
                            c["page_label"] for c in (state.get("citations") or [])
                        ],
                        "abstained": bool(state.get("abstained")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written
