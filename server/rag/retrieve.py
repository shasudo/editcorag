"""Retrieval and context assembly for answering.

Two jobs beyond running the retriever:

* decide whether the best hit is good enough to answer from at all, and
* pack the surviving chunks into a numbered context block whose ids the model
  can cite and we can validate afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..ingest.quality import detect_script
from ..settings import Settings, get_settings
from ..util.logging import get_logger

log = get_logger(__name__)

_CHARS_PER_TOKEN = 3.5
_DENSE_CHARS_PER_TOKEN = 1.2
_DENSE_SCRIPTS = frozenset({"han", "kana", "hangul"})


@dataclass
class Source:
    """One citable chunk."""

    ref: str
    chunk_id: str
    doc_id: int
    page_no: int
    page_span: list[int]
    kind: str
    section: str | None
    text: str
    bbox: list[float] | None
    asset_path: str | None
    score: float

    @property
    def page_label(self) -> str:
        """Cite a range when the chunk genuinely spans pages.

        The chunker merges adjacent content, so a chunk can cover two pages;
        citing only the first would point at the wrong page for half its text.
        """
        if len(self.page_span) > 1 and self.page_span[0] != self.page_span[-1]:
            return f"p.{self.page_span[0]}-{self.page_span[-1]}"
        return f"p.{self.page_no}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"page_label": self.page_label, "score": round(self.score, 4)}


@dataclass
class Retrieved:
    sources: list[Source] = field(default_factory=list)
    top_score: float = 0.0


def retrieve(
    question: str,
    cfg: Settings | None = None,
    *,
    doc_ids: list[int] | None = None,
    rerank: bool = True,
    dense_only: bool = False,
    fts_only: bool = False,
    top_n: int | None = None,
) -> Retrieved:
    """Run the hybrid+rerank pipeline and wrap the results as citable sources."""
    from ..index.retrievers import build_retrieval_pipeline

    cfg = cfg or get_settings()
    pipeline = build_retrieval_pipeline(
        cfg,
        doc_ids=doc_ids,
        rerank=rerank,
        dense_only=dense_only,
        fts_only=fts_only,
        top_n=top_n,
    )
    docs = pipeline.invoke(question)

    if docs and "rerank_score" in (docs[0].metadata or {}):
        docs = sorted(
            docs,
            key=lambda d: (
                -float((d.metadata or {}).get("rerank_score", 0.0) or 0.0),
                str((d.metadata or {}).get("chunk_id", "")),
            ),
        )

    sources: list[Source] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        score = float(meta.get("rerank_score", meta.get("retrieval_score", 0.0)) or 0.0)
        span = [int(p) for p in (meta.get("page_span") or [meta.get("page_no", 0)])]
        sources.append(
            Source(
                ref=f"S{i}",
                chunk_id=str(meta.get("chunk_id", "")),
                doc_id=int(meta.get("doc_id", 0)),
                page_no=int(meta.get("page_no", 0)),
                page_span=span,
                kind=str(meta.get("kind", "text")),
                section=meta.get("section"),
                text=doc.page_content,
                bbox=meta.get("bbox"),
                asset_path=meta.get("asset_path"),
                score=score,
            )
        )

    top = max((s.score for s in sources), default=0.0)
    log.debug("retrieved", n=len(sources), top_score=round(top, 4))
    return Retrieved(sources=sources, top_score=top)


def should_abstain(n_sources: int, top_score: float, cfg: Settings | None = None) -> bool:
    """True when nothing retrieved is relevant enough to answer from."""
    cfg = cfg or get_settings()
    return n_sources == 0 or top_score < cfg.abstain_score_min


def _chars_per_token(sources: list[Source]) -> float:
    """Density of the retrieved text, from its dominant script.

    CJK, kana and hangul pack roughly three times more tokens per character than
    Latin text, so budgeting them at the Latin ratio overruns the model's context
    window by about the same factor.
    """
    sample = " ".join(s.text for s in sources[:4])[:2000]
    return (
        _DENSE_CHARS_PER_TOKEN
        if detect_script(sample) in _DENSE_SCRIPTS
        else _CHARS_PER_TOKEN
    )


def pack_context(
    sources: list[Source], cfg: Settings | None = None
) -> tuple[str, list[Source]]:
    """Render sources into a numbered context block within the token budget.

    Returns the block and the sources that actually fitted, so citations can be
    validated against what the model was really shown. The first source is kept
    even when it alone exceeds the budget: an empty context block would leave the
    model answering from nothing.
    """
    cfg = cfg or get_settings()
    budget_chars = int(cfg.context_token_budget * _chars_per_token(sources))

    blocks: list[str] = []
    used: list[Source] = []
    spent = 0
    for source in sources:
        header = f"[{source.ref}] ({source.page_label}"
        if source.section:
            header += f", section: {source.section}"
        if source.kind in ("figure", "table"):
            header += f", {source.kind}"
        header += ")"
        block = f"{header}\n{source.text.strip()}"
        if used and spent + len(block) > budget_chars:
            break
        blocks.append(block)
        used.append(source)
        spent += len(block)

    if len(used) < len(sources):
        log.debug("context truncated", kept=len(used), of=len(sources))
    return "\n\n".join(blocks), used
