"""Self-corrective RAG graph.

    START -> condense -> retrieve -> grade_documents
                            ^             |
                            |             +-- relevant ---> generate -> verify
                            |             |                              |
                            +-- rewrite <-+ weak (bounded)               +-> END
                                          |                             |
                                          +-- exhausted --> abstain --> END

The loops are what make this a graph rather than a chain, and each is bounded in
state (`rewrites`, `regenerations`) so it can never spin:

* **grade_documents** asks whether the retrieved excerpts actually answer the
  question. A weak result rewrites the query and retries instead of answering
  from whatever came back.
* **verify** re-reads the drafted answer against the excerpts and rejects claims
  they do not support, which is the difference between "sounds right" and
  "is in the document".

Both graders need reliable JSON-schema output. Small local models are uneven at
that, so `enable_graders` turns them off and `answer_mode=fast` collapses the
graph to retrieve -> generate. Score-threshold abstention still applies in fast
mode because it costs no LLM call.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from ..llm.models import build_chat_model, message_text
from ..rag import prompt as P
from ..rag.retrieve import Source, pack_context, retrieve, should_abstain
from ..settings import Settings, get_settings
from ..util.logging import get_logger
from ._stream import emit
from .state import RESET_NOTES, QueryState

log = get_logger(__name__)

_CITE_RE = re.compile(r"[\[［【(]\s*S\s*(\d+)\s*[\]］】)]")
_SOURCE_FIELDS = {f.name for f in dataclasses.fields(Source)}


def _graders_enabled(cfg: Settings, state: QueryState) -> bool:
    mode = state.get("mode") or cfg.answer_mode
    return bool(cfg.enable_graders) and mode == "careful"


def _sources(state: QueryState) -> list[Source]:
    """Rebuild Source objects from the dicts carried in graph state.

    to_dict() adds derived keys (page_label); filtering on the dataclass fields
    keeps that list from having to be maintained by hand.
    """
    return [
        Source(**{k: v for k, v in d.items() if k in _SOURCE_FIELDS})
        for d in state.get("documents", [])
    ]


def _ask(cfg: Settings, system: str, user: str, *, temperature: float, what: str) -> str:
    """One-shot LLM call that degrades to an empty string instead of raising.

    Query rewriting and follow-up condensing are both optimisations: if the
    model is unavailable the caller falls back to the original question rather
    than failing the request.
    """
    try:
        reply = build_chat_model(cfg, temperature=temperature).invoke(
            [("system", system), ("user", user)]
        )
        return message_text(reply).strip().strip('"')
    except Exception as exc:  # noqa: BLE001 - optional step, never fatal
        log.warning(f"{what} failed", error=str(exc))
        return ""


def condense_node(state: QueryState) -> dict[str, Any]:
    """Turn a follow-up into a standalone query.

    Without this, "what about the second one?" retrieves on the words "second
    one" and returns noise.
    """
    cfg = get_settings()
    question = state["question"]
    history = state.get("history") or []
    if not history:
        return {"standalone_question": question, "notes": [RESET_NOTES]}

    emit(stage="condense")
    standalone = (
        _ask(
            cfg,
            P.CONDENSE_SYSTEM,
            P.CONDENSE_USER.format(history=P.format_history(history), question=question),
            temperature=0.0,
            what="condense",
        )
        or question
    )
    if standalone != question:
        log.debug("condensed", original=question, standalone=standalone)
    return {
        "standalone_question": standalone,
        "notes": [RESET_NOTES, f"query: {standalone}"],
    }


def retrieve_node(state: QueryState) -> dict[str, Any]:
    cfg = get_settings()
    query = state.get("standalone_question") or state["question"]
    emit(stage="retrieve", query=query)

    result = retrieve(query, cfg, doc_ids=state.get("doc_ids") or None)
    emit(
        stage="reranked",
        n=len(result.sources),
        top_score=round(result.top_score, 4),
        pages=[s.page_no for s in result.sources[:5]],
    )
    return {
        "documents": [s.to_dict() for s in result.sources],
        "top_score": result.top_score,
    }


def grade_documents_node(state: QueryState) -> dict[str, Any]:
    """Ask the model whether the excerpts actually answer the question.

    The grader fails open: a local model that errors or returns unparseable JSON
    would otherwise push every query into the rewrite loop and then abstain. An
    empty retrieval is left to the score threshold in _after_grade for the same
    reason.
    """
    cfg = get_settings()
    if not _graders_enabled(cfg, state):
        return {}
    sources = _sources(state)
    if not sources:
        return {"relevant": True, "notes": ["no candidates retrieved"]}

    context, _used = pack_context(sources, cfg)
    emit(stage="grade_documents")
    try:
        grader = build_chat_model(cfg, temperature=0.0).with_structured_output(
            P.DocumentRelevance
        )
        verdict = grader.invoke(
            [
                ("system", P.GRADE_DOCS_SYSTEM),
                (
                    "user",
                    P.GRADE_DOCS_USER.format(
                        question=state.get("standalone_question") or state["question"],
                        context=context,
                    ),
                ),
            ]
        )
        relevant = bool(getattr(verdict, "relevant", True))
        reason = str(getattr(verdict, "reason", ""))
    except Exception as exc:  # noqa: BLE001
        log.warning("document grading failed; proceeding", error=str(exc))
        return {"relevant": True, "notes": [f"grader unavailable: {type(exc).__name__}"]}

    emit(stage="graded", relevant=relevant, reason=reason)
    return {"relevant": relevant, "notes": [f"relevance: {relevant} ({reason})"]}


def rewrite_node(state: QueryState) -> dict[str, Any]:
    """Reformulate and try again, at most max_query_rewrites times.

    Every attempt rewrites the original question and passes the previous attempt
    only as something to avoid. Rewriting the rewrite compounds drift until the
    query no longer resembles what was asked.
    """
    cfg = get_settings()
    attempts = int(state.get("rewrites", 0)) + 1
    original = state.get("standalone_question") or state["question"]
    emit(stage="rewrite", attempt=attempts)

    rewritten = (
        _ask(
            cfg,
            P.REWRITE_SYSTEM,
            P.REWRITE_USER.format(
                question=state["question"], tried=f"Already tried: {original}"
            ),
            temperature=0.3,
            what="rewrite",
        )
        or original
    )
    return {
        "standalone_question": rewritten,
        "rewrites": attempts,
        "notes": [f"rewrite {attempts}: {rewritten}"],
    }


def generate_node(state: QueryState) -> dict[str, Any]:
    """Draft the answer, then keep only citations it was actually shown."""
    cfg = get_settings()
    sources = _sources(state)
    context, used = pack_context(sources, cfg)
    stricter = int(state.get("regenerations", 0)) > 0
    emit(stage="generate", stricter=stricter, sources=len(used))

    template = P.STRICTER_ANSWER_USER if stricter else P.ANSWER_USER
    reply = build_chat_model(cfg, temperature=0.0).invoke(
        [
            ("system", P.ANSWER_SYSTEM),
            (
                "user",
                template.format(
                    question=state.get("standalone_question") or state["question"],
                    context=context,
                ),
            ),
        ]
    )
    answer = message_text(reply).strip()

    if P.NOT_IN_DOCUMENT in answer:
        return {
            "answer": P.ABSTAIN_MESSAGE,
            "abstained": True,
            "citations": [s.to_dict() for s in used[:3]],
            "notes": ["model reported the answer is not in the document"],
        }

    answer, citations = _validate_citations(answer, used)
    return {"answer": answer, "citations": citations, "abstained": False}


def verify_node(state: QueryState) -> dict[str, Any]:
    """Check the draft against the excerpts before it is returned.

    Fails open like document grading: an unavailable verifier must not turn every
    answer into an abstention.
    """
    cfg = get_settings()
    context, _used = pack_context(_sources(state), cfg)
    emit(stage="verify")
    try:
        grader = build_chat_model(cfg, temperature=0.0).with_structured_output(P.Groundedness)
        verdict = grader.invoke(
            [
                ("system", P.GROUNDED_SYSTEM),
                ("user", P.GROUNDED_USER.format(context=context, answer=state["answer"])),
            ]
        )
        grounded = bool(getattr(verdict, "grounded", True))
        unsupported = str(getattr(verdict, "unsupported", ""))
    except Exception as exc:  # noqa: BLE001
        log.warning("groundedness check failed; accepting answer", error=str(exc))
        return {"grounded": True, "notes": [f"verifier unavailable: {type(exc).__name__}"]}

    emit(stage="verified", grounded=grounded, unsupported=unsupported[:120])
    note = "grounded" if grounded else f"ungrounded: {unsupported[:160]}"
    return {"grounded": grounded, "notes": [note]}


def abstain_node(state: QueryState) -> dict[str, Any]:
    """Say so, and show the near misses rather than inventing an answer."""
    emit(stage="abstain", top_score=round(float(state.get("top_score", 0.0)), 4))
    return {
        "answer": P.ABSTAIN_MESSAGE,
        "abstained": True,
        "grounded": True,
        "citations": [s.to_dict() for s in _sources(state)[:3]],
        "notes": ["abstained"],
    }


def _validate_citations(
    answer: str, used: list[Source]
) -> tuple[str, list[dict[str, Any]]]:
    """Remove citations the model invented; report the ones it really used.

    A model under pressure will cite [S9] when it was shown five excerpts. Left
    in, that looks like a verifiable reference and is not one.
    """
    valid = {s.ref: s for s in used}
    cited: list[str] = []

    def keep(match: re.Match[str]) -> str:
        ref = f"S{match.group(1)}"
        if ref in valid:
            if ref not in cited:
                cited.append(ref)
            return f"[{ref}]"
        log.debug("dropped hallucinated citation", ref=ref)
        return ""

    cleaned = _CITE_RE.sub(keep, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if not cited and used:
        log.warning("answer contained no valid citations")

    ordered = [valid[ref].to_dict() for ref in cited]
    return cleaned, ordered


def _after_grade(state: QueryState) -> str:
    """Route on retrieval quality: answer, rewrite, or give up.

    The score threshold is checked before the grader verdict, so fast mode still
    abstains on weak retrieval without spending an LLM call. Exhausting
    max_query_rewrites is what terminates the retrieve/rewrite loop.
    """
    cfg = get_settings()

    weak = should_abstain(
        len(state.get("documents", [])), float(state.get("top_score", 0.0)), cfg
    )
    if not weak and _graders_enabled(cfg, state):
        weak = state.get("relevant") is False

    if not weak:
        return "generate"
    if int(state.get("rewrites", 0)) < cfg.max_query_rewrites:
        return "rewrite"
    return "abstain"


def _after_verify(state: QueryState) -> str:
    cfg = get_settings()
    if state.get("grounded", True):
        return "__end__"
    if int(state.get("regenerations", 0)) < cfg.max_regenerations:
        return "regenerate"
    return "abstain"


def _regenerate_node(state: QueryState) -> dict[str, Any]:
    """Count the retry before generate re-runs.

    The decision to retry is made downstream in _after_verify, so generate cannot
    bump this itself; the count is what makes the next pass use the stricter
    prompt and what bounds the loop.
    """
    return {"regenerations": int(state.get("regenerations", 0)) + 1}


def _verify_or_end(state: QueryState) -> str:
    """Skip verification for an abstention.

    An abstention asserts nothing the excerpts could support, so the verifier
    would call it ungrounded and send it round the regenerate loop to produce the
    same abstention again.
    """
    cfg = get_settings()
    if state.get("abstained"):
        return "__end__"
    return "verify" if _graders_enabled(cfg, state) else "__end__"


def build_query_graph(checkpointer: Any | None = None) -> Any:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(QueryState)
    builder.add_node("condense", condense_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("regenerate", _regenerate_node)
    builder.add_node("abstain", abstain_node)

    builder.add_edge(START, "condense")
    builder.add_edge("condense", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        _after_grade,
        {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges(
        "generate", _verify_or_end, {"verify": "verify", "__end__": END}
    )
    builder.add_conditional_edges(
        "verify",
        _after_verify,
        {"regenerate": "regenerate", "abstain": "abstain", "__end__": END},
    )
    builder.add_edge("regenerate", "generate")
    builder.add_edge("abstain", END)

    return builder.compile(checkpointer=checkpointer)


def async_query_checkpointer(cfg: Settings | None = None) -> Any:
    """Conversation memory, one thread_id per chat, as an async context manager.

    Async because the API drives the graph with astream(); SqliteSaver is
    sync-only and raises there.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    cfg = cfg or get_settings()
    cfg.ensure_dirs()
    return AsyncSqliteSaver.from_conn_string(str(cfg.checkpoint_path))


def studio_graph() -> Any:
    """Zero-argument entrypoint for `langgraph dev` / LangGraph Studio.

    The platform inspects the factory's signature and injects a RunnableConfig
    into any parameter, so build_query_graph(checkpointer) cannot be referenced
    directly. Studio manages persistence itself, hence no checkpointer here.
    """
    return build_query_graph()


def answer_question(
    question: str,
    *,
    cfg: Settings | None = None,
    doc_ids: list[int] | None = None,
    history: list[dict[str, str]] | None = None,
    mode: str | None = None,
) -> QueryState:
    """Single-shot convenience wrapper used by the CLI and tests."""
    cfg = cfg or get_settings()
    graph = build_query_graph()
    state: QueryState = {
        "question": question,
        "history": history or [],
        "doc_ids": doc_ids or [],
        "mode": mode or cfg.answer_mode,
        "rewrites": 0,
        "regenerations": 0,
    }
    return graph.invoke(state, config={"recursion_limit": 40})
