"""LangGraph state schemas."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

RESET_NOTES = "\x00reset"


def add_notes(existing: list[str] | None, new: list[str]) -> list[str]:
    """Accumulate notes within one run; start clean when a run says so.

    notes is a per-answer trace, not conversation memory, but the checkpointer
    persists every channel under the thread id. With a plain append reducer,
    turn 5 of a chat returned the traces of turns 1-4 alongside its own.
    """
    if RESET_NOTES in new:
        return [n for n in new if n != RESET_NOTES]
    return (existing or []) + new


class IngestState(TypedDict, total=False):
    """State for the ingest graph.

    Note what is *not* here: per-page progress. Page state lives in SQLite
    because it has to survive across runs, not just across nodes of this one -
    the pages table is what lets a completely new process pick up at page 612.
    This graph runs without a checkpointer; see the module docstring in
    ingest_graph.py for why one cannot help.
    """

    pdf_path: str
    job_id: str
    doc_id: int
    doc_sha: str
    force: bool
    display_name: str

    pending: list[tuple[int, str]]

    pages_total: int
    pages_done: int


class QueryState(TypedDict, total=False):
    """State for the self-corrective query graph."""

    question: str
    standalone_question: str
    history: list[dict[str, str]]
    doc_ids: list[int]
    mode: str

    documents: list[dict[str, Any]]
    top_score: float
    relevant: bool

    answer: str
    citations: list[dict[str, Any]]
    abstained: bool
    grounded: bool

    rewrites: int
    regenerations: int
    notes: Annotated[list[str], add_notes]
