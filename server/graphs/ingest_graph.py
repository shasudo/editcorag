"""Ingest graph: PDF -> routed pages -> extracted chunks -> searchable index.

    START -> prepare -> extract_and_index -> finalize -> END

Durability is the **SQLite pages table**, and only that. A page moves
pending -> routed -> done|failed|skipped and is keyed on
(sha256, page_no, pipeline_version), so a brand-new process resumes exactly
where a killed one stopped.

This graph deliberately runs without a LangGraph checkpointer. One was tried:
it cannot help here, because LangGraph only resumes a thread when the input is
None or a Command, and every caller passes a populated dict from a fresh
process - so the graph re-enters at START and re-runs `prepare` regardless.
Making it genuinely resume would be worse than useless: it would skip `prepare`
and extract against a `pending` list computed before the interruption, silently
dropping the pages whose status changed. Re-running `prepare` is cheap
(inventory is ~1 ms/page and pure), and it recomputes `pending` from the table
that actually knows.

A note on the fan-out: the plan called for `Send` to parallelise batches. In
practice each `Send` branch would need its own worker process, and because
`spawn` re-imports and reloads the layout/table models, that costs several
seconds per batch - roughly 11 minutes of pure model loading across an
800-page document. Instead a single node drives a persistent process pool that
loads models once per worker and recycles every `worker_max_batches`. The
parallelism, crash isolation and bounded concurrency are identical; the model
loading is paid once instead of 133 times.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from ..db import store as db
from ..index.embed import embed_dimension, get_embeddings
from ..index.store import ChunkStore, rows_from_chunks, sqlite_chunk_rows
from ..ingest import ocr
from ..ingest.extract import BatchResult
from ..ingest.inventory import inventory
from ..ingest.route import Route, histogram, route_pages
from ..ingest.runner import BatchRunner, jobs_for
from ..settings import Settings, get_settings
from ..util.logging import get_logger
from ._stream import emit
from .state import IngestState

log = get_logger(__name__)


def prepare_node(state: IngestState) -> dict[str, Any]:
    """Identify the document, classify every page, and work out what is left.

    Inventory and routing live in one node because routing needs the inventory:
    splitting them meant opening the PDF, re-reading every page and re-hashing
    the whole file a second time. Routing is cheap and pure, so nothing is lost
    by not checkpointing between the two.
    """
    cfg = get_settings()
    cfg.ensure_dirs()
    pdf_path = Path(state["pdf_path"])
    conn = db.get_conn(cfg.sqlite_path)

    inv = inventory(pdf_path, min_chars=cfg.route_min_chars)
    sha = inv.sha256
    doc = db.upsert_document(
        conn,
        sha256=sha,
        filename=state.get("display_name") or inv.filename,
        path=str(pdf_path.resolve()),
        page_count=inv.page_count,
        byte_size=inv.byte_size,
        pipeline_version=cfg.pipeline_version,
        title=inv.title,
    )
    doc_id = int(doc["id"])
    db.ensure_pages(conn, doc_id, inv.page_count)

    job_id = state.get("job_id") or f"ingest-{sha[:12]}"
    db.upsert_job(
        conn,
        job_id=job_id,
        doc_id=doc_id,
        kind="ingest",
        status="running",
        pages_total=inv.page_count,
    )

    if state.get("force"):
        ChunkStore(cfg).delete_document(doc_id)
        for rel in db.document_asset_paths(conn, doc_id):
            (cfg.assets_dir / rel).unlink(missing_ok=True)
        with db.transaction(conn):
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM assets WHERE doc_id=?", (doc_id,))
            conn.execute(
                "UPDATE pages"
                "   SET status='pending', attempts=0, error=NULL, n_chunks=0"
                " WHERE doc_id=?",
                (doc_id,),
            )

    emit(stage="inventory", doc_id=doc_id, pages=inv.page_count, filename=inv.filename)

    decisions = route_pages(inv.pages, cfg)
    db.save_routing(
        conn, doc_id, [d.as_row() for d in decisions], max_attempts=cfg.max_page_attempts
    )

    empties = [d.page_no for d in decisions if d.route is Route.EMPTY]
    for page_no in empties:
        db.set_page_status(conn, doc_id, page_no, "skipped")

    hist = histogram(decisions)
    routes = {d.page_no: d.route.value for d in decisions}
    todo = db.pending_pages(conn, doc_id, cfg.max_page_attempts)
    pending = [(int(r["page_no"]), routes.get(int(r["page_no"]), "digital")) for r in todo]
    settled = db.status_counts(conn, doc_id)
    already = max(0, sum(settled.get(s, 0) for s in db.PAGE_TERMINAL) - len(empties))

    db.set_document_status(conn, doc_id, "extracting")
    emit(
        stage="route",
        histogram=hist,
        pending=len(pending),
        skipped=len(empties),
        already_done=already,
    )
    log.info(
        "prepared",
        file=inv.filename,
        pages=inv.page_count,
        sha=sha[:12],
        histogram=hist,
        pending=len(pending),
        resumed=already,
    )
    return {
        "doc_id": doc_id,
        "doc_sha": sha,
        "job_id": job_id,
        "pages_total": inv.page_count,
        "pending": pending,
        "pages_done": already + len(empties),
    }


def extract_and_index_node(state: IngestState) -> dict[str, Any]:
    """Extract, embed and index every pending page.

    Each batch is committed as it completes - chunks into LanceDB and SQLite,
    page status flipped to done - so killing the process loses at most the
    batches currently in flight.
    """
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    doc_id = int(state["doc_id"])
    doc_sha = str(state["doc_sha"])
    pending: list[tuple[int, str]] = [tuple(p) for p in state.get("pending", [])]  # type: ignore[misc]

    if not pending:
        return {}

    unreadable = ocr.unsupported(cfg.ocr_languages)
    if unreadable:
        warning = f"RapidOCR cannot read: {', '.join(unreadable)}; those pages will be empty"
        log.warning("ocr", detail=warning)
        emit(stage="warning", message=warning)

    embeddings = get_embeddings(cfg)
    dim = embed_dimension(cfg)
    chunk_store = ChunkStore(cfg)
    chunk_store.open(create_with_dim=dim)
    db.set_document_status(conn, doc_id, "extracting", embed_model=cfg.embed_model, embed_dim=dim)

    jobs = jobs_for(Path(state["pdf_path"]), doc_sha, pending, cfg.batch_pages)
    total_pages = sum(len(j.pages) for j in jobs)
    done_pages = int(state.get("pages_done", 0))

    emit(stage="extract_start", batches=len(jobs), pages=total_pages)

    with BatchRunner(cfg) as runner:
        for result in runner.run(jobs):
            pages_in_batch = list(range(result.page_from, result.page_to + 1))

            if not result.ok:
                for page_no in pages_in_batch:
                    db.set_page_status(
                        conn, doc_id, page_no, "failed", error=result.error, bump_attempts=True
                    )
                emit(
                    stage="batch",
                    page_from=result.page_from,
                    page_to=result.page_to,
                    ok=False,
                    error=result.error,
                )
                continue

            n_indexed = _commit_batch(
                conn=conn,
                cfg=cfg,
                chunk_store=chunk_store,
                embeddings=embeddings,
                doc_id=doc_id,
                doc_sha=doc_sha,
                result=result,
                pages_in_batch=pages_in_batch,
            )
            done_pages += len(pages_in_batch)

            db.update_job(conn, str(state["job_id"]), pages_done=done_pages)
            emit(
                stage="batch",
                page_from=result.page_from,
                page_to=result.page_to,
                ok=True,
                route=result.route,
                chunks=n_indexed,
                figures=len(result.assets),
                seconds=round(result.duration_s, 1),
                pages_done=done_pages,
                pages_total=state.get("pages_total", total_pages),
            )

    return {"pages_done": done_pages}


def _commit_batch(
    *,
    conn: sqlite3.Connection,
    cfg: Settings,
    chunk_store: ChunkStore,
    embeddings: Any,
    doc_id: int,
    doc_sha: str,
    result: BatchResult,
    pages_in_batch: list[int],
) -> int:
    """Embed and persist one batch, then mark its pages done.

    Assets and chunks for these pages are deleted before the new ones land, so a
    re-run after a crash replaces the batch rather than doubling it. Pages are
    flipped to done last, so anything interrupted mid-commit is simply redone.
    """
    db.delete_page_assets(conn, doc_id, pages_in_batch)
    asset_ids: dict[str, int] = {}
    for asset in result.assets:
        asset_id = db.insert_asset(
            conn,
            doc_id=doc_id,
            page_no=asset.page_no,
            kind="figure",
            path=asset.path,
            bbox=asset.bbox,
            ocr_text=asset.ocr_text,
        )
        if asset.path:
            asset_ids[asset.path] = asset_id

    chunks = result.chunks
    if chunks:
        chunk_store.delete_pages(doc_id, pages_in_batch)
        vectors = embeddings.embed_documents([c.embed_text for c in chunks])
        chunk_store.upsert(
            rows_from_chunks(chunks, doc_id=doc_id, doc_sha=doc_sha, vectors=vectors)
        )

        by_page: dict[int, list[Any]] = {}
        for chunk in chunks:
            by_page.setdefault(chunk.page_no, []).append(chunk)
        for page_no, page_chunks in by_page.items():
            db.replace_page_chunks(
                conn, doc_id, page_no, sqlite_chunk_rows(page_chunks, asset_ids)
            )

    counts = Counter(c.page_no for c in result.chunks)
    for page_no in pages_in_batch:
        db.set_page_status(conn, doc_id, page_no, "done", n_chunks=counts.get(page_no, 0))
    return len(chunks)


def finalize_node(state: IngestState) -> dict[str, Any]:
    """Build the search indexes and close the job out."""
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    doc_id = int(state["doc_id"])

    chunk_store = ChunkStore(cfg)
    if chunk_store.exists():
        chunk_store.ensure_fts_index()
        chunk_store.ensure_vector_index()

    counts = db.status_counts(conn, doc_id)
    unfinished = sum(n for s, n in counts.items() if s not in db.PAGE_TERMINAL)
    total_chunks = db.count_chunks(conn, doc_id)
    status = "done" if not unfinished else "partial"

    db.set_document_status(conn, doc_id, status)
    db.update_job(
        conn,
        str(state["job_id"]),
        status="done" if not unfinished else "failed",
        message=f"{total_chunks} chunks indexed; {unfinished} page(s) unfinished",
        finished=True,
    )
    emit(
        stage="finalize",
        status=status,
        chunks=total_chunks,
        failed_pages=unfinished,
        page_status=counts,
    )
    log.info("ingest finished", status=status, chunks=total_chunks, page_status=counts)
    return {}


def build_ingest_graph() -> Any:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(IngestState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("extract_and_index", extract_and_index_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "extract_and_index")
    builder.add_edge("extract_and_index", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


studio_graph = build_ingest_graph
