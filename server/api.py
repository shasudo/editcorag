"""FastAPI application: REST + SSE, and it serves the built SPA.

One process runs everything, which is the point - the whole system is a single
`editcorag serve` with no database server, no broker and no container.

Streaming uses two channels off the same graph run:

* ``custom`` events, which the graph nodes emit, drive the "retrieving ->
  reranking -> verifying" status the UI shows; and
* ``messages`` events carry the answer tokens.

Job progress is read from SQLite rather than an in-memory queue, so progress
survives a server restart and several browser tabs can watch the same ingest.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from . import __version__
from .db import store as db
from .llm.models import message_text
from .settings import AnswerMode, get_settings
from .util.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up: logging, data directories, then the SQLite schema.

    get_conn is called for its side effect - connections are thread-local, so
    this one is never reused - but it creates the schema before the first
    request, rather than leaving several handler threads to race on it.
    """
    from . import setup as provisioning

    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    cfg.ensure_dirs()
    # Re-apply a key saved by an earlier session before anything reads settings.
    provisioning.load_api_key(cfg)
    cfg = get_settings()
    db.get_conn(cfg.sqlite_path)
    log.info("api ready", data_dir=str(cfg.data_dir), model=cfg.llm_model)
    yield


app = FastAPI(title="EditcoRAG", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str | None = None
    doc_ids: list[int] = Field(default_factory=list)
    mode: AnswerMode | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


def _doc_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "sha256": row["sha256"],
        "pages": row["page_count"],
        "bytes": row["byte_size"],
        "title": row["title"],
        "status": row["status"],
        "error": row["error"],
        "embed_model": row["embed_model"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness, store sizes, and whether the app can actually answer yet.

    This is what the desktop shell polls to decide between the setup screen and
    the app, so it must never raise: every part is reported independently and a
    failure becomes a field, not a 500. It previously returned ok:true without
    consulting the LLM at all, so the UI claimed to be ready while nothing
    could answer.

    Sync on purpose, so FastAPI runs it in a worker thread. Every part of it
    blocks - importing lancedb, the httpx call in llm_status, the snapshot
    scans - and as an async endpoint it froze the loop for seconds at a time,
    which meant the shell's own start-up probes timed out against a server that
    was running perfectly well.
    """
    from . import setup as provisioning
    from .index.store import ChunkStore
    from .llm.models import llm_status

    cfg = get_settings()

    documents = chunks = 0
    index_error: str | None = None
    try:
        conn = db.get_conn(cfg.sqlite_path)
        documents = len(db.list_documents(conn))
        chunks = ChunkStore(cfg).count()
    except Exception as exc:  # noqa: BLE001 - a broken index must not hide the rest
        index_error = str(exc)

    llm = llm_status(cfg)
    models = provisioning.status(cfg)
    return {
        "ok": True,
        "ready": bool(models["ready"] and llm.get("status") == "ready" and not index_error),
        "documents": documents,
        "chunks": chunks,
        "index_error": index_error,
        "llm": llm,
        "models": models,
        "llm_model": cfg.llm_model,
        "embed_model": cfg.embed_model,
        "answer_mode": cfg.answer_mode,
        "graders_enabled": cfg.enable_graders,
    }


def _stream_blocking(
    request: Request, produce: Callable[[], Iterator[dict[str, Any]]]
) -> EventSourceResponse:
    """Relay a blocking generator as SSE without stalling the event loop.

    Both provisioning steps are network-bound synchronous code. Iterating them
    directly in an async endpoint would freeze the loop, and so freeze the very
    progress bar they exist to feed.
    """

    async def events() -> AsyncIterator[dict[str, str]]:
        loop = asyncio.get_running_loop()
        relay: asyncio.Queue = asyncio.Queue()

        def pump() -> None:
            try:
                for event in produce():
                    loop.call_soon_threadsafe(relay.put_nowait, event)
            finally:
                loop.call_soon_threadsafe(relay.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(pump))
        try:
            while True:
                event = await relay.get()
                if event is None or await request.is_disconnected():
                    break
                yield {"event": event.get("stage", "progress"), "data": json.dumps(event)}
        finally:
            task.cancel()

    return EventSourceResponse(events())


@app.get("/api/setup/stream")
async def setup_stream(request: Request) -> EventSourceResponse:
    """Download any missing embedding and reranking models, streaming progress."""
    from . import setup as provisioning

    cfg = get_settings()
    return _stream_blocking(request, lambda: provisioning.download_models(cfg))


class ApiKey(BaseModel):
    key: str = Field(min_length=1)


@app.post("/api/setup/llm-key")
async def set_llm_key(payload: ApiKey) -> dict[str, Any]:
    """Accept a Groq API key and report whether it actually works.

    Applied to the live process rather than written to disk: the shell owns
    persistence, and a packaged app's own directory is read-only anyway. The
    settings cache and the status cache are both dropped so the very next poll
    reflects the new key instead of a stale rejection.
    """
    from . import setup as provisioning
    from .llm.models import llm_status

    provisioning.save_api_key(payload.key.strip())
    return llm_status(get_settings())


@app.get("/api/documents")
async def list_documents() -> list[dict[str, Any]]:
    """Every document, each enriched with its page, route and chunk counts.

    The counts are derived per row rather than stored, so this is the endpoint
    that reflects an ingest in progress.
    """
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    out = []
    for row in db.list_documents(conn):
        item = _doc_row(row)
        item["page_status"] = db.status_counts(conn, int(row["id"]))
        item["routes"] = db.route_counts(conn, int(row["id"]))
        item["chunks"] = db.count_chunks(conn, int(row["id"]))
        out.append(item)
    return out


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int) -> dict[str, Any]:
    """One document, with its counts and the per-page errors behind them.

    failed_pages is the detail the list endpoint omits: which pages failed and
    why, which is what turns a 'partial' status into something actionable.
    """
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    row = db.get_document(conn, doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    item = _doc_row(row)
    item["page_status"] = db.status_counts(conn, doc_id)
    item["routes"] = db.route_counts(conn, doc_id)
    item["chunks"] = db.count_chunks(conn, doc_id)
    item["failed_pages"] = [
        {"page_no": p["page_no"], "error": p["error"], "attempts": p["attempts"]}
        for p in db.get_pages(conn, doc_id, statuses=["failed"])
    ]
    return item


@app.get("/api/documents/{doc_id}/pdf")
async def get_pdf(doc_id: int) -> FileResponse:
    """Serve the original PDF so pdf.js can render citation pages client-side."""
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    row = db.get_document(conn, doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"file no longer at {path}")
    return FileResponse(path, media_type="application/pdf", filename=row["filename"])


@app.get("/api/assets/{asset_path:path}")
async def get_asset(asset_path: str) -> FileResponse:
    """Serve a cropped figure image.

    asset_path is client-supplied, so the resolved target must still sit under
    assets_dir - without that check a '../..' path would serve any file the
    server process can read.
    """
    cfg = get_settings()
    target = (cfg.assets_dir / asset_path).resolve()
    if not target.is_relative_to(cfg.assets_dir.resolve()) or not target.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(target)


def _run_ingest(pdf_path: str, job_id: str, force: bool, display_name: str = "") -> None:
    """Background ingest. Progress is recorded in SQLite by the graph itself.

    The response has already been sent, so a failure here has nowhere to
    propagate: it must land on the job row, or /api/jobs/{id}/stream polls a
    job that stays 'running' for ever.
    """
    from .graphs.ingest_graph import build_ingest_graph

    cfg = get_settings()
    try:
        build_ingest_graph().invoke(
            {
                "pdf_path": pdf_path,
                "job_id": job_id,
                "force": force,
                "display_name": display_name,
            }
        )
    except Exception as exc:
        log.exception("ingest failed", job=job_id)
        conn = db.get_conn(cfg.sqlite_path)
        db.update_job(conn, job_id, status="failed", error=str(exc)[:800], finished=True)


def _queue_ingest(
    background: BackgroundTasks,
    *,
    sha: str,
    path: str,
    doc_id: int | None,
    filename: str,
    force: bool,
) -> dict[str, Any]:
    """Queue an ingest job, unless one for the same document is already going.

    Shared by upload and reingest: the job id is derived from the content hash,
    so both entry points name the same job for the same bytes, and two runs
    would race on the same page rows.
    """
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    job_id = f"ingest-{sha[:12]}"
    running = db.get_job(conn, job_id)
    if running and running["status"] in ("queued", "running"):
        return {"job_id": job_id, "already_running": True}

    db.upsert_job(conn, job_id=job_id, doc_id=doc_id, kind="ingest", status="queued")
    background.add_task(_run_ingest, path, job_id, force, filename)
    return {"job_id": job_id}


@app.post("/api/documents")
def upload_document(
    background: BackgroundTasks, file: UploadFile, force: bool = False
) -> dict[str, Any]:
    """Accept a PDF and start ingesting it.

    Bytes land in a temp file inside pdf_dir - same directory, so the rename is
    atomic - and are only moved to <sha256>.pdf once they actually open as a
    PDF. The store therefore never holds a half-written or unreadable file, and
    the same document uploaded twice collapses onto one copy.
    """
    from .ingest.inventory import PdfOpenError, file_sha256, open_pdf

    cfg = get_settings()
    cfg.ensure_dirs()
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="only PDF files are supported")

    fd, tmp_name = tempfile.mkstemp(dir=cfg.pdf_dir, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            open_pdf(tmp).close()
        except PdfOpenError as exc:
            log.warning("rejected upload", filename=file.filename, error=str(exc))
            raise HTTPException(
                status_code=400, detail=f"{file.filename}: not a readable PDF"
            ) from None
        sha = file_sha256(tmp)
        target = cfg.pdf_dir / f"{sha}.pdf"
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

    result = _queue_ingest(
        background, sha=sha, path=str(target), doc_id=None,
        filename=file.filename or "", force=force,
    )
    return result | {"filename": file.filename}


@app.post("/api/documents/{doc_id}/reingest")
async def reingest(doc_id: int, background: BackgroundTasks, force: bool = True) -> dict[str, Any]:
    """Re-run ingest for a document already in the store."""
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    row = db.get_document(conn, doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _queue_ingest(
        background, sha=row["sha256"], path=row["path"], doc_id=doc_id,
        filename=row["filename"], force=force,
    )


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int) -> dict[str, Any]:
    """Delete a document, its chunks, its figure crops and its stored PDF.

    Ordering matters: the asset paths are read before anything that can fail, and
    the rows themselves go only with the document via ON DELETE CASCADE.
    Connections autocommit, so deleting the rows up front would strand the PNG
    files on disk with nothing left pointing at them if a later step raised. The
    PDF is removed only when it lives in our own store - an ingest run against a
    path elsewhere on disk must not delete the user's original.
    """
    cfg = get_settings()
    conn = db.get_conn(cfg.sqlite_path)
    row = db.get_document(conn, doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    from .index.store import ChunkStore

    asset_paths = db.document_asset_paths(conn, doc_id)
    pdf_path = Path(row["path"]).resolve()
    ChunkStore(cfg).delete_document(doc_id)
    with db.transaction(conn):
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    for rel in asset_paths:
        (cfg.assets_dir / rel).unlink(missing_ok=True)
    if pdf_path.is_relative_to(cfg.pdf_dir.resolve()):
        pdf_path.unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request) -> EventSourceResponse:
    """Poll SQLite and push job progress.

    Reading committed state rather than an in-memory queue means progress is
    correct for a page that reloads mid-ingest, and for two tabs at once.
    """
    cfg = get_settings()

    async def events() -> AsyncIterator[dict[str, str]]:
        last: str | None = None
        while True:
            if await request.is_disconnected():
                break
            conn = db.get_conn(cfg.sqlite_path)
            row = db.get_job(conn, job_id)
            if row is None:
                yield {"event": "error", "data": json.dumps({"detail": "unknown job"})}
                break
            payload = dict(row)
            if payload.get("doc_id"):
                payload["page_status"] = db.status_counts(conn, int(payload["doc_id"]))
            body = json.dumps(payload, default=str)
            if body != last:
                yield {"event": "progress", "data": body}
                last = body
            if row["status"] in ("done", "failed", "cancelled"):
                yield {"event": "done", "data": body}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(events())


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> EventSourceResponse:
    """Stream an answer: node status, then tokens, then the final sources."""
    cfg = get_settings()
    from .graphs.query_graph import async_query_checkpointer, build_query_graph

    thread_id = payload.thread_id or f"chat-{uuid.uuid4().hex[:12]}"

    async def events() -> AsyncIterator[dict[str, str]]:
        yield {"event": "start", "data": json.dumps({"thread_id": thread_id})}
        final: dict[str, Any] = {}
        try:
            async with async_query_checkpointer(cfg) as checkpointer:
                graph = build_query_graph(checkpointer)
                async for kind, chunk in graph.astream(
                    {
                        "question": payload.question,
                        "history": payload.history,
                        "doc_ids": payload.doc_ids,
                        "mode": payload.mode or cfg.answer_mode,
                        "rewrites": 0,
                        "regenerations": 0,
                    },
                    config={
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 40,
                    },
                    stream_mode=["custom", "messages", "values"],
                ):
                    if await request.is_disconnected():
                        break
                    if kind == "custom":
                        yield {"event": "status", "data": json.dumps(chunk, default=str)}
                    elif kind == "messages":
                        message, meta = chunk
                        text = message_text(message)
                        if text and meta.get("langgraph_node") == "generate":
                            yield {"event": "token", "data": json.dumps({"text": text})}
                    elif kind == "values":
                        final = chunk
            yield {
                "event": "final",
                "data": json.dumps(
                    {
                        "answer": final.get("answer", ""),
                        "citations": final.get("citations", []),
                        "abstained": bool(final.get("abstained", False)),
                        "grounded": final.get("grounded"),
                        "top_score": float(final.get("top_score", 0.0) or 0.0),
                        "rewrites": int(final.get("rewrites", 0) or 0),
                        "notes": final.get("notes", []),
                        "thread_id": thread_id,
                    },
                    default=str,
                ),
            }
        except Exception as exc:
            log.exception("chat failed")
            yield {"event": "error", "data": json.dumps({"detail": str(exc)[:500]})}

    return EventSourceResponse(events())


def mount_spa() -> None:
    """Serve the built frontend, if it has been built.

    Mounted at '/', so this must run after every /api route is registered - a
    StaticFiles mount at the root matches first and shadows anything added
    after it.
    """
    dist = get_settings().web_dist
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
    else:

        @app.get("/")
        async def _no_ui() -> JSONResponse:
            return JSONResponse(
                {
                    "detail": "frontend not built",
                    "hint": "cd web && pnpm install && pnpm build",
                    "api": "/docs",
                }
            )


mount_spa()
