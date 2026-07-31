"""SQLite access layer.

One connection per thread (SQLite objects are not safely shared across
threads), WAL so readers never block the ingest writer, and a generous
busy_timeout because bounded-parallel extraction means several writers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SCHEMA_SQL = Path(__file__).with_name("schema.sql")

_local = threading.local()

PAGE_TERMINAL = ("done", "skipped")


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")


def get_conn(db_path: Path) -> sqlite3.Connection:
    """Thread-local connection, initialising the schema on first use."""
    key = str(db_path)
    cache: dict[str, sqlite3.Connection] | None = getattr(_local, "conns", None)
    if cache is None:
        cache = _local.conns = {}

    conn = cache.get(key)
    if conn is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(key, timeout=15.0, isolation_level=None)
        _configure(conn)
        conn.executescript(_SCHEMA_SQL.read_text(encoding="utf-8"))
        cache[key] = conn
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction (connections run in autocommit otherwise)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def upsert_document(
    conn: sqlite3.Connection,
    *,
    sha256: str,
    filename: str,
    path: str,
    page_count: int,
    byte_size: int,
    pipeline_version: str,
    title: str | None = None,
) -> sqlite3.Row:
    """Insert or fetch the document row for this (sha256, pipeline_version).

    Returning the existing row unchanged is what makes re-running `ingest` on
    the same file cheap instead of destructive.
    """
    return conn.execute(
        """
        INSERT INTO documents
            (sha256, filename, path, page_count, byte_size, pipeline_version, title)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (sha256, pipeline_version) DO UPDATE SET
            path       = excluded.path,
            filename   = excluded.filename,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
        RETURNING *
        """,
        (sha256, filename, path, page_count, byte_size, pipeline_version, title),
    ).fetchone()


def get_document(conn: sqlite3.Connection, doc_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()


def set_document_status(
    conn: sqlite3.Connection,
    doc_id: int,
    status: str,
    *,
    error: str | None = None,
    embed_model: str | None = None,
    embed_dim: int | None = None,
) -> None:
    """Move a document's status; the recorded embedder survives, the error does not.

    embed_model/embed_dim are COALESCEd because finalize_node re-sets the status
    with neither of them, and a plain assignment would blank the embedder the
    document was actually indexed with. error is assigned outright, so a
    successful re-run clears the previous failure.
    """
    conn.execute(
        """
        UPDATE documents
           SET status=?, error=?,
               embed_model=COALESCE(?, embed_model),
               embed_dim=COALESCE(?, embed_dim),
               updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
         WHERE id=?
        """,
        (status, error, embed_model, embed_dim, doc_id),
    )


def ensure_pages(conn: sqlite3.Connection, doc_id: int, page_count: int) -> None:
    """Create the page rows once; existing rows keep their state (resume)."""
    with transaction(conn):
        conn.executemany(
            "INSERT OR IGNORE INTO pages (doc_id, page_no) VALUES (?, ?)",
            [(doc_id, n) for n in range(1, page_count + 1)],
        )


def save_routing(
    conn: sqlite3.Connection,
    doc_id: int,
    rows: Sequence[dict[str, Any]],
    *,
    max_attempts: int,
) -> None:
    """Persist router output for a batch of pages.

    Routing runs on every ingest (it is cheap, and thresholds may have changed),
    so it must never demote a page whose outcome is already settled. Two cases:

    * a finished page reset to 'routed' would be re-extracted, which re-does the
      whole document on every re-run; and
    * a page that has exhausted its retries, reset to 'routed' with its error
      cleared, disappears from the failure count entirely - the run then reports
      success while silently having dropped those pages.
    """
    settled = (
        f"status IN ({', '.join('?' * len(PAGE_TERMINAL))}) "
        "OR (status = 'failed' AND attempts >= ?)"
    )
    params = [*PAGE_TERMINAL, max_attempts]
    with transaction(conn):
        conn.executemany(
            f"""
            UPDATE pages
               SET route=?,
                   status=CASE WHEN {settled} THEN status ELSE 'routed' END,
                   width=?, height=?, rotation=?, n_chars=?, text_cover=?,
                   image_cover=?, quality=?,
                   error=CASE WHEN {settled} THEN error ELSE NULL END,
                   updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE doc_id=? AND page_no=?
            """,
            [
                (
                    r["route"],
                    *params,
                    r["width"],
                    r["height"],
                    r["rotation"],
                    r["n_chars"],
                    r["text_cover"],
                    r["image_cover"],
                    r["quality"],
                    *params,
                    doc_id,
                    r["page_no"],
                )
                for r in rows
            ],
        )


def set_page_status(
    conn: sqlite3.Connection,
    doc_id: int,
    page_no: int,
    status: str,
    *,
    error: str | None = None,
    n_chunks: int | None = None,
    bump_attempts: bool = False,
) -> None:
    """Move a page's status, spending a retry attempt only when asked.

    attempts counts failures, not visits. pending_pages skips anything at
    attempts >= max_attempts and save_routing then refuses to reset it, so a
    caller that bumped on every status change would retire pages that never
    actually failed.
    """
    conn.execute(
        """
        UPDATE pages
           SET status=?, error=?,
               n_chunks=COALESCE(?, n_chunks),
               attempts=attempts + ?,
               updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
         WHERE doc_id=? AND page_no=?
        """,
        (status, error, n_chunks, 1 if bump_attempts else 0, doc_id, page_no),
    )


def get_pages(
    conn: sqlite3.Connection,
    doc_id: int,
    *,
    statuses: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM pages WHERE doc_id=?"
    params: list[Any] = [doc_id]
    if statuses:
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        params.extend(statuses)
    return conn.execute(sql + " ORDER BY page_no", params).fetchall()


def pending_pages(
    conn: sqlite3.Connection, doc_id: int, max_attempts: int
) -> list[sqlite3.Row]:
    """Pages still needing work, skipping ones that exhausted their retries."""
    placeholders = ",".join("?" * len(PAGE_TERMINAL))
    return conn.execute(
        f"""
        SELECT * FROM pages
         WHERE doc_id=?
           AND status NOT IN ({placeholders})
           AND attempts < ?
         ORDER BY page_no
        """,
        (doc_id, *PAGE_TERMINAL, max_attempts),
    ).fetchall()


def status_counts(conn: sqlite3.Connection, doc_id: int) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM pages WHERE doc_id=? GROUP BY status",
        (doc_id,),
    ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def route_counts(conn: sqlite3.Connection, doc_id: int) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT COALESCE(route,'unrouted') AS route, COUNT(*) AS n
          FROM pages WHERE doc_id=? GROUP BY route
        """,
        (doc_id,),
    ).fetchall()
    return {r["route"]: r["n"] for r in rows}


def replace_page_chunks(
    conn: sqlite3.Connection, doc_id: int, page_no: int, chunks: Sequence[dict[str, Any]]
) -> None:
    """Swap in a page's chunks atomically so a retry cannot duplicate rows."""
    with transaction(conn):
        conn.execute("DELETE FROM chunks WHERE doc_id=? AND page_no=?", (doc_id, page_no))
        if chunks:
            conn.executemany(
                """
                INSERT INTO chunks
                    (id, doc_id, page_no, page_span, kind, section, bbox, asset_id)
                VALUES
                    (:id, :doc_id, :page_no, :page_span, :kind, :section, :bbox,
                     :asset_id)
                -- Chunk ids are a hash of (doc, version, page, kind, text), so two
                -- chunks collide only when they are genuinely the same text on the
                -- same page. Dedup deliberately keeps short fragments ("Q1", "N/A")
                -- because they fuzzy-match everything, so a table of repeated cells
                -- really does produce collisions. LanceDB merges them; without this
                -- the plain INSERT raised IntegrityError and killed the whole run.
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    {
                        "doc_id": doc_id,
                        "page_no": page_no,
                        "page_span": json.dumps(c.get("page_span") or [page_no]),
                        "bbox": json.dumps(c["bbox"]) if c.get("bbox") else None,
                        "asset_id": c.get("asset_id"),
                        "section": c.get("section"),
                        "kind": c.get("kind", "text"),
                        "id": c["id"],
                    }
                    for c in chunks
                ],
            )


def count_chunks(conn: sqlite3.Connection, doc_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE doc_id=?", (doc_id,)
    ).fetchone()["n"]


def insert_asset(
    conn: sqlite3.Connection,
    *,
    doc_id: int,
    page_no: int,
    kind: str,
    path: str,
    bbox: list[float] | None,
    ocr_text: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO assets (doc_id, page_no, kind, path, bbox, ocr_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (doc_id, page_no, kind, path, json.dumps(bbox) if bbox else None, ocr_text),
    )
    return int(cur.lastrowid or 0)


def delete_page_assets(conn: sqlite3.Connection, doc_id: int, pages: Sequence[int]) -> None:
    """Clear a page's asset rows before re-extracting it, so retries cannot stack."""
    if not pages:
        return
    placeholders = ",".join("?" * len(pages))
    conn.execute(
        f"DELETE FROM assets WHERE doc_id=? AND page_no IN ({placeholders})",
        (doc_id, *pages),
    )


def document_asset_paths(conn: sqlite3.Connection, doc_id: int) -> list[str]:
    """Every figure crop belonging to a document, as paths under assets_dir.

    Read-only on purpose. Deleting the rows here would commit immediately -
    connections run in autocommit - so any later step that failed would strand
    the PNG files on disk with nothing left pointing at them.
    """
    return [
        r["path"]
        for r in conn.execute("SELECT path FROM assets WHERE doc_id=?", (doc_id,))
        if r["path"]
    ]


def upsert_job(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    doc_id: int | None,
    kind: str,
    status: str,
    pages_total: int = 0,
) -> None:
    """Create or re-queue a job row.

    doc_id is COALESCEd rather than assigned: an upload queues the job before the
    document row exists and passes NULL, so the id only arrives on the graph's
    later call, and re-queueing that same upload passes NULL again. A plain
    assignment blanked it, leaving the progress stream unable to report pages.
    Re-queueing clears the previous outcome so a job that succeeds on retry does
    not still report the old error text.
    """
    conn.execute(
        """
        INSERT INTO jobs (id, doc_id, kind, status, pages_total, started_at)
        VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        ON CONFLICT (id) DO UPDATE SET
            status=excluded.status,
            pages_total=excluded.pages_total,
            -- An upload queues the job before the document row exists, so the
            -- id only arrives on the graph's later call. COALESCE, not a plain
            -- assignment: re-queueing that same upload passes NULL again and
            -- must not blank an id we already learned. Without this the row
            -- keeps doc_id NULL and the progress stream never reports pages.
            doc_id=COALESCE(excluded.doc_id, doc_id),
            -- Re-queueing clears the previous outcome; otherwise a job that
            -- succeeds on retry still reports the old error text.
            message=NULL,
            error=NULL,
            finished_at=NULL,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
        """,
        (job_id, doc_id, kind, status, pages_total),
    )


def update_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    status: str | None = None,
    pages_done: int | None = None,
    message: str | None = None,
    error: str | None = None,
    finished: bool = False,
) -> None:
    """Apply a partial job update; None leaves a column as it was.

    Progress ticks carry only pages_done, so every column COALESCEs - which also
    means this can never clear an error. upsert_job does that when a job is
    re-queued.
    """
    conn.execute(
        """
        UPDATE jobs
           SET status=COALESCE(?, status),
               pages_done=COALESCE(?, pages_done),
               message=COALESCE(?, message),
               error=COALESCE(?, error),
               finished_at=CASE WHEN ? THEN strftime('%Y-%m-%dT%H:%M:%fZ','now')
                                ELSE finished_at END,
               updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
         WHERE id=?
        """,
        (status, pages_done, message, error, 1 if finished else 0, job_id),
    )


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
