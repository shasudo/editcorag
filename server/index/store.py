"""LanceDB vector + full-text store.

One embedded, on-disk table holding both the dense vectors and a full-text
index, which is what lets hybrid retrieval run with no server process.

Two design points worth stating:

* **Upserts are idempotent.** Rows are merged on the deterministic chunk id, so
  re-ingesting a document (or retrying a failed batch) updates rows in place
  instead of duplicating them. Duplicate rows would distort BM25 term
  statistics for every other query, not just the affected chunk.
* **The embedding model and dimension are recorded in the table.** Querying an
  index built with a different embedder returns plausible nonsense rather than
  an error, so the mismatch is detected explicitly on open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..settings import Settings
from ..util.logging import get_logger

log = get_logger(__name__)

TABLE_NAME = "chunks"
_META_TABLE = "index_meta"


@dataclass
class SearchHit:
    id: str
    doc_id: int
    page_no: int
    page_span: list[int]
    kind: str
    section: str | None
    text: str
    bbox: list[float] | None
    asset_path: str | None
    score: float


def _doc_filter(doc_ids: list[int]) -> str:
    """SQL predicate restricting a search to specific documents."""
    return f"doc_id IN ({', '.join(str(int(d)) for d in doc_ids)})"


def _schema(dim: int) -> Any:
    """Arrow schema for the chunk table.

    `text` is what BM25 indexes and what the reader is shown; `embed_text` is the
    heading-prefixed variant that was actually embedded, so the vector carries
    section context without the heading text skewing BM25 term statistics. The
    vector width is fixed here, which is why a changed embedding dimension needs
    a new table rather than a migration.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("doc_id", pa.int64(), nullable=False),
            pa.field("doc_sha", pa.string(), nullable=False),
            pa.field("page_no", pa.int32(), nullable=False),
            pa.field("page_span", pa.list_(pa.int32())),
            pa.field("kind", pa.string()),
            pa.field("section", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embed_text", pa.string()),
            pa.field("bbox", pa.list_(pa.float32(), 4)),
            pa.field("asset_path", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
        ]
    )


class ChunkStore:
    """Thin, explicit wrapper over the LanceDB table."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._db: Any = None
        self._table: Any = None

    @property
    def db(self) -> Any:
        if self._db is None:
            import lancedb

            self.cfg.lance_dir.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.cfg.lance_dir))
        return self._db

    def exists(self) -> bool:
        return TABLE_NAME in self.db.table_names()

    def _open_table(self) -> Any:
        """Table handle with no compatibility check.

        Removing rows does not depend on which embedder wrote them, and checking
        here would make a changed embed model unrecoverable: _verify_compatible's
        own remedy is to re-ingest, and re-ingesting starts by deleting.
        """
        if self._table is None:
            self._table = self.db.open_table(TABLE_NAME)
        return self._table

    def open(self, *, create_with_dim: int | None = None) -> Any:
        """Open the table, creating it for the given vector width if absent."""
        if self._table is not None:
            return self._table

        if self.exists():
            table = self.db.open_table(TABLE_NAME)
            self._verify_compatible()
            self._table = table
            return self._table

        if create_with_dim is None:
            raise RuntimeError("index does not exist yet and no embedding dimension was given")
        self._table = self.db.create_table(
            TABLE_NAME, schema=_schema(create_with_dim), exist_ok=True
        )
        self._write_meta(create_with_dim)
        log.info("created chunk index", dim=create_with_dim, path=str(self.cfg.lance_dir))
        return self._table

    def _write_meta(self, dim: int) -> None:
        import pyarrow as pa

        schema = pa.schema(
            [pa.field("key", pa.string()), pa.field("value", pa.string())]
        )
        rows = [
            {"key": "embed_model", "value": self.cfg.embed_model},
            {"key": "embed_dim", "value": str(dim)},
            {"key": "pipeline_version", "value": self.cfg.pipeline_version},
        ]
        table = self.db.create_table(_META_TABLE, schema=schema, exist_ok=True)
        table.merge_insert("key").when_matched_update_all().when_not_matched_insert_all().execute(
            rows
        )

    def meta(self) -> dict[str, str]:
        if _META_TABLE not in self.db.table_names():
            return {}
        rows = self.db.open_table(_META_TABLE).search().limit(50).to_list()
        return {r["key"]: r["value"] for r in rows}

    def _verify_compatible(self) -> None:
        """Refuse to mix embedders: the failure would otherwise be invisible."""
        meta = self.meta()
        stored_model = meta.get("embed_model")
        if stored_model and stored_model != self.cfg.embed_model:
            raise RuntimeError(
                f"index was built with embeddings {stored_model!r} but settings say "
                f"{self.cfg.embed_model!r}. Re-ingest, or point LOCALRAG_EMBED_MODEL back. "
                f"Mixing embedders returns confident nonsense rather than an error."
            )

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        table = self.open()
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        return len(rows)

    def delete_document(self, doc_id: int) -> None:
        if not self.exists():
            return
        self._open_table().delete(f"doc_id = {int(doc_id)}")

    def delete_pages(self, doc_id: int, pages: list[int]) -> None:
        """Clear a page's rows before rewriting them, so a retry cannot double up."""
        if not pages or not self.exists():
            return
        page_list = ", ".join(str(int(p)) for p in pages)
        self._open_table().delete(f"doc_id = {int(doc_id)} AND page_no IN ({page_list})")

    def count(self) -> int:
        return self.open().count_rows() if self.exists() else 0

    def ensure_fts_index(self) -> None:
        """Build the full-text index, falling back to the 'simple' tokeniser.

        The configured tokeniser may not be available in the installed lancedb build.
        Raising here would leave the table with no full-text index at all, silently
        degrading hybrid retrieval to dense-only - losing exactly the exact-token
        recall BM25 is here for - so a failure downgrades the tokeniser instead.
        """
        from lancedb.index import FTS

        table = self.open()
        config = FTS(
            base_tokenizer=self.cfg.fts_tokenizer,
            language=self.cfg.fts_language,
            stem=self.cfg.fts_stem,
            remove_stop_words=self.cfg.fts_remove_stopwords,
            ascii_folding=True,
            lower_case=True,
        )
        try:
            table.create_index("text", config=config, replace=True)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "FTS index with configured tokenizer failed; falling back to 'simple'",
                tokenizer=self.cfg.fts_tokenizer,
                error=str(exc),
            )
            table.create_index(
                "text",
                config=FTS(base_tokenizer="simple", stem=False, remove_stop_words=False),
                replace=True,
            )

    def ensure_vector_index(self) -> None:
        """Create an ANN index only once brute force stops being the better option."""
        from lancedb.index import IvfPq

        table = self.open()
        rows = table.count_rows()
        if rows < self.cfg.ann_index_min_rows:
            log.debug("skipping ANN index; brute force is exact and faster", rows=rows)
            return
        table.create_index("vector", config=IvfPq(distance_type="cosine"), replace=True)

    def _to_hits(self, rows: list[dict[str, Any]], *, dense: bool) -> list[SearchHit]:
        """Normalise LanceDB rows into SearchHit, higher score always better.

        Dense search reports a distance (lower is better) and FTS a BM25 score (higher
        is better); mapping the distance through 1/(1+d) puts both on one descending
        scale, so ranking and tie-breaking downstream never need to know which
        retriever produced a hit.
        """
        hits: list[SearchHit] = []
        for row in rows:
            if dense:
                distance = float(row.get("_distance", 0.0) or 0.0)
                score = 1.0 / (1.0 + max(0.0, distance))
            else:
                score = float(row.get("_score", 0.0) or 0.0)
            span = list(row.get("page_span") or [row.get("page_no", 0)])
            bbox = row.get("bbox")
            hits.append(
                SearchHit(
                    id=row["id"],
                    doc_id=int(row.get("doc_id", 0)),
                    page_no=int(row.get("page_no", 0)),
                    page_span=[int(p) for p in span],
                    kind=row.get("kind") or "text",
                    section=row.get("section") or None,
                    text=row.get("text") or "",
                    bbox=[float(v) for v in bbox] if bbox is not None else None,
                    asset_path=row.get("asset_path") or None,
                    score=score,
                )
            )
        return hits

    def dense_search(
        self, vector: list[float], k: int, *, doc_ids: list[int] | None = None
    ) -> list[SearchHit]:
        if not self.exists():
            return []
        query = self.open().search(vector, vector_column_name="vector").limit(k)
        if doc_ids:
            query = query.where(_doc_filter(doc_ids))
        return self._to_hits(query.to_list(), dense=True)

    def fts_search(
        self, text: str, k: int, *, doc_ids: list[int] | None = None
    ) -> list[SearchHit]:
        if not self.exists() or not text.strip():
            return []
        try:
            query = self.open().search(text, query_type="fts").limit(k)
            if doc_ids:
                query = query.where(_doc_filter(doc_ids))
            return self._to_hits(query.to_list(), dense=False)
        except Exception as exc:  # noqa: BLE001 - missing index must not kill a query
            log.warning("full-text search unavailable", error=str(exc))
            return []


def rows_from_chunks(
    chunks: list[Any], *, doc_id: int, doc_sha: str, vectors: list[list[float]]
) -> list[dict[str, Any]]:
    """ChunkOut + vectors -> LanceDB rows."""
    rows: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        rows.append(
            {
                "id": chunk.id,
                "doc_id": doc_id,
                "doc_sha": doc_sha,
                "page_no": int(chunk.page_no),
                "page_span": [int(p) for p in (chunk.page_span or [chunk.page_no])],
                "kind": chunk.kind,
                "section": chunk.section,
                "text": chunk.text,
                "embed_text": chunk.embed_text,
                "bbox": [float(v) for v in chunk.bbox] if chunk.bbox else None,
                "asset_path": chunk.asset_path,
                "vector": vector,
            }
        )
    return rows


def sqlite_chunk_rows(
    chunks: list[Any], asset_ids: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """ChunkOut -> the dict shape db.store.replace_page_chunks expects.

    asset_ids maps an asset path to its row id, so a figure chunk links to the
    cropped image it was read from.
    """
    lookup = asset_ids or {}
    return [
        {
            "id": c.id,
            "kind": c.kind,
            "section": c.section,
            "bbox": c.bbox,
            "page_span": c.page_span,
            "asset_id": lookup.get(c.asset_path) if c.asset_path else None,
        }
        for c in chunks
    ]
