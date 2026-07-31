"""Retrievers over the LanceDB chunk store.

Dense and full-text are exposed as separate LangChain retrievers so that
EnsembleRetriever can fuse them with reciprocal rank fusion, and
ContextualCompressionRetriever can layer the cross-encoder on top. Both are
reused rather than reimplemented; the custom part is only the LanceDB access,
which is where control over filters and scores is needed.

Why hybrid rather than dense alone: dense embeddings are weak at exact tokens -
part numbers, section references, "42 Nm", a name spelled one way - which are
precisely what someone looks up in a technical document. BM25 nails those and
misses paraphrases; each covers the other's blind spot.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..settings import Settings, get_settings
from .store import ChunkStore, SearchHit

ID_KEY = "chunk_id"


def _stable(hits: list[SearchHit]) -> list[SearchHit]:
    """Deterministic order within one retriever.

    Safe here and only here: a single retriever's scores share one scale, so
    breaking ties on the chunk id changes nothing but reproducibility. Rebuilding
    the full-text index otherwise reshuffles equally-scoring hits, which moves
    eval numbers between runs for no real reason.
    """
    return sorted(hits, key=lambda h: (-h.score, h.id))


def hit_to_document(hit: SearchHit, *, source: str) -> Document:
    return Document(
        page_content=hit.text,
        metadata={
            ID_KEY: hit.id,
            "doc_id": hit.doc_id,
            "page_no": hit.page_no,
            "page_span": hit.page_span,
            "kind": hit.kind,
            "section": hit.section,
            "bbox": hit.bbox,
            "asset_path": hit.asset_path,
            "retrieval_source": source,
            "retrieval_score": hit.score,
        },
    )


class LanceDenseRetriever(BaseRetriever):
    """Vector similarity over bge-m3 embeddings."""

    store: ChunkStore
    embeddings: Any
    k: int = 50
    doc_ids: list[int] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        hits = self.store.dense_search(vector, self.k, doc_ids=self.doc_ids)
        return [hit_to_document(h, source="dense") for h in _stable(hits)]


class LanceFtsRetriever(BaseRetriever):
    """BM25 over LanceDB's full-text index."""

    store: ChunkStore
    k: int = 50
    doc_ids: list[int] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        hits = self.store.fts_search(query, self.k, doc_ids=self.doc_ids)
        return [hit_to_document(h, source="fts") for h in _stable(hits)]


def build_retrieval_pipeline(
    cfg: Settings | None = None,
    *,
    doc_ids: list[int] | None = None,
    rerank: bool = True,
    dense_only: bool = False,
    fts_only: bool = False,
    top_n: int | None = None,
) -> Any:
    """The full retrieve-then-rerank chain as a single LangChain retriever.

    Dense + BM25 are fused with RRF. dense_only / fts_only exist for the eval
    harness's ablation mode, so the claim "hybrid beats dense" is measured rather
    than assumed.

    id_key matters: EnsembleRetriever otherwise fuses on page_content, and
    repeated table cells give distinct chunks identical text - two pages' hits
    would collapse into one and a citation would vanish.
    """
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
    )

    from .embed import get_embeddings
    from .rerank import build_reranker

    cfg = cfg or get_settings()
    store = ChunkStore(cfg)
    if fts_only:
        base: Any = LanceFtsRetriever(store=store, k=cfg.fts_k, doc_ids=doc_ids)
    else:
        base = LanceDenseRetriever(
            store=store, embeddings=get_embeddings(cfg), k=cfg.dense_k, doc_ids=doc_ids
        )
        if not dense_only:
            base = EnsembleRetriever(
                retrievers=[base, LanceFtsRetriever(store=store, k=cfg.fts_k, doc_ids=doc_ids)],
                weights=[0.5, 0.5],
                c=cfg.rrf_k,
                id_key=ID_KEY,
            )
    if not rerank:
        return base
    return ContextualCompressionRetriever(
        base_compressor=build_reranker(cfg, top_n=top_n), base_retriever=base
    )
