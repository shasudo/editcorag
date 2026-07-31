"""Cross-encoder reranking that keeps its scores.

LangChain ships a CrossEncoderReranker, but it sorts by score and then throws
the scores away:

    result = sorted(docs_with_scores, key=itemgetter(1), reverse=True)
    return [doc for doc, _ in result[: self.top_n]]

Relative order alone is not enough here. Abstention needs the *absolute*
relevance of the best hit: "these are the 8 least-bad chunks" and "these 8
chunks actually answer the question" are indistinguishable from ordering, and
telling them apart is what stops the system inventing an answer when the
document simply does not contain one.

So this subclasses langchain-core's BaseDocumentCompressor directly (also
avoiding langchain-community, which upstream has declared sunset) and writes
each score into document metadata as `rerank_score`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from ..settings import Settings, get_settings
from ..util.device import resolve_device
from ..util.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str) -> Any:
    """Load a cross-encoder once per (model, device).

    max_length=512 truncates the query+chunk pair, so a chunk well past that
    budget is scored on its opening alone - chunk size and rerank quality are
    coupled.
    """
    from sentence_transformers import CrossEncoder

    log.info("loading reranker", model=model_name, device=device)
    return CrossEncoder(model_name, device=device, max_length=512)


class ScoringCrossEncoderReranker(BaseDocumentCompressor):
    """Rerank documents and record each score in `metadata["rerank_score"]`.

    bge-reranker-v2-m3 has a single output label, so sentence-transformers
    applies a sigmoid and scores land in 0..1 - directly comparable against
    settings.abstain_score_min.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 8
    batch_size: int = 16
    device: str = "auto"

    def _model(self) -> Any:
        return _load_cross_encoder(self.model_name, resolve_device(self.device))

    def compress_documents(
        self,
        documents: Any,
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        docs = list(documents)
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        ranked = sorted(
            zip(docs, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        out: list[Document] = []
        for doc, score in ranked[: self.top_n]:
            out.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_score": float(score)},
                )
            )
        return out


def build_reranker(
    cfg: Settings | None = None, *, top_n: int | None = None
) -> ScoringCrossEncoderReranker:
    cfg = cfg or get_settings()
    return ScoringCrossEncoderReranker(
        model_name=cfg.rerank_model,
        top_n=top_n if top_n is not None else cfg.rerank_top_n,
        batch_size=cfg.rerank_batch_size,
        device=cfg.rerank_device,
    )
