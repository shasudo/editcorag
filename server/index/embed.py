"""Embedding model access.

Wrapped in langchain-huggingface's Embeddings interface so the same object can
be handed to retrievers and to the graph, and swapped for any other LangChain
embeddings implementation without touching call sites.

The model is a process-level singleton: bge-m3 is 2.3 GB and on a 16 GB machine
loading it twice is the difference between working and swapping.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..settings import Settings, get_settings
from ..util.device import resolve_device
from ..util.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=2)
def _load(model_name: str, device: str, normalize: bool, batch_size: int, max_seq: int) -> Any:
    """Load and cache the embeddings handle, keyed on the tuning, not on cfg.

    Settings is not hashable, so lru_cache gets the scalars instead. max_seq is
    forced down from bge-m3's 8192 default: chunks are capped at 600 tokens, so
    the lower ceiling only bounds what one pathological chunk can cost a batch.
    It is a private attribute with no setter, hence best-effort.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("loading embedding model", model=model_name, device=device)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize, "batch_size": batch_size},
    )
    try:
        embeddings._client.max_seq_length = max_seq  # noqa: SLF001
    except Exception:  # noqa: BLE001 - purely an optimisation
        pass
    return embeddings


def get_embeddings(cfg: Settings | None = None) -> Any:
    cfg = cfg or get_settings()
    return _load(
        cfg.embed_model,
        resolve_device(cfg.embed_device),
        cfg.embed_normalize,
        cfg.embed_batch_size,
        cfg.embed_max_seq_length,
    )


def embed_dimension(cfg: Settings | None = None) -> int:
    """Actual width of the loaded model, not the configured guess.

    A mismatch between this and the index's stored dimension is a silent
    corruption source, so callers validate against it rather than trusting
    settings.
    """
    return len(get_embeddings(cfg).embed_query("dimension probe"))
