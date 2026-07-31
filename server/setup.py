"""First-run model provisioning.

The embedder and reranker are roughly 2.3 GB each and are otherwise fetched
lazily, on the first ingest, with no output at all. In a terminal that reads as
a long pause; in a desktop window it is indistinguishable from a hang, and the
usual response is to force-quit halfway through and corrupt the cache.

So the desktop shell provisions them up front and shows real progress. Nothing
here is required by the CLI, which is happy to download lazily.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .settings import Settings, get_settings
from .util.logging import get_logger

log = get_logger(__name__)

# Sentinel closing the progress queue; None is a legitimate payload elsewhere.
_DONE = object()
# How often to sample on-disk size while a download is in flight.
_POLL_S = 0.5


def required_models(cfg: Settings | None = None) -> list[tuple[str, str]]:
    """(role, repo_id) for every model the app cannot run without."""
    cfg = cfg or get_settings()
    return [("embeddings", cfg.embed_model), ("reranker", cfg.rerank_model)]


def is_cached(repo_id: str) -> bool:
    """Is this repo fully downloaded already?

    Asks huggingface_hub to resolve the snapshot offline rather than testing for
    the directory: a run interrupted mid-download leaves the folder in place
    with blobs missing, which a directory check reports as ready and which then
    fails much later, during an ingest.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any failure means "fetch it"
        return False


def missing_models(cfg: Settings | None = None) -> list[tuple[str, str]]:
    return [(role, repo) for role, repo in required_models(cfg) if not is_cached(repo)]


def _repo_bytes(repo_id: str) -> int:
    """Total size of a repo, for the progress denominator.

    Read from the API rather than hooked out of huggingface_hub's own progress
    bars: it drives those with a mix of byte counters and a file *counter*, and
    a hook cannot reliably tell them apart, so totals came out nonsensical.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        return sum(f.size or 0 for f in (info.siblings or []))
    except Exception:  # noqa: BLE001 - progress is a nicety, never a blocker
        return 0


def _cache_bytes(repo_id: str) -> int:
    """Bytes on disk for this repo so far."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        folder = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
        if not folder.exists():
            return 0
        # Skip symlinks: the cache stores each file once as a blob and again as
        # a snapshot symlink pointing at it, and is_file() follows the link, so
        # counting both reported exactly double the real size.
        return sum(
            f.stat().st_size
            for f in folder.rglob("*")
            if f.is_file() and not f.is_symlink()
        )
    except Exception:  # noqa: BLE001
        return 0


def download_models(cfg: Settings | None = None) -> Iterator[dict[str, Any]]:
    """Fetch every missing model, yielding progress events as they arrive.

    The download runs on a worker thread because huggingface_hub blocks; the
    queue is what turns it back into something streamable to a progress bar.
    """
    cfg = cfg or get_settings()
    pending = missing_models(cfg)
    if not pending:
        yield {"stage": "complete", "downloaded": 0}
        return

    sink: queue.Queue = queue.Queue()
    sizes = {repo: _repo_bytes(repo) for _role, repo in pending}
    grand_total = sum(sizes.values())

    def worker() -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        # Its bars would otherwise render into the app's log stream.
        disable_progress_bars()
        try:
            for role, repo in pending:
                sink.put({"stage": "start", "role": role, "repo": repo})
                snapshot_download(repo)
                sink.put({"stage": "model_done", "role": role, "repo": repo})
            sink.put({"stage": "complete", "downloaded": len(pending)})
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
            log.warning("model download failed", error=str(exc))
            sink.put({"stage": "error", "detail": f"{type(exc).__name__}: {exc}"})
        finally:
            sink.put(_DONE)

    thread = threading.Thread(target=worker, daemon=True, name="model-download")
    thread.start()

    done_repos: set[str] = set()
    while True:
        try:
            event = sink.get(timeout=_POLL_S)
        except queue.Empty:
            yield _progress(pending, sizes, grand_total, done_repos)
            continue
        if event is _DONE:
            return
        if event.get("stage") == "model_done":
            done_repos.add(event["repo"])
        yield event


def _progress(
    pending: list[tuple[str, str]],
    sizes: dict[str, int],
    grand_total: int,
    done_repos: set[str],
) -> dict[str, Any]:
    """Aggregate on-disk progress across every model still being fetched."""
    got = sum(
        sizes[repo] if repo in done_repos else min(_cache_bytes(repo), sizes[repo] or 1 << 62)
        for _role, repo in pending
    )
    # Held below 100 until the worker actually confirms: the repo total counts
    # every file in the repo, and a snapshot may legitimately fetch fewer.
    percent = min(99.0, round(100 * got / grand_total, 1)) if grand_total else 0.0
    return {
        "stage": "download",
        "done_bytes": got,
        "total_bytes": grand_total,
        "percent": percent,
    }


def status(cfg: Settings | None = None) -> dict[str, Any]:
    """What the setup screen needs: which models are here, which are not."""
    cfg = cfg or get_settings()
    models = [
        {"role": role, "repo": repo, "cached": is_cached(repo)}
        for role, repo in required_models(cfg)
    ]
    return {"models": models, "ready": all(m["cached"] for m in models)}
