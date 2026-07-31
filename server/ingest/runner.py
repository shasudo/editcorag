"""Batch execution with crash isolation and hard timeouts.

Why subprocesses: docling's PDF parsing, layout models and OCR are native code.
A segfault or an unbounded native loop in any of them would take down an
in-process 800-page run with no way to recover. Running each batch in a worker
means a fatal fault costs one batch, not the whole document. It also means
torch's allocator is genuinely reclaimed when a worker is recycled.

Two timeouts guard a batch:

* docling's own ``document_timeout``, set from settings, which unwinds cleanly
  and reports a failed conversion; and
* the deadline here, a backstop for a worker wedged in native code where
  Python-level timeouts cannot fire. Hitting it rebuilds the pool, so any
  batches that were in flight are re-queued rather than lost.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import batched, groupby
from pathlib import Path
from time import monotonic

from ..settings import Settings
from ..util.logging import get_logger
from .extract import BatchResult, extract_batch
from .route import Route

log = get_logger(__name__)

_MAX_POOL_RESTARTS = 3


@dataclass(frozen=True)
class BatchJob:
    """One contiguous, single-route page range."""

    pdf_path: str
    page_from: int
    page_to: int
    route: str
    doc_sha: str

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_from, self.page_to + 1))

    def describe(self) -> str:
        """Short label for logs, and the key retry attempts are counted under.

        It must stay stable across requeues: a label that varied per submission would
        reset the attempt count, so a batch that crashes its worker every time would be
        retried for ever instead of failing after max_page_attempts.
        """
        span = (
            f"{self.page_from}"
            if self.page_from == self.page_to
            else f"{self.page_from}-{self.page_to}"
        )
        return f"{self.route}:{span}"


def _worker(
    job: BatchJob, cfg: Settings, want_markdown: bool = False
) -> BatchResult:
    """Entry point executed inside a worker process (must be importable)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(cfg.worker_threads))
    return extract_batch(
        job.pdf_path,
        job.page_from,
        job.page_to,
        job.route,
        doc_sha=job.doc_sha,
        cfg=cfg,
        want_markdown=want_markdown,
    )


def _failed(job: BatchJob, error: str) -> BatchResult:
    return BatchResult(
        page_from=job.page_from,
        page_to=job.page_to,
        route=job.route,
        error=error,
    )


def plan_batches(
    routed_pages: list[tuple[int, str]], batch_size: int
) -> list[tuple[int, int, str]]:
    """Group (page_no, route) pairs into contiguous single-route batches.

    docling's page_range is a contiguous span, so a batch cannot mix routes or
    skip pages. Pages already finished are simply absent from the input, which
    naturally splits the runs and is what makes resume cheap.
    """
    runs = groupby(enumerate(sorted(routed_pages)), key=lambda t: (t[1][1], t[1][0] - t[0]))
    return [
        (batch[0], batch[-1], route)
        for (route, _), grp in runs
        for batch in batched((p for _, (p, _) in grp), batch_size)
    ]


class BatchRunner:
    """Runs BatchJobs, isolated in worker processes unless told otherwise."""

    def __init__(
        self, cfg: Settings, *, want_markdown: bool = False
    ) -> None:
        """Set up the pool parameters and the hard-timeout backstop.

        docling already gets page_timeout_s * batch_pages to unwind a slow conversion
        cleanly (see pipelines.converter_for); the 1.5x margin keeps this deadline
        strictly looser, so an ordinary slow batch fails with an error instead of
        tearing down and rebuilding the pool. The 60s floor covers model load on a
        one-page batch.
        """
        self.cfg = cfg
        self.want_markdown = want_markdown
        self._pool: ProcessPoolExecutor | None = None
        self._deadline = max(60.0, cfg.page_timeout_s * cfg.batch_pages * 1.5)

    def _start_pool(self) -> ProcessPoolExecutor:
        """Fresh pool: spawn-based workers, recycled after worker_max_batches tasks.

        spawn gives each worker a clean interpreter rather than a fork of a parent that
        already holds native docling and torch state. Recycling is what actually returns
        that state: a worker kept alive for a whole document grows its allocator batch
        by batch.
        """
        return ProcessPoolExecutor(
            max_workers=max(1, self.cfg.max_workers),
            mp_context=mp.get_context("spawn"),
            max_tasks_per_child=max(1, self.cfg.worker_max_batches),
        )

    def _reset_pool(self, *, force: bool = False) -> None:
        """Tear the pool down.

        shutdown(wait=False) does not stop a worker already wedged in native
        code - it just stops waiting for it. Without force, every timeout would
        leave that process running and holding its memory for the rest of the
        run, so abnormal teardown escalates to terminate() and then kill().
        """
        pool, self._pool = self._pool, None
        if pool is None:
            return
        procs = []
        if force:
            try:
                procs = list(getattr(pool, "_processes", {}).values())
            except RuntimeError:
                procs = []
        pool.shutdown(wait=False, cancel_futures=True)
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                log.debug("worker teardown", error=str(exc))

    def close(self) -> None:
        self._reset_pool()

    def __enter__(self) -> BatchRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run(self, jobs: list[BatchJob]) -> Iterator[BatchResult]:
        """Yield results as they complete."""
        if not jobs:
            return
        if self.cfg.ingest_in_process:
            for job in jobs:
                yield _worker(job, self.cfg, self.want_markdown)
            return
        yield from self._run_pooled(list(jobs))

    def _run_pooled(self, queue: list[BatchJob]) -> Iterator[BatchResult]:
        """Drive the queue, rebuilding the pool whenever a worker dies.

        A dead worker (native crash, OOM kill) breaks the whole pool, so the
        queue is the durable thing here: whatever was in flight is re-queued and
        a fresh pool picks it up. Without this a single segfault on page 400
        would lose the other 399 pages' worth of pending work.
        """
        attempts: Counter[str] = Counter()
        restarts = 0
        capacity = max(1, self.cfg.max_workers)

        while queue:
            pending_queue, queue = queue, []
            pool = self._pool = self._start_pool()
            in_flight: dict[object, tuple[BatchJob, float]] = {}
            cursor = 0
            broken: str | None = None
            drained = False

            try:
                while cursor < len(pending_queue) or in_flight:
                    while len(in_flight) < capacity and cursor < len(pending_queue):
                        job = pending_queue[cursor]
                        try:
                            future = pool.submit(
                                _worker, job, self.cfg, self.want_markdown
                            )
                        except Exception as exc:  # noqa: BLE001 - pool is gone
                            broken = f"pool unusable: {type(exc).__name__}: {exc}"
                            break
                        cursor += 1
                        in_flight[future] = (job, monotonic() + self._deadline)
                    if broken or not in_flight:
                        break

                    done, _still = wait(
                        set(in_flight), timeout=5.0, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        job, _deadline = in_flight.pop(future)
                        try:
                            yield future.result()
                        except Exception as exc:  # noqa: BLE001 - worker died
                            error = f"worker failed: {type(exc).__name__}: {exc}"
                            log.warning(
                                "batch worker failed", batch=job.describe(), error=str(exc)
                            )
                            attempts[job.describe()] += 1
                            if attempts[job.describe()] >= self.cfg.max_page_attempts:
                                yield _failed(job, error)
                            else:
                                queue.append(job)
                            if "process pool" in str(exc).lower():
                                broken = error
                    if broken:
                        break

                    now = monotonic()
                    overdue = [f for f, (_j, dl) in in_flight.items() if dl < now]
                    if overdue:
                        for future in overdue:
                            job, _dl = in_flight.pop(future)
                            attempts[job.describe()] += 1
                            log.warning("batch timed out", batch=job.describe())
                            yield _failed(job, f"timed out after {self._deadline:.0f}s")
                        broken = "timeout"
                        break
                else:
                    drained = True
            finally:
                self._reset_pool(force=not drained)

            collateral = [job for job, _dl in in_flight.values()]
            queue = queue + collateral + pending_queue[cursor:]

            if broken:
                restarts += 1
                if restarts > _MAX_POOL_RESTARTS:
                    for job in queue:
                        yield _failed(
                            job, f"abandoned after {restarts} pool restarts: {broken}"
                        )
                    return
                log.warning(
                    "restarting worker pool", reason=broken, requeued=len(queue)
                )


def jobs_for(
    pdf_path: Path,
    doc_sha: str,
    routed_pages: list[tuple[int, str]],
    batch_size: int,
) -> list[BatchJob]:
    """Build batch jobs, skipping routes with nothing to extract.

    Empty pages must not reach the converter: loading the layout pipeline to
    produce zero chunks costs seconds per batch, which on a document with a lot
    of blank or separator pages is minutes of pure waste. Callers mark these
    pages 'skipped' directly instead.
    """
    eligible = [(n, r) for n, r in routed_pages if r != Route.EMPTY.value]
    return [
        BatchJob(
            pdf_path=str(pdf_path),
            page_from=lo,
            page_to=hi,
            route=route,
            doc_sha=doc_sha,
        )
        for lo, hi, route in plan_batches(eligible, batch_size)
    ]
