"""Regression tests for defects found in the code audit.

Each test here failed before its fix. They are grouped by the failure they lock
down rather than by module, because the point is to stop the specific bug
coming back.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from server.db import store as db
from server.settings import Settings


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return db.get_conn(tmp_path / "t.db")


def _make_doc(conn: sqlite3.Connection, pages: int = 3) -> int:
    row = db.upsert_document(
        conn,
        sha256="a" * 64,
        filename="f.pdf",
        path="/tmp/f.pdf",
        page_count=pages,
        byte_size=1,
        pipeline_version="1",
    )
    doc_id = int(row["id"])
    db.ensure_pages(conn, doc_id, pages)
    return doc_id


def _routing(pages: int) -> list[dict[str, object]]:
    """Distinct value per column per page, so a positional mix-up cannot pass."""
    return [
        {
            "page_no": n,
            "route": "digital",
            "width": 100.0 + n,
            "height": 200.0 + n,
            "rotation": n * 10,
            "n_chars": 1000 + n,
            "text_cover": round(0.10 * n, 4),
            "image_cover": round(0.90 - 0.10 * n, 4),
            "quality": round(0.50 + 0.01 * n, 4),
        }
        for n in range(1, pages + 1)
    ]


class TestRoutingDoesNotUndoSettledPages:
    """save_routing runs on every ingest and must not rewrite settled outcomes."""

    def test_finished_pages_are_not_reset(self, conn: sqlite3.Connection) -> None:
        doc_id = _make_doc(conn)
        db.save_routing(conn, doc_id, _routing(3), max_attempts=2)
        db.set_page_status(conn, doc_id, 1, "done", n_chunks=4)

        db.save_routing(conn, doc_id, _routing(3), max_attempts=2)

        assert db.status_counts(conn, doc_id) == {"done": 1, "routed": 2}

    def test_exhausted_failures_keep_their_error(self, conn: sqlite3.Connection) -> None:
        """Otherwise the run reports success having silently dropped the page."""
        doc_id = _make_doc(conn)
        db.save_routing(conn, doc_id, _routing(3), max_attempts=2)
        for _ in range(2):
            db.set_page_status(
                conn, doc_id, 3, "failed", error="boom", bump_attempts=True
            )

        db.save_routing(conn, doc_id, _routing(3), max_attempts=2)

        page3 = next(p for p in db.get_pages(conn, doc_id) if p["page_no"] == 3)
        assert page3["status"] == "failed"
        assert page3["error"] == "boom"
        assert page3["attempts"] == 2
        assert db.status_counts(conn, doc_id).get("failed") == 1

    def test_retryable_failure_is_still_retried(self, conn: sqlite3.Connection) -> None:
        """A page with retries left must go back into the queue."""
        doc_id = _make_doc(conn)
        db.save_routing(conn, doc_id, _routing(3), max_attempts=2)
        db.set_page_status(conn, doc_id, 2, "failed", error="transient", bump_attempts=True)

        db.save_routing(conn, doc_id, _routing(3), max_attempts=2)

        page2 = next(p for p in db.get_pages(conn, doc_id) if p["page_no"] == 2)
        assert page2["status"] == "routed"
        assert page2["error"] is None
        assert page2["page_no"] in [p["page_no"] for p in db.pending_pages(conn, doc_id, 2)]


class TestRoutingWritesEveryColumnCorrectly:
    """save_routing builds its UPDATE positionally with a repeated CASE
    predicate; an off-by-one in the parameter order would silently write the
    wrong column rather than raise."""

    def test_all_metric_columns_round_trip(self, conn: sqlite3.Connection) -> None:
        doc_id = _make_doc(conn, pages=3)
        rows = _routing(3)
        db.save_routing(conn, doc_id, rows, max_attempts=2)

        stored = {p["page_no"]: p for p in db.get_pages(conn, doc_id)}
        for expected in rows:
            page = stored[expected["page_no"]]
            assert page["route"] == expected["route"]
            assert page["width"] == pytest.approx(expected["width"])
            assert page["height"] == pytest.approx(expected["height"])
            assert page["rotation"] == expected["rotation"]
            assert page["n_chars"] == expected["n_chars"]
            assert page["text_cover"] == pytest.approx(expected["text_cover"])
            assert page["image_cover"] == pytest.approx(expected["image_cover"])
            assert page["quality"] == pytest.approx(expected["quality"])


class TestRetrievalOrdering:
    """The tie-break must not re-rank across retrievers with unlike scales."""

    def test_tie_break_only_applies_to_rerank_scores(self) -> None:
        import inspect

        from server.rag import retrieve as mod

        src = inspect.getsource(mod.retrieve)
        assert '"rerank_score" in' in src, "tie-break must be gated on rerank_score"
        assert "retrieval_score" not in src.split("sources: list[Source]")[0], (
            "sorting the fused list by raw retrieval_score discards RRF: BM25 is "
            "unbounded while dense is 1/(1+distance)"
        )

    def test_within_retriever_ordering_is_deterministic(self) -> None:
        from server.index.retrievers import _stable
        from server.index.store import SearchHit

        def hit(chunk_id: str, score: float) -> SearchHit:
            return SearchHit(
                id=chunk_id,
                doc_id=1,
                page_no=1,
                page_span=[1],
                kind="text",
                section=None,
                text="t",
                bbox=None,
                asset_path=None,
                score=score,
            )

        # Same scores, different input order -> same output order.
        a = _stable([hit("c", 0.5), hit("a", 0.5), hit("b", 0.9)])
        b = _stable([hit("a", 0.5), hit("b", 0.9), hit("c", 0.5)])
        assert [h.id for h in a] == [h.id for h in b] == ["b", "a", "c"]


class TestJobLifecycle:
    def test_upload_then_graph_records_the_doc_id(self, conn: sqlite3.Connection) -> None:
        """An upload queues the job before the document exists, so doc_id
        arrives on the graph's later upsert. If ON CONFLICT does not carry it
        over, the job row stays NULL and /api/jobs/{id}/stream never attaches
        page_status - the UI's live page counter silently never fires."""
        doc_id = _make_doc(conn)
        db.upsert_job(conn, job_id="j2", doc_id=None, kind="ingest", status="queued")
        db.upsert_job(conn, job_id="j2", doc_id=doc_id, kind="ingest", status="running")

        row = db.get_job(conn, "j2")
        assert row is not None
        assert row["doc_id"] == doc_id

    def test_requeueing_does_not_blank_a_known_doc_id(
        self, conn: sqlite3.Connection
    ) -> None:
        """Re-uploading the same file passes doc_id=None again."""
        doc_id = _make_doc(conn)
        db.upsert_job(conn, job_id="j3", doc_id=doc_id, kind="ingest", status="running")
        db.upsert_job(conn, job_id="j3", doc_id=None, kind="ingest", status="queued")

        row = db.get_job(conn, "j3")
        assert row is not None
        assert row["doc_id"] == doc_id

    def test_requeue_clears_the_previous_error(self, conn: sqlite3.Connection) -> None:
        doc_id = _make_doc(conn)
        db.upsert_job(conn, job_id="j1", doc_id=doc_id, kind="ingest", status="running")
        db.update_job(conn, "j1", status="failed", error="old failure", finished=True)

        db.upsert_job(conn, job_id="j1", doc_id=doc_id, kind="ingest", status="queued")

        row = db.get_job(conn, "j1")
        assert row is not None
        assert row["status"] == "queued"
        assert row["error"] is None
        assert row["finished_at"] is None


class TestAssetLifecycle:
    def test_reingesting_a_page_does_not_stack_asset_rows(
        self, conn: sqlite3.Connection
    ) -> None:
        doc_id = _make_doc(conn)
        for _ in range(2):
            db.delete_page_assets(conn, doc_id, [1])
            db.insert_asset(
                conn, doc_id=doc_id, page_no=1, kind="figure", path="a/p1-f0.png", bbox=None
            )
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM assets WHERE doc_id=?", (doc_id,)
        ).fetchone()
        assert rows["n"] == 1

    def test_listing_asset_paths_does_not_delete_them(
        self, conn: sqlite3.Connection
    ) -> None:
        """Connections autocommit, so a delete here would land before the work
        that can still fail - stranding the PNGs with nothing pointing at them."""
        doc_id = _make_doc(conn)
        db.insert_asset(
            conn, doc_id=doc_id, page_no=1, kind="figure", path="a/x.png", bbox=None
        )
        assert db.document_asset_paths(conn, doc_id) == ["a/x.png"]
        assert db.document_asset_paths(conn, doc_id) == ["a/x.png"]

    def test_deleting_the_document_cascades_its_assets(
        self, conn: sqlite3.Connection
    ) -> None:
        doc_id = _make_doc(conn)
        db.insert_asset(
            conn, doc_id=doc_id, page_no=1, kind="figure", path="a/x.png", bbox=None
        )
        with db.transaction(conn):
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        assert db.document_asset_paths(conn, doc_id) == []


class TestDuplicateChunkIds:
    """Dedup deliberately keeps fragments too short to compare, so a page of
    repeated cells really does emit two chunks with the same hashed id. SQLite's
    primary key aborted the whole ingest on the second one, and LanceDB's
    merge_insert does not dedupe within a batch, so it kept both and inflated
    that term's document frequency for every other query."""

    @staticmethod
    def _chunk(text: str, page_no: int = 1):
        from server.ingest.extract import ChunkOut, chunk_id

        return ChunkOut(
            id=chunk_id("sha", "1", page_no, "text", text),
            page_no=page_no,
            page_span=[page_no],
            kind="text",
            text=text,
            embed_text=text,
        )

    def test_identical_short_chunks_collapse_to_one(self) -> None:
        from server.ingest.extract import _dedupe_per_page

        kept = _dedupe_per_page([self._chunk("Q1"), self._chunk("Q1")], 0.90)
        assert len(kept) == 1
        assert len({c.id for c in kept}) == 1

    def test_distinct_short_chunks_are_all_kept(self) -> None:
        """The short-fragment exemption itself must survive: "Q1" and "Q2" are
        different data, and fuzzy matching alone would collapse them."""
        from server.ingest.extract import _dedupe_per_page

        kept = _dedupe_per_page(
            [self._chunk("Q1"), self._chunk("Q2"), self._chunk("Q3")], 0.90
        )
        assert [c.text for c in kept] == ["Q1", "Q2", "Q3"]

    def test_same_text_on_different_pages_is_kept(self) -> None:
        """A repeated heading on two pages is two citable chunks, not one."""
        from server.ingest.extract import _dedupe_per_page

        kept = _dedupe_per_page(
            [self._chunk("Notes", page_no=1), self._chunk("Notes", page_no=2)], 0.90
        )
        assert {c.page_no for c in kept} == {1, 2}

    def test_a_colliding_id_no_longer_aborts_the_write(
        self, conn: sqlite3.Connection
    ) -> None:
        """Belt and braces: even if a duplicate reaches the store, the insert
        must not raise and take the whole document down with it."""
        doc_id = _make_doc(conn)
        row = {
            "id": "collide", "kind": "text", "section": None,
            "bbox": None, "page_span": [1], "asset_id": None,
        }
        db.replace_page_chunks(conn, doc_id, 1, [row, dict(row)])
        assert db.count_chunks(conn, doc_id) == 1


class TestWorkerPoolTeardown:
    """A consumer that stops iterating (a raise in _commit_batch, a KeyboardInterrupt)
    finalises the generator with workers still mid-batch. Tearing the pool down
    without force only stops *waiting* for them, so docling workers kept running
    with their models loaded and the process hung at exit joining them."""

    @staticmethod
    def _runner(monkeypatch, forced: list[bool]):
        from concurrent.futures import Future

        from server.ingest.extract import BatchResult
        from server.ingest.runner import BatchRunner

        class FakePool:
            _processes: dict = {}

            def submit(self, _fn, job, *_a):
                future: Future = Future()
                future.set_result(
                    BatchResult(
                        page_from=job.page_from, page_to=job.page_to, route=job.route
                    )
                )
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                pass

        runner = BatchRunner(Settings())
        monkeypatch.setattr(runner, "_start_pool", lambda: FakePool())
        monkeypatch.setattr(
            runner, "_reset_pool", lambda *, force=False: forced.append(force)
        )
        return runner

    @staticmethod
    def _jobs(n: int) -> list:
        from server.ingest.runner import BatchJob

        return [
            BatchJob(pdf_path="x.pdf", page_from=i, page_to=i, route="digital", doc_sha="s")
            for i in range(1, n + 1)
        ]

    def test_abandoning_the_generator_forces_teardown(self, monkeypatch) -> None:
        import gc

        forced: list[bool] = []
        runner = self._runner(monkeypatch, forced)
        for _result in runner.run(self._jobs(5)):
            break  # consumer gives up while batches are still queued
        gc.collect()

        assert forced, "the pool was never torn down"
        assert all(forced), f"workers left running: force={forced}"

    def test_a_clean_drain_does_not_force(self, monkeypatch) -> None:
        """Terminating workers that already finished would be wrong."""
        forced: list[bool] = []
        runner = self._runner(monkeypatch, forced)
        results = list(runner.run(self._jobs(3)))

        assert len(results) == 3
        assert forced == [False]


class TestNotesDoNotLeakBetweenTurns:
    """notes is a per-answer trace, but the checkpointer persists every channel
    under the thread id, so an appending reducer replayed old turns' traces."""

    def test_a_new_run_clears_the_previous_trace(self) -> None:
        from server.graphs.state import RESET_NOTES, add_notes

        carried = add_notes(["turn 1: retrieved", "turn 1: grounded"], [RESET_NOTES])
        assert carried == []

    def test_notes_still_accumulate_within_a_run(self) -> None:
        from server.graphs.state import RESET_NOTES, add_notes

        state = add_notes(None, [RESET_NOTES, "query: x"])
        state = add_notes(state, ["relevance: True"])
        state = add_notes(state, ["grounded"])
        assert state == ["query: x", "relevance: True", "grounded"]


class TestQueryStateChannels:
    """LangGraph silently drops keys a node returns that the state does not declare."""

    def test_grader_verdict_channel_exists(self) -> None:
        from server.graphs.query_graph import build_query_graph

        channels = set(build_query_graph().builder.channels)
        for key in ("relevant", "documents", "top_score", "answer", "citations"):
            assert key in channels, (
                f"a node returns {key!r}; without the channel LangGraph discards it "
                "and the self-correction loop is unreachable"
            )


class TestAbstentionGate:
    def test_no_sources_abstains(self) -> None:
        from server.rag.retrieve import should_abstain

        assert should_abstain(0, 0.99, Settings()) is True

    def test_low_score_abstains(self) -> None:
        from server.rag.retrieve import should_abstain

        cfg = Settings()
        assert should_abstain(8, cfg.abstain_score_min - 0.01, cfg) is True
        assert should_abstain(8, cfg.abstain_score_min, cfg) is False


class TestContextBudgetIsScriptAware:
    """A character budget derived from Latin density overfills the model's
    window on CJK by 2-3x. The server then truncates the excerpts silently, so
    the answer cites passages it was never actually shown."""

    @staticmethod
    def _sources(text: str, n: int = 40) -> list:
        from server.rag.retrieve import Source

        return [
            Source(
                ref=f"S{i}", chunk_id=f"c{i}", doc_id=1, page_no=i, page_span=[i],
                kind="text", section=None, text=text, bbox=None,
                asset_path=None, score=0.9,
            )
            for i in range(1, n + 1)
        ]

    def test_cjk_context_stays_within_the_token_budget(self) -> None:
        from server.rag.retrieve import pack_context

        cfg = Settings()
        han = "本文件规定了法兰螺栓的紧固扭矩要求。干燥状态下为四十二牛米。" * 4
        context, _used = pack_context(self._sources(han), cfg)

        # Han runs ~1.2 chars/token, so the budget in characters must be about
        # that multiple - not the ~3.5 that Latin text allows.
        assert len(context) <= cfg.context_token_budget * 1.5, (
            f"{len(context)} chars of Han text is roughly "
            f"{len(context) / 1.2:.0f} tokens against a "
            f"{cfg.context_token_budget}-token budget"
        )

    def test_latin_context_still_uses_the_full_budget(self) -> None:
        """The fix must not shrink the context for Latin documents."""
        from server.rag.retrieve import pack_context

        cfg = Settings()
        latin = "The flange bolts shall be torqued to 42 Nm when dry. " * 8
        context, _used = pack_context(self._sources(latin, n=200), cfg)

        assert len(context) > cfg.context_token_budget * 3


class TestSettingsLoadOutsideTheRepo:
    def test_env_file_is_anchored_to_the_project(self) -> None:
        """A CWD-relative .env silently drops all config when run elsewhere."""
        from server.settings import PROJECT_ROOT

        env_files = Settings.model_config["env_file"]
        assert PROJECT_ROOT / ".env" in tuple(env_files)


class TestOcrIsConsistentAcrossPlatforms:
    """One backend everywhere. A document must not index differently depending
    on whose machine opened it."""

    def test_every_route_builds_a_converter(self) -> None:
        from server.ingest.pipelines import converter_for

        cfg = Settings()
        for route in ("digital", "scanned", "hybrid"):
            assert converter_for(route, cfg) is not None, route

    def test_model_set_is_chosen_by_language_not_platform(self) -> None:
        from server.ingest import ocr

        # Narrowest covering set wins, so English never loads the Chinese
        # recogniser - which drops the spaces between Latin words.
        assert ocr.model_set_for(["en"]) == "english"
        assert ocr.model_set_for(["de", "fr"]) == "latin"
        assert ocr.model_set_for(["zh"]) == "chinese"
        assert ocr.model_set_for(["English"]) == "english"
        assert ocr.model_set_for([]) == "english"

    def test_unreadable_languages_are_reported_not_silently_wrong(self) -> None:
        from server.ingest import ocr

        assert ocr.unsupported(["en", "ja", "ko"]) == ["ja", "ko"]
        assert ocr.unsupported(["en", "de"]) == []


class TestCitationParsing:
    """Models do not all emit ASCII brackets; a parser that assumes they do
    reports zero citations while the answer still reads plausibly."""

    def test_accepts_the_bracket_styles_models_actually_emit(self) -> None:
        from server.graphs.query_graph import _validate_citations
        from server.rag.retrieve import Source

        used = [
            Source(
                ref=f"S{i}",
                chunk_id=f"c{i}",
                doc_id=1,
                page_no=i,
                page_span=[i],
                kind="text",
                section=None,
                text="t",
                bbox=None,
                asset_path=None,
                score=0.9,
            )
            for i in (1, 2, 3)
        ]
        for answer in (
            "Rating agencies failed [S1][S2].",       # ASCII
            "Rating agencies failed 【S1】【S2】.",   # fullwidth CJK (gpt-oss)
            "Rating agencies failed ［S1］［S2］.",   # fullwidth ASCII variants
            "Rating agencies failed (S1)(S2).",       # parenthesised
        ):
            cleaned, cites = _validate_citations(answer, used)
            refs = [c["ref"] for c in cites]
            assert refs == ["S1", "S2"], f"failed to parse: {answer!r}"
            assert "[S1][S2]" in cleaned, f"not normalised: {cleaned!r}"

    def test_hallucinated_citation_is_still_dropped(self) -> None:
        from server.graphs.query_graph import _validate_citations
        from server.rag.retrieve import Source

        used = [
            Source(
                ref="S1", chunk_id="c1", doc_id=1, page_no=1, page_span=[1],
                kind="text", section=None, text="t", bbox=None,
                asset_path=None, score=0.9,
            )
        ]
        cleaned, cites = _validate_citations("Claim 【S1】 and invented 【S9】.", used)
        assert [c["ref"] for c in cites] == ["S1"]
        assert "S9" not in cleaned


class TestHealthDoesNotBlockTheLoop:
    """The desktop shell polls /api/health to decide the engine is up.

    Every part of that handler blocks - lancedb's import, llm_status' httpx
    call, the snapshot scans. Declared `async def` it froze uvicorn's event
    loop, the shell's probes timed out, and a perfectly healthy engine was
    reported as "started but never became reachable".
    """

    def test_handler_is_sync_so_fastapi_threads_it(self) -> None:
        import inspect

        from server.api import health

        assert not inspect.iscoroutinefunction(health)
