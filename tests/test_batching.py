"""Batch planning and duplicate suppression."""

from __future__ import annotations

from server.ingest.dedup import drop_duplicate_texts, drop_lines_present_in, normalize
from server.ingest.runner import plan_batches


class TestPlanBatches:
    def test_single_route_splits_by_size(self) -> None:
        pages = [(n, "digital") for n in range(1, 8)]
        assert plan_batches(pages, 3) == [
            (1, 3, "digital"),
            (4, 6, "digital"),
            (7, 7, "digital"),
        ]

    def test_route_change_forces_a_new_batch(self) -> None:
        """docling applies one pipeline per call, so a batch cannot mix routes."""
        pages = [(1, "digital"), (2, "digital"), (3, "scanned"), (4, "digital")]
        assert plan_batches(pages, 10) == [
            (1, 2, "digital"),
            (3, 3, "scanned"),
            (4, 4, "digital"),
        ]

    def test_gaps_split_batches(self) -> None:
        """page_range is contiguous; a gap means a separate call."""
        pages = [(1, "digital"), (2, "digital"), (5, "digital"), (6, "digital")]
        assert plan_batches(pages, 10) == [(1, 2, "digital"), (5, 6, "digital")]

    def test_unsorted_input_is_ordered(self) -> None:
        pages = [(3, "digital"), (1, "digital"), (2, "digital")]
        assert plan_batches(pages, 5) == [(1, 3, "digital")]

    def test_empty(self) -> None:
        assert plan_batches([], 4) == []

    def test_every_page_appears_exactly_once(self) -> None:
        routes = ["digital"] * 30 + ["scanned"] * 12 + ["hybrid"] * 8
        pages = list(enumerate(routes, start=1))
        batches = plan_batches(pages, 6)
        covered: list[int] = []
        for lo, hi, _route in batches:
            covered.extend(range(lo, hi + 1))
        assert covered == list(range(1, 51))
        assert all(hi - lo + 1 <= 6 for lo, hi, _ in batches)


class TestNormalize:
    def test_casefold_and_punctuation(self) -> None:
        assert normalize("Torque:  42 Nm!") == normalize("torque 42 nm")

    def test_whitespace_collapse(self) -> None:
        assert normalize("a\n\n  b\tc") == "a b c"


class TestDropLinesPresentIn:
    def test_removes_labels_already_in_the_text_layer(self) -> None:
        """Figure OCR must not re-index a caption the PDF already provides."""
        page_text = "Figure 7-1 bolt A max 42 Nm\nTorque compliance by quarter"
        ocr = ["Figure 7-1 bolt A max 42 Nm", "88%", "Q1", "Torque compliance by quarter"]
        kept = drop_lines_present_in(ocr, page_text, threshold=0.90)
        assert "88%" in kept
        assert "Q1" in kept
        assert not any("Figure 7-1" in k for k in kept)
        assert not any("compliance by quarter" in k for k in kept)

    def test_keeps_everything_when_there_is_no_reference(self) -> None:
        ocr = ["88% Q1 label", "62% Q2 label"]
        assert drop_lines_present_in(ocr, "", threshold=0.9) == ocr

    def test_near_match_is_dropped(self) -> None:
        """OCR rarely reproduces text character-for-character."""
        kept = drop_lines_present_in(
            ["Torque compliance by quarler"],  # OCR misread 't' as 'l'
            "Torque compliance by quarter",
            threshold=0.85,
        )
        assert kept == []

    def test_blank_candidates_dropped(self) -> None:
        assert drop_lines_present_in(["", "   "], "anything at all here") == []


class TestDropDuplicateTexts:
    def test_keeps_first_of_a_duplicate_pair(self) -> None:
        texts = ["The bolts are torqued to 42 Nm.", "The bolts are torqued to 42 Nm.", "Other"]
        assert drop_duplicate_texts(texts, threshold=0.9) == [0, 2]

    def test_distinct_texts_all_kept(self) -> None:
        texts = ["torque spec is 42 Nm", "gasket seating surface check", "pitting depth 0.2 mm"]
        assert drop_duplicate_texts(texts, threshold=0.9) == [0, 1, 2]

    def test_short_fragments_are_never_deduped(self) -> None:
        """Short strings fuzzy-match everything; dropping them loses real data."""
        assert drop_duplicate_texts(["Q1", "Q2", "Q3"], threshold=0.9) == [0, 1, 2]


class TestJobsForSkipsEmptyPages:
    def test_empty_pages_never_reach_the_converter(self, tmp_path) -> None:
        """Loading the layout pipeline to produce zero chunks is pure waste."""
        from server.ingest.runner import jobs_for

        routed = [(1, "digital"), (2, "empty"), (3, "empty"), (4, "digital")]
        jobs = jobs_for(tmp_path / "x.pdf", "sha", routed, batch_size=6)
        spans = [(j.page_from, j.page_to, j.route) for j in jobs]
        assert spans == [(1, 1, "digital"), (4, 4, "digital")]

    def test_all_empty_yields_no_work(self, tmp_path) -> None:
        from server.ingest.runner import jobs_for

        routed = [(n, "empty") for n in range(1, 6)]
        assert jobs_for(tmp_path / "x.pdf", "sha", routed, batch_size=6) == []
