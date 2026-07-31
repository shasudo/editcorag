"""Page routing, driven through real PDFs rather than mocked metrics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from server.ingest.inventory import inventory
from server.ingest.quality import score_text
from server.ingest.route import Route, classify, histogram, needs_ocr, route_pages
from server.settings import Settings


def _route_of(path: Path, cfg: Settings, page_no: int = 1) -> tuple[Route, str]:
    inv = inventory(path, min_chars=cfg.route_min_chars)
    decision = classify(inv.pages[page_no - 1], cfg)
    return decision.route, decision.reason


class TestRoutingRealPdfs:
    def test_digital_page(self, digital_pdf: Path, settings: Settings) -> None:
        route, reason = _route_of(digital_pdf, settings)
        assert route is Route.DIGITAL, reason
        assert not needs_ocr(route)

    def test_scanned_page_has_no_text_layer(
        self, scanned_pdf: Path, settings: Settings
    ) -> None:
        inv = inventory(scanned_pdf, min_chars=settings.route_min_chars)
        page = inv.pages[0]
        # The premise of the fixture: pixels, but genuinely nothing extractable.
        assert page.n_chars < settings.route_min_chars
        assert page.image_cover > 0.9
        route, reason = _route_of(scanned_pdf, settings)
        assert route is Route.SCANNED, reason
        assert needs_ocr(route)

    def test_hybrid_page(self, hybrid_pdf: Path, settings: Settings) -> None:
        inv = inventory(hybrid_pdf, min_chars=settings.route_min_chars)
        page = inv.pages[0]
        assert page.n_chars > settings.route_min_chars  # real text layer
        assert page.image_cover >= settings.route_hybrid_image_cover  # and a figure
        route, reason = _route_of(hybrid_pdf, settings)
        assert route is Route.HYBRID, reason
        assert needs_ocr(route)

    def test_empty_page(self, empty_pdf: Path, settings: Settings) -> None:
        route, reason = _route_of(empty_pdf, settings)
        assert route is Route.EMPTY, reason
        assert not needs_ocr(route)

    def test_all_routes_in_one_document(
        self, multi_pdf: Path, settings: Settings
    ) -> None:
        inv = inventory(multi_pdf, min_chars=settings.route_min_chars)
        assert inv.page_count == 4
        decisions = route_pages(inv.pages, settings)
        assert [d.route for d in decisions] == [
            Route.DIGITAL,
            Route.SCANNED,
            Route.HYBRID,
            Route.EMPTY,
        ]
        assert histogram(decisions) == {
            "digital": 1,
            "scanned": 1,
            "hybrid": 1,
            "empty": 1,
        }
        assert all(d.reason for d in decisions)


class TestPoisonedTextLayerForcesOcr:
    """The trap: extractable text that is garbage must be re-read via OCR."""

    def test_garbled_text_layer_is_rerouted_to_ocr(
        self, digital_pdf: Path, settings: Settings
    ) -> None:
        inv = inventory(digital_pdf, min_chars=settings.route_min_chars)
        good = inv.pages[0]
        assert classify(good, settings).route is Route.DIGITAL

        # Same page geometry and character count, but a broken encoding.
        poisoned = replace(
            good, quality=score_text("".join(f"(cid:{i})" for i in range(40, 140)))
        )
        decision = classify(poisoned, settings)
        assert decision.route is Route.SCANNED
        assert "unreliable" in decision.reason
        assert "cid" in decision.reason

    def test_lost_spacing_is_rerouted_to_ocr(
        self, digital_pdf: Path, settings: Settings
    ) -> None:
        inv = inventory(digital_pdf, min_chars=settings.route_min_chars)
        text = "TheflangeboltsshallbetorquedtoNmwhendryandNmwhenlubricated" * 3
        poisoned = replace(inv.pages[0], quality=score_text(text))
        assert classify(poisoned, settings).route is Route.SCANNED


class TestVectorOnlyPages:
    def test_vector_drawn_text_is_ocred_not_skipped(
        self, empty_pdf: Path, settings: Settings
    ) -> None:
        """No text layer and no raster image, but the page draws something."""
        inv = inventory(empty_pdf, min_chars=settings.route_min_chars)
        blank = inv.pages[0]
        assert classify(blank, settings).route is Route.EMPTY

        vector_page = replace(blank, has_drawings=True)
        decision = classify(vector_page, settings)
        assert decision.route is Route.SCANNED
        assert "vector" in decision.reason


class TestInventoryRobustness:
    def test_sha256_is_stable_and_content_addressed(
        self, digital_pdf: Path, tmp_path: Path
    ) -> None:
        copy = tmp_path / "copy.pdf"
        copy.write_bytes(digital_pdf.read_bytes())
        assert inventory(digital_pdf).sha256 == inventory(copy).sha256

    def test_page_range_subset(self, multi_pdf: Path) -> None:
        inv = inventory(multi_pdf, pages=range(1, 3))
        assert [p.page_no for p in inv.pages] == [2, 3]

    def test_corrupt_file_raises_clean_error(self, tmp_path: Path) -> None:
        from server.ingest.inventory import PdfOpenError

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"this is definitely not a pdf")
        try:
            inventory(bad)
        except PdfOpenError as exc:
            assert "bad.pdf" in str(exc)
        else:
            raise AssertionError("expected PdfOpenError")
