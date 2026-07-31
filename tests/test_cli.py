"""CLI smoke tests.

`audit` is the operator's main window onto an 800-page document, so its
contract (histogram totals, per-page reasons, machine-readable output) is worth
pinning down.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from server.cli import app

runner = CliRunner()


class TestAudit:
    def test_json_output_covers_every_page(self, multi_pdf: Path) -> None:
        result = runner.invoke(app, ["audit", str(multi_pdf), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)

        assert payload["page_count"] == 4
        assert payload["audited"] == 4
        assert payload["histogram"] == {
            "digital": 1,
            "scanned": 1,
            "hybrid": 1,
            "empty": 1,
        }
        # scanned + hybrid are the OCR workload
        assert payload["ocr_pages"] == 2
        assert len(payload["sha256"]) == 64

        assert [p["page_no"] for p in payload["pages"]] == [1, 2, 3, 4]
        assert all(p["reason"] for p in payload["pages"])

    def test_page_range_limits_the_audit(self, multi_pdf: Path) -> None:
        result = runner.invoke(app, ["audit", str(multi_pdf), "--pages", "2-3", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["page_count"] == 4  # document is still 4 pages
        assert payload["audited"] == 2  # but only 2 were examined
        assert [p["page_no"] for p in payload["pages"]] == [2, 3]

    def test_bad_page_range_is_rejected(self, multi_pdf: Path) -> None:
        result = runner.invoke(app, ["audit", str(multi_pdf), "--pages", "nonsense"])
        assert result.exit_code != 0

    def test_human_output_explains_each_route(self, multi_pdf: Path) -> None:
        result = runner.invoke(app, ["audit", str(multi_pdf), "--samples", "1"])
        assert result.exit_code == 0, result.output
        for route in ("digital", "scanned", "hybrid", "empty"):
            assert route in result.output
        assert "pages needing OCR" in result.output

    def test_verbose_lists_every_page(self, multi_pdf: Path) -> None:
        result = runner.invoke(app, ["audit", str(multi_pdf), "--verbose"])
        assert result.exit_code == 0, result.output
        assert "Per-page detail" in result.output

    def test_corrupt_pdf_exits_nonzero(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf at all")
        result = runner.invoke(app, ["audit", str(bad)])
        assert result.exit_code == 1


class TestOtherCommands:
    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "editcorag" in result.output

    def test_doctor_runs_without_an_llm(self) -> None:
        """A missing LLM is a warning, not a crash - it is chosen later."""
        result = runner.invoke(app, ["doctor", "--no-check-llm"])
        assert result.exit_code == 0, result.output
        assert "RapidOCR" in result.output
        assert "Environment" in result.output
