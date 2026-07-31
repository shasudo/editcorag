"""Text-layer quality scoring.

These cases are the whole reason the scorer exists: each "bad" sample is text
that a PDF really can hand you, and that a naive pipeline would happily index.
"""

from __future__ import annotations

import pytest

from server.ingest.quality import detect_script, score_text

CLEAN_EN = (
    "The flange bolts shall be torqued to 42 Nm when dry and 38 Nm when "
    "lubricated. Verify the gasket seating surface before assembly."
)
CLEAN_ZH = "本文件规定了法兰螺栓的紧固扭矩要求。干燥状态下为四十二牛米，润滑状态下为三十八牛米。"
CLEAN_DE = (
    "Die Flanschschrauben sind trocken mit 42 Nm und geschmiert mit 38 Nm "
    "anzuziehen. Prüfen Sie die Dichtfläche vor dem Zusammenbau."
)


class TestGoodTextPasses:
    @pytest.mark.parametrize(
        ("sample", "script"),
        [(CLEAN_EN, "latin"), (CLEAN_DE, "latin"), (CLEAN_ZH, "han")],
    )
    def test_clean_text_scores_high(self, sample: str, script: str) -> None:
        report = score_text(sample)
        assert report.score >= 0.9, report.reasons
        assert report.reasons == []
        assert report.script == script

    def test_cjk_not_penalised_for_lacking_spaces(self) -> None:
        """Han script has no inter-word spaces; the Latin heuristic must not fire."""
        report = score_text(CLEAN_ZH)
        assert report.space_ratio < 0.03  # genuinely almost no spaces
        assert report.score >= 0.9  # yet not penalised
        assert not any("space" in r for r in report.reasons)

    def test_short_page_is_not_punished(self) -> None:
        """A chapter divider is valid, not broken."""
        report = score_text("Chapter 4")
        assert report.score >= 0.9
        assert report.n_chars < 50


class TestBrokenTextLayerIsCaught:
    def test_cid_encoded_glyphs(self) -> None:
        """A font with no ToUnicode CMap extracts as (cid:NN) runs."""
        sample = "".join(f"(cid:{i})" for i in range(40, 90))
        report = score_text(sample)
        assert report.score < 0.6
        assert any("cid" in r for r in report.reasons)

    def test_replacement_characters(self) -> None:
        report = score_text("The fl" + "�" * 30 + "ange bolts shall be torqued to 42 Nm.")
        assert report.score < 0.6
        assert any("replacement" in r for r in report.reasons)

    def test_private_use_area_glyphs(self) -> None:
        """Custom-encoded subset fonts map glyphs into the PUA."""
        sample = "".join(chr(0xE000 + (i % 200)) for i in range(120))
        report = score_text(sample)
        assert report.score < 0.6
        assert any("private-use" in r for r in report.reasons)

    def test_lost_word_spacing(self) -> None:
        """The exact corruption docling warns about for the wrong OCR model."""
        report = score_text(CLEAN_EN.replace(" ", ""))
        assert report.score < 0.6
        assert any("space" in r for r in report.reasons)

    def test_symbol_soup(self) -> None:
        report = score_text("@#$%^&*<>{}[]|\\~`" * 20)
        assert report.score < 0.6

    def test_empty_text(self) -> None:
        report = score_text("")
        assert report.score == 0.0
        assert report.reasons == ["empty"]


class TestScriptDetection:
    @pytest.mark.parametrize(
        ("sample", "expected"),
        [
            ("Hello world, this is plain English text.", "latin"),
            ("これは日本語のテキストです", "kana"),
            ("한국어 텍스트입니다", "hangul"),
            ("Это текст на русском языке", "cyrillic"),
            ("هذا نص عربي", "arabic"),
            ("यह हिंदी पाठ है", "devanagari"),
            ("Grüße aus München, Straße", "latin"),
            ("", "unknown"),
        ],
    )
    def test_detect(self, sample: str, expected: str) -> None:
        assert detect_script(sample) == expected


class TestThresholdSeparation:
    def test_good_and_bad_straddle_the_default_threshold(self) -> None:
        """Whatever the tuning, clean text must land above and garbage below."""
        from server.settings import Settings

        threshold = Settings().route_quality_min
        assert score_text(CLEAN_EN).score >= threshold
        assert score_text(CLEAN_ZH).score >= threshold
        assert score_text("".join(f"(cid:{i})" for i in range(40, 90))).score < threshold
        assert score_text(CLEAN_EN.replace(" ", "")).score < threshold
