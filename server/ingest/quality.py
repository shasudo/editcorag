"""Text-layer quality scoring.

The failure mode this exists to catch: a PDF page *has* an extractable text
layer, but that layer is garbage - a subset font with no ToUnicode CMap, a
broken encoding, or glyphs mapped into the Unicode private-use area. Naive
pipelines see "text found" and skip OCR, silently indexing gibberish that then
poisons retrieval in a way that is very hard to trace back.

Scoring is deliberately **script-aware**. Latin-specific heuristics (space
ratio, vowel presence, word length) are meaningless for Chinese or Japanese,
where applying them would force pointless OCR on perfectly good text.

Score is 0..1, higher is better. Pages below settings.route_quality_min get
routed to OCR instead of trusting their text layer.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_CID_RE = re.compile(r"\(cid:\d+\)")
_WS_RE = re.compile(r"\s+")

_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("han", 0x4E00, 0x9FFF),
    ("han", 0x3400, 0x4DBF),
    ("kana", 0x3040, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
    ("hangul", 0x1100, 0x11FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("cyrillic", 0x0400, 0x04FF),
    ("devanagari", 0x0900, 0x097F),
    ("thai", 0x0E00, 0x0E7F),
    ("hebrew", 0x0590, 0x05FF),
    ("greek", 0x0370, 0x03FF),
)

_SPACE_DELIMITED = {"latin", "cyrillic", "greek", "devanagari", "hebrew", "unknown"}

_PUA = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))


@dataclass
class QualityReport:
    """Why a page's text layer was trusted or rejected."""

    score: float
    n_chars: int
    script: str
    cid_ratio: float = 0.0
    replacement_ratio: float = 0.0
    control_ratio: float = 0.0
    pua_ratio: float = 0.0
    space_ratio: float = 0.0
    longword_ratio: float = 0.0
    alnum_ratio: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _in_pua(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _PUA)


def detect_script(text: str) -> str:
    """Dominant script of the sample, by counted code points.

    Only letters vote: ASCII digits and punctuation are script-neutral, so a page
    that is mostly a numeric table still reports the script of its prose. A sample
    with no letters at all comes back "unknown", which the scorer then treats as
    space-delimited.
    """
    counts: Counter[str] = Counter()
    latin = 0
    for ch in text:
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                latin += 1
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break
        else:
            if unicodedata.category(ch).startswith("L") and 0x00C0 <= cp <= 0x024F:
                latin += 1
    if latin:
        counts["latin"] += latin
    return counts.most_common(1)[0][0] if counts else "unknown"


def score_text(text: str, *, min_chars: int = 50) -> QualityReport:
    """Score a page's extracted text layer.

    Encoding damage - cid glyphs, replacement chars, PUA, controls - is penalised
    at any length, but the statistical heuristics (spaces, token length,
    alphanumerics) only start at min_chars: penalising a short-but-valid page, a
    title or a chapter divider, would force OCR it does not need. Deciding that a
    page has too little text to judge is the router's call, not this function's.
    """
    n = len(text)
    if n == 0:
        return QualityReport(score=0.0, n_chars=0, script="unknown", reasons=["empty"])

    script = detect_script(text)
    report = QualityReport(score=1.0, n_chars=n, script=script)

    cid_chars = sum(len(m.group(0)) for m in _CID_RE.finditer(text))
    report.cid_ratio = cid_chars / n
    report.replacement_ratio = text.count("�") / n
    report.pua_ratio = sum(1 for ch in text if _in_pua(ord(ch))) / n
    report.control_ratio = (
        sum(
            1
            for ch in text
            if unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn") and ch not in "\n\r\t"
        )
        / n
    )

    stripped = _WS_RE.sub("", text)
    report.space_ratio = (n - len(stripped)) / n if n else 0.0
    report.alnum_ratio = sum(1 for ch in text if ch.isalnum()) / n

    tokens = [t for t in _WS_RE.split(text) if t]
    if tokens:
        long_chars = sum(len(t) for t in tokens if len(t) > 30)
        report.longword_ratio = long_chars / max(1, len(stripped))

    penalty = 0.0

    if report.cid_ratio > 0.01:
        penalty += min(1.0, report.cid_ratio * 5)
        report.reasons.append(f"cid-encoded glyphs ({report.cid_ratio:.1%})")
    if report.replacement_ratio > 0.01:
        penalty += min(1.0, report.replacement_ratio * 5)
        report.reasons.append(f"replacement chars ({report.replacement_ratio:.1%})")
    if report.pua_ratio > 0.01:
        penalty += min(1.0, report.pua_ratio * 5)
        report.reasons.append(f"private-use glyphs ({report.pua_ratio:.1%})")
    if report.control_ratio > 0.02:
        penalty += min(0.8, report.control_ratio * 4)
        report.reasons.append(f"control chars ({report.control_ratio:.1%})")

    if n >= min_chars and report.alnum_ratio < 0.45:
        penalty += 0.5 * (0.45 - report.alnum_ratio) / 0.45
        report.reasons.append(f"low alphanumeric ratio ({report.alnum_ratio:.1%})")

    if script in _SPACE_DELIMITED and n >= min_chars:
        if report.space_ratio < 0.03:
            penalty += 0.6
            report.reasons.append(f"almost no spaces ({report.space_ratio:.1%})")
        elif report.space_ratio < 0.07:
            penalty += 0.25
            report.reasons.append(f"few spaces ({report.space_ratio:.1%})")
        if report.longword_ratio > 0.35:
            penalty += 0.4
            report.reasons.append(f"runaway tokens ({report.longword_ratio:.1%})")

    report.score = max(0.0, min(1.0, 1.0 - penalty))
    return report
