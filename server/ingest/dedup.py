"""Near-duplicate suppression.

Two places produce duplicate text on a hybrid page:

1. Figure label OCR re-reads a caption that the PDF's own text layer already
   provides, so the same sentence enters the index twice.
2. Retries or overlapping extraction paths emit near-identical chunks.

Duplicates are not harmless: BM25 term statistics are computed over the corpus,
so indexing the same sentence twice inflates its document frequency and quietly
distorts ranking for every other query.

docling's PDF_AWARE_LAYOUT_REGIONS mode prevents most of case 1 upstream; this
module is the safety net that catches what slips through.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_MIN_COMPARE_LEN = 8


def normalize(text: str) -> str:
    """Casefold, strip punctuation and collapse whitespace for comparison only."""
    return _WS.sub(" ", _PUNCT.sub(" ", text.casefold())).strip()


def drop_lines_present_in(
    candidates: Sequence[str], reference: str, *, threshold: float = 0.90
) -> list[str]:
    """Return candidate lines that are NOT already present in reference text.

    Used for figure-label OCR against the page's text layer.
    """
    kept: list[str] = []
    ref_lines = [
        norm
        for line in reference.splitlines()
        if len(norm := normalize(line)) >= _MIN_COMPARE_LEN
    ]
    if not ref_lines:
        return [c for c in candidates if c.strip()]

    from rapidfuzz import fuzz, process

    cutoff = threshold * 100
    for candidate in candidates:
        if not candidate.strip():
            continue
        norm = normalize(candidate)
        if len(norm) < _MIN_COMPARE_LEN:
            kept.append(candidate)
            continue
        match = process.extractOne(norm, ref_lines, scorer=fuzz.token_set_ratio)
        if match is not None and match[1] >= cutoff:
            continue
        kept.append(candidate)
    return kept


def drop_duplicate_texts(
    texts: Sequence[str], *, threshold: float = 0.90
) -> list[int]:
    """Indices of texts to keep, dropping later near-duplicates of earlier ones.

    First occurrence wins, so callers should pass their best text first - chunks from
    the PDF's own text layer ahead of figure OCR. Fragments shorter than
    _MIN_COMPARE_LEN skip comparison entirely: token_set_ratio scores short strings
    against each other far too generously, and dropping them would lose distinct
    table cells and axis labels.
    """
    from rapidfuzz import fuzz

    kept_idx: list[int] = []
    kept_norm: list[str] = []
    cutoff = threshold * 100
    for i, text in enumerate(texts):
        norm = normalize(text)
        if len(norm) < _MIN_COMPARE_LEN:
            kept_idx.append(i)
            kept_norm.append(norm)
            continue
        if any(fuzz.token_set_ratio(norm, seen) >= cutoff for seen in kept_norm):
            continue
        kept_idx.append(i)
        kept_norm.append(norm)
    return kept_idx
