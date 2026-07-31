"""OCR, via RapidOCR.

One backend on every platform, deliberately. RapidOCR is ONNX with no system
binaries, so a scanned page reads the same on Windows as on macOS. Apple Vision
was faster on macOS and Tesseract covered more languages, but a document that
indexes differently depending on whose machine opened it is worth neither.

The one thing that still has to be got right is the model set. RapidOCR bundles
three and loads exactly one per pass, and picking the wrong one does not fail -
it returns plausible nonsense. Its Chinese recogniser in particular drops the
spaces between Latin words.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

# The bundled sets, narrowest first so a request is served by the most specific
# one that covers it.
MODEL_SETS: dict[str, set[str]] = {
    "english": {"en"},
    "latin": {
        "en", "de", "fr", "es", "it", "pt", "nl", "fi", "sv", "da", "no", "pl",
        "cs", "ro", "hu", "tr", "hr", "sk", "sl", "ca", "id",
    },
    "chinese": {"zh"},
}
_ORDER = ("english", "latin", "chinese")

# What people actually type into a config file, mapped onto ISO 639-1.
_ALIASES = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "polish": "pl",
    "chinese": "zh", "eng": "en", "deu": "de", "ger": "de", "fra": "fr",
    "fre": "fr", "spa": "es", "ita": "it", "por": "pt", "nld": "nl",
    "zho": "zh", "chi": "zh",
}


def canonical_langs(values: list[str]) -> list[str]:
    """'en-GB', 'eng', 'English' -> 'en', keeping order and dropping repeats.

    Order matters: the first entry is treated as the primary language when a
    request spans more than one model set.
    """
    out: list[str] = []
    for value in values:
        s = value.strip().lower().replace("_", "-")
        code = _ALIASES.get(s) or _ALIASES.get(s.split("-")[0]) or s.split("-")[0]
        if code and code not in out:
            out.append(code)
    return out


def model_set_for(languages: list[str]) -> str:
    """The bundled set that best covers these languages."""
    wanted = [c for c in canonical_langs(languages) if c in supported_languages()] or ["en"]
    for name in _ORDER:
        if set(wanted) <= MODEL_SETS[name]:
            return name
    # Spans several sets, and only one loads per pass: serve the primary.
    return next((n for n in _ORDER if wanted[0] in MODEL_SETS[n]), "latin")


def supported_languages() -> set[str]:
    return set().union(*MODEL_SETS.values())


def unsupported(languages: list[str]) -> list[str]:
    """Requested languages RapidOCR cannot read at all."""
    known = supported_languages()
    return [c for c in canonical_langs(languages) if c not in known]


def docling_options(languages: list[str], *, full_page: bool, pdf_aware: bool) -> Any:
    """Pipeline options for docling.

    full_page  -> OcrMode.FULL_PAGE, for pages with no usable text layer.
    pdf_aware  -> OcrMode.PDF_AWARE_LAYOUT_REGIONS, which OCRs layout regions
                  while respecting text the PDF already provides; this is what
                  stops a diagram-with-a-caption page being indexed twice.
    """
    from docling.datamodel.pipeline_options import OcrMode, RapidOcrOptions

    if full_page:
        mode = OcrMode.FULL_PAGE
    elif pdf_aware:
        mode = OcrMode.PDF_AWARE_LAYOUT_REGIONS
    else:
        mode = OcrMode.DEFAULT
    return RapidOcrOptions(lang=[model_set_for(languages)], mode=mode)


@lru_cache(maxsize=3)
def _engine(model_set: str) -> Any:
    """One engine per set; construction loads the ONNX weights."""
    from rapidocr import LangDet, LangRec, RapidOCR

    params = {
        "english": {"Det.lang_type": LangDet.EN, "Rec.lang_type": LangRec.EN},
        "latin": {"Det.lang_type": LangDet.EN, "Rec.lang_type": LangRec.LATIN},
        "chinese": {"Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.CH},
    }[model_set]
    return RapidOCR(params=params)


def recognize(image: Image.Image, languages: list[str]) -> list[str]:
    """OCR a cropped image directly, for figure labels."""
    import numpy as np

    result = _engine(model_set_for(languages))(np.array(image.convert("RGB")))
    return [str(t) for t in (getattr(result, "txts", None) or [])]
