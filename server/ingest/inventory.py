"""PDF inventory: identity, structure, and the per-page signals the router needs.

Kept deliberately cheap - roughly a millisecond per page - so that auditing an
800-page document is interactive and so that a full ingest can re-derive
routing without re-running extraction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .quality import QualityReport, score_text

_SAMPLE_CHARS = 400


class PdfOpenError(RuntimeError):
    """The file could not be opened as a PDF (corrupt, or password protected)."""


@dataclass
class PageMetrics:
    """Everything the router decides on, plus enough to explain the decision."""

    page_no: int
    width: float
    height: float
    rotation: int
    n_chars: int
    text_cover: float
    image_cover: float
    n_images: int
    has_drawings: bool
    quality: QualityReport
    sample: str = ""
    error: str | None = None


@dataclass
class DocumentInventory:
    sha256: str
    path: Path
    filename: str
    page_count: int
    byte_size: int
    title: str | None
    encrypted: bool
    repaired: bool
    pages: list[PageMetrics] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    """Content hash: the identity half of the (sha256, pipeline_version) key."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def open_pdf(path: Path) -> pymupdf.Document:
    """Open a PDF, tolerating the things real-world files do.

    Empty-password encryption is extremely common in documents exported from
    office suites; those decrypt without any user input, so we try that before
    giving up.
    """
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PdfOpenError(f"cannot open {path.name}: {exc}") from exc

    if doc.needs_pass:
        if not doc.authenticate(""):
            doc.close()
            raise PdfOpenError(
                f"{path.name} is password protected; decrypt it before ingesting"
            )
    return doc


def _rect_area(rect: pymupdf.Rect) -> float:
    return max(0.0, rect.width) * max(0.0, rect.height)


def _page_metrics(page: pymupdf.Page, page_no: int, min_chars: int) -> PageMetrics:
    """Per-page signals for the router, computed without rendering the page.

    Cover ratios sum bounding-box areas without subtracting overlaps and clamp at
    1.0, so they are threshold proxies rather than true coverage. Each probe is
    guarded separately because one malformed page must degrade to weaker signals,
    never abort the inventory of an 800-page document.
    """
    rect = page.rect
    page_area = _rect_area(rect) or 1.0

    text = page.get_text("text") or ""
    quality = score_text(text, min_chars=min_chars)

    text_area = 0.0
    try:
        for x0, y0, x1, y1, *_ in page.get_text("words"):
            text_area += abs(x1 - x0) * abs(y1 - y0)
    except Exception:  # noqa: BLE001 - a malformed page must not abort inventory
        pass

    image_area = 0.0
    n_images = 0
    try:
        infos = page.get_image_info()
        n_images = len(infos)
        for info in infos:
            bbox = info.get("bbox")
            if bbox:
                image_area += _rect_area(pymupdf.Rect(bbox))
    except Exception:  # noqa: BLE001
        pass

    has_drawings = False
    try:
        has_drawings = bool(page.get_cdrawings())
    except Exception:  # noqa: BLE001
        pass

    return PageMetrics(
        page_no=page_no,
        width=float(rect.width),
        height=float(rect.height),
        rotation=int(page.rotation or 0),
        n_chars=len(text.strip()),
        text_cover=min(1.0, text_area / page_area),
        image_cover=min(1.0, image_area / page_area),
        n_images=n_images,
        has_drawings=has_drawings,
        quality=quality,
        sample=" ".join(text.split())[:_SAMPLE_CHARS],
    )


def iter_page_metrics(
    doc: pymupdf.Document,
    *,
    min_chars: int = 50,
    pages: range | None = None,
) -> Iterator[PageMetrics]:
    """Yield metrics page by page so memory stays flat on huge documents.

    A page that fails to load still yields a PageMetrics with `error` set and
    zeroed signals: the router turns that into EMPTY and the rest of the document
    still ingests, where raising would cost all 800 pages. Indices outside the
    document are skipped silently.
    """
    indices = pages if pages is not None else range(doc.page_count)
    for idx in indices:
        if idx < 0 or idx >= doc.page_count:
            continue
        page_no = idx + 1
        try:
            yield _page_metrics(doc.load_page(idx), page_no, min_chars)
        except Exception as exc:  # noqa: BLE001 - one bad page, not a dead run
            yield PageMetrics(
                page_no=page_no,
                width=0.0,
                height=0.0,
                rotation=0,
                n_chars=0,
                text_cover=0.0,
                image_cover=0.0,
                n_images=0,
                has_drawings=False,
                quality=score_text(""),
                error=f"{type(exc).__name__}: {exc}",
            )


def inventory(
    path: Path, *, min_chars: int = 50, pages: range | None = None
) -> DocumentInventory:
    """Full inventory of a PDF, including per-page routing signals.

    `page_count` is the document's own count, so a `pages` slice leaves it larger
    than `len(pages)`; callers must not use one as the length of the other.
    """
    path = Path(path)
    doc = open_pdf(path)
    try:
        meta = doc.metadata or {}
        inv = DocumentInventory(
            sha256=file_sha256(path),
            path=path,
            filename=path.name,
            page_count=doc.page_count,
            byte_size=path.stat().st_size,
            title=(meta.get("title") or "").strip() or None,
            encrypted=bool(doc.is_encrypted),
            repaired=bool(doc.is_repaired),
            pages=list(iter_page_metrics(doc, min_chars=min_chars, pages=pages)),
        )
    finally:
        doc.close()
    return inv
