"""Synthetic PDF fixtures.

The four routes are built from scratch with PyMuPDF so the router can be tested
without needing anyone's real 800-page document:

  digital  text drawn normally -> a real text layer
  scanned  text rasterised into an image, then placed on a blank page, so the
           page has pixels but no text layer at all
  hybrid   a real text layer plus a rasterised image that itself contains text
  empty    nothing at all
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pymupdf
import pytest

from server.settings import Settings

LOREM = (
    "The flange bolts shall be torqued to 42 Nm when dry and 38 Nm when "
    "lubricated. Verify the gasket seating surface before assembly and replace "
    "any component showing pitting deeper than 0.2 mm. Record the measured "
    "values in the maintenance log for each service interval."
)


def _text_pixmap(
    text: str, width: int = 420, height: int = 220, dpi: int = 150, wrap: int = 46
):
    """Rasterise some text, so we can embed it as an image with no text layer.

    Note insert_image() keeps the source aspect ratio, letterboxing inside the
    target rect. A "full-page scan" fixture therefore has to be rasterised at
    the page's own proportions or its image coverage will not reach 1.0.
    """
    src = pymupdf.open()
    page = src.new_page(width=width, height=height)
    y = 40
    for line in textwrap.wrap(text, wrap):
        page.insert_text((24, y), line, fontsize=13)
        y += 18
    pix = page.get_pixmap(dpi=dpi)
    src.close()
    return pix


def _full_page_scan_pixmap(text: str):
    """A4-proportioned raster, so it fills the page rect exactly."""
    return _text_pixmap(text * 3, width=595, height=842, dpi=110, wrap=64)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def digital_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in textwrap.wrap(LOREM + " " + LOREM, 70):
        page.insert_text((60, y), line, fontsize=11)
        y += 16
    out = tmp_path / "digital.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """A page that is purely an image of text - the OCR path's reason to exist."""
    pix = _full_page_scan_pixmap(LOREM)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), pixmap=pix)
    out = tmp_path / "scanned.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def hybrid_pdf(tmp_path: Path) -> Path:
    """Text layer plus a figure that itself contains text."""
    pix = _text_pixmap("Figure 6-2  Torque sequence  bolt A B C  42 Nm", 400, 200)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    for line in textwrap.wrap(LOREM, 70):
        page.insert_text((60, y), line, fontsize=11)
        y += 16
    page.insert_image(pymupdf.Rect(60, 400, 520, 700), pixmap=pix)
    out = tmp_path / "hybrid.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page()
    out = tmp_path / "empty.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def multi_pdf(tmp_path: Path) -> Path:
    """One document exercising every route, for audit/histogram tests."""
    pix = _full_page_scan_pixmap(LOREM)
    fig = _text_pixmap("Figure 1  labelled diagram  see note 4", 400, 200)
    doc = pymupdf.open()

    digital = doc.new_page(width=595, height=842)
    y = 72
    for line in textwrap.wrap(LOREM + " " + LOREM, 70):
        digital.insert_text((60, y), line, fontsize=11)
        y += 16

    scanned = doc.new_page(width=595, height=842)
    scanned.insert_image(pymupdf.Rect(0, 0, 595, 842), pixmap=pix)

    hybrid = doc.new_page(width=595, height=842)
    y = 72
    for line in textwrap.wrap(LOREM, 70):
        hybrid.insert_text((60, y), line, fontsize=11)
        y += 16
    hybrid.insert_image(pymupdf.Rect(60, 400, 520, 700), pixmap=fig)

    doc.new_page(width=595, height=842)  # empty

    out = tmp_path / "multi.pdf"
    doc.save(out)
    doc.close()
    return out
