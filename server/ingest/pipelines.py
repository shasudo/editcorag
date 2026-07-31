"""Docling converter construction - one pipeline per route.

The three OCR-relevant routes need genuinely different pipelines:

  digital  do_ocr=False. Trusting the text layer is what makes the fast path fast.
  scanned  OcrMode.FULL_PAGE. There is no text layer, so read the whole raster.
  hybrid   OcrMode.PDF_AWARE_LAYOUT_REGIONS. OCR the layout regions while
           respecting text the PDF already provides, which is what stops a
           diagram-with-a-caption page being indexed twice.

Converters are cached per configuration because constructing one loads the
layout and table models.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..settings import Settings
from ..util.device import resolve_device
from ..util.logging import get_logger
from .route import Route

log = get_logger(__name__)


@lru_cache(maxsize=6)
def _build_converter(
    route: str,
    languages: tuple[str, ...],
    images_scale: float,
    table_mode: str,
    document_timeout: float,
    device: str,
    num_threads: int,
) -> Any:
    """Cached converter. Arguments are primitives so the cache key is hashable."""
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from . import ocr

    ocr_options = None
    if route in (Route.SCANNED.value, Route.HYBRID.value):
        ocr_options = ocr.docling_options(
            list(languages),
            full_page=route == Route.SCANNED.value,
            pdf_aware=route == Route.HYBRID.value,
        )
    want_ocr = ocr_options is not None

    options = PdfPipelineOptions(
        do_ocr=want_ocr,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST,
        ),
        generate_picture_images=True,
        generate_page_images=False,
        images_scale=images_scale,
        do_picture_classification=False,
        do_picture_description=False,
        do_formula_enrichment=False,
        do_code_enrichment=False,
        document_timeout=document_timeout,
        accelerator_options=AcceleratorOptions(device=device, num_threads=num_threads),
    )

    if ocr_options is not None:
        options.ocr_options = ocr_options

    log.debug(
        "built docling converter",
        route=route,
        ocr=want_ocr,
        device=device,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def converter_for(route: str, cfg: Settings) -> Any:
    """Converter appropriate to this route under the current settings.

    docling's timeout covers a whole conversion, so the per-page budget is scaled by
    batch_pages here. It must stay under BatchRunner's own deadline: this one
    unwinds cleanly and reports a failed batch, the runner's kills the pool.
    """
    return _build_converter(
        route=route,
        languages=tuple(cfg.ocr_languages),
        images_scale=cfg.figure_images_scale,
        table_mode=cfg.table_mode,
        document_timeout=float(cfg.page_timeout_s * cfg.batch_pages),
        device=resolve_device("auto"),
        num_threads=cfg.worker_threads,
    )
