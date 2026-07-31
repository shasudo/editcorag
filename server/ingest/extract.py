"""Batch extraction: PDF page range -> chunks ready to embed.

Runs entirely within one process (the parent may host it in a worker
subprocess, see runner.py). Everything returned is a plain dataclass so it can
cross a process boundary cheaply; figure images are written to disk here and
referenced by path rather than shipped through the pickle.

Coordinate convention: every bbox that leaves this module is **normalised to
0..1 with a top-left origin**, which is what pdf.js needs to draw a citation
highlight and what makes the value independent of zoom and page size. docling
reports bottom-left origin, so the conversion happens exactly once, here.
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..settings import Settings
from ..util.logging import get_logger
from . import ocr
from .dedup import drop_duplicate_texts, drop_lines_present_in
from .pipelines import converter_for
from .route import Route

log = get_logger(__name__)

_SKIP_LABELS = {"page_header", "page_footer"}


@dataclass
class ChunkOut:
    """One indexable unit."""

    id: str
    page_no: int
    page_span: list[int]
    kind: str
    text: str
    embed_text: str
    section: str | None = None
    bbox: list[float] | None = None
    asset_path: str | None = None


@dataclass
class AssetOut:
    """A cropped figure written to disk, awaiting a DB row in the parent."""

    page_no: int
    path: str
    bbox: list[float] | None
    ocr_text: str | None


@dataclass
class BatchResult:
    page_from: int
    page_to: int
    route: str
    chunks: list[ChunkOut] = field(default_factory=list)
    assets: list[AssetOut] = field(default_factory=list)
    markdown: str = ""
    duration_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def chunk_id(doc_sha: str, pipeline_version: str, page_no: int, kind: str, text: str) -> str:
    """Deterministic id.

    Stable across runs for identical input, so re-ingesting the same document
    produces the same ids and the LanceDB upsert is idempotent rather than
    duplicating rows.
    """
    digest = hashlib.sha1(
        "\x1f".join([doc_sha, pipeline_version, str(page_no), kind, text]).encode("utf-8"),
        usedforsecurity=False,
    )
    return digest.hexdigest()[:24]


def _norm_bbox(bbox: Any, page_size: Any) -> list[float] | None:
    """docling bbox -> normalised, top-left-origin [x0, y0, x1, y1]."""
    try:
        height = float(page_size.height)
        width = float(page_size.width)
        if height <= 0 or width <= 0:
            return None
        tl = bbox.to_top_left_origin(page_height=height)
        x0, x1 = sorted((float(tl.l) / width, float(tl.r) / width))
        y0, y1 = sorted((float(tl.t) / height, float(tl.b) / height))
        return [max(0.0, min(1.0, v)) for v in (x0, y0, x1, y1)]
    except Exception:  # noqa: BLE001 - a missing bbox must not fail the page
        return None


def _union(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _kind_for(label: str, route: str) -> str:
    if label == "table":
        return "table"
    if label in ("picture", "chart"):
        return "figure"
    return "ocr" if route == Route.SCANNED.value else "text"


@lru_cache(maxsize=2)
def _build_chunker(model: str, max_tokens: int, merge_peers: bool) -> Any:
    """Cached chunker. Arguments are primitives so the cache key is hashable.

    Rebuilding this re-parses the ~17 MB XLM-R tokenizer, which measured at
    ~2.1s. A worker handles many batches before it is recycled, so without the
    cache an 800-page document spends minutes doing nothing but reloading the
    same tokenizer.
    """
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    return HybridChunker(
        tokenizer=HuggingFaceTokenizer.from_pretrained(model, max_tokens=max_tokens),
        merge_peers=merge_peers,
    )


def _chunker(cfg: Settings) -> Any:
    return _build_chunker(cfg.embed_model, cfg.chunk_max_tokens, cfg.chunk_merge_peers)


def _save_figure(
    image: Any, assets_dir: Path, doc_sha: str, page_no: int, index: int
) -> str | None:
    """Write a figure crop; return its path relative to assets_dir."""
    rel = Path(doc_sha[:12]) / f"p{page_no:04d}-f{index}.png"
    target = assets_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        target.write_bytes(buffer.getvalue())
    except Exception as exc:  # noqa: BLE001
        log.warning("figure save failed", page=page_no, error=str(exc))
        return None
    return str(rel)


def extract_batch(
    pdf_path: str | Path,
    page_from: int,
    page_to: int,
    route: str,
    *,
    doc_sha: str,
    cfg: Settings,
    want_markdown: bool = False,
) -> BatchResult:
    """Convert and chunk one contiguous, single-route page range.

    Never raises: a failure is reported on the result so the surrounding run can
    mark these pages failed and carry on with the remaining 790.

    want_markdown is off by default because the rendered Markdown is pickled back
    from the worker for every batch and only `editcorag extract` ever reads it.
    """
    started = time.perf_counter()
    result = BatchResult(page_from=page_from, page_to=page_to, route=route)

    try:
        from docling.datamodel.base_models import ConversionStatus

        converter = converter_for(route, cfg)
        conversion = converter.convert(
            str(pdf_path),
            page_range=(page_from, page_to),
            raises_on_error=False,
        )
        if conversion.status in (ConversionStatus.FAILURE, ConversionStatus.SKIPPED):
            errors = "; ".join(str(e) for e in (conversion.errors or [])) or "conversion failed"
            result.error = f"docling {conversion.status.value}: {errors[:400]}"
            result.duration_s = time.perf_counter() - started
            return result

        doc = conversion.document
        result.chunks, result.assets = _harvest(
            doc, route=route, doc_sha=doc_sha, cfg=cfg
        )
        if want_markdown:
            try:
                result.markdown = doc.export_to_markdown()
            except Exception:  # noqa: BLE001 - markdown is a debugging nicety
                result.markdown = ""
    except Exception as exc:  # noqa: BLE001 - one batch must not kill the run
        result.error = f"{type(exc).__name__}: {exc}"

    result.duration_s = time.perf_counter() - started
    return result


def _harvest(
    doc: Any, *, route: str, doc_sha: str, cfg: Settings
) -> tuple[list[ChunkOut], list[AssetOut]]:
    """Pull chunks, tables and figures out of a converted document."""
    chunks: list[ChunkOut] = []

    chunker = _chunker(cfg)
    for chunk in chunker.chunk(doc):
        meta = getattr(chunk, "meta", None)
        items = list(getattr(meta, "doc_items", []) or [])
        label = str(getattr(items[0], "label", "text")) if items else "text"
        if label in _SKIP_LABELS:
            continue

        pages_seen: set[int] = set()
        page_boxes: dict[int, list[list[float]]] = {}
        for item in items:
            for prov in getattr(item, "prov", []) or []:
                page_no = int(getattr(prov, "page_no", 0) or 0)
                if not page_no:
                    continue
                page = doc.pages.get(page_no)
                if page is None:
                    continue
                pages_seen.add(page_no)
                box = _norm_bbox(prov.bbox, page.size)
                if box:
                    page_boxes.setdefault(page_no, []).append(box)

        text = (chunk.text or "").strip()
        if not text:
            continue
        span = sorted(pages_seen)
        if not span:
            log.warning("dropping chunk with no page provenance", preview=text[:80])
            continue
        primary = span[0]
        try:
            embed_text = chunker.contextualize(chunk) or text
        except Exception:  # noqa: BLE001
            embed_text = text

        headings = list(getattr(meta, "headings", []) or [])
        kind = _kind_for(label, route)
        chunks.append(
            ChunkOut(
                id=chunk_id(doc_sha, cfg.pipeline_version, primary, kind, text),
                page_no=primary,
                page_span=span,
                kind=kind,
                text=text,
                embed_text=embed_text,
                section=" > ".join(h for h in headings if h) or None,
                bbox=_union(page_boxes.get(primary, [])),
            )
        )

    figure_chunks, assets = _harvest_figures(
        doc, route=route, doc_sha=doc_sha, cfg=cfg
    )
    chunks.extend(figure_chunks)

    return _dedupe_per_page(chunks, cfg.dedup_similarity), assets


def _harvest_figures(
    doc: Any, *, route: str, doc_sha: str, cfg: Settings
) -> tuple[list[ChunkOut], list[AssetOut]]:
    """Crop each significant figure, OCR its labels, and make it searchable.

    A chart or diagram carries meaning that no surrounding prose repeats. Making
    the labels inside it retrievable - and keeping the crop so the UI can show
    the actual figure beside the answer - is most of the value of ingesting a
    technical document at all.
    """
    chunks: list[ChunkOut] = []
    assets: list[AssetOut] = []
    pictures = list(getattr(doc, "pictures", []) or [])
    if not pictures:
        return chunks, assets

    assets_dir = cfg.assets_dir
    per_page_index: dict[int, int] = {}

    for picture in pictures:
        prov = next(iter(getattr(picture, "prov", []) or []), None)
        if prov is None:
            continue
        page_no = int(getattr(prov, "page_no", 0) or 0)
        page = doc.pages.get(page_no) if page_no else None
        if page is None:
            continue
        bbox = _norm_bbox(prov.bbox, page.size)
        if bbox is None:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < cfg.figure_min_area:
            continue

        index = per_page_index.get(page_no, 0)
        per_page_index[page_no] = index + 1

        image = None
        try:
            image = picture.get_image(doc)
        except Exception as exc:  # noqa: BLE001
            log.debug("figure image unavailable", page=page_no, error=str(exc))
        if image is None:
            continue
        if image.width * image.height > cfg.max_image_pixels:
            continue

        rel_path = _save_figure(image, assets_dir, doc_sha, page_no, index)

        caption = ""
        try:
            caption = (picture.caption_text(doc) or "").strip()
        except Exception:  # noqa: BLE001
            caption = ""

        ocr_text = ""
        try:
            lines = ocr.recognize(image, cfg.ocr_languages)
            candidates = [t.strip() for t in lines if t.strip()]
            # Drop labels the page's own text layer already carries.
            kept = drop_lines_present_in(
                candidates, _page_text(doc, page_no), threshold=cfg.dedup_similarity
            )
            ocr_text = " ".join(kept).strip()
        except Exception as exc:  # noqa: BLE001 - a figure without labels is still worth keeping
            log.debug("figure OCR failed", page=page_no, error=str(exc))

        assets.append(
            AssetOut(
                page_no=page_no,
                path=rel_path or "",
                bbox=bbox,
                ocr_text=ocr_text or None,
            )
        )

        parts = [p for p in (caption, ocr_text) if p]
        if not parts:
            continue
        body = " ".join(parts)
        text = f"[Figure, page {page_no}] {body}"
        chunks.append(
            ChunkOut(
                id=chunk_id(doc_sha, cfg.pipeline_version, page_no, "figure", text),
                page_no=page_no,
                page_span=[page_no],
                kind="figure",
                text=text,
                embed_text=text,
                section=None,
                bbox=bbox,
                asset_path=rel_path,
            )
        )

    return chunks, assets


def _page_text(doc: Any, page_no: int) -> str:
    """All text docling found on one page, used as the dedup reference."""
    parts: list[str] = []
    try:
        for item, _level in doc.iterate_items(page_no=page_no):
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(parts)


def _dedupe_per_page(chunks: list[ChunkOut], threshold: float) -> list[ChunkOut]:
    """Suppress near-duplicates, comparing only chunks that share a page.

    Scoping per page is deliberate: boilerplate repeated across pages is
    legitimate, and each page needs its own chunk to anchor a citation.

    The id set is a second, separate guard. Fuzzy dedup keeps fragments shorter
    than its comparison floor ("Q1", "N/A") because short strings match
    everything - but the id is a hash of (doc, version, page, kind, text), so
    identical fragments on one page collide. SQLite's primary key then rejected
    the second row and aborted the whole ingest, while LanceDB's merge_insert
    kept both and inflated that term's document frequency for every query.
    """
    by_page: dict[int, list[ChunkOut]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk.page_no, []).append(chunk)

    kept: list[ChunkOut] = []
    seen_ids: set[str] = set()
    for page_no in sorted(by_page):
        group = by_page[page_no]
        keep_idx = set(drop_duplicate_texts([c.text for c in group], threshold=threshold))
        for i, chunk in enumerate(group):
            if i not in keep_idx or chunk.id in seen_ids:
                continue
            seen_ids.add(chunk.id)
            kept.append(chunk)
    return kept
