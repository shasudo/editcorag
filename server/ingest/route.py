"""Per-page routing: decide how each page should be read.

Four routes, each mapping to a different extraction strategy:

  DIGITAL  trust the text layer; run layout + table structure, no OCR
  SCANNED  no usable text layer; render and OCR the whole page
  HYBRID   real text layer *and* imagery; OCR only the picture regions
  EMPTY    nothing renderable; record and skip

Every decision carries a human-readable reason so `editcorag audit` can explain
itself over 800 pages instead of asking to be trusted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from ..settings import Settings
from .inventory import PageMetrics

_FULL_PAGE_IMAGE_COVER = 0.85
_MIN_RENDERABLE_IMAGE_COVER = 0.02


class Route(StrEnum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"
    EMPTY = "empty"


@dataclass
class RoutingDecision:
    page_no: int
    route: Route
    reason: str
    metrics: PageMetrics

    def as_row(self) -> dict[str, object]:
        """Shape expected by db.store.save_routing."""
        m = self.metrics
        return {
            "page_no": self.page_no,
            "route": self.route.value,
            "width": m.width,
            "height": m.height,
            "rotation": m.rotation,
            "n_chars": m.n_chars,
            "text_cover": round(m.text_cover, 4),
            "image_cover": round(m.image_cover, 4),
            "quality": round(m.quality.score, 4),
        }


def classify(metrics: PageMetrics, cfg: Settings) -> RoutingDecision:
    """Assign one page to a route.

    Order is the contract. The quality gate runs before the hybrid check, so a
    page with both a bad text layer and imagery gets full-page OCR rather than a
    hybrid pass that would keep the gibberish. The has_drawings fallback matters
    because vector-only scans render fine but report almost no image cover;
    calling them EMPTY would drop the page from the index without a trace.
    """
    m = metrics

    def decide(route: Route, reason: str) -> RoutingDecision:
        return RoutingDecision(page_no=m.page_no, route=route, reason=reason, metrics=m)

    if m.error:
        return decide(Route.EMPTY, f"inventory failed: {m.error}")

    if m.n_chars < cfg.route_min_chars:
        if m.image_cover >= cfg.route_scanned_image_cover:
            return decide(
                Route.SCANNED,
                f"no text layer ({m.n_chars} chars); image covers {m.image_cover:.0%}",
            )
        if m.image_cover >= _MIN_RENDERABLE_IMAGE_COVER or m.has_drawings:
            what = "imagery" if m.image_cover >= _MIN_RENDERABLE_IMAGE_COVER else "vector drawings"
            return decide(Route.SCANNED, f"no text layer; renderable {what} present")
        return decide(Route.EMPTY, "no text, no imagery, no vector content")

    if m.quality.score < cfg.route_quality_min:
        why = "; ".join(m.quality.reasons) or "low quality score"
        return decide(
            Route.SCANNED,
            f"text layer unreliable (score {m.quality.score:.2f}): {why}",
        )

    if (
        m.image_cover >= cfg.route_hybrid_image_cover
        and m.text_cover >= cfg.route_hybrid_text_cover
    ):
        if m.n_images <= 2 and m.image_cover >= _FULL_PAGE_IMAGE_COVER:
            return decide(
                Route.HYBRID,
                f"searchable scan: full-page image with text layer "
                f"(text {m.text_cover:.0%})",
            )
        return decide(
            Route.HYBRID,
            f"text layer plus {m.n_images} image(s) covering {m.image_cover:.0%}",
        )

    return decide(
        Route.DIGITAL,
        f"good text layer ({m.n_chars} chars, quality {m.quality.score:.2f})",
    )


def route_pages(metrics: list[PageMetrics], cfg: Settings) -> list[RoutingDecision]:
    return [classify(m, cfg) for m in metrics]


def histogram(decisions: list[RoutingDecision]) -> dict[str, int]:
    """Counts per route, zero-filled so audit output keeps a stable shape."""
    counts = Counter(d.route.value for d in decisions)
    return {route.value: counts.get(route.value, 0) for route in Route}


def needs_ocr(route: Route) -> bool:
    return route in (Route.SCANNED, Route.HYBRID)
