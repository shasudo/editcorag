"""Single source of truth for compute-device selection.

Every model load goes through resolve_device() so that one code path decides
cuda / mps / cpu on every platform.
"""

from __future__ import annotations

import platform
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=8)
def resolve_device(preference: str = "auto") -> str:
    """Return a concrete torch device string.

    "auto" prefers cuda, then Apple mps, then cpu. An explicit preference is
    honoured only if it is actually usable, otherwise we fall back rather than
    crash deep inside a model load.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    def usable(name: str) -> bool:
        if name == "cuda":
            return bool(torch.cuda.is_available())
        if name == "mps":
            return bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
        return name == "cpu"

    if preference != "auto":
        if usable(preference):
            return preference
        return "cpu"

    for candidate in ("cuda", "mps"):
        if usable(candidate):
            return candidate
    return "cpu"


def device_report() -> dict[str, Any]:
    """Human-facing diagnostics for `editcorag doctor`."""
    report: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "resolved": resolve_device("auto"),
    }
    try:
        import torch
    except ImportError:
        report["torch"] = "not installed"
        return report

    report["torch"] = torch.__version__
    if torch.cuda.is_available():
        report["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return report
