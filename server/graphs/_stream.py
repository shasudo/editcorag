"""Progress events from inside graph nodes."""

from __future__ import annotations

from typing import Any


def emit(**event: Any) -> None:
    """Push a progress event onto the graph's custom stream, if one is attached.

    Nodes run both under `graph.stream(stream_mode="custom")` (the API and the
    CLI progress bar) and under a plain `invoke` (tests, `answer_question`).
    A node called outside a runnable context has no config to read, hence the
    bare except: not streaming is normal, not an error.
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(event)
    except Exception:  # noqa: BLE001 - not streaming, which is fine
        return
