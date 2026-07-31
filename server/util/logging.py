"""structlog configuration.

Console-friendly by default; set LOCALRAG_LOG_JSON=true for machine-readable
JSONL.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog.

    Entry points call this explicitly: importing anything that calls
    get_logger() at module scope would otherwise lock in the defaults and make
    LOCALRAG_LOG_LEVEL and LOCALRAG_LOG_JSON silently inert.
    """
    global _configured

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    for noisy in ("httpx", "urllib3", "transformers", "sentence_transformers", "docling", "lance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Structlog logger, falling back to the default configuration on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
