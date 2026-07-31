"""Chat model construction and health probing.

The answering model is Groq, reached through its OpenAI-compatible endpoint, so
langchain-openai is the only client needed and there is no provider branching
left to get wrong.

This is the one part of the system that is not local: retrieved excerpts are
sent to Groq to be turned into an answer. Documents, the index and retrieval
all stay on the machine.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..settings import Settings, get_settings
from ..util.logging import get_logger

log = get_logger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Status is polled by the desktop setup screen, so the probe is cheap and its
# result is reused for a few seconds rather than hitting Groq on every request.
_STATUS_TTL_S = 5.0
_status_cache: tuple[float, dict[str, Any]] | None = None


def message_text(reply: Any) -> str:
    """Flatten a chat reply's content to plain text.

    Content arrives either as a string or as a list of blocks, and the blocks
    may be dicts. Getting this wrong puts a stringified dict in the answer.
    """
    content = getattr(reply, "content", reply)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            parts.append(str(part.get("text", "")) if isinstance(part, dict) else str(part))
        return "".join(parts)
    return str(content or "")


def build_chat_model(cfg: Settings | None = None, *, temperature: float | None = None) -> Any:
    """Instantiate the chat model.

    temperature is explicit because the graph wants 0.0 for answering but a
    little slack for query rewriting.
    """
    from langchain_openai import ChatOpenAI

    cfg = cfg or get_settings()
    return ChatOpenAI(
        model=cfg.llm_model,
        base_url=cfg.llm_base_url or GROQ_BASE_URL,
        api_key=cfg.llm_api_key or "missing",
        temperature=cfg.llm_temperature if temperature is None else temperature,
        timeout=cfg.llm_timeout_s,
        max_retries=cfg.llm_max_retries,
    )


def invalidate_status() -> None:
    """Drop the cached probe, so a newly saved API key takes effect at once."""
    global _status_cache
    _status_cache = None


def llm_status(cfg: Settings | None = None) -> dict[str, Any]:
    """Cheap, cached check of whether answering will actually work.

    Deliberately not probe_llm, which runs a real generation and is far too slow
    to poll. Listing models separates the states a user fixes differently:
    no key at all, a rejected key, no network, or a model name Groq does not
    serve. A failed generation cannot tell those apart.

    status is one of: ready | no_key | bad_key | offline | model_missing
    """
    global _status_cache

    cfg = cfg or get_settings()
    now = time.monotonic()
    if _status_cache and now - _status_cache[0] < _STATUS_TTL_S:
        return _status_cache[1]

    base = (cfg.llm_base_url or GROQ_BASE_URL).rstrip("/")
    out: dict[str, Any] = {"model": cfg.llm_model, "base_url": base}

    if not cfg.llm_api_key:
        out |= {"status": "no_key", "detail": "no API key configured"}
        _status_cache = (now, out)
        return out

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                f"{base}/models", headers={"Authorization": f"Bearer {cfg.llm_api_key}"}
            )
    except Exception as exc:  # noqa: BLE001 - a status probe never raises
        out |= {"status": "offline", "detail": type(exc).__name__}
    else:
        if response.status_code in (401, 403):
            out |= {"status": "bad_key", "detail": "the API key was rejected"}
        elif response.status_code != 200:
            out |= {"status": "offline", "detail": f"HTTP {response.status_code}"}
        else:
            served = [m.get("id", "") for m in response.json().get("data", [])]
            if served and cfg.llm_model not in served:
                out |= {
                    "status": "model_missing",
                    "detail": f"{cfg.llm_model} is not served",
                    "available": served[:20],
                }
            else:
                out |= {"status": "ready", "detail": ""}

    _status_cache = (now, out)
    return out


def probe_llm(cfg: Settings | None = None) -> dict[str, Any]:
    """Real round-trip for `editcorag doctor`.

    Separates "misconfigured" (cannot even build a client) from "unreachable"
    (built fine, the call failed), because the fixes differ.
    """
    cfg = cfg or get_settings()
    result: dict[str, Any] = {
        "model": cfg.llm_model,
        "base_url": cfg.llm_base_url or GROQ_BASE_URL,
        "constructed": False,
        "reachable": False,
        "detail": "",
    }
    try:
        model = build_chat_model(cfg)
        result["constructed"] = True
    except Exception as exc:  # noqa: BLE001 - report, never crash the doctor
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        reply = model.invoke("Reply with the single word: ok")
        result["reachable"] = True
        result["detail"] = message_text(reply).strip()[:120]
    except Exception as exc:  # noqa: BLE001
        result["detail"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return result


def supports_structured_output(cfg: Settings | None = None) -> tuple[bool, str]:
    """Check the model can hold a JSON schema, which the grader nodes need."""
    from pydantic import BaseModel, Field

    class _Probe(BaseModel):
        relevant: bool = Field(description="true if the text mentions cats")
        reason: str = Field(description="one short sentence")

    cfg = cfg or get_settings()
    try:
        structured = build_chat_model(cfg, temperature=0.0).with_structured_output(_Probe)
        out = structured.invoke(
            "Text: 'The cat sat on the mat.' Question: does the text mention cats?"
        )
        if isinstance(out, _Probe):
            return True, f"ok (relevant={out.relevant})"
        if isinstance(out, dict) and "relevant" in out:
            return True, f"ok (relevant={out['relevant']})"
        return False, f"unexpected shape: {type(out).__name__}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
