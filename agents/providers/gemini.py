"""Gemini backend via google-genai in Vertex API-key mode.

Lifted from agents/llm.py (original single-provider implementation).
Auth: VERTEX_AI_API_KEY environment variable.
"""
from __future__ import annotations

import os
import re
import threading

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass


_client_lock = threading.Lock()
_genai_client = None

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRY_HINTS = (
    "rate_limit",
    "too many requests",
    "high demand",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
)


def _vertex_api_key() -> str:
    key = os.environ.get("VERTEX_AI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set VERTEX_AI_API_KEY for Vertex AI / Gemini mode")
    return key


def _client():
    global _genai_client
    if genai is None:
        raise RuntimeError("google-genai package not installed. Run: pip install google-genai")
    with _client_lock:
        if _genai_client is None:
            _genai_client = genai.Client(vertexai=True, api_key=_vertex_api_key())
        return _genai_client


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


def _extract_text(resp: object) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    candidates = getattr(resp, "candidates", None) or []
    parts_out: list[str] = []
    for c in candidates:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            t = getattr(p, "text", None)
            if isinstance(t, str) and t:
                parts_out.append(t)
    return "".join(parts_out)


def generate(
    model_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    json_mode: bool,
) -> tuple[str, int, int]:
    """Call the Gemini model and return (text, input_tokens, output_tokens)."""
    client = _client()
    generation_config: dict[str, object] = {
        "max_output_tokens": max_tokens,
        "system_instruction": system,
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=user,
            config=generation_config,
        )
    except Exception as e:
        status = _status_code(e)
        message = str(e).strip() or e.__class__.__name__
        if status == 403 and (
            "api_key_service_blocked" in message.lower()
            or "permission_denied" in message.lower()
            or "generativelanguage.googleapis.com" in message.lower()
            or "aiplatform.googleapis.com" in message.lower()
        ):
            message = (
                "LLM access is blocked for the configured provider/project. "
                "Enable required APIs and remove restrictive key/application restrictions. "
                f"Details: {message}"
            )
        err = RuntimeError(
            f"Gemini request failed ({status if status is not None else 'error'}): {message}. "
            "If this is a quota/rate-limit issue, reduce request volume and verify Vertex billing/quota."
        )
        setattr(err, "status_code", status)
        raise err from e

    text = _extract_text(resp)
    if not text:
        raise RuntimeError("Gemini request returned empty response")

    usage = getattr(resp, "usage_metadata", None)
    in_tok = 0
    out_tok = 0
    if usage is not None:
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0

    return text, in_tok, out_tok
