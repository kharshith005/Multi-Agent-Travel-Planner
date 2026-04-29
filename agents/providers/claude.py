"""Claude backend via anthropic[vertex] (AnthropicVertex client).

Auth: Application Default Credentials (ADC) + GOOGLE_CLOUD_PROJECT + CLAUDE_VERTEX_REGION.
Structured output uses forced tool-use: the Pydantic schema is declared as a single
tool and the model is forced to call it, so the response is always valid JSON.

If GOOGLE_CLOUD_PROJECT or CLAUDE_VERTEX_REGION are not set, all calls raise
RuntimeError immediately (caller should check available_models() first).
"""
from __future__ import annotations

import functools
import json
import os
import sys

try:
    from anthropic import AnthropicVertex
except ImportError:
    AnthropicVertex = None  # type: ignore[assignment,misc]

# Fixed 2-minute timeout for any single API call.
API_TIMEOUT_SECONDS = 120.0


def _project() -> str:
    p = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not p:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT for Claude on Vertex AI. "
            "See .env.example for all required Claude env vars."
        )
    return p


def _region() -> str:
    r = os.environ.get("CLAUDE_VERTEX_REGION", "").strip()
    if not r:
        raise RuntimeError(
            "Set CLAUDE_VERTEX_REGION (e.g. global) for Claude on Vertex AI."
        )
    return r


@functools.lru_cache(maxsize=1)
def _build_client():
    if AnthropicVertex is None:
        raise RuntimeError(
            "anthropic[vertex] package not installed. "
            "Run: pip install 'anthropic[vertex]'"
        )
    # max_retries=0: agents/llm.py owns the retry loop. Without this,
    # every 429 burns N×2 quota slots (SDK default max_retries=2),
    # which on a tight Vertex quota rapidly self-perpetuates the 429.
    return AnthropicVertex(
        region=_region(),
        project_id=_project(),
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _client():
    return _build_client()


def clear_client_cache() -> None:
    """Clear the cached AnthropicVertex client (for tests or auth rotation)."""
    _build_client.cache_clear()


def _vertex_model_id(model_id: str) -> str:
    """Strip @revision suffix — Vertex SDK expects bare model names (e.g. claude-sonnet-4-6)."""
    return model_id.split("@", 1)[0]


def generate(
    model_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    schema_cls: type | None = None,
) -> tuple[str, int, int]:
    """Call Claude on Vertex and return (text, input_tokens, output_tokens).

    When schema_cls is provided, uses forced tool-use structured output so the
    response is always valid JSON matching the Pydantic schema.
    When schema_cls is None, returns raw text (for call_text).
    """
    client = _client()
    vertex_id = _vertex_model_id(model_id)

    if os.environ.get("CLAUDE_DEBUG") == "1":
        print(
            f"[claude] model={vertex_id} region={_region()} project={_project()[:8]}…",
            file=sys.stderr,
        )

    messages = [{"role": "user", "content": user}]

    try:
        if schema_cls is not None:
            schema = schema_cls.model_json_schema()
            tool = {
                "name": "structured_output",
                "description": f"Produce {schema_cls.__name__} structured output.",
                "input_schema": schema,
            }
            resp = client.messages.create(
                model=vertex_id,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": "structured_output"},
                timeout=API_TIMEOUT_SECONDS,
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "structured_output":
                    text = json.dumps(block.input)
                    break
            else:
                raise RuntimeError(
                    "Claude did not call the structured_output tool. "
                    "This indicates a tool-use forcing failure."
                )
        else:
            resp = client.messages.create(
                model=vertex_id,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            parts = [
                block.text
                for block in resp.content
                if getattr(block, "type", None) == "text"
            ]
            text = "".join(parts)
            if not text.strip():
                raise RuntimeError("Claude returned empty response")

    except RuntimeError:
        raise
    except Exception as e:
        status = getattr(e, "status_code", None)
        err = RuntimeError(
            f"Claude request failed ({status or 'error'}): {e}. "
            "Check GOOGLE_CLOUD_PROJECT, CLAUDE_VERTEX_REGION (e.g. global), and ADC configuration."
        )
        setattr(err, "status_code", status)
        raise err from e

    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0

    return text, in_tok, out_tok
