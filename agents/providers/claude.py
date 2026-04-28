"""Claude backend via anthropic[vertex] (AnthropicVertex client).

Auth: Application Default Credentials (ADC) + GOOGLE_CLOUD_PROJECT + CLAUDE_VERTEX_REGION.
Structured output uses forced tool-use: the Pydantic schema is declared as a single
tool and the model is forced to call it, so the response is always valid JSON.

If GOOGLE_CLOUD_PROJECT or CLAUDE_VERTEX_REGION are not set, all calls raise
RuntimeError immediately (caller should check available_models() first).
"""
from __future__ import annotations

import json
import os
import threading

try:
    from anthropic import AnthropicVertex
except ImportError:
    AnthropicVertex = None  # type: ignore[assignment,misc]

_client_lock = threading.Lock()
_claude_client = None


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
            "Set CLAUDE_VERTEX_REGION (e.g. us-east5) for Claude on Vertex AI."
        )
    return r


def _client():
    global _claude_client
    if AnthropicVertex is None:
        raise RuntimeError(
            "anthropic[vertex] package not installed. "
            "Run: pip install 'anthropic[vertex]'"
        )
    with _client_lock:
        if _claude_client is None:
            _claude_client = AnthropicVertex(region=_region(), project_id=_project())
        return _claude_client


def _vertex_model_id(model_id: str) -> str:
    """Strip registry @revision suffix if the SDK expects bare model names."""
    return model_id


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
            "Check GOOGLE_CLOUD_PROJECT, CLAUDE_VERTEX_REGION, and ADC configuration."
        )
        setattr(err, "status_code", status)
        raise err from e

    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0

    return text, in_tok, out_tok
