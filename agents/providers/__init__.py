"""Provider router — dispatches to Gemini or Claude backends.

Usage (via agents/llm.py facade only):
    from agents.providers import call_provider
    text, in_tok, out_tok = call_provider(model_entry, system, user, ...)

To invalidate cached singleton clients (e.g., after auth rotation or in tests):
    from agents.providers import clear_caches
    clear_caches()
"""
from __future__ import annotations

from agents.models import ModelEntry


def clear_caches() -> None:
    """Clear all cached LLM client singletons so the next call re-initializes them."""
    try:
        from .gemini import clear_client_cache as _g
        _g()
    except Exception:
        pass
    try:
        from .claude import clear_client_cache as _c
        _c()
    except Exception:
        pass


def call_provider(
    model: ModelEntry,
    system: str,
    user: str,
    *,
    max_tokens: int,
    json_mode: bool,
    schema_cls: type | None = None,
) -> tuple[str, int, int]:
    """Route to the correct provider backend.

    Returns (raw_text, input_tokens, output_tokens).
    raw_text is the full model response (JSON string for json_mode calls).
    """
    if model.provider == "gemini":
        from .gemini import generate as gemini_generate
        return gemini_generate(model.id, system, user, max_tokens=max_tokens, json_mode=json_mode)
    elif model.provider == "claude":
        from .claude import generate as claude_generate
        return claude_generate(
            model.id, system, user,
            max_tokens=max_tokens,
            schema_cls=schema_cls,
        )
    else:
        raise ValueError(
            f"Unknown provider {model.provider!r} for model {model.id!r}. "
            "Must be 'gemini' or 'claude'."
        )
