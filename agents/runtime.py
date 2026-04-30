"""Per-request model context variable.

Sets the active LLM model for a planning call without threading model_id through
every function signature. Safe for LangGraph parallel nodes — the ContextVar is
set within _run_specialist (called inside each node's thread) so each thread sees
the correct value.

Usage:
    from agents.runtime import use_model
    with use_model("gemini-2.5-flash"):
        result = plan_trip(query)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

_current_model: ContextVar[str] = ContextVar("llm_model", default="")


def current_model() -> str:
    """Return the canonical model ID for this context (env fallback → registry default).

    Short aliases (e.g. 'llama-3.3', 'mistral-small-3.1') are resolved to their
    full registry IDs so cache keys and provider lookups are always consistent.
    """
    from agents.models import resolve_model_id
    val = _current_model.get()
    if val:
        return resolve_model_id(val)
    raw = os.environ.get("LLM_MODEL", "").strip()
    return resolve_model_id(raw) if raw else "gemini-2.5-flash-lite"


@contextmanager
def use_model(model_id: str) -> Generator[None, None, None]:
    """Context manager that sets the active model for the duration of the block.

    Accepts short aliases — they are resolved to canonical IDs by current_model().
    """
    token = _current_model.set(model_id)
    try:
        yield
    finally:
        _current_model.reset(token)
