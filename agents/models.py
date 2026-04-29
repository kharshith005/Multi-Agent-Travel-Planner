"""Model registry for the multi-provider LLM layer.

Lists all supported models with their metadata and auth requirements.
`available_models()` filters to those whose auth is configured in the environment.

GPT/OpenAI models are intentionally excluded: Vertex AI does not host them.
Claude entries are silently dropped when Vertex ADC + GCP project are not configured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    id: str
    display_name: str
    provider: str            # "gemini" | "claude"
    tier: str                # "lite" | "standard"
    family: str
    default_temperature: float
    est_relative_cost: float  # relative to gemini-2.5-flash-lite = 1.0
    auth_mode: str           # "vertex_api_key" | "vertex_adc"
    available_regions: tuple[str, ...] = ()   # empty = no regional restriction
    est_latency_ms_per_call: int = 1000       # rough per-call latency estimate


# Registry — exact GA model IDs pinned per Phase 0 checklist.
# Claude IDs follow Vertex AI Model Garden format: <model>@<revision>.
# GPT/OpenAI is excluded: Vertex AI does not host OpenAI models (see ARCHITECTURE.md §4.3).
#
# Cost calibration (relative to gemini-2.5-flash-lite ≈ $0.10/1M input tokens):
#   Gemini 2.0 Flash       ~$0.25/M  → 2.5×
#   Gemini 2.5 Flash-Lite  ~$0.10/M  → 1.0× (baseline)
#   Gemini 2.5 Flash       ~$0.40/M  → 4.0×
#   Claude Haiku 4.5       ~$0.80/M  → 8.0× (input); output 5× pricier — blended ≈ 12×
#   Claude Sonnet 4.5/4.6  ~$3.00/M  → 30× (input); output 5× pricier — blended ≈ 37×
# Claude models require GOOGLE_CLOUD_PROJECT + CLAUDE_VERTEX_REGION=global (ADC auth).
# region="global" uses Anthropic's pooled Vertex endpoint with higher token quota.
REGISTRY: list[ModelEntry] = [
    ModelEntry(
        id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite (default)",
        provider="gemini",
        tier="lite",
        family="gemini-2.5",
        default_temperature=0.0,
        est_relative_cost=1.0,
        auth_mode="vertex_api_key",
        est_latency_ms_per_call=600,
    ),
    ModelEntry(
        id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        provider="gemini",
        tier="standard",
        family="gemini-2.5",
        default_temperature=0.0,
        est_relative_cost=4.0,
        auth_mode="vertex_api_key",
        est_latency_ms_per_call=900,
    ),
    ModelEntry(
        id="claude-haiku-4-5@20251001",
        display_name="Claude Haiku 4.5",
        provider="claude",
        tier="lite",
        family="claude-4",
        default_temperature=0.0,
        est_relative_cost=12.0,
        auth_mode="vertex_adc",
        available_regions=("global",),
        est_latency_ms_per_call=500,
    ),
]

_REGISTRY_BY_ID: dict[str, ModelEntry] = {m.id.split("@", 1)[0]: m for m in REGISTRY}


def get_model(model_id: str) -> ModelEntry:
    """Look up a model by ID. Raises KeyError if not found."""
    bare = model_id.split("@", 1)[0]
    entry = _REGISTRY_BY_ID.get(bare)
    if entry is None:
        raise KeyError(
            f"Model {model_id!r} not in registry. Available: {sorted(_REGISTRY_BY_ID)}"
        )
    return entry


def _gemini_auth_ok() -> bool:
    return bool(os.environ.get("VERTEX_AI_API_KEY", "").strip())


def _claude_auth_ok() -> bool:
    return bool(
        os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        and os.environ.get("CLAUDE_VERTEX_REGION", "").strip()
    )


def available_models() -> list[ModelEntry]:
    """Return models whose auth requirements are satisfied in the current environment.

    Claude entries are silently omitted when GOOGLE_CLOUD_PROJECT or
    CLAUDE_VERTEX_REGION are not set (Vertex ADC not configured).
    """
    gem_ok = _gemini_auth_ok()
    cl_ok = _claude_auth_ok()
    return [
        m for m in REGISTRY
        if (m.auth_mode == "vertex_api_key" and gem_ok)
        or (m.auth_mode == "vertex_adc" and cl_ok)
    ]


def default_model_id() -> str:
    """Return the default model ID from env or fall back to gemini-2.5-flash-lite."""
    return os.environ.get("LLM_MODEL", "").strip() or "gemini-2.5-flash-lite"
