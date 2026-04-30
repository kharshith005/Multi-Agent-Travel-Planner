"""Model registry for the multi-provider LLM layer.

Lists all supported models with their metadata and auth requirements.
`available_models()` filters to those whose auth is configured in the environment.

GPT/OpenAI models are intentionally excluded: Vertex AI does not host them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    id: str
    display_name: str
    provider: str            # "gemini" | "meta" | "mistral"
    tier: str                # "lite" | "standard"
    family: str
    default_temperature: float
    est_relative_cost: float  # relative to gemini-2.5-flash-lite = 1.0
    auth_mode: str           # "vertex_api_key" | "vertex_oauth"
    available_regions: tuple[str, ...] = ()   # empty = no regional restriction
    est_latency_ms_per_call: int = 1000       # rough per-call latency estimate


# Registry — exact GA model IDs pinned per Phase 0 checklist.
# GPT/OpenAI is excluded: Vertex AI does not host OpenAI models (see ARCHITECTURE.md §4.3).
#
# Cost calibration (relative to gemini-2.5-flash-lite ≈ $0.10/1M input tokens):
#   Gemini 2.5 Flash-Lite  ~$0.10/M  → 1.0× (baseline)
#   Mistral Small 3.1      ~$0.10/M  → ~1× (Vertex MaaS; calibrate after first run)
#   Llama 3.3 70B Instruct ~$0.53/M  → ~5× (Vertex MaaS; calibrate after first run)
# Vertex MaaS models (Llama/Mistral) require GOOGLE_CLOUD_PROJECT + google-auth (ADC).
# Models must be enabled in Vertex AI Model Garden before first use.
REGISTRY: list[ModelEntry] = [
    ModelEntry(
        id="gemini-3.1-flash-lite-preview",
        display_name="Gemini 3.1 Flash-Lite-Preview (default)",
        provider="gemini",
        tier="lite",
        family="gemini-3.1",
        default_temperature=0.0,
        est_relative_cost=1.0,
        auth_mode="vertex_api_key",
        est_latency_ms_per_call=600,
    ),
    ModelEntry(
        id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        provider="gemini",
        tier="lite",
        family="gemini-2.5",
        default_temperature=0.0,
        est_relative_cost=1.0,
        auth_mode="vertex_api_key",
        est_latency_ms_per_call=600,
    ),
    ModelEntry(
        id="llama-3.3-70b-instruct-maas",
        display_name="Llama 3.3 70B Instruct",
        provider="meta",
        tier="standard",
        family="llama-3",
        default_temperature=0.0,
        est_relative_cost=5.0,
        auth_mode="vertex_oauth",
        available_regions=("us-central1",),
        est_latency_ms_per_call=1200,
    ),
    ModelEntry(
        id="mistral-small-2503",
        display_name="Mistral Small 3.1",
        provider="mistral",
        tier="lite",
        family="mistral-3",
        default_temperature=0.0,
        est_relative_cost=1.0,
        auth_mode="vertex_oauth",
        available_regions=("us-central1", "europe-west4"),
        est_latency_ms_per_call=800,
    ),
]

_REGISTRY_BY_ID: dict[str, ModelEntry] = {m.id.split("@", 1)[0]: m for m in REGISTRY}

# Short aliases accepted in LLM_MODEL env var and --model CLI flag.
# Maps alias → full registry ID (before @revision stripping).
_ALIASES: dict[str, str] = {
    "llama-3.3":         "llama-3.3-70b-instruct-maas",
    "mistral-small-3.1": "mistral-small-2503",
    "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
}


def resolve_model_id(model_id: str) -> str:
    """Expand a short alias to the full registry ID, or return the input unchanged."""
    return _ALIASES.get(model_id.strip(), model_id.strip())


def get_model(model_id: str) -> ModelEntry:
    """Look up a model by ID or short alias. Raises KeyError if not found."""
    resolved = resolve_model_id(model_id)
    bare = resolved.split("@", 1)[0]
    entry = _REGISTRY_BY_ID.get(bare)
    if entry is None:
        raise KeyError(
            f"Model {model_id!r} not in registry. "
            f"Available IDs: {sorted(_REGISTRY_BY_ID)}. "
            f"Available aliases: {sorted(_ALIASES)}."
        )
    return entry


def _gemini_auth_ok() -> bool:
    return bool(os.environ.get("VERTEX_AI_API_KEY", "").strip())


def _vertex_oauth_ok() -> bool:
    """Llama/Mistral: need GOOGLE_CLOUD_PROJECT + google-auth (ADC) installed."""
    if not os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip():
        return False
    try:
        import google.auth  # noqa: F401
        return True
    except ImportError:
        return False


def available_models() -> list[ModelEntry]:
    """Return models whose auth requirements are satisfied in the current environment.

    Vertex OAuth entries (Llama/Mistral) are omitted when GOOGLE_CLOUD_PROJECT
    is unset or google-auth is not installed.
    """
    gem_ok = _gemini_auth_ok()
    oauth_ok = _vertex_oauth_ok()
    return [
        m for m in REGISTRY
        if (m.auth_mode == "vertex_api_key" and gem_ok)
        or (m.auth_mode == "vertex_oauth" and oauth_ok)
    ]


def default_model_id() -> str:
    """Return the resolved model ID from env or fall back to gemini-3.1-flash-lite-preview.

    Accepts short aliases (e.g. 'llama-3.3', 'mistral-small-3.1') defined in
    _ALIASES and expands them to the full registry ID automatically.
    """
    raw = os.environ.get("LLM_MODEL", "").strip()
    return resolve_model_id(raw) if raw else "gemini-3.1-flash-lite-preview"
