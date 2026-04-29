"""Tests for agents/models.py — model registry and filtering."""
from __future__ import annotations

import os

import pytest

from agents.models import REGISTRY, ModelEntry, available_models, default_model_id, get_model


# ── REGISTRY integrity ────────────────────────────────────────────────────────

def test_registry_not_empty():
    assert len(REGISTRY) >= 1

def test_registry_all_have_required_fields():
    for m in REGISTRY:
        assert m.id and m.display_name and m.provider
        assert m.provider in {"gemini", "claude"}
        assert m.auth_mode in {"vertex_api_key", "vertex_adc"}

def test_registry_ids_unique():
    ids = [m.id for m in REGISTRY]
    assert len(ids) == len(set(ids))

def test_registry_gemini_entries():
    gemini = [m for m in REGISTRY if m.provider == "gemini"]
    assert len(gemini) >= 1

def test_registry_claude_entries():
    claude = [m for m in REGISTRY if m.provider == "claude"]
    assert len(claude) >= 3  # Haiku 4.5, Sonnet 4.5, Sonnet 4.6

def test_registry_has_haiku_45():
    m = get_model("claude-haiku-4-5")
    assert m.tier == "lite"
    assert m.est_relative_cost == 12.0

def test_registry_has_sonnet_45():
    m = get_model("claude-sonnet-4-5")
    assert m.tier == "standard"
    assert m.est_relative_cost == 37.0

def test_registry_has_sonnet_46():
    m = get_model("claude-sonnet-4-6")
    assert m.tier == "standard"
    assert m.est_relative_cost == 37.0

def test_registry_available_regions_field():
    haiku = get_model("claude-haiku-4-5")
    assert "global" in haiku.available_regions
    sonnet = get_model("claude-sonnet-4-6")
    assert "global" in sonnet.available_regions

def test_registry_latency_field():
    haiku = get_model("claude-haiku-4-5")
    assert haiku.est_latency_ms_per_call < 1000  # haiku is faster than sonnet
    sonnet = get_model("claude-sonnet-4-6")
    assert sonnet.est_latency_ms_per_call >= 1000

def test_registry_gemini_no_region_restriction():
    m = get_model("gemini-2.5-flash-lite")
    assert m.available_regions == ()  # no restriction


# ── get_model ────────────────────────────────────────────────────────────────

def test_get_model_known():
    m = get_model("gemini-2.5-flash-lite")
    assert m.provider == "gemini"

def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        get_model("nonexistent-model-id")


# ── available_models ──────────────────────────────────────────────────────────

def test_available_models_empty_without_keys(monkeypatch):
    monkeypatch.delenv("VERTEX_AI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("CLAUDE_VERTEX_REGION", raising=False)
    result = available_models()
    assert result == []

def test_available_models_gemini_with_api_key(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_API_KEY", "fake-key")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    result = available_models()
    providers = {m.provider for m in result}
    assert "gemini" in providers
    assert "claude" not in providers

def test_available_models_claude_needs_both_vars(monkeypatch):
    monkeypatch.delenv("VERTEX_AI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("CLAUDE_VERTEX_REGION", "us-east5")
    result = available_models()
    providers = {m.provider for m in result}
    assert "claude" in providers

def test_available_models_claude_missing_region(monkeypatch):
    monkeypatch.delenv("VERTEX_AI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.delenv("CLAUDE_VERTEX_REGION", raising=False)
    result = available_models()
    providers = {m.provider for m in result}
    assert "claude" not in providers


# ── default_model_id ──────────────────────────────────────────────────────────

def test_default_model_id_fallback(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert default_model_id() == "gemini-2.5-flash-lite"

def test_default_model_id_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    assert default_model_id() == "gemini-2.0-flash"

def test_default_model_id_ignores_whitespace(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "  gemini-2.0-flash  ")
    assert default_model_id() == "gemini-2.0-flash"
