"""Integration tests: Claude provider code path with stubbed AnthropicVertex.

Verifies:
1. Claude backend is selected when a claude model_id is used via call_provider.
2. Forced tool-use response parsing produces valid JSON output.
3. Cache key separation: same prompt on Gemini vs Claude model = two distinct misses.
"""
from __future__ import annotations

import json
import types
from unittest.mock import MagicMock, patch

import pytest

from agents.llm import _cache_key, _cache_get
from agents.models import get_model
from agents.schemas import Intent


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_fake_anthropic_response(input_data: dict) -> MagicMock:
    """Build a mock that looks like the AnthropicVertex messages.create response."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "structured_output"
    block.input = input_data

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50

    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    return resp


_VALID_INTENT_DICT = {
    "org": "Washington DC",
    "dest": "Myrtle Beach",
    "days": 3,
    "dates": ["2026-06-01", "2026-06-02", "2026-06-03"],
    "people": 2,
    "budget": 1400,
    "house_rule": None,
    "cuisine": None,
    "room_type": None,
    "transportation": None,
}


# ── Claude generate() ─────────────────────────────────────────────────────────

def test_claude_generate_with_schema_returns_json(monkeypatch):
    """claude.generate() with schema_cls uses forced tool-use and returns JSON text."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("CLAUDE_VERTEX_REGION", "global")

    fake_resp = _make_fake_anthropic_response(_VALID_INTENT_DICT)

    import agents.providers.claude as claude_mod
    claude_mod.clear_client_cache()

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp

    with patch.object(claude_mod, "_build_client", return_value=fake_client):
        # _client() calls _build_client() but the lru_cache means we need to clear first
        claude_mod.clear_client_cache()
        # Now _build_client is patched to return fake_client directly
        # Bypass the lru_cache by patching _client
        with patch.object(claude_mod, "_client", return_value=fake_client):
            text, in_tok, out_tok = claude_mod.generate(
                "claude-haiku-4-5",
                system="Extract intent.",
                user="3-day trip from Washington DC to Myrtle Beach under $1400",
                max_tokens=1024,
                schema_cls=Intent,
            )

    parsed = json.loads(text)
    assert parsed["org"] == "Washington DC"
    assert parsed["dest"] == "Myrtle Beach"
    assert in_tok == 100
    assert out_tok == 50


def test_claude_generate_without_schema_returns_text(monkeypatch):
    """claude.generate() without schema_cls returns raw text."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("CLAUDE_VERTEX_REGION", "global")

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello from Claude!"

    usage = MagicMock()
    usage.input_tokens = 20
    usage.output_tokens = 5

    resp = MagicMock()
    resp.content = [text_block]
    resp.usage = usage

    fake_client = MagicMock()
    fake_client.messages.create.return_value = resp

    import agents.providers.claude as claude_mod
    with patch.object(claude_mod, "_client", return_value=fake_client):
        text, in_tok, out_tok = claude_mod.generate(
            "claude-haiku-4-5",
            system="You are helpful.",
            user="Say hello.",
            max_tokens=128,
            schema_cls=None,
        )

    assert text == "Hello from Claude!"
    assert in_tok == 20


def test_claude_generate_missing_tool_call_raises(monkeypatch):
    """If Claude doesn't call the tool, generate() raises RuntimeError."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("CLAUDE_VERTEX_REGION", "global")

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I refuse to use the tool."

    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 10

    resp = MagicMock()
    resp.content = [text_block]
    resp.usage = usage

    fake_client = MagicMock()
    fake_client.messages.create.return_value = resp

    import agents.providers.claude as claude_mod
    with patch.object(claude_mod, "_client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="structured_output tool"):
            claude_mod.generate(
                "claude-haiku-4-5",
                system="Extract.",
                user="trip",
                max_tokens=256,
                schema_cls=Intent,
            )


# ── provider routing via call_provider ────────────────────────────────────────

def test_call_provider_routes_to_claude(monkeypatch):
    """call_provider dispatches claude models to claude.generate."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("CLAUDE_VERTEX_REGION", "global")

    model = get_model("claude-haiku-4-5")
    assert model.provider == "claude"

    import agents.providers.claude as claude_mod
    with patch.object(claude_mod, "generate", return_value=('{"test": 1}', 10, 5)) as mock_gen:
        from agents.providers import call_provider
        result = call_provider(
            model,
            system="test sys",
            user="test user",
            max_tokens=512,
            json_mode=True,
            schema_cls=None,
        )
        mock_gen.assert_called_once()
        assert result[0] == '{"test": 1}'


def test_call_provider_routes_gemini_not_claude(monkeypatch):
    """call_provider dispatches gemini models to gemini.generate, not claude."""
    monkeypatch.setenv("VERTEX_AI_API_KEY", "fake-gemini-key")

    model = get_model("gemini-2.5-flash-lite")
    assert model.provider == "gemini"

    import agents.providers.gemini as gemini_mod
    with patch.object(gemini_mod, "generate", return_value=('{"ok": true}', 15, 8)) as mock_gen:
        from agents.providers import call_provider
        result = call_provider(
            model,
            system="test sys",
            user="test user",
            max_tokens=512,
            json_mode=True,
            schema_cls=None,
        )
        mock_gen.assert_called_once()
        assert result[0] == '{"ok": true}'


# ── cache key separation ─────────────────────────────────────────────────────

def test_cache_key_gemini_vs_claude_are_different():
    """The same prompt/schema on two different providers = distinct cache keys."""
    sys_prompt = "Extract intent."
    user_prompt = "3-day trip Washington DC to Myrtle Beach $1400"
    schema = "Intent"
    max_tok = 1024

    k_gemini = _cache_key("gemini-2.5-flash-lite", sys_prompt, user_prompt, schema, max_tok, False)
    k_claude = _cache_key("claude-haiku-4-5", sys_prompt, user_prompt, schema, max_tok, False)

    assert k_gemini != k_claude, (
        "Gemini and Claude must produce different cache keys for the same prompt. "
        "If they match, a Claude response could be served from Gemini's cache."
    )


def test_cache_key_two_claude_models_are_different():
    """Two different Claude model IDs also produce different cache keys."""
    k_haiku = _cache_key("claude-haiku-4-5", "sys", "user", "Intent", 512, False)
    k_sonnet = _cache_key("claude-sonnet-4-6", "sys", "user", "Intent", 512, False)
    assert k_haiku != k_sonnet


def test_cache_populated_for_gemini_not_returned_for_claude(monkeypatch, tmp_path):
    """A cache entry for model A must not be returned for model B."""
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)

    from agents.llm import _cache_put, _cache_get, _cache_key as ck

    sys_p = "sys-isolation-test"
    user_p = "user-isolation-test-abc123"
    schema = "IsolationSchema"

    k_gemini = ck("gemini-2.5-flash-lite", sys_p, user_p, schema, 1024, False)
    k_claude = ck("claude-haiku-4-5", sys_p, user_p, schema, 1024, False)

    # Populate gemini's slot
    _cache_put(k_gemini, '{"gemini": true}', _write_disk=False)

    # Claude's key must be a miss
    assert _cache_get(k_claude) is None, (
        "Claude cache lookup returned Gemini's cached value — cache key separation broken."
    )
