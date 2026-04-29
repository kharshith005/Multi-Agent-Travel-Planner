"""Tests for the LLM cache layer in agents/llm.py.

These tests exercise the cache machinery without making real API calls.
"""
from __future__ import annotations

import hashlib
import json
import threading

import pytest

from agents.llm import (
    _cache_get,
    _cache_key,
    _cache_put,
    get_call_stats,
    reset_call_stats,
)


# ── cache key ────────────────────────────────────────────────────────────────

def test_cache_key_is_deterministic():
    k1 = _cache_key("gemini-2.5-flash-lite", "sys", "user", "Intent", 1024, False)
    k2 = _cache_key("gemini-2.5-flash-lite", "sys", "user", "Intent", 1024, False)
    assert k1 == k2

def test_cache_key_differs_by_model():
    k1 = _cache_key("gemini-2.5-flash-lite", "sys", "user", "Intent", 1024, False)
    k2 = _cache_key("gemini-2.0-flash", "sys", "user", "Intent", 1024, False)
    assert k1 != k2

def test_cache_key_differs_by_provider():
    # Claude and Gemini with same text should produce different keys
    k1 = _cache_key("claude-haiku-4-5", "sys", "user", "Intent", 1024, False)
    k2 = _cache_key("gemini-2.5-flash-lite", "sys", "user", "Intent", 1024, False)
    assert k1 != k2

def test_cache_key_differs_by_schema():
    k1 = _cache_key("gemini-2.5-flash-lite", "sys", "user", "Intent", 1024, False)
    k2 = _cache_key("gemini-2.5-flash-lite", "sys", "user", "TransportPlan", 1024, False)
    assert k1 != k2

def test_cache_key_differs_by_content():
    k1 = _cache_key("gemini-2.5-flash-lite", "sys A", "user", "Intent", 1024, False)
    k2 = _cache_key("gemini-2.5-flash-lite", "sys B", "user", "Intent", 1024, False)
    assert k1 != k2

def test_cache_key_is_hex_sha256():
    k = _cache_key("m", "s", "u", "Schema", 1024, False)
    assert len(k) == 64
    int(k, 16)  # should not raise


# ── in-memory cache put/get ───────────────────────────────────────────────────

def test_cache_put_get_roundtrip(monkeypatch):
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    key = _cache_key("gemini-2.5-flash-lite", "test-sys", "test-user-" + __name__, "Sc", 512, False)
    _cache_put(key, '{"test": 1}', _write_disk=False)
    result = _cache_get(key)
    assert result == '{"test": 1}'

def test_cache_miss_returns_none(monkeypatch):
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    key = _cache_key("model-x", "nonexistent-sys-xyz", "nonexistent-user-xyz", "Sc", 1, False)
    assert _cache_get(key) is None


# ── stats ────────────────────────────────────────────────────────────────────

def test_reset_clears_all_counters():
    reset_call_stats()
    stats = get_call_stats()
    assert stats["llm_calls"] == 0
    assert stats["input_tokens"] == 0
    assert stats["output_tokens"] == 0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0

def test_get_call_stats_returns_copy():
    reset_call_stats()
    s1 = get_call_stats()
    s1["llm_calls"] = 999
    s2 = get_call_stats()
    assert s2["llm_calls"] == 0  # mutation of returned dict must not affect internal state


# ── disk cache (in temp dir) ──────────────────────────────────────────────────

def test_disk_cache_put_and_get(tmp_path, monkeypatch):
    import agents.llm as llm_mod
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    key = _cache_key("g", "disk-sys", "disk-user-" + str(tmp_path), "Sc", 512, False)
    _cache_put(key, '{"disk": true}')
    # Clear in-memory cache so we're forced to load from disk
    import agents.llm as _llm
    with _llm._cache_lock:
        _llm._json_cache.pop(key, None)
    result = _cache_get(key)
    assert result is not None
    assert '"disk"' in result
