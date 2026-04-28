"""Tests for agents/runtime.py — ContextVar model switching."""
from __future__ import annotations

import threading

import pytest

from agents.runtime import current_model, use_model


def test_default_model_is_gemini_flash_lite(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert current_model() == "gemini-2.5-flash-lite"

def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    assert current_model() == "gemini-2.0-flash"

def test_use_model_sets_and_restores():
    original = current_model()
    with use_model("gemini-2.5-flash"):
        assert current_model() == "gemini-2.5-flash"
    assert current_model() == original

def test_use_model_nested():
    with use_model("gemini-2.5-flash"):
        assert current_model() == "gemini-2.5-flash"
        with use_model("gemini-2.0-flash"):
            assert current_model() == "gemini-2.0-flash"
        assert current_model() == "gemini-2.5-flash"

def test_use_model_restores_on_exception():
    original = current_model()
    with pytest.raises(ValueError):
        with use_model("gemini-2.5-flash"):
            raise ValueError("test")
    assert current_model() == original

def test_use_model_thread_isolation():
    """Each thread sees its own model even when running simultaneously."""
    results: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def thread_a():
        with use_model("gemini-2.5-flash"):
            barrier.wait()
            results["a"] = current_model()

    def thread_b():
        with use_model("gemini-2.0-flash"):
            barrier.wait()
            results["b"] = current_model()

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start(); tb.start()
    ta.join(); tb.join()

    assert results["a"] == "gemini-2.5-flash"
    assert results["b"] == "gemini-2.0-flash"
