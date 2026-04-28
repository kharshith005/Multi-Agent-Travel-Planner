"""Tests for agents/runtime_status.py — env auth feature flags."""
from __future__ import annotations

import pytest

from agents.runtime_status import flights_available, llm_available, places_available


def test_llm_available_false_when_missing(monkeypatch):
    monkeypatch.delenv("VERTEX_AI_API_KEY", raising=False)
    assert not llm_available()

def test_llm_available_true_when_set(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_API_KEY", "some-key")
    assert llm_available()

def test_llm_available_false_for_empty_string(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_API_KEY", "   ")
    assert not llm_available()

def test_flights_available_false_when_missing(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert not flights_available()

def test_flights_available_true_when_set(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "some-key")
    assert flights_available()

def test_places_available_false_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    assert not places_available()

def test_places_available_true_when_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "some-key")
    assert places_available()

def test_all_false_with_no_env(monkeypatch):
    for key in ("VERTEX_AI_API_KEY", "SERPAPI_API_KEY", "GOOGLE_MAPS_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert not llm_available()
    assert not flights_available()
    assert not places_available()
