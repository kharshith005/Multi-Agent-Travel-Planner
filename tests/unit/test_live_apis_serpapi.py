"""Tests for SerpAPI (flight search) integration in tools/live_apis.py.

Key scenarios:
- Missing key raises immediately (no silent swallow)
- Valid key returns a list (or empty) — live test, skipped when key absent
- Coordinator's research_node uses flights_available() to surface status
"""
from __future__ import annotations

import os

import pytest

from agents.runtime_status import flights_available


# -- Key validation -----------------------------------------------------------

def test_serpapi_key_missing_raises(monkeypatch):
    """flight_search() must raise when SERPAPI_API_KEY is absent, never silently return []."""
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    from tools.live_apis import LiveTravelAPIs

    apis = LiveTravelAPIs(google_maps_api_key="", serpapi_api_key="")
    with pytest.raises(Exception):
        # Should raise RuntimeError -- not return an empty list
        apis.flight_search("Washington DC", "Myrtle Beach", "2026-06-01")


def test_flights_available_false_without_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert not flights_available()


def test_flights_available_true_with_key(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake-key")
    assert flights_available()


# -- Live integration (skipped without real key) ------------------------------

@pytest.mark.skipif(
    not os.environ.get("SERPAPI_API_KEY"),
    reason="SERPAPI_API_KEY not set -- skipping live flight search test",
)
def test_live_flight_search_returns_list():
    from tools.live_apis import default_live_apis
    apis = default_live_apis()
    # Use IATA codes to bypass city-name to airport resolution
    results = apis.flight_search("JFK", "LAX", "2026-07-01")
    assert isinstance(results, list)


@pytest.mark.skipif(
    not os.environ.get("SERPAPI_API_KEY"),
    reason="SERPAPI_API_KEY not set -- skipping live flight search test",
)
def test_live_flight_search_result_format():
    from tools.live_apis import default_live_apis
    apis = default_live_apis()
    results = apis.flight_search("JFK", "LAX", "2026-07-15")
    if results:
        # Results should be formatted strings (not raw dicts)
        assert isinstance(results[0], str)
        assert len(results[0]) > 5
