"""Tests for Google Maps (places/routes) integration in tools/live_apis.py."""
from __future__ import annotations

import os

import pytest

from agents.runtime_status import places_available


# ── Key validation ────────────────────────────────────────────────────────────

def test_maps_key_missing_raises():
    """search_places() must raise when GOOGLE_MAPS_API_KEY is absent."""
    from tools.live_apis import LiveTravelAPIs
    apis = LiveTravelAPIs(google_maps_api_key="", serpapi_api_key="")
    with pytest.raises(Exception):
        apis.search_places("Myrtle Beach", "hotels")


def test_places_available_false_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    assert not places_available()


def test_places_available_true_with_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")
    assert places_available()


# ── Live integration (skipped without real key) ───────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("GOOGLE_MAPS_API_KEY"),
    reason="GOOGLE_MAPS_API_KEY not set — skipping live maps test",
)
def test_live_search_places_hotels():
    from tools.live_apis import default_live_apis
    apis = default_live_apis()
    results = apis.search_places("Myrtle Beach, SC", "hotels", max_results=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "name" in results[0]


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_MAPS_API_KEY"),
    reason="GOOGLE_MAPS_API_KEY not set — skipping live maps test",
)
def test_live_route_summary():
    from tools.live_apis import default_live_apis
    apis = default_live_apis()
    result = apis.route_summary("Washington DC", "Myrtle Beach", "driving")
    assert result is not None
    assert "Distance" in result
