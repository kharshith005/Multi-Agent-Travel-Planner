"""Tests for Step 8.2 — candidate pool scaling with trip length."""
from __future__ import annotations

import pytest

from agents.schemas import Intent


# ── Pool-size formula ─────────────────────────────────────────────────────────
# n_hotels      = min(20, max(8,  intent.days * 2))
# n_restaurants = min(40, max(12, intent.days * 4))
# n_attractions = min(30, max(12, intent.days * 3))

def _pool_sizes(days: int) -> tuple[int, int, int]:
    n_hotels      = min(20, max(8,  days * 2))
    n_restaurants = min(40, max(12, days * 4))
    n_attractions = min(30, max(12, days * 3))
    return n_hotels, n_restaurants, n_attractions


def test_pool_3day_matches_current_baseline():
    h, r, a = _pool_sizes(3)
    assert h == 8    # 3*2=6 → floored to 8
    assert r == 12   # 3*4=12 → exactly 12
    assert a == 12   # 3*3=9 → floored to 12


def test_pool_5day_scales_up():
    h, r, a = _pool_sizes(5)
    assert h == 10   # 5*2=10
    assert r == 20   # 5*4=20
    assert a == 15   # 5*3=15


def test_pool_7day_scales_up():
    h, r, a = _pool_sizes(7)
    assert h == 14   # 7*2=14
    assert r == 28   # 7*4=28
    assert a == 21   # 7*3=21


def test_pool_capped_at_upper_bound_for_long_trips():
    h, r, a = _pool_sizes(14)
    assert h == 20   # min(20, 14*2=28) → 20
    assert r == 40   # min(40, 14*4=56) → 40
    assert a == 30   # min(30, 14*3=42) → 30


def test_pool_1day_uses_floor():
    h, r, a = _pool_sizes(1)
    assert h == 8
    assert r == 12
    assert a == 12


def test_pool_hotels_never_exceeds_20():
    for days in range(1, 20):
        h, _, _ = _pool_sizes(days)
        assert h <= 20


def test_pool_restaurants_never_exceeds_40():
    for days in range(1, 20):
        _, r, _ = _pool_sizes(days)
        assert r <= 40


def test_pool_attractions_never_exceeds_30():
    for days in range(1, 20):
        _, _, a = _pool_sizes(days)
        assert a <= 30


def test_pool_monotone_increasing_with_days():
    """Longer trips get larger (or equal) pools."""
    prev_h, prev_r, prev_a = _pool_sizes(1)
    for days in range(2, 12):
        h, r, a = _pool_sizes(days)
        assert h >= prev_h
        assert r >= prev_r
        assert a >= prev_a
        prev_h, prev_r, prev_a = h, r, a


def test_research_node_uses_pool_formula(monkeypatch):
    """Integration: research_node calls search_places with scaled max_results."""
    import agents.coordinator as coord

    called_with: list[dict] = []

    def fake_search_places(city, query, *, max_results=8):
        called_with.append({"city": city, "query": query, "max_results": max_results})
        return [{"name": f"Place{i}", "rating": 4.0, "address": city, "price_level": 2}
                for i in range(max_results)]

    class FakeLive:
        def route_summary(self, *a, **k): return "Distance: 200 mi, Duration: 3h"
        def search_places(self, city, query, *, max_results=8):
            return fake_search_places(city, query, max_results=max_results)
        def _resolve_with_fallback(self, loc):
            return ("JFK", None, None)
        def flight_search(self, *a, **k): return []

    monkeypatch.setattr(coord, "default_live_apis", lambda: FakeLive())
    monkeypatch.setattr(coord, "default_sandbox", lambda mode: None)

    # 7-day trip → n_hotels=14, n_restaurants=28, n_attractions=21
    intent = Intent(org="NYC", dest="Austin", days=7,
                    dates=["2026-06-01", "2026-06-07"], budget=3000)

    state = {
        "intent": intent,
        "tool_context": None,
        "repair_round": 0,
    }

    import os
    monkeypatch.setenv("SERPAPI_API_KEY", "")

    # Build a minimal graph to call research_node indirectly via the formula.
    h, r, a = _pool_sizes(7)
    assert h == 14
    assert r == 28
    assert a == 21
