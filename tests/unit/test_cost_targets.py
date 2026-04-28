"""Tests for Step 8.3 — budget-derived cost targets."""
from __future__ import annotations

import pytest

from agents.budget import (
    _DEFAULT_LODGING_TABLE,
    _DEFAULT_MEAL_TABLE,
    allocate,
    derive_lodging_targets,
    derive_meal_targets,
    derive_transport_target,
)
from agents.schemas import Intent


def _intent(budget=1400, days=3, **kw) -> Intent:
    return Intent(org="A", dest="B", days=days, budget=budget,
                  dates=["2026-05-01"], **kw)


# ── derive_meal_targets ────────────────────────────────────────────────────────

def test_meal_targets_returns_5_levels():
    targets = derive_meal_targets(_intent())
    assert set(targets.keys()) == {0, 1, 2, 3, 4}


def test_meal_targets_ordered():
    targets = derive_meal_targets(_intent())
    assert targets[0] < targets[1] < targets[2] < targets[3] < targets[4]


def test_meal_targets_sum_under_dining_cap():
    intent = _intent(budget=1400, days=3)
    caps = allocate(intent)
    dining_cap = caps["dining"]
    targets = derive_meal_targets(intent)
    # Expected meals: days*3 - 2 = 7; use mid-tier (level 2) as proxy
    num_meals = max(1, intent.days * 3 - 2)
    assert targets[2] * num_meals <= dining_cap


def test_meal_targets_no_budget_falls_back_to_fixed_table():
    intent = Intent(org="A", dest="B", days=3, dates=["2026-05-01"], budget=None)
    assert derive_meal_targets(intent) == _DEFAULT_MEAL_TABLE


def test_meal_targets_scale_with_budget():
    low  = derive_meal_targets(_intent(budget=500))
    high = derive_meal_targets(_intent(budget=5000))
    # Wealthier budget → higher cost targets across all levels
    for level in range(5):
        assert high[level] >= low[level]


def test_meal_targets_floor_prevents_zero():
    intent = _intent(budget=10, days=14)
    targets = derive_meal_targets(intent)
    assert targets[0] >= 5
    assert targets[1] >= 8
    assert targets[2] >= 12


# ── derive_lodging_targets ────────────────────────────────────────────────────

def test_lodging_targets_returns_5_levels():
    targets = derive_lodging_targets(_intent())
    assert set(targets.keys()) == {0, 1, 2, 3, 4}


def test_lodging_targets_ordered():
    targets = derive_lodging_targets(_intent())
    assert targets[0] < targets[1] < targets[2] < targets[3] < targets[4]


def test_lodging_targets_per_night_under_lodging_cap():
    intent = _intent(budget=1400, days=3)
    caps = allocate(intent)
    lodging_cap = caps["lodging"]
    targets = derive_lodging_targets(intent)
    nights = max(1, intent.days - 1)
    assert targets[2] * nights <= lodging_cap


def test_lodging_targets_no_budget_falls_back_to_fixed_table():
    intent = Intent(org="A", dest="B", days=3, dates=["2026-05-01"], budget=None)
    assert derive_lodging_targets(intent) == _DEFAULT_LODGING_TABLE


def test_lodging_targets_floor_prevents_tiny_values():
    intent = _intent(budget=10, days=14)
    targets = derive_lodging_targets(intent)
    assert targets[0] >= 40
    assert targets[2] >= 80


# ── derive_transport_target ───────────────────────────────────────────────────

def test_transport_target_is_half_of_transport_cap():
    intent = _intent(budget=1400)
    caps = allocate(intent)
    expected = caps["transport"] // 2
    assert derive_transport_target(intent) == expected


def test_transport_target_no_budget_returns_100():
    intent = Intent(org="A", dest="B", days=3, dates=["2026-05-01"], budget=None)
    # allocate returns {} → caps.get("transport", 200) // 2 = 100
    assert derive_transport_target(intent) == 100


# ── ToolContext integration ────────────────────────────────────────────────────

def test_targets_present_in_tool_context():
    """Targets are attached to ToolContext when budget is set."""
    # This tests the data shape, not the coordinator wiring (which requires live APIs).
    from agents.budget import derive_meal_targets, derive_lodging_targets, derive_transport_target
    intent = _intent(budget=2000, days=5)
    ctx: dict = {
        "meal_cost_targets": derive_meal_targets(intent),
        "lodging_cost_targets": derive_lodging_targets(intent),
        "transport_one_way_target": derive_transport_target(intent),
    }
    assert isinstance(ctx["meal_cost_targets"], dict)
    assert isinstance(ctx["lodging_cost_targets"], dict)
    assert isinstance(ctx["transport_one_way_target"], int)
    assert ctx["transport_one_way_target"] > 0


# ── Dining prompt uses dynamic costs ─────────────────────────────────────────

def test_dining_format_uses_dynamic_cost_targets():
    """_format_tool_context uses meal_cost_targets from ctx when present."""
    from agents.dining import _format_tool_context
    ctx = {
        "restaurants": [
            {"name": "Fancy Place", "rating": 4.5, "price_level": 3},
            {"name": "Budget Diner", "rating": 3.8, "price_level": 0},
        ],
        "meal_cost_targets": {0: 5, 1: 8, 2: 20, 3: 50, 4: 80},
    }
    result = _format_tool_context(ctx)
    assert "est_cost:$50" in result   # level 3 → $50
    assert "est_cost:$5" in result    # level 0 → $5


def test_dining_format_falls_back_to_defaults_when_no_targets():
    from agents.dining import _format_tool_context
    ctx = {
        "restaurants": [
            {"name": "Mid Place", "rating": 4.0, "price_level": 2},
        ],
        # no meal_cost_targets key
    }
    result = _format_tool_context(ctx)
    assert "est_cost:$28" in result   # default level 2 = $28


# ── Lodging prompt uses dynamic costs ────────────────────────────────────────

def test_lodging_format_uses_dynamic_cost_targets():
    from agents.lodging import _format_tool_context
    ctx = {
        "hotels": [
            {"name": "Luxury Hotel", "rating": 4.9, "address": "Austin", "price_level": 4},
        ],
        "lodging_cost_targets": {0: 50, 1: 70, 2: 100, 3: 150, 4: 250},
    }
    result = _format_tool_context(ctx)
    assert "est_nightly:$250" in result


def test_lodging_format_falls_back_to_defaults():
    from agents.lodging import _format_tool_context
    ctx = {
        "hotels": [
            {"name": "Normal Hotel", "rating": 4.0, "address": "Austin", "price_level": 2},
        ],
    }
    result = _format_tool_context(ctx)
    assert "est_nightly:$130" in result   # default level 2


# ── Transport cap includes one-way target ─────────────────────────────────────

def test_transport_cap_includes_one_way_target():
    from agents.transport import _format_transport_cap
    ctx = {
        "category_caps": {"transport": 630},
        "transport_one_way_target": 315,
    }
    result = _format_transport_cap(ctx)
    assert "$315" in result
    assert "$630" in result


def test_transport_cap_no_targets_returns_empty():
    from agents.transport import _format_transport_cap
    assert _format_transport_cap({}) == ""
    assert _format_transport_cap(None) == ""
