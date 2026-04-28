"""Integration tests: coordinator running in sandbox mode (no live APIs, no LLM calls).

These tests verify that the coordinator graph wires up correctly and the
InvalidUpdateError (H1) is permanently fixed for parallel execution mode.
They mock the LLM calls so they complete instantly without API keys.
"""
from __future__ import annotations

import pytest

from agents.coordinator import plan_trip
from agents.perf import Timeline, merge_timelines
from agents.schemas import (
    DayAttractions,
    DayLodging,
    DayMeals,
    DayTransport,
    DiningPlan,
    Intent,
    LodgingPlan,
    SightseeingPlan,
    TransportPlan,
    ToolContext,
)


def _make_tool_context(dest: str = "Myrtle Beach") -> ToolContext:
    """Minimal ToolContext that satisfies the coordinator without live APIs."""
    return {  # type: ignore[return-value]
        "flights_outbound": ["Delta 08:00->10:00 DCA-MYR Duration: 120 Cost: $200"],
        "flights_return": ["Delta 14:00->16:00 MYR-DCA Duration: 120 Cost: $200"],
        "route_drive": "Distance: 620 mi, Duration: 10 hours",
        "route_taxi": None,
        "hotels": [
            {"name": "Marriott Resort", "rating": 4.5, "address": dest, "price_level": 3},
            {"name": "Hilton Oceanfront", "rating": 4.2, "address": dest, "price_level": 2},
        ],
        "restaurants": [
            {"name": "Sea Captain's House", "rating": 4.5, "price_level": 2},
            {"name": "Croissants Bistro", "rating": 4.3, "price_level": 2},
            {"name": "Bubba's Fish Shack", "rating": 4.1, "price_level": 1},
            {"name": "Flying Fish Market", "rating": 4.4, "price_level": 2},
            {"name": "Perrone's Restaurant", "rating": 4.2, "price_level": 2},
            {"name": "Ciao Bella", "rating": 4.0, "price_level": 2},
        ],
        "attractions": [
            {"name": "Broadway at the Beach", "rating": 4.5, "address": dest},
            {"name": "Ripley's Aquarium", "rating": 4.4, "address": dest},
            {"name": "Myrtle Beach Boardwalk", "rating": 4.3, "address": dest},
        ],
    }


def _make_specialists():
    """Return specialist stub functions that return minimal valid plans."""

    def transport_fn(intent, tool_context, repair_note=""):
        return TransportPlan(days=[
            DayTransport(day=1, description="Delta 08:00->10:00; Cost: $200"),
            DayTransport(day=2, description="-"),
            DayTransport(day=3, description="Delta 14:00->16:00; Cost: $200"),
        ])

    def lodging_fn(intent, tool_context, repair_note=""):
        return LodgingPlan(days=[
            DayLodging(day=1, city=intent.dest,
                       description="Marriott Resort; Cost: $150 for 2 nights"),
            DayLodging(day=2, city=intent.dest, description="-"),
            DayLodging(day=3, city=intent.dest, description="-"),
        ])

    def dining_fn(intent, tool_context, repair_note=""):
        return DiningPlan(days=[
            DayMeals(day=1, city=intent.dest, lunch="Sea Captain's House; Cost: $25",
                     dinner="Croissants Bistro; Cost: $30"),
            DayMeals(day=2, city=intent.dest, breakfast="Cafe; Cost: $15",
                     lunch="Bubba's; Cost: $20", dinner="Flying Fish; Cost: $35"),
            DayMeals(day=3, city=intent.dest, breakfast="Bagels; Cost: $10"),
        ])

    def sightseeing_fn(intent, tool_context, repair_note=""):
        return SightseeingPlan(days=[
            DayAttractions(day=1, city=intent.dest, description="Broadway at the Beach"),
            DayAttractions(day=2, city=intent.dest, description="Ripley's Aquarium"),
            DayAttractions(day=3, city=intent.dest, description="-"),
        ])

    return transport_fn, lodging_fn, dining_fn, sightseeing_fn


@pytest.fixture()
def _patch_specialists(monkeypatch):
    """Monkey-patch all specialist run() functions to avoid LLM calls."""
    transport_fn, lodging_fn, dining_fn, sightseeing_fn = _make_specialists()
    import agents.transport as _transport
    import agents.lodging as _lodging
    import agents.dining as _dining
    import agents.sightseeing as _sightseeing
    import agents.coordinator as _coord

    # Also patch parse_intent to avoid the LLM call
    from agents.schemas import Intent as _Intent

    def mock_parse(query: str) -> _Intent:
        return _Intent(
            org="Washington DC", dest="Myrtle Beach",
            days=3, dates=["2026-06-01", "2026-06-02", "2026-06-03"],
            people=2, budget=1400,
        )

    monkeypatch.setattr(_transport, "run", transport_fn)
    monkeypatch.setattr(_lodging, "run", lodging_fn)
    monkeypatch.setattr(_dining, "run", dining_fn)
    monkeypatch.setattr(_sightseeing, "run", sightseeing_fn)
    monkeypatch.setattr(_coord, "parse_intent", mock_parse)


@pytest.mark.usefixtures("_patch_specialists")
def test_coordinator_parallel_no_invalid_update_error():
    """H1 regression: parallel mode must not raise InvalidUpdateError."""
    intent = Intent(
        org="Washington DC", dest="Myrtle Beach",
        days=3, dates=["2026-06-01", "2026-06-02", "2026-06-03"],
        people=2, budget=1400,
    )
    ctx = _make_tool_context()
    # This was raising: InvalidUpdateError: At key 'timeline': Can receive only one value
    plan, report = plan_trip(
        "Plan a 3-day trip from Washington DC to Myrtle Beach under $1400",
        execution_mode="parallel",
        parsed_intent=intent,
        tool_context=ctx,
        bypass_date_check=True,
    )
    assert plan is not None
    assert len(plan.plan) == 3


@pytest.mark.usefixtures("_patch_specialists")
def test_coordinator_sequential_runs_cleanly():
    """Sequential mode also works without InvalidUpdateError."""
    intent = Intent(
        org="Washington DC", dest="Myrtle Beach",
        days=3, dates=["2026-06-01", "2026-06-02", "2026-06-03"],
        people=2, budget=1400,
    )
    ctx = _make_tool_context()
    plan, report = plan_trip(
        "Plan a 3-day trip from Washington DC to Myrtle Beach under $1400",
        execution_mode="sequential",
        parsed_intent=intent,
        tool_context=ctx,
        bypass_date_check=True,
    )
    assert len(plan.plan) == 3


@pytest.mark.usefixtures("_patch_specialists")
def test_coordinator_verifier_report_returned():
    intent = Intent(
        org="Washington DC", dest="Myrtle Beach",
        days=3, dates=["2026-06-01", "2026-06-02", "2026-06-03"],
        people=2, budget=1400,
    )
    ctx = _make_tool_context()
    plan, report = plan_trip(
        "test",
        parsed_intent=intent,
        tool_context=ctx,
        bypass_date_check=True,
    )
    assert report is not None
    assert isinstance(report.passed, bool)
    assert isinstance(report.violations, list)
