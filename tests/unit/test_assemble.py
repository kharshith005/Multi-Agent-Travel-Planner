"""Tests for coordinator.assemble and _apply_timing_adjustments."""
from __future__ import annotations

import pytest

from agents.coordinator import assemble
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
)


def _intent(days: int = 3) -> Intent:
    return Intent(
        org="Washington DC", dest="Myrtle Beach",
        days=days,
        dates=["2026-06-01", "2026-06-02", "2026-06-03"][:days],
        people=2,
        budget=1400,
    )


def test_assemble_day_count(transport_plan, lodging_plan, dining_plan, sightseeing_plan):
    plan = assemble(_intent(3), transport_plan, lodging_plan, dining_plan, sightseeing_plan, "q")
    assert len(plan.plan) == 3


def test_assemble_current_city_labeling(transport_plan, lodging_plan, dining_plan, sightseeing_plan):
    plan = assemble(_intent(3), transport_plan, lodging_plan, dining_plan, sightseeing_plan, "q")
    assert "from Washington DC to Myrtle Beach" in plan.plan[0].current_city
    assert "Myrtle Beach" == plan.plan[1].current_city
    assert "from Myrtle Beach to Washington DC" in plan.plan[2].current_city


def test_assemble_transportation_forwarded(transport_plan, lodging_plan, dining_plan, sightseeing_plan):
    plan = assemble(_intent(3), transport_plan, lodging_plan, dining_plan, sightseeing_plan, "q")
    assert "Cost: $200" in plan.plan[0].transportation
    assert plan.plan[1].transportation == "-"


def test_assemble_last_day_no_accommodation(transport_plan, lodging_plan, dining_plan, sightseeing_plan):
    plan = assemble(_intent(3), transport_plan, lodging_plan, dining_plan, sightseeing_plan, "q")
    assert plan.plan[-1].accommodation == "-"


def test_assemble_lodging_cost_on_first_stay_only(transport_plan, lodging_plan, dining_plan, sightseeing_plan):
    plan = assemble(_intent(3), transport_plan, lodging_plan, dining_plan, sightseeing_plan, "q")
    # Day 1 should have the full lodging string with Cost
    assert "Cost:" in plan.plan[0].accommodation
    # Day 2 should have just the name (no Cost duplicate)
    assert "Cost:" not in plan.plan[1].accommodation
    assert "Marriott Resort" in plan.plan[1].accommodation


def test_assemble_query_preserved(transport_plan, lodging_plan, dining_plan, sightseeing_plan):
    plan = assemble(_intent(3), transport_plan, lodging_plan, dining_plan, sightseeing_plan, "my query")
    assert plan.query == "my query"


def test_assemble_2day_trip():
    t = TransportPlan(days=[
        DayTransport(day=1, description="Bus; Cost: $50"),
        DayTransport(day=2, description="Bus return; Cost: $50"),
    ])
    l = LodgingPlan(days=[
        DayLodging(day=1, city="B", description="Hotel B; Cost: $100 for 1 night"),
        DayLodging(day=2, city="B", description="-"),
    ])
    d = DiningPlan(days=[
        DayMeals(day=1, city="B", lunch="Cafe; Cost: $15", dinner="Bar; Cost: $20"),
        DayMeals(day=2, city="B", breakfast="Bakery; Cost: $10"),
    ])
    s = SightseeingPlan(days=[
        DayAttractions(day=1, city="B", description="Park"),
        DayAttractions(day=2, city="B", description="-"),
    ])
    plan = assemble(Intent(org="A", dest="B", days=2, dates=["2026-06-01", "2026-06-02"], budget=500), t, l, d, s, "q")
    assert len(plan.plan) == 2
    assert plan.plan[-1].accommodation == "-"
