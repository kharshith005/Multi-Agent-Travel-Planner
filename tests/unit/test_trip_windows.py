"""Tests for Step 8.4 — time-aware YES/NO slot grid in format_trip_windows."""
from __future__ import annotations

import pytest

from agents.rules import format_trip_windows
from agents.coordinator import _apply_timing_adjustments
from agents.schemas import FullPlan, Intent, PlanDay


def _ctx(arr_min=None, dep_min=None):
    def _hhmm(m):
        return f"{m // 60:02d}:{m % 60:02d}" if m is not None else None

    tw = {}
    if arr_min is not None:
        tw["arrival_minutes_day1"] = arr_min
        tw["arrival_hhmm_day1"] = _hhmm(arr_min)
    if dep_min is not None:
        tw["departure_minutes_last"] = dep_min
        tw["departure_hhmm_last"] = _hhmm(dep_min)
    return {"trip_windows": tw} if tw else {}


def _arr(hh, mm=0): return hh * 60 + mm
def _dep(hh, mm=0): return hh * 60 + mm


# ── Grid content tests ────────────────────────────────────────────────────────

def test_grid_emits_yes_no_for_arrival_18_30():
    ctx = _ctx(arr_min=_arr(18, 30))
    grid = format_trip_windows(ctx)
    assert "Day 1 (arrival):" in grid
    assert "breakfast: NO" in grid
    assert "lunch:      NO" in grid    # >= 14:00
    assert "dinner:     YES" in grid   # < 19:00
    assert "attraction: NO" in grid    # >= 16:00


def test_grid_emits_dinner_no_when_arrival_after_19():
    ctx = _ctx(arr_min=_arr(19, 30))
    grid = format_trip_windows(ctx)
    assert "dinner:     NO" in grid


def test_grid_emits_dinner_yes_when_arrival_before_19():
    ctx = _ctx(arr_min=_arr(16, 0))
    grid = format_trip_windows(ctx)
    assert "dinner:     YES" in grid


def test_grid_emits_lunch_yes_when_arrival_before_14():
    ctx = _ctx(arr_min=_arr(10, 0))
    grid = format_trip_windows(ctx)
    assert "lunch:      YES" in grid


def test_grid_omits_lunch_when_departure_before_13_30():
    ctx = _ctx(dep_min=_dep(12, 0))
    grid = format_trip_windows(ctx)
    assert "Last day (departure):" in grid
    assert "lunch:      NO" in grid


def test_grid_lunch_yes_when_departure_after_13_30():
    ctx = _ctx(dep_min=_dep(14, 0))
    grid = format_trip_windows(ctx)
    assert "lunch:      YES" in grid


def test_grid_dinner_yes_when_departure_after_20():
    ctx = _ctx(dep_min=_dep(20, 30))
    grid = format_trip_windows(ctx)
    assert "dinner:     YES" in grid


def test_grid_dinner_no_when_departure_before_20():
    ctx = _ctx(dep_min=_dep(18, 0))
    grid = format_trip_windows(ctx)
    assert "dinner:     NO" in grid


def test_grid_attraction_yes_when_departure_after_16():
    ctx = _ctx(dep_min=_dep(17, 0))
    grid = format_trip_windows(ctx)
    assert "attraction: YES" in grid


def test_grid_attraction_no_when_departure_before_16():
    ctx = _ctx(dep_min=_dep(11, 0))
    grid = format_trip_windows(ctx)
    assert "attraction: NO" in grid


def test_grid_empty_when_no_trip_windows():
    assert format_trip_windows({}) == ""
    assert format_trip_windows(None) == ""
    assert format_trip_windows({"trip_windows": {}}) == ""


def test_grid_shows_both_days_when_both_times_given():
    ctx = _ctx(arr_min=_arr(17, 0), dep_min=_dep(11, 0))
    grid = format_trip_windows(ctx)
    assert "Day 1 (arrival):" in grid
    assert "Last day (departure):" in grid


# ── Safety clamp tests ────────────────────────────────────────────────────────

def _make_plan(
    transport_d1="Southwest 08:00->10:30 Cost: $120",
    transport_last="Southwest 11:00->13:30 Cost: $120",
    days=3,
    lunch_d1="La Paloma, Austin; Cost: $22",
    dinner_d1="Uchiko, Austin; Cost: $55",
    attraction_d1="Barton Springs Pool, Austin;",
    lunch_last="Franklin BBQ, Austin; Cost: $18",
    dinner_last="-",
) -> FullPlan:
    plan_days = [
        PlanDay(
            days=1, current_city="from NYC to Austin",
            transportation=transport_d1,
            breakfast="-", lunch=lunch_d1, dinner=dinner_d1,
            attraction=attraction_d1, accommodation="Hotel, Austin; Cost: $150 for 2 nights",
        ),
    ]
    for i in range(2, days):
        plan_days.append(PlanDay(
            days=i, current_city="Austin",
            transportation="-",
            breakfast="Tacodeli, Austin; Cost: $12",
            lunch="Easy Tiger, Austin; Cost: $15",
            dinner="Uchiko, Austin; Cost: $55",
            attraction="South Congress Ave, Austin;",
            accommodation="Hotel, Austin",
        ))
    plan_days.append(PlanDay(
        days=days, current_city="from Austin to NYC",
        transportation=transport_last,
        breakfast="Hotel, Austin; Cost: $12",
        lunch=lunch_last,
        dinner=dinner_last,
        attraction="-",
        accommodation="-",
    ))
    return FullPlan(query="test", plan=plan_days)


def test_safety_clamp_idempotent_when_grid_obeyed():
    """If specialist already put '-' for slots the grid marks NO, clamp is a no-op."""
    plan = _make_plan(
        transport_d1="Southwest 08:00->10:30 Cost: $120",   # arr 10:30 — lunch YES
        lunch_d1="-",   # specialist correctly set to '-'
    )
    result = _apply_timing_adjustments(plan, None)
    assert result.plan[0].lunch == "-"


def test_safety_clamp_fixes_lunch_when_arrival_after_14():
    plan = _make_plan(
        transport_d1="Southwest 12:00->15:00 Cost: $120",  # arr 15:00 >= 14:00
        lunch_d1="La Paloma, Austin; Cost: $22",            # specialist left lunch
    )
    result = _apply_timing_adjustments(plan, None)
    # Phase 7: clamp to "-" — transportation is the single source of truth for arrival time
    assert result.plan[0].lunch == "-"


def test_safety_clamp_fixes_dinner_when_arrival_after_19():
    plan = _make_plan(
        transport_d1="Southwest 16:00->19:30 Cost: $120",  # arr 19:30 >= 19:00
        dinner_d1="Uchiko, Austin; Cost: $55",
    )
    result = _apply_timing_adjustments(plan, None)
    # Phase 7: clamp to "-" — no echoed arrival time in meal fields
    assert result.plan[0].dinner == "-"


def test_safety_clamp_fixes_attraction_when_arrival_after_16():
    plan = _make_plan(
        transport_d1="Southwest 14:00->16:30 Cost: $120",  # arr 16:30 >= 16:00
        attraction_d1="Barton Springs Pool, Austin;",
    )
    result = _apply_timing_adjustments(plan, None)
    assert result.plan[0].attraction == "-"


def test_safety_clamp_fixes_lunch_when_departure_before_13_30():
    plan = _make_plan(
        transport_last="Southwest 11:00->13:30 Cost: $120",  # dep 11:00 < 13:30
        lunch_last="Franklin BBQ, Austin; Cost: $18",
    )
    result = _apply_timing_adjustments(plan, None)
    assert result.plan[-1].lunch == "-"


def test_safety_clamp_allows_lunch_when_departure_after_13_30():
    plan = _make_plan(
        transport_last="Southwest 15:00->17:30 Cost: $120",  # dep 15:00 >= 13:30
        lunch_last="Franklin BBQ, Austin; Cost: $18",
    )
    result = _apply_timing_adjustments(plan, None)
    assert result.plan[-1].lunch != "-"
