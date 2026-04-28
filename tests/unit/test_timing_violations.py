"""Tests for Step 9.3 — timing violations as verifier rules."""
from __future__ import annotations

import pytest

from agents.rules import (
    RULE_ARRIVAL_ATTRACTION,
    RULE_ARRIVAL_DINNER,
    RULE_ARRIVAL_LUNCH,
    RULE_DEPARTURE_ATTRACTION,
    RULE_DEPARTURE_BREAKFAST,
    RULE_DEPARTURE_LUNCH,
)
from agents.schemas import FullPlan, Intent, PlanDay, TripWindows
from agents.verifier import verify


def _intent(**kw):
    defaults = dict(org="New York", dest="Austin", days=3, dates=["2026-05-01", "2026-05-03"], budget=1400)
    defaults.update(kw)
    return Intent(**defaults)


def _plan(days=3, transport_d1="WN 1234 08:00->10:30 Cost: $120",
          transport_last="WN 5678 12:00->14:30 Cost: $120",
          lunch_d1="La Paloma; Cost: $22", dinner_d1="Uchi; Cost: $55",
          attr_d1="Barton Springs;", breakfast_last="Hotel; Cost: $12",
          lunch_last="-", attr_last="-") -> FullPlan:
    plan_days = [
        PlanDay(
            days=1, current_city="NYC→Austin",
            transportation=transport_d1,
            breakfast="-", lunch=lunch_d1, dinner=dinner_d1,
            attraction=attr_d1,
            accommodation="Hyatt; Cost: $150",
        ),
    ]
    for i in range(2, days):
        plan_days.append(PlanDay(
            days=i, current_city="Austin",
            transportation="-",
            breakfast="Tacodeli; Cost: $12",
            lunch="Easy Tiger; Cost: $15",
            dinner="Uchi 2; Cost: $55",
            attraction="South Congress;",
            accommodation="Hyatt",
        ))
    plan_days.append(PlanDay(
        days=days, current_city="Austin→NYC",
        transportation=transport_last,
        breakfast=breakfast_last,
        lunch=lunch_last,
        dinner="-",
        attraction=attr_last,
        accommodation="-",
    ))
    return FullPlan(query="test", plan=plan_days)


def _tw(arr_min=None, dep_min=None) -> TripWindows:
    def hhmm(m):
        return f"{m // 60:02d}:{m % 60:02d}" if m is not None else None
    tw: TripWindows = {}
    if arr_min is not None:
        tw["arrival_minutes_day1"] = arr_min
        tw["arrival_hhmm_day1"] = hhmm(arr_min)
    if dep_min is not None:
        tw["departure_minutes_last"] = dep_min
        tw["departure_hhmm_last"] = hhmm(dep_min)
    return tw


# ── Arrival-day violations ─────────────────────────────────────────────────────

def test_lunch_after_14_flagged_as_dining():
    plan = _plan(lunch_d1="La Paloma; Cost: $22")
    tw = _tw(arr_min=15 * 60)  # 15:00 → lunch NO
    report = verify(plan, _intent(), trip_windows=tw)
    rules = [v.rule for v in report.violations]
    assert RULE_ARRIVAL_LUNCH in rules
    lunch_v = next(v for v in report.violations if v.rule == RULE_ARRIVAL_LUNCH)
    assert lunch_v.responsible == "dining"


def test_dinner_after_19_flagged_as_dining():
    plan = _plan(dinner_d1="Uchi; Cost: $55")
    tw = _tw(arr_min=19 * 60 + 30)  # 19:30 → dinner NO
    report = verify(plan, _intent(), trip_windows=tw)
    rules = [v.rule for v in report.violations]
    assert RULE_ARRIVAL_DINNER in rules
    v = next(v for v in report.violations if v.rule == RULE_ARRIVAL_DINNER)
    assert v.responsible == "dining"


def test_attraction_after_16_flagged_as_sightseeing():
    plan = _plan(attr_d1="Barton Springs;")
    tw = _tw(arr_min=16 * 60 + 30)  # 16:30 → attraction NO
    report = verify(plan, _intent(), trip_windows=tw)
    rules = [v.rule for v in report.violations]
    assert RULE_ARRIVAL_ATTRACTION in rules
    v = next(v for v in report.violations if v.rule == RULE_ARRIVAL_ATTRACTION)
    assert v.responsible == "sightseeing"


def test_no_violation_when_grid_obeyed():
    """Arrival at 10:00 — lunch/dinner/attraction are all fine."""
    plan = _plan(lunch_d1="La Paloma; Cost: $22", dinner_d1="Uchi; Cost: $55", attr_d1="Barton;")
    tw = _tw(arr_min=10 * 60)  # arrival at 10:00
    report = verify(plan, _intent(), trip_windows=tw)
    timing_rules = {
        RULE_ARRIVAL_LUNCH, RULE_ARRIVAL_DINNER, RULE_ARRIVAL_ATTRACTION,
        RULE_DEPARTURE_BREAKFAST, RULE_DEPARTURE_LUNCH, RULE_DEPARTURE_ATTRACTION,
    }
    fired = {v.rule for v in report.violations}
    assert not (timing_rules & fired)


# ── Departure-day violations ───────────────────────────────────────────────────

def test_departure_lunch_before_1330_flagged():
    plan = _plan(lunch_last="Franklin BBQ; Cost: $18")
    tw = _tw(dep_min=11 * 60)  # 11:00 → lunch NO
    report = verify(plan, _intent(), trip_windows=tw)
    rules = [v.rule for v in report.violations]
    assert RULE_DEPARTURE_LUNCH in rules
    v = next(v for v in report.violations if v.rule == RULE_DEPARTURE_LUNCH)
    assert v.responsible == "dining"


def test_breakfast_before_08_flagged():
    plan = _plan(breakfast_last="Hotel; Cost: $12")
    tw = _tw(dep_min=6 * 60)  # 06:00 → breakfast NO
    report = verify(plan, _intent(), trip_windows=tw)
    rules = [v.rule for v in report.violations]
    assert RULE_DEPARTURE_BREAKFAST in rules


def test_departure_attraction_before_16_flagged():
    plan = _plan(attr_last="South Congress;")
    tw = _tw(dep_min=11 * 60)  # 11:00 → attraction NO
    report = verify(plan, _intent(), trip_windows=tw)
    rules = [v.rule for v in report.violations]
    assert RULE_DEPARTURE_ATTRACTION in rules
    v = next(v for v in report.violations if v.rule == RULE_DEPARTURE_ATTRACTION)
    assert v.responsible == "sightseeing"


def test_no_trip_windows_skips_timing_rules():
    """When trip_windows is None, no timing rules fire."""
    plan = _plan(lunch_d1="La Paloma; Cost: $22")  # would normally fire at 15:00
    report = verify(plan, _intent(), trip_windows=None)
    timing_rules = {
        RULE_ARRIVAL_LUNCH, RULE_ARRIVAL_DINNER, RULE_ARRIVAL_ATTRACTION,
        RULE_DEPARTURE_BREAKFAST, RULE_DEPARTURE_LUNCH, RULE_DEPARTURE_ATTRACTION,
    }
    fired = {v.rule for v in report.violations}
    assert not (timing_rules & fired)


# ── Sentinel values are not flagged ───────────────────────────────────────────

def test_timing_sentinel_not_flagged_as_timing_violation():
    """'In transit (arrives 15:00)' was set by the safety clamp — skip it."""
    plan = _plan(lunch_d1="In transit (arrives 15:00)")
    tw = _tw(arr_min=15 * 60)
    report = verify(plan, _intent(), trip_windows=tw)
    rules = [v.rule for v in report.violations]
    assert RULE_ARRIVAL_LUNCH not in rules
