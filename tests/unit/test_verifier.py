"""Tests for agents/verifier.py — rule-based constraint checking."""
from __future__ import annotations

import pytest

from agents.rules import (
    RULE_BUDGET,
    RULE_COMPLETE_INFORMATION,
    RULE_CUISINE,
    RULE_DIVERSE_ATTRACTIONS,
    RULE_DIVERSE_RESTAURANTS,
    RULE_MIN_NIGHTS,
    RULE_MISSING_COST,
    RULE_TRANSPORTATION,
)
from agents.schemas import FullPlan, Intent, PlanDay
from agents.verifier import verify


def _simple_plan(days: int = 3) -> FullPlan:
    """Minimal valid plan with costs everywhere so missing_cost never fires."""
    plan_days = []
    for i in range(1, days + 1):
        plan_days.append(PlanDay(
            days=i,
            current_city="Test City",
            transportation=f"Airline; Cost: $100" if i in {1, days} else "-",
            breakfast=f"Cafe {i}; Cost: $10" if i > 1 else "-",
            attraction=f"Attraction {i}; Cost: $5" if i < days else "-",
            lunch=f"Restaurant {i}; Cost: $15",
            dinner=f"Bistro {i}; Cost: $20" if i < days else "-",
            accommodation=f"Hotel A; Cost: $80 for {days - 1} nights" if i == 1 else
                          ("Hotel A" if i < days else "-"),
        ))
    return FullPlan(query="test", plan=plan_days)


def _intent(**kwargs) -> Intent:
    defaults = dict(org="A", dest="B", days=3, budget=2000,
                    dates=["2026-06-01", "2026-06-02", "2026-06-03"])
    defaults.update(kwargs)
    return Intent(**defaults)


# ── completeness ──────────────────────────────────────────────────────────────

def test_verify_passes_clean_plan():
    plan = _simple_plan(3)
    report = verify(plan, _intent(days=3))
    rule_names = [v.rule for v in report.violations]
    assert RULE_COMPLETE_INFORMATION not in rule_names

def test_verify_wrong_day_count():
    plan = _simple_plan(2)
    report = verify(plan, _intent(days=3))
    assert any(v.rule == RULE_COMPLETE_INFORMATION for v in report.violations)


# ── budget ────────────────────────────────────────────────────────────────────

def test_verify_over_budget():
    plan = _simple_plan(3)
    report = verify(plan, _intent(days=3, budget=1))
    assert any(v.rule == RULE_BUDGET for v in report.violations)

def test_verify_within_budget():
    plan = _simple_plan(3)
    report = verify(plan, _intent(days=3, budget=10_000))
    assert not any(v.rule == RULE_BUDGET for v in report.violations)


# ── diversity ─────────────────────────────────────────────────────────────────

def test_verify_duplicate_attractions():
    plan = FullPlan(query="t", plan=[
        PlanDay(days=1, current_city="C", attraction="The Museum; Cost: $5",
                transportation="Airline; Cost: $100", lunch="Cafe A; Cost: $10",
                dinner="Cafe B; Cost: $20",
                accommodation="Hotel A; Cost: $80 for 2 nights"),
        PlanDay(days=2, current_city="C", attraction="The Museum; Cost: $5",
                breakfast="Cafe C; Cost: $10", lunch="Cafe D; Cost: $15",
                dinner="Cafe E; Cost: $20", accommodation="Hotel A"),
        PlanDay(days=3, current_city="C", attraction="-",
                transportation="Airline; Cost: $100",
                breakfast="Cafe F; Cost: $10", lunch="Cafe G; Cost: $15", dinner="-",
                accommodation="-"),
    ])
    report = verify(plan, _intent(days=3))
    assert any(v.rule == RULE_DIVERSE_ATTRACTIONS for v in report.violations)


def test_verify_duplicate_restaurants():
    plan = FullPlan(query="t", plan=[
        PlanDay(days=1, current_city="C", transportation="Airline; Cost: $100",
                lunch="Same Place; Cost: $15", dinner="Same Place; Cost: $20",
                accommodation="Hotel A; Cost: $80 for 2 nights"),
        PlanDay(days=2, current_city="C", breakfast="Cafe B; Cost: $10",
                lunch="Another; Cost: $15", dinner="Yet Another; Cost: $20",
                accommodation="Hotel A"),
        PlanDay(days=3, current_city="C", transportation="Airline; Cost: $100",
                breakfast="Morning Spot; Cost: $10", lunch="-", dinner="-",
                accommodation="-"),
    ])
    report = verify(plan, _intent(days=3))
    assert any(v.rule == RULE_DIVERSE_RESTAURANTS for v in report.violations)


# ── min nights ────────────────────────────────────────────────────────────────

def test_verify_missing_accommodation_non_final_day():
    plan = FullPlan(query="t", plan=[
        PlanDay(days=1, current_city="C", transportation="Airline; Cost: $100",
                lunch="Cafe; Cost: $10", dinner="Bistro; Cost: $20",
                accommodation="-"),  # missing on non-final day
        PlanDay(days=2, current_city="C", breakfast="Cafe; Cost: $10",
                lunch="Bistro; Cost: $15", dinner="Bar; Cost: $20",
                accommodation="-"),  # missing on non-final day
        PlanDay(days=3, current_city="C", transportation="Airline; Cost: $100",
                breakfast="Cafe; Cost: $10", lunch="-", dinner="-", accommodation="-"),
    ])
    report = verify(plan, _intent(days=3))
    assert any(v.rule == RULE_MIN_NIGHTS for v in report.violations)


# ── cuisine ───────────────────────────────────────────────────────────────────

def test_verify_cuisine_present():
    plan = _simple_plan(3)
    # "Seafood" is not in any meal field of _simple_plan
    report = verify(plan, _intent(days=3, cuisine="seafood"))
    assert any(v.rule == RULE_CUISINE for v in report.violations)

def test_verify_cuisine_found():
    plan = FullPlan(query="t", plan=[
        PlanDay(days=1, current_city="C", transportation="Airline; Cost: $100",
                lunch="Seafood Shack; Cost: $20", dinner="Bistro; Cost: $25",
                accommodation="Hotel A; Cost: $80 for 2 nights"),
        PlanDay(days=2, current_city="C", breakfast="Bakery; Cost: $10",
                lunch="Cafe; Cost: $15", dinner="Pizza; Cost: $20",
                accommodation="Hotel A"),
        PlanDay(days=3, current_city="C", transportation="Airline; Cost: $100",
                breakfast="Bagels; Cost: $8", lunch="-", dinner="-",
                accommodation="-"),
    ])
    report = verify(plan, _intent(days=3, cuisine="seafood"))
    assert not any(v.rule == RULE_CUISINE for v in report.violations)


# ── transportation ────────────────────────────────────────────────────────────

def test_verify_required_transport_missing():
    plan = _simple_plan(3)
    report = verify(plan, _intent(days=3, transportation="train"))
    assert any(v.rule == RULE_TRANSPORTATION for v in report.violations)

def test_verify_forbidden_transport():
    plan = FullPlan(query="t", plan=[
        PlanDay(days=1, current_city="C",
                transportation="Self-driving from A to B; Cost: $50",
                lunch="Cafe; Cost: $15", dinner="Bistro; Cost: $20",
                accommodation="Hotel A; Cost: $80 for 2 nights"),
        PlanDay(days=2, current_city="C", breakfast="Cafe; Cost: $10",
                lunch="Bistro; Cost: $15", dinner="Bar; Cost: $20",
                accommodation="Hotel A"),
        PlanDay(days=3, current_city="C",
                transportation="Self-driving return; Cost: $50",
                breakfast="Bagels; Cost: $8", lunch="-", dinner="-",
                accommodation="-"),
    ])
    report = verify(plan, _intent(days=3, transportation="no self-driving"))
    assert any(v.rule == RULE_TRANSPORTATION for v in report.violations)
