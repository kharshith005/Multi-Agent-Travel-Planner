"""Tests for agents/budget.py — cost extraction, allocation, evaluation."""
from __future__ import annotations

import pytest

from agents.budget import allocate, evaluate, extract_cost
from agents.schemas import FullPlan, Intent, PlanDay


# ── extract_cost ─────────────────────────────────────────────────────────────

def test_extract_cost_basic():
    assert extract_cost("Hotel Marriott; Cost: $180 for 2 nights") == 180

def test_extract_cost_multiple():
    assert extract_cost("Flight; Cost: $200\nTax; Cost: $15") == 215

def test_extract_cost_missing():
    assert extract_cost("-") == 0
    assert extract_cost("") == 0
    assert extract_cost("No cost mentioned here") == 0

def test_extract_cost_case_insensitive():
    assert extract_cost("Airline; cost: $300") == 300

def test_extract_cost_no_dollar_sign():
    assert extract_cost("Hotel; Cost: 150") == 150


# ── allocate ─────────────────────────────────────────────────────────────────

def test_allocate_splits_correctly():
    caps = allocate(Intent(org="A", dest="B", days=3, budget=1000))
    assert caps["transport"] == 450
    assert caps["lodging"] == 350
    assert caps["dining"] == 200

def test_allocate_no_budget():
    assert allocate(Intent(org="A", dest="B", days=3, budget=None)) == {}

def test_allocate_zero_budget():
    assert allocate(Intent(org="A", dest="B", days=3, budget=0)) == {}

def test_allocate_small_budget_floors_at_minimums():
    caps = allocate(Intent(org="A", dest="B", days=3, budget=10))
    assert caps["transport"] >= 50
    assert caps["lodging"] >= 50
    assert caps["dining"] >= 30


# ── evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_within_budget(three_day_plan, basic_intent):
    report = evaluate(three_day_plan, basic_intent)
    assert report.passed
    assert report.total_cost > 0
    assert report.budget == 1400

def test_evaluate_over_budget(three_day_plan):
    tight_intent = Intent(org="A", dest="B", days=3, budget=1)
    report = evaluate(three_day_plan, tight_intent)
    assert not report.passed
    assert report.total_cost > 1

def test_evaluate_no_budget(three_day_plan):
    no_budget_intent = Intent(org="A", dest="B", days=3, budget=None)
    report = evaluate(three_day_plan, no_budget_intent)
    assert report.passed

def test_evaluate_highest_category(three_day_plan, basic_intent):
    report = evaluate(three_day_plan, basic_intent)
    assert report.highest_category in {"transport", "dining", "lodging"}
    # transport costs 200+200=400 in the fixture, should be highest
    assert report.category_costs["transport"] == 400

def test_evaluate_category_breakdown(three_day_plan, basic_intent):
    report = evaluate(three_day_plan, basic_intent)
    total_check = sum(report.category_costs.values())
    assert report.total_cost == total_check
