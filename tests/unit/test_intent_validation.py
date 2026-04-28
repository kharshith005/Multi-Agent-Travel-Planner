"""Tests for Step 8.1 — from-date validation in intent_constraint_violations."""
from __future__ import annotations

import pytest

from agents.coordinator import intent_constraint_violations
from agents.schemas import Intent


def _intent(**kwargs) -> Intent:
    defaults = dict(org="New York", dest="Austin", days=3, budget=1400)
    defaults.update(kwargs)
    return Intent(**defaults)


# ── No dates at all ────────────────────────────────────────────────────────────

def test_intent_no_dates_rejected():
    intent = _intent(dates=[])
    violations = intent_constraint_violations(intent)
    assert any("start date" in v for v in violations)


def test_intent_blank_dates_rejected():
    intent = _intent(dates=["", "  "])
    violations = intent_constraint_violations(intent)
    assert any("start date" in v for v in violations)


# ── Only end date given — LLM returns empty dates per updated INTENT_SYSTEM ───

def test_intent_only_end_date_rejected():
    # Simulates: LLM correctly returns empty dates when user gave only an end date.
    intent = _intent(days=3, dates=[])
    violations = intent_constraint_violations(intent)
    assert any("start date" in v for v in violations)


# ── Single date is valid when it is the start date ────────────────────────────

def test_intent_one_date_one_day_trip_accepted():
    intent = _intent(days=1, dates=["2026-04-15"])
    violations = intent_constraint_violations(intent)
    assert not any("date" in v.lower() for v in violations)


def test_intent_one_date_multi_day_accepted():
    # User said "3-day trip starting March 15" — start date is present.
    intent = _intent(days=3, dates=["2026-03-15"])
    violations = intent_constraint_violations(intent)
    assert not any("date" in v.lower() for v in violations)


# ── Full date range ────────────────────────────────────────────────────────────

def test_intent_full_date_range_accepted():
    intent = _intent(days=3, dates=["2026-05-10", "2026-05-12"])
    violations = intent_constraint_violations(intent)
    assert not any("date" in v.lower() for v in violations)


# ── Violation message content ──────────────────────────────────────────────────

def test_constraint_message_mentions_start_date():
    intent = _intent(dates=[])
    violations = intent_constraint_violations(intent)
    msg = " ".join(violations)
    assert "start date" in msg.lower()


# ── Other constraints still fire independently ────────────────────────────────

def test_missing_budget_still_fires_with_valid_dates():
    intent = Intent(org="NYC", dest="Austin", days=3, dates=["2026-05-10"], budget=None)
    violations = intent_constraint_violations(intent)
    assert any("budget" in v for v in violations)
    assert not any("date" in v.lower() for v in violations)


def test_multiple_violations_collected():
    intent = Intent(org="", dest="", days=3, dates=[], budget=None)
    violations = intent_constraint_violations(intent)
    assert len(violations) >= 3
