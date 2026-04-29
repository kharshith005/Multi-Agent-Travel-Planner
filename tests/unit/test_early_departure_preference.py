"""Tests for Step 10.2 + 10.3 — early-outbound preference and layered fallback."""
from __future__ import annotations

import pytest

from tools.live_apis import LiveTravelAPIs, _FlightOption


def _opt(dep="08:00", price=200, dur=120, airline="DL"):
    return _FlightOption(airline=airline, dep_time=dep, arr_time="10:00", price=price, duration_min=dur)


# ── Early departure selection ─────────────────────────────────────────────────

def test_early_departure_picks_morning_when_available():
    """prefer_early_departure=True should select ≤11:00 option over later ones."""
    morning = _opt(dep="08:00", price=150, dur=90)
    evening = _opt(dep="16:00", price=130, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [morning, evening], prefer_early_departure=True
    )
    assert ranked[0].dep_time == "08:00"


def test_early_departure_falls_back_to_pre14_when_no_morning():
    """When no option departs ≤11:00, select the earliest option ≤14:00."""
    noon = _opt(dep="12:00", price=200, dur=90)
    evening = _opt(dep="17:00", price=150, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [noon, evening], prefer_early_departure=True
    )
    assert ranked[0].dep_time == "12:00"


def test_early_departure_falls_back_to_earliest_when_all_late():
    """When all options depart after 14:00, select options[0] (duration-first)."""
    afternoon = _opt(dep="15:00", price=200, dur=90)
    late_eve = _opt(dep="19:00", price=150, dur=120)
    ranked = LiveTravelAPIs._rank_flight_options(
        [afternoon, late_eve], prefer_early_departure=True
    )
    # duration-first: afternoon is shorter (90 < 120), so it wins
    assert ranked[0] is afternoon


def test_early_departure_does_not_bias_when_flag_false():
    """prefer_early_departure=False (default) should not bias toward early options."""
    short_late = _opt(dep="18:00", price=150, dur=60)
    long_morning = _opt(dep="06:00", price=100, dur=240)
    ranked = LiveTravelAPIs._rank_flight_options([short_late, long_morning])
    # duration-first: short_late (60min) beats long_morning (240min)
    assert ranked[0] is short_late


def test_early_departure_within_duration_cap():
    """Duration cap is applied before the early preference."""
    long_early = _opt(dep="07:00", price=100, dur=300)  # 300min > cap(180) for JFK→BOS
    short_mid = _opt(dep="13:00", price=150, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [long_early, short_mid],
        dep_id="JFK", arr_id="BOS",
        prefer_early_departure=True,
    )
    # long_early filtered by cap; only short_mid survives → wins despite late dep
    assert ranked[0] is short_mid


def test_early_picks_earliest_in_morning_pool():
    """When multiple ≤11:00 options exist, pick the one with the earliest departure."""
    early1 = _opt(dep="06:00", price=200, dur=90)
    early2 = _opt(dep="09:00", price=150, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [early1, early2], prefer_early_departure=True
    )
    assert ranked[0].dep_time == "06:00"


# ── Layered late-return fallback (10.3) ─────────────────────────────────────

def test_late_return_layered_fallback_to_1100():
    """When no option ≥14:00, fall back to ≥11:00 within price ceiling."""
    mid_morning = _opt(dep="09:00", price=200, dur=90)
    midday = _opt(dep="11:30", price=200, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [mid_morning, midday], prefer_late_departure=True
    )
    # No ≥14:00 option; ≥11:00: midday (11:30) → selected
    assert ranked[0].dep_time == "11:30"


def test_late_return_layered_fallback_to_duration_first():
    """When no option ≥11:00, fall back to duration-first (options[0])."""
    early = _opt(dep="07:00", price=200, dur=120)
    mid = _opt(dep="09:30", price=150, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [early, mid], prefer_late_departure=True
    )
    # No ≥14:00 or ≥11:00 options → duration-first → mid (90min) wins
    assert ranked[0] is mid


def test_late_return_prefers_later_of_two_qualifying():
    """When two options qualify (≥14:00 within ceiling), pick the latest departure."""
    opt_14 = _opt(dep="14:00", price=200, dur=90)
    opt_17 = _opt(dep="17:00", price=200, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [opt_14, opt_17], prefer_late_departure=True
    )
    # Both ≥14:00 within ceiling → pick latest departure = 17:00
    assert ranked[0].dep_time == "17:00"
