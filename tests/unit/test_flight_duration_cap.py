"""Tests for Step 9.1 — distance-aware flight duration cap in _rank_flight_options."""
from __future__ import annotations

import pytest

from tools.live_apis import LiveTravelAPIs, _FlightOption, _duration_cap_minutes


def _opt(airline="DL", dep="08:00", arr="10:00", price=200, dur=120):
    return _FlightOption(airline=airline, dep_time=dep, arr_time=arr, price=price, duration_min=dur)


# ── _duration_cap_minutes ─────────────────────────────────────────────────────

def test_cap_short_haul_under_500mi():
    # JFK → BOS is ~200 mi
    cap = _duration_cap_minutes("JFK", "BOS")
    assert cap == 180


def test_cap_medium_haul_under_1000mi():
    # JFK → ATL is ~760 mi
    cap = _duration_cap_minutes("JFK", "ATL")
    assert cap == 240


def test_cap_long_haul_under_2000mi():
    # JFK → DFW is ~1400 mi
    cap = _duration_cap_minutes("JFK", "DFW")
    assert cap == 360


def test_cap_cross_country_over_2000mi():
    # JFK → LAX is ~2450 mi
    cap = _duration_cap_minutes("JFK", "LAX")
    assert cap == 600


def test_cap_unknown_iata_returns_none():
    cap = _duration_cap_minutes("ZZZ", "ZZZ")
    assert cap is None


def test_cap_none_when_dep_id_missing():
    assert _duration_cap_minutes(None, "JFK") is None
    assert _duration_cap_minutes("JFK", None) is None


# ── _rank_flight_options with cap ────────────────────────────────────────────

def test_short_haul_filters_10hr_option():
    """A 10-hour option on a <500 mi route (cap=180 min) should be filtered."""
    long_opt = _opt(dur=600, price=100)
    short_opt = _opt(dur=90, price=200)
    ranked = LiveTravelAPIs._rank_flight_options(
        [long_opt, short_opt], dep_id="JFK", arr_id="BOS"
    )
    # long option should be filtered out; short survives
    assert ranked[0].duration_min == 90


def test_short_haul_keeps_2hr_option():
    opt = _opt(dur=120, price=150)
    ranked = LiveTravelAPIs._rank_flight_options([opt], dep_id="JFK", arr_id="BOS")
    assert ranked[0] is opt


def test_cross_country_keeps_6hr_option():
    """JFK→LAX cap=600 min; a 360-min option passes."""
    opt = _opt(dur=360, price=300)
    ranked = LiveTravelAPIs._rank_flight_options([opt], dep_id="JFK", arr_id="LAX")
    assert ranked[0] is opt


def test_when_all_options_exceed_cap_keeps_shortest():
    """When all options exceed the cap, skip the cap (return something)."""
    o1 = _opt(dur=300, price=100)
    o2 = _opt(dur=250, price=120)
    # JFK→BOS cap=180; both exceed cap → cap is skipped
    ranked = LiveTravelAPIs._rank_flight_options(
        [o1, o2], dep_id="JFK", arr_id="BOS"
    )
    assert len(ranked) == 2  # neither filtered out


def test_distance_unknown_falls_back_to_no_cap():
    """Unknown IATA → cap is None → all options survive."""
    long_opt = _opt(dur=900, price=50)
    ranked = LiveTravelAPIs._rank_flight_options(
        [long_opt], dep_id="ZZZ", arr_id="ZZZ"
    )
    assert ranked[0] is long_opt


def test_cap_applied_before_budget_filter():
    """The duration cap runs before the 50%-of-budget hard filter.

    A 10-hour option at $50 should be filtered by the cap (JFK→BOS=180 min),
    even though it would easily fit within the budget.
    """
    cheap_long = _opt(dur=600, price=50)
    ok_priced = _opt(dur=90, price=200)
    ranked = LiveTravelAPIs._rank_flight_options(
        [cheap_long, ok_priced], budget=1000, dep_id="JFK", arr_id="BOS"
    )
    # cheap_long filtered by cap; ok_priced survives and wins
    assert ranked[0].duration_min == 90
