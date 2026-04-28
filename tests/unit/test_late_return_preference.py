"""Tests for Step 9.2 — late-return departure preference in _rank_flight_options."""
from __future__ import annotations

import pytest

from tools.live_apis import LiveTravelAPIs, _FlightOption


def _opt(dep="08:00", price=200, dur=120, airline="DL"):
    return _FlightOption(airline=airline, dep_time=dep, arr_time="10:00", price=price, duration_min=dur)


# ── Late preference selection ─────────────────────────────────────────────────

def test_late_return_picks_evening_when_available():
    """prefer_late_departure=True should select the 16:00 option over the cheaper 08:00."""
    morning = _opt(dep="08:00", price=150)
    evening = _opt(dep="16:00", price=180)
    ranked = LiveTravelAPIs._rank_flight_options(
        [morning, evening], prefer_late_departure=True
    )
    assert ranked[0].dep_time == "16:00"


def test_late_return_falls_back_when_only_morning_options():
    """When no option departs ≥14:00, select by duration/price as usual."""
    o1 = _opt(dep="07:00", price=200, dur=120)
    o2 = _opt(dep="09:00", price=150, dur=90)
    ranked = LiveTravelAPIs._rank_flight_options(
        [o1, o2], prefer_late_departure=True
    )
    # Falls back to duration-first: o2 has shorter duration at lower price
    assert ranked[0] is o2


def test_late_return_respects_1_3x_ceiling():
    """If the late option costs >1.3× cheapest, fall back to duration-first."""
    morning = _opt(dep="08:00", price=100, dur=90)
    evening = _opt(dep="17:00", price=200, dur=120)  # 200 > 1.3 * 100 = 130
    ranked = LiveTravelAPIs._rank_flight_options(
        [morning, evening], prefer_late_departure=True
    )
    # evening is too expensive (>130); morning selected by duration-first
    assert ranked[0] is morning


def test_late_return_accepted_at_exactly_1_3x():
    """Late option at exactly 1.3× cheapest should be accepted."""
    morning = _opt(dep="08:00", price=100)
    evening = _opt(dep="15:00", price=130)  # 130 == 1.3 * 100
    ranked = LiveTravelAPIs._rank_flight_options(
        [morning, evening], prefer_late_departure=True
    )
    assert ranked[0].dep_time == "15:00"


def test_outbound_unaffected_by_late_return_flag():
    """prefer_late_departure=False (default) should NOT bias toward late options."""
    morning = _opt(dep="08:00", price=100, dur=90)
    evening = _opt(dep="18:00", price=105, dur=120)
    ranked = LiveTravelAPIs._rank_flight_options([morning, evening])
    # Duration-first: morning is shorter and cheaper → wins
    assert ranked[0] is morning


def test_late_return_within_duration_cap():
    """Late-return selection respects the duration cap applied first (9.1+9.2 interaction)."""
    # JFK→BOS cap=180 min. A 300-min late option should be filtered out,
    # leaving only an 80-min morning option.
    long_late = _opt(dep="18:00", price=100, dur=300)
    short_morning = _opt(dep="07:00", price=150, dur=80)
    ranked = LiveTravelAPIs._rank_flight_options(
        [long_late, short_morning],
        dep_id="JFK", arr_id="BOS",
        prefer_late_departure=True,
    )
    # long_late filtered by cap; only short_morning survives
    assert ranked[0] is short_morning
