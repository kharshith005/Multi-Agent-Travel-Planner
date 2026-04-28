"""Tests for Step 9.4 — round-trip flight search and cost attribution."""
from __future__ import annotations

import re
import pytest

from tools.live_apis import LiveTravelAPIs, _FlightOption, _RoundTripPair


# ── Helper: build a minimal fake SerpAPI round-trip payload ──────────────────

def _rt_payload(pairs: list[tuple]) -> dict:
    """Build a SerpAPI-style round-trip payload.

    Each pair: (out_airline, out_dep, out_arr, out_dur, price,
                ret_airline, ret_dep, ret_arr, ret_dur)
    """
    best = []
    for (oa, od, oa_arr, odr, price, ra, rd, ra_arr, rdr) in pairs:
        best.append({
            "flights": [{"airline": oa,
                         "departure_airport": {"time": od},
                         "arrival_airport": {"time": oa_arr}}],
            "total_duration": odr,
            "price": price,
            "return_flights": {
                "flights": [{"airline": ra,
                             "departure_airport": {"time": rd},
                             "arrival_airport": {"time": ra_arr}}],
                "total_duration": rdr,
            },
        })
    return {"best_flights": best, "other_flights": []}


def _mock_rt(api: LiveTravelAPIs, payload: dict):
    """Monkeypatch the requests.get call for the round-trip search."""
    import unittest.mock as mock
    fake_resp = mock.MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = payload
    fake_resp.raise_for_status = mock.MagicMock()
    return mock.patch("tools.live_apis.requests.get", return_value=fake_resp)


# ── flight_search_round_trip parsing ─────────────────────────────────────────

def test_round_trip_search_returns_paired_options():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    payload = _rt_payload([
        ("Delta", "2026-05-01 08:00", "2026-05-01 09:30", 90, 350,
         "Delta", "2026-05-03 17:00", "2026-05-03 18:30", 90),
    ])
    with _mock_rt(api, payload):
        out_strs, ret_strs = api.flight_search_round_trip("JFK", "ATL", "2026-05-01", "2026-05-03")
    assert len(out_strs) == 1
    assert len(ret_strs) == 1


def test_round_trip_cost_attributed_to_day_1_only():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    payload = _rt_payload([
        ("WN", "2026-05-01 09:00", "2026-05-01 10:15", 75, 280,
         "WN", "2026-05-03 16:00", "2026-05-03 17:15", 75),
    ])
    with _mock_rt(api, payload):
        out_strs, ret_strs = api.flight_search_round_trip("NYC", "DAL", "2026-05-01", "2026-05-03")
    # Outbound should have "Cost: $280"
    assert "Cost: $280" in out_strs[0]


def test_round_trip_return_leg_has_no_cost_fragment():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    payload = _rt_payload([
        ("AA", "2026-05-01 07:00", "2026-05-01 09:00", 120, 400,
         "AA", "2026-05-03 15:00", "2026-05-03 17:00", 120),
    ])
    with _mock_rt(api, payload):
        out_strs, ret_strs = api.flight_search_round_trip("NYC", "LAX", "2026-05-01", "2026-05-03")
    # Return should NOT have a Cost fragment
    assert "Cost:" not in ret_strs[0]


def test_round_trip_falls_back_on_serpapi_error():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    import unittest.mock as mock
    with mock.patch("tools.live_apis.requests.get", side_effect=Exception("timeout")):
        with pytest.raises(RuntimeError, match="round-trip"):
            api.flight_search_round_trip("JFK", "LAX", "2026-05-01", "2026-05-03")


def test_round_trip_falls_back_when_empty_results():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    payload = {"best_flights": [], "other_flights": []}
    with _mock_rt(api, payload):
        with pytest.raises(RuntimeError, match="no paired"):
            api.flight_search_round_trip("JFK", "LAX", "2026-05-01", "2026-05-03")


def test_round_trip_late_return_preference():
    """prefer_late_return=True should prefer the 17:00 return over 08:00."""
    api = LiveTravelAPIs(serpapi_api_key="fake")
    payload = _rt_payload([
        ("DL", "2026-05-01 08:00", "2026-05-01 09:30", 90, 300,
         "DL", "2026-05-03 08:00", "2026-05-03 09:30", 90),  # early return
        ("WN", "2026-05-01 09:00", "2026-05-01 10:30", 90, 310,
         "WN", "2026-05-03 17:00", "2026-05-03 18:30", 90),  # late return
    ])
    with _mock_rt(api, payload):
        _out, ret_strs = api.flight_search_round_trip(
            "JFK", "ATL", "2026-05-01", "2026-05-03", prefer_late_return=True
        )
    # Late-return pair should be first
    assert "17:00" in ret_strs[0]


def test_round_trip_respects_duration_cap_on_both_legs():
    """Duration cap filters long-haul pairs; when multiple options exist, short ones win.

    The cap skips when it would empty the pool (plan design: better a long
    flight than no result). So we test filtering by providing both a short and
    a long pair — only the short one should survive.
    """
    api = LiveTravelAPIs(serpapi_api_key="fake")
    # JFK→BOS: cap=180 min.
    payload = _rt_payload([
        ("DL", "2026-05-01 08:00", "2026-05-01 09:00", 60, 200,  # 60-min outbound — passes
         "DL", "2026-05-03 17:00", "2026-05-03 18:00", 60),
        ("AA", "2026-05-01 06:00", "2026-05-01 11:00", 300, 100,  # 300-min outbound — filtered
         "AA", "2026-05-03 17:00", "2026-05-03 22:00", 90),
    ])
    with _mock_rt(api, payload):
        out_strs, ret_strs = api.flight_search_round_trip(
            "JFK", "BOS", "2026-05-01", "2026-05-03",
            dep_id="JFK", arr_id="BOS",
        )
    # Only the 60-min DL pair should survive the cap
    assert len(out_strs) == 1
    assert "DL" in out_strs[0]
    assert "09:00" in out_strs[0]  # arrival of the short flight


# ── missing_cost rule with round-trip flag ────────────────────────────────────

def test_verifier_missing_cost_skips_last_day_when_round_trip():
    """Return leg without Cost: $N should NOT trigger missing_cost when flights_round_trip=True."""
    from agents.schemas import FullPlan, Intent, PlanDay
    from agents.verifier import verify

    intent = Intent(org="NYC", dest="ATL", days=2, dates=["2026-05-01", "2026-05-02"], budget=1000)
    plan = FullPlan(query="test", plan=[
        PlanDay(days=1, current_city="NYC→ATL",
                transportation="Delta 08:00->09:30 Duration: 1h 30m Cost: $350",
                breakfast="-", lunch="Restaurant; Cost: $20", dinner="Dinner; Cost: $40",
                attraction="Park;", accommodation="Hotel ATL; Cost: $120"),
        PlanDay(days=2, current_city="ATL→NYC",
                transportation="Delta 17:00->18:30 Duration: 1h 30m",  # no cost — RT
                breakfast="Hotel; Cost: $12", lunch="-", dinner="-",
                attraction="-", accommodation="-"),
    ])
    report = verify(plan, intent, flights_round_trip=True)
    missing_cost_v = [v for v in report.violations if v.rule == "missing_cost"]
    # The last-day transport missing cost should NOT fire
    assert not any("day(s)" in v.detail and "2" in v.detail for v in missing_cost_v)


def test_one_way_path_unchanged_when_round_trip_disabled():
    """When flights_round_trip=False, missing cost on last day IS flagged normally."""
    from agents.schemas import FullPlan, Intent, PlanDay
    from agents.verifier import verify

    intent = Intent(org="NYC", dest="ATL", days=2, dates=["2026-05-01", "2026-05-02"], budget=1000)
    plan = FullPlan(query="test", plan=[
        PlanDay(days=1, current_city="NYC→ATL",
                transportation="Delta 08:00->09:30 Cost: $180",
                breakfast="-", lunch="Restaurant; Cost: $20", dinner="Dinner; Cost: $40",
                attraction="Park;", accommodation="Hotel; Cost: $120"),
        PlanDay(days=2, current_city="ATL→NYC",
                transportation="Delta 17:00->18:30",  # no cost — one-way should be flagged
                breakfast="Hotel; Cost: $12", lunch="-", dinner="-",
                attraction="-", accommodation="-"),
    ])
    report = verify(plan, intent, flights_round_trip=False)
    rules = [v.rule for v in report.violations]
    assert "missing_cost" in rules
