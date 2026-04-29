"""Tests for round-trip flight search — two-call departure_token protocol."""
from __future__ import annotations

import unittest.mock as mock

import pytest

from tools.live_apis import LiveTravelAPIs, _FlightOption


# ── Helpers: build minimal SerpAPI payloads ──────────────────────────────────

def _outbound_payload(flights: list[dict]) -> dict:
    """Call-1 payload: best_flights with departure_token on each option."""
    best = []
    for f in flights:
        best.append({
            "flights": [
                {
                    "airline": f["airline"],
                    "departure_airport": {"time": f["dep"]},
                    "arrival_airport": {"time": f["arr"]},
                }
            ],
            "total_duration": f.get("dur", 90),
            "price": f.get("price"),
            "departure_token": f.get("token", "tok_abc"),
        })
    return {"best_flights": best, "other_flights": []}


def _return_payload(flights: list[dict]) -> dict:
    """Call-2 payload: best_flights for return legs (no departure_token)."""
    best = []
    for f in flights:
        best.append({
            "flights": [
                {
                    "airline": f["airline"],
                    "departure_airport": {"time": f["dep"]},
                    "arrival_airport": {"time": f["arr"]},
                }
            ],
            "total_duration": f.get("dur", 90),
        })
    return {"best_flights": best, "other_flights": []}


def _mock_two_calls(payload1: dict, payload2: dict):
    """Return a context manager that yields different payloads for call 1 and call 2."""
    def _make_resp(data: dict):
        r = mock.MagicMock()
        r.status_code = 200
        r.json.return_value = data
        r.raise_for_status = mock.MagicMock()
        return r

    responses = [_make_resp(payload1), _make_resp(payload2)]
    return mock.patch("tools.live_apis.requests.get", side_effect=responses)


# ── Basic round-trip parsing ──────────────────────────────────────────────────

def test_round_trip_search_returns_paired_strings():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    p1 = _outbound_payload([
        {"airline": "Delta", "dep": "2026-05-01 08:00", "arr": "2026-05-01 09:30",
         "dur": 90, "price": 350, "token": "tok1"},
    ])
    p2 = _return_payload([
        {"airline": "Delta", "dep": "2026-05-03 17:00", "arr": "2026-05-03 18:30", "dur": 90},
    ])
    with _mock_two_calls(p1, p2):
        out_strs, ret_strs = api.flight_search_round_trip("JFK", "ATL", "2026-05-01", "2026-05-03")
    assert len(out_strs) == 1
    assert len(ret_strs) == 1


def test_round_trip_cost_attributed_to_day_1_only():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    p1 = _outbound_payload([
        {"airline": "WN", "dep": "2026-05-01 09:00", "arr": "2026-05-01 10:15",
         "dur": 75, "price": 280, "token": "tok2"},
    ])
    p2 = _return_payload([
        {"airline": "WN", "dep": "2026-05-03 16:00", "arr": "2026-05-03 17:15", "dur": 75},
    ])
    with _mock_two_calls(p1, p2):
        out_strs, ret_strs = api.flight_search_round_trip("NYC", "DAL", "2026-05-01", "2026-05-03")
    assert "Cost: $280" in out_strs[0]


def test_round_trip_return_leg_has_no_cost_fragment():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    p1 = _outbound_payload([
        {"airline": "AA", "dep": "2026-05-01 07:00", "arr": "2026-05-01 09:00",
         "dur": 120, "price": 400, "token": "tok3"},
    ])
    p2 = _return_payload([
        {"airline": "AA", "dep": "2026-05-03 15:00", "arr": "2026-05-03 17:00", "dur": 120},
    ])
    with _mock_two_calls(p1, p2):
        out_strs, ret_strs = api.flight_search_round_trip("NYC", "LAX", "2026-05-01", "2026-05-03")
    assert "Cost:" not in ret_strs[0]


def test_round_trip_falls_back_on_serpapi_error():
    api = LiveTravelAPIs(serpapi_api_key="fake")
    with mock.patch("tools.live_apis.requests.get", side_effect=Exception("timeout")):
        with pytest.raises(RuntimeError, match="round-trip"):
            api.flight_search_round_trip("JFK", "LAX", "2026-05-01", "2026-05-03")


def test_round_trip_falls_back_when_no_departure_token():
    """Call 1 returns options with no departure_token → error."""
    api = LiveTravelAPIs(serpapi_api_key="fake")
    # Options without departure_token are silently dropped; all dropped → error
    p1 = {"best_flights": [
        {"flights": [{"airline": "DL",
                      "departure_airport": {"time": "2026-05-01 08:00"},
                      "arrival_airport": {"time": "2026-05-01 09:30"}}],
         "total_duration": 90, "price": 300}
        # No "departure_token" field
    ], "other_flights": []}
    p2 = _return_payload([
        {"airline": "DL", "dep": "2026-05-03 17:00", "arr": "2026-05-03 18:30"},
    ])
    with _mock_two_calls(p1, p2):
        with pytest.raises(RuntimeError, match="departure_token"):
            api.flight_search_round_trip("JFK", "LAX", "2026-05-01", "2026-05-03")


def test_round_trip_late_return_preference():
    """prefer_late_return=True picks the 17:00 return option over 08:00."""
    api = LiveTravelAPIs(serpapi_api_key="fake")
    p1 = _outbound_payload([
        {"airline": "DL", "dep": "2026-05-01 08:00", "arr": "2026-05-01 09:30",
         "dur": 90, "price": 300, "token": "tok4"},
    ])
    # Return payload has two options: early and late
    p2 = {"best_flights": [
        {"flights": [{"airline": "DL",
                      "departure_airport": {"time": "2026-05-03 08:00"},
                      "arrival_airport": {"time": "2026-05-03 09:30"}}],
         "total_duration": 90},
        {"flights": [{"airline": "WN",
                      "departure_airport": {"time": "2026-05-03 17:00"},
                      "arrival_airport": {"time": "2026-05-03 18:30"}}],
         "total_duration": 90},
    ], "other_flights": []}
    with _mock_two_calls(p1, p2):
        _out, ret_strs = api.flight_search_round_trip(
            "JFK", "ATL", "2026-05-01", "2026-05-03", prefer_late_return=True
        )
    assert "17:00" in ret_strs[0]


def test_round_trip_respects_duration_cap_on_outbound():
    """Duration cap filters long outbound options; short one wins."""
    api = LiveTravelAPIs(serpapi_api_key="fake")
    # JFK→BOS: cap ≈180 min.
    p1 = _outbound_payload([
        {"airline": "DL", "dep": "2026-05-01 08:00", "arr": "2026-05-01 09:00",
         "dur": 60, "price": 200, "token": "tok_short"},
        {"airline": "AA", "dep": "2026-05-01 06:00", "arr": "2026-05-01 11:00",
         "dur": 300, "price": 100, "token": "tok_long"},  # too long
    ])
    p2 = _return_payload([
        {"airline": "DL", "dep": "2026-05-03 17:00", "arr": "2026-05-03 18:00", "dur": 60},
    ])
    with _mock_two_calls(p1, p2):
        out_strs, _ret = api.flight_search_round_trip(
            "JFK", "BOS", "2026-05-01", "2026-05-03",
            dep_id="JFK", arr_id="BOS",
        )
    # The winner outbound should be the short (60-min) flight
    assert "DL" in out_strs[0]
    assert "09:00" in out_strs[0]


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
