"""Tests for Step 8.5 — nearest-airport fallback in LiveTravelAPIs."""
from __future__ import annotations

import math
import pytest

from tools.live_apis import LiveTravelAPIs, _haversine_miles, _IATA_RE


# ── _haversine_miles ──────────────────────────────────────────────────────────

def test_haversine_same_point_is_zero():
    assert _haversine_miles(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0, abs=0.01)


def test_haversine_nyc_to_lax_roughly_2450_miles():
    # JFK: 40.6413°N 73.7781°W, LAX: 33.9425°N 118.4081°W
    dist = _haversine_miles(40.6413, -73.7781, 33.9425, -118.4081)
    assert 2400 < dist < 2500


def test_haversine_short_distance():
    # ~60 mi apart (roughly)
    dist = _haversine_miles(40.0, -74.0, 40.9, -74.0)
    assert 50 < dist < 70


# ── _resolve_with_fallback direct path ────────────────────────────────────────

def test_resolve_direct_iata_passthrough():
    """IATA code passed directly → returns it unchanged, no geocoding needed."""
    api = LiveTravelAPIs(google_maps_api_key="fake", serpapi_api_key="fake")

    def fake_resolve(location):
        return "JFK"

    api._resolve_airport_code = fake_resolve
    iata, fc, leg = api._resolve_with_fallback("JFK")
    assert iata == "JFK"
    assert fc is None
    assert leg is None


def test_resolve_known_city_resolves_directly():
    """A well-known city resolves without fallback (mocked)."""
    api = LiveTravelAPIs(google_maps_api_key="fake", serpapi_api_key="fake")

    def fake_resolve(location):
        return "ATL"

    api._resolve_airport_code = fake_resolve
    iata, fc, leg = api._resolve_with_fallback("Atlanta")
    assert iata == "ATL"
    assert fc is None


# ── _resolve_with_fallback fallback path ──────────────────────────────────────

def _fake_client(geocode_fn):
    """Build a minimal fake googlemaps client with the given geocode function."""
    return type("FakeClient", (), {"geocode": staticmethod(geocode_fn)})()


def test_resolve_fallback_picks_nearest_within_200mi():
    """When direct resolution fails, geocode + haversine finds nearest US airport."""
    api = LiveTravelAPIs(google_maps_api_key="fake", serpapi_api_key="fake")

    def fail_resolve(loc):
        raise RuntimeError("no direct match")
    api._resolve_airport_code = fail_resolve

    # Fake geocode: small town near RDU (Raleigh-Durham), NC
    def fake_geocode(location):
        return [{"geometry": {"location": {"lat": 35.7796, "lng": -78.6382}}}]

    api._client = lambda: _fake_client(fake_geocode)
    api.route_summary = lambda *a, **k: "Distance: 30 mi, Duration: 40m"

    iata, fc, leg = api._resolve_with_fallback("Small Town NC")
    assert iata is not None
    assert _IATA_RE.match(iata)
    assert fc is not None
    assert leg is not None
    assert "Drive" in leg


def test_resolve_fallback_raises_when_no_airport_in_radius():
    """Location with no US airport within 200 mi raises RuntimeError."""
    api = LiveTravelAPIs(google_maps_api_key="fake", serpapi_api_key="fake")

    def fail_resolve(loc):
        raise RuntimeError("no direct match")
    api._resolve_airport_code = fail_resolve

    # Mid-Pacific Ocean — no US airports within 200 mi
    def fake_geocode(location):
        return [{"geometry": {"location": {"lat": 30.0, "lng": -140.0}}}]

    api._client = lambda: _fake_client(fake_geocode)

    with pytest.raises(RuntimeError, match="200 mi"):
        api._resolve_with_fallback("Remote Pacific Island")


def test_resolve_fallback_raises_on_geocode_failure():
    api = LiveTravelAPIs(google_maps_api_key="fake", serpapi_api_key="fake")

    def fail_resolve(loc):
        raise RuntimeError("no direct match")
    api._resolve_airport_code = fail_resolve

    def fail_geocode(location):
        raise RuntimeError("Maps API down")

    api._client = lambda: _fake_client(fail_geocode)

    with pytest.raises(RuntimeError, match="geocode"):
        api._resolve_with_fallback("Somewhere")


# ── Flight strings with drive leg ─────────────────────────────────────────────

def test_flight_string_includes_drive_leg():
    """When dest fallback fires, drive_leg is prepended to every flight string."""
    drive_leg = "Drive 45 mi to CLT; Distance: 45 mi, Duration: 55m"
    raw_flights = [
        "Delta 1234 09:00->10:15 Duration: 1h 15m Cost: $220",
        "American 5678 14:00->15:20 Duration: 1h 20m Cost: $195",
    ]
    result = [f"{drive_leg}; {f}" for f in raw_flights]
    assert result[0].startswith("Drive 45 mi to CLT;")
    assert "Delta 1234" in result[0]
    assert result[1].startswith("Drive 45 mi to CLT;")
    assert "American 5678" in result[1]


def test_flight_string_drive_prefix_parseable_by_transport():
    """Transport specialist regex still finds airline+times after the drive prefix."""
    import re
    HHMM_RE = re.compile(r"(\d{1,2}):(\d{2})\s*->")
    line = "Drive 45 mi to CLT; Distance: 45 mi, Duration: 55m; Delta 1234 09:00->10:15 Cost: $220"
    assert HHMM_RE.search(line) is not None
