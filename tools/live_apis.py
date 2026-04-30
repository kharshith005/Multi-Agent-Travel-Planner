"""Strict live API helpers (Google Maps + SerpAPI flights).

No fallback logic is used in this module. If required APIs are unavailable,
methods raise RuntimeError so callers can fail fast.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests

try:
    import googlemaps
except ImportError:
    googlemaps = None  # type: ignore[assignment]

try:
    import airportsdata
except ImportError:
    airportsdata = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


_IATA_RE = re.compile(r"^[A-Z]{3}$")

# Common multi-airport metros and cities whose airportsdata city field doesn't
# match the colloquial name (e.g. "New York City" vs. city="New York").
# Checked before substring search so ambiguous names never land on seaplane bases.
_CITY_AIRPORT_ALIASES: dict[str, list[str]] = {
    "new york":         ["JFK", "LGA", "EWR"],
    "new york city":    ["JFK", "LGA", "EWR"],
    "nyc":              ["JFK", "LGA", "EWR"],
    "washington":       ["IAD", "DCA", "BWI"],
    "washington dc":    ["IAD", "DCA", "BWI"],
    "washington d.c.":  ["IAD", "DCA", "BWI"],
    "chicago":          ["ORD", "MDW"],
    "houston":          ["IAH", "HOU"],
    "dallas":           ["DFW", "DAL"],
    "fort worth":       ["DFW"],
    "los angeles":      ["LAX", "BUR", "LGB", "SNA"],
    "san francisco":    ["SFO", "OAK", "SJC"],
    "miami":            ["MIA", "FLL"],
    "boston":           ["BOS"],
    "seattle":          ["SEA"],
    "denver":           ["DEN"],
    "atlanta":          ["ATL"],
    "phoenix":          ["PHX"],
    "minneapolis":      ["MSP"],
    "detroit":          ["DTW"],
    "orlando":          ["MCO"],
    "london":           ["LHR", "LGW", "STN", "LCY"],
    "paris":            ["CDG", "ORY"],
    "tokyo":            ["HND", "NRT"],
    "osaka":            ["KIX", "ITM"],
}

# Name substrings that identify non-commercial fields with no scheduled service.
_NON_COMMERCIAL_TAGS = (
    "seaplane", " spb", "heliport", "helipad", "heli ",
    "airpark", " strip", "float plane", "floatplane",
)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles between two (lat, lon) points."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class _FlightOption:
    airline: str
    dep_time: str
    arr_time: str
    price: int | None
    duration_min: int | None
    departure_token: str | None = None
    flight_number: str | None = None



def _dep_time_minutes(dep_time: str) -> int | None:
    """Extract minutes-since-midnight from a time string like '2024-04-15 14:30' or '14:30'."""
    if not dep_time:
        return None
    m = re.search(r"(\d{1,2}):(\d{2})(?:\s|$)", dep_time)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def _duration_cap_minutes(dep_id: str | None, arr_id: str | None) -> int | None:
    """Return bracket-based max flight duration (minutes) for the route, or None if unknown."""
    if not dep_id or not arr_id or airportsdata is None:
        return None
    airports = airportsdata.load("IATA")
    dep_meta = airports.get(dep_id.upper(), {})
    arr_meta = airports.get(arr_id.upper(), {})
    try:
        dep_lat = float(dep_meta["lat"])
        dep_lon = float(dep_meta["lon"])
        arr_lat = float(arr_meta["lat"])
        arr_lon = float(arr_meta["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    dist = _haversine_miles(dep_lat, dep_lon, arr_lat, arr_lon)
    if dist < 500:
        return 180
    if dist < 1000:
        return 240
    if dist < 2000:
        return 360
    return 600


@dataclass
class LiveTravelAPIs:
    google_maps_api_key: str = ""
    serpapi_api_key: str = ""

    @classmethod
    def from_env(cls) -> "LiveTravelAPIs":
        return cls(
            google_maps_api_key=os.environ.get("GOOGLE_MAPS_API_KEY", "").strip(),
            serpapi_api_key=os.environ.get("SERPAPI_API_KEY", "").strip(),
        )

    def _client(self):
        if not self.google_maps_api_key:
            raise RuntimeError("Set GOOGLE_MAPS_API_KEY")
        if googlemaps is None:
            raise RuntimeError("googlemaps package not installed. Run: pip install googlemaps")
        return googlemaps.Client(key=self.google_maps_api_key)

    # -------------------- Places --------------------
    def search_places(self, city: str, query: str, *, max_results: int = 8) -> list[dict[str, Any]]:
        client = self._client()
        q = f"{query} in {city}"
        try:
            payload = client.places(q)
        except Exception as e:
            raise RuntimeError(f"Google Maps places API failed: {e}") from e

        rows = payload.get("results") or []
        out: list[dict[str, Any]] = []
        for r in rows[:max_results]:
            out.append(
                {
                    "name": r.get("name"),
                    "address": r.get("formatted_address"),
                    "rating": r.get("rating"),
                    "price_level": r.get("price_level"),
                }
            )
        return out

    # -------------------- Lodging geo search --------------------
    def search_lodging_near(
        self,
        latlng: tuple[float, float],
        *,
        radius_m: int = 48000,
        max_results: int = 20,
        budget_levels: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for hotels within `radius_m` metres of `latlng` using Places Nearby.

        Over-fetches (2× max_results) then filters by `budget_levels` and sorts
        by rating descending, price_level ascending. Hotels with no price_level
        are kept as "tier unknown".
        """
        client = self._client()
        try:
            payload = client.places_nearby(
                location=latlng,
                radius=radius_m,
                type="lodging",
            )
        except Exception as e:
            raise RuntimeError(f"Google Maps places_nearby (lodging) failed: {e}") from e

        rows = payload.get("results") or []
        out: list[dict[str, Any]] = []
        for r in rows[: max_results * 2]:
            level = r.get("price_level")
            if budget_levels and level is not None and level not in budget_levels:
                continue
            out.append({
                "name": r.get("name"),
                "address": r.get("vicinity") or r.get("formatted_address"),
                "rating": r.get("rating"),
                "price_level": level,
            })

        # Highest-rated first; within same rating, cheapest tier first.
        out.sort(key=lambda h: (-(h.get("rating") or 0), h.get("price_level") or 99))
        return out[:max_results]

    # -------------------- Distance/Routes --------------------
    def route_summary(self, origin: str, destination: str, mode: str = "driving") -> str | None:
        client = self._client()
        matrix_mode = "driving" if mode.lower() == "taxi" else mode.lower()
        try:
            payload = client.distance_matrix(
                origins=[origin],
                destinations=[destination],
                mode=matrix_mode,
                units="imperial",
            )
        except Exception as e:
            raise RuntimeError(f"Google Maps distance matrix API failed: {e}") from e

        rows = payload.get("rows") or []
        if not rows:
            return None
        elements = rows[0].get("elements") or []
        if not elements:
            return None
        e = elements[0]
        if e.get("status") != "OK":
            return None

        distance = (e.get("distance") or {}).get("text", "")
        duration = (e.get("duration") or {}).get("text", "")
        if mode.lower() == "taxi":
            est = self._estimate_taxi_cost(distance)
            if est is not None:
                return f"Distance: {distance}, Duration: {duration}, Cost: ${est}"
        return f"Distance: {distance}, Duration: {duration}"

    # -------------------- SerpAPI helper --------------------
    def _serpapi_get(self, params: dict, *, timeout: int = 20) -> dict:
        """Execute a SerpAPI request with automatic past-date retry.

        Raises RuntimeError on non-200 responses or network errors.
        """
        try:
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=timeout)
            if resp.status_code == 400 and "cannot be in the past" in resp.text.lower():
                params = {**params, "outbound_date": datetime.utcnow().strftime("%Y-%m-%d")}
                resp = requests.get("https://serpapi.com/search.json", params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"SerpAPI request failed: {e}") from e

    # -------------------- Flight options (SerpAPI Google Flights) --------------------
    def flight_search(
        self,
        origin: str,
        destination: str,
        date: str,
        *,
        max_results: int = 10,
        budget: int | None = None,
        prefer_late_departure: bool = False,
        prefer_early_departure: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        if not self.serpapi_api_key:
            raise RuntimeError("Set SERPAPI_API_KEY for flight search")

        dep_id = self._resolve_airport_code(origin)
        arr_id = self._resolve_airport_code(destination)

        outbound_date = self._normalize_outbound_date(date)
        params = {
            "engine": "google_flights",
            "departure_id": dep_id,
            "arrival_id": arr_id,
            "outbound_date": outbound_date,
            "type": "2",
            "currency": "USD",
            "hl": "en",
            "adults": 1,
            "api_key": self.serpapi_api_key,
        }
        # Server-side time-of-day filter: morning outbound (4–13) so Day 1 has
        # time for activities; afternoon/evening return (12–21) so last day
        # gets at least breakfast/lunch. SerpAPI accepts "start,end" hours.
        if prefer_early_departure:
            params["outbound_times"] = "4,13"
        elif prefer_late_departure:
            params["outbound_times"] = "12,21"

        try:
            payload = self._serpapi_get(params)
            # If a tight time filter returns nothing, broaden by dropping it.
            if params.get("outbound_times") and not (
                (payload.get("best_flights") or payload.get("other_flights"))
            ):
                if on_progress:
                    on_progress(
                        f"No flights matched outbound_times={params['outbound_times']} — broadening."
                    )
                params.pop("outbound_times", None)
                payload = self._serpapi_get(params)
        except Exception as e:
            raise RuntimeError(f"SerpAPI flights request failed: {e}") from e

        options: list[_FlightOption] = []
        for bucket in ("best_flights", "other_flights"):
            for option in payload.get(bucket, []) or []:
                flights = option.get("flights") or []
                if not flights:
                    continue
                first = flights[0]
                last = flights[-1]
                dep = (first.get("departure_airport") or {}).get("time", "")
                arr = (last.get("arrival_airport") or {}).get("time", "")
                airline = first.get("airline", "Flight")
                price = self._parse_price(option.get("price"))
                duration_min = self._extract_duration_minutes(option)
                options.append(
                    _FlightOption(
                        airline=airline,
                        dep_time=dep,
                        arr_time=arr,
                        price=price,
                        duration_min=duration_min,
                        flight_number=first.get("flight_number") or None,
                    )
                )

        options = self._dedupe_flight_options(options)
        flight_warnings: list[str] = []
        ranked = self._rank_flight_options(
            options, budget=budget,
            dep_id=dep_id, arr_id=arr_id,
            prefer_late_departure=prefer_late_departure,
            prefer_early_departure=prefer_early_departure,
            _warnings=flight_warnings,
        )
        for w in flight_warnings:
            if on_progress:
                on_progress(w)

        out: list[str] = []
        for o in ranked[:max_results]:
            duration_txt = self._format_duration(o.duration_min)
            fn = f" {o.flight_number}" if o.flight_number else ""
            seg = f"{o.airline}{fn} {o.dep_time}->{o.arr_time}"
            if duration_txt:
                seg += f" Duration: {duration_txt}"
            if o.price is not None:
                seg += f" Cost: ${o.price}"
            out.append(seg)

        if not out:
            if on_progress:
                on_progress(
                    f"SerpAPI has no Google Flights inventory for "
                    f"{dep_id}↔{arr_id} — airport may lack scheduled service."
                )
            raise RuntimeError("SerpAPI flights returned no options")
        return out

    @staticmethod
    def _parse_price(raw: object) -> int | None:
        if isinstance(raw, (int, float)):
            return int(raw)
        if isinstance(raw, str):
            m = re.search(r"\d+", raw.replace(",", ""))
            if m:
                return int(m.group(0))
        return None

    @staticmethod
    def _duration_from_value(raw: object) -> int | None:
        if isinstance(raw, (int, float)):
            return int(raw)
        if not isinstance(raw, str):
            return None
        txt = raw.lower().strip()
        if txt.isdigit():
            return int(txt)

        hours = 0
        mins = 0
        h = re.search(r"(\d+)\s*h", txt)
        m = re.search(r"(\d+)\s*m", txt)
        if h:
            hours = int(h.group(1))
        if m:
            mins = int(m.group(1))
        total = hours * 60 + mins
        return total if total > 0 else None

    def _extract_duration_minutes(self, option: dict[str, Any]) -> int | None:
        direct = self._duration_from_value(option.get("total_duration"))
        if direct is not None:
            return direct

        direct = self._duration_from_value(option.get("duration"))
        if direct is not None:
            return direct

        flights = option.get("flights") or []
        total = 0
        any_part = False
        for f in flights:
            if not isinstance(f, dict):
                continue
            part = self._duration_from_value(f.get("duration"))
            if part is not None:
                total += part
                any_part = True
        if any_part:
            return total
        return None

    @staticmethod
    def _format_duration(minutes: int | None) -> str:
        if minutes is None or minutes <= 0:
            return ""
        h, m = divmod(minutes, 60)
        if h and m:
            return f"{h}h {m}m"
        if h:
            return f"{h}h"
        return f"{m}m"

    @staticmethod
    def _dedupe_flight_options(options: list[_FlightOption]) -> list[_FlightOption]:
        seen: set[tuple[str, str, str, int | None, int | None]] = set()
        out: list[_FlightOption] = []
        for o in options:
            key = (o.airline, o.dep_time, o.arr_time, o.price, o.duration_min)
            if key in seen:
                continue
            seen.add(key)
            out.append(o)
        return out

    @staticmethod
    def _rank_flight_options(
        options: list[_FlightOption],
        budget: int | None = None,
        dep_id: str | None = None,
        arr_id: str | None = None,
        prefer_late_departure: bool = False,
        prefer_early_departure: bool = False,
        _warnings: list[str] | None = None,
    ) -> list[_FlightOption]:
        if not options:
            return []

        # 10.1 — Duration cap (soft): skip when it would empty the pool; warn when skipped.
        cap_min = _duration_cap_minutes(dep_id, arr_id)
        if cap_min is not None:
            capped = [o for o in options if o.duration_min is None or o.duration_min <= cap_min]
            if capped:
                options = capped
            elif _warnings is not None:
                longest = max((o.duration_min or 0) for o in options)
                _warnings.append(
                    f"⚠️ All flight options exceed the {cap_min}min cap for this route "
                    f"(longest: {longest}min). Returning best available."
                )

        # Hard budget filter: drop options costing more than 50% of trip budget.
        if budget and budget > 0:
            hard_cap = int(0.50 * budget)
            within_cap = [o for o in options if o.price is None or o.price <= hard_cap]
            if within_cap:
                options = within_cap

        # 10.4 — Duration-first sort: shortest travel time is most valuable.
        options = sorted(options, key=lambda o: (o.duration_min or 10**9, o.price or 10**9, o.dep_time))

        # 10.2 + 10.3 — Time-window preference with layered fallback.
        if prefer_early_departure:
            # Outbound morning preference: ≤11:00 → ≤14:00 → earliest in pool.
            EARLY = 11 * 60
            EARLY_FB = 14 * 60
            early = [o for o in options if (_dep_time_minutes(o.dep_time) or 10**9) <= EARLY]
            if not early:
                early = [o for o in options if (_dep_time_minutes(o.dep_time) or 10**9) <= EARLY_FB]
            selected = (
                min(early, key=lambda o: (_dep_time_minutes(o.dep_time) or 10**9, o.duration_min or 10**9))
                if early else options[0]
            )
        elif prefer_late_departure:
            # Return leg: ≥14:00 within 1.3× cheapest → ≥11:00 within ceiling → duration-first.
            LATE = 14 * 60
            LATE_FB = 11 * 60
            cheapest_p = min((o.price for o in options if o.price is not None), default=None)
            ceiling = cheapest_p * 1.3 if cheapest_p is not None else None
            late = [
                o for o in options
                if (_dep_time_minutes(o.dep_time) or 0) >= LATE
                and (ceiling is None or o.price is None or o.price <= ceiling)
            ]
            if not late:
                late = [
                    o for o in options
                    if (_dep_time_minutes(o.dep_time) or 0) >= LATE_FB
                    and (ceiling is None or o.price is None or o.price <= ceiling)
                ]
            selected = (
                max(late, key=lambda o: (_dep_time_minutes(o.dep_time) or 0))
                if late else options[0]
            )
        else:
            selected = options[0]

        remainder = [o for o in options if o is not selected]
        return [selected] + remainder

    def flight_search_round_trip(
        self,
        origin: str,
        dest: str,
        depart_date: str,
        return_date: str,
        *,
        max_results: int = 10,
        budget: int | None = None,
        prefer_late_return: bool = True,
        dep_id: str | None = None,
        arr_id: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Round-trip flight search via SerpAPI two-call departure_token protocol.

        Call 1 (type=1): fetch outbound itineraries; each carries a departure_token
        and the round-trip total price.
        Call 2: same params + departure_token → fetch matching return options.

        Returns (outbound_strings, return_strings). Outbound strings carry
        'Cost: $N' (round-trip total). Return strings omit the cost fragment —
        cost is attributed to Day 1 only.

        Raises RuntimeError on SerpAPI failure or empty results; the coordinator
        falls back to two one-way searches on any exception.
        """
        if not self.serpapi_api_key:
            raise RuntimeError("Set SERPAPI_API_KEY for flight search")

        dep = (dep_id or origin).upper()
        arr = (arr_id or dest).upper()

        base_params: dict = {
            "engine": "google_flights",
            "departure_id": dep,
            "arrival_id": arr,
            "outbound_date": self._normalize_outbound_date(depart_date),
            "return_date": self._normalize_outbound_date(return_date),
            "type": "1",
            "currency": "USD",
            "hl": "en",
            "adults": 1,
            "api_key": self.serpapi_api_key,
        }
        # Server-side time-of-day filter: morning outbound (4–13) AND
        # afternoon/evening return (12–21). Round-trip format takes 4 values:
        # "out_start,out_end,ret_start,ret_end".
        if prefer_late_return:
            call1_params = {**base_params, "outbound_times": "4,13,12,21"}
        else:
            call1_params = {**base_params, "outbound_times": "4,13"}

        # ── Call 1: outbound itineraries ──────────────────────────────────────
        try:
            payload1 = self._serpapi_get(call1_params)
            # If the time filter is too tight, broaden by dropping it.
            if not (payload1.get("best_flights") or payload1.get("other_flights")):
                if on_progress:
                    on_progress(
                        f"No round-trip itineraries matched outbound_times="
                        f"{call1_params['outbound_times']} — broadening."
                    )
                payload1 = self._serpapi_get(base_params)
        except Exception as e:
            raise RuntimeError(f"SerpAPI round-trip outbound request failed: {e}") from e

        outbound_candidates: list[_FlightOption] = []
        for bucket in ("best_flights", "other_flights"):
            for option in payload1.get(bucket, []) or []:
                flights = option.get("flights") or []
                if not flights:
                    continue
                first = flights[0]
                last = flights[-1]
                token = option.get("departure_token")
                if not token:
                    continue
                outbound_candidates.append(_FlightOption(
                    airline=first.get("airline", "Flight"),
                    dep_time=(first.get("departure_airport") or {}).get("time", ""),
                    arr_time=(last.get("arrival_airport") or {}).get("time", ""),
                    price=self._parse_price(option.get("price")),
                    duration_min=self._extract_duration_minutes(option),
                    departure_token=token,
                    flight_number=first.get("flight_number") or None,
                ))

        if not outbound_candidates:
            raise RuntimeError("SerpAPI round-trip search returned no outbound options with departure_token")

        # Apply duration cap + budget filter + late-return preference to pick winner outbound.
        out_cap = _duration_cap_minutes(dep, arr)
        if out_cap is not None:
            capped = [o for o in outbound_candidates if o.duration_min is None or o.duration_min <= out_cap]
            if capped:
                outbound_candidates = capped
            elif on_progress:
                longest = max((o.duration_min or 0) for o in outbound_candidates)
                on_progress(
                    f"All outbound options exceed the {out_cap}min cap "
                    f"(longest: {longest}min). Returning best available."
                )

        if budget and budget > 0:
            within = [o for o in outbound_candidates if o.price is None or o.price <= budget]
            if within:
                outbound_candidates = within

        outbound_candidates.sort(key=lambda o: (o.duration_min or 10**9, o.price or 10**9))

        # Pick the single best outbound; keep N runners-up for the result list.
        top_outbounds = outbound_candidates[:max_results]

        if on_progress:
            on_progress(f"Round-trip: {len(outbound_candidates)} outbound options found; fetching return flights.")

        # ── Call 2: return itineraries for the best outbound ─────────────────
        winner = top_outbounds[0]
        try:
            payload2 = self._serpapi_get({**base_params, "departure_token": winner.departure_token})
        except Exception as e:
            raise RuntimeError(f"SerpAPI round-trip return request failed: {e}") from e

        return_candidates: list[_FlightOption] = []
        for bucket in ("best_flights", "other_flights"):
            for option in payload2.get(bucket, []) or []:
                flights = option.get("flights") or []
                if not flights:
                    continue
                first = flights[0]
                last = flights[-1]
                return_candidates.append(_FlightOption(
                    airline=first.get("airline", "Flight"),
                    dep_time=(first.get("departure_airport") or {}).get("time", ""),
                    arr_time=(last.get("arrival_airport") or {}).get("time", ""),
                    price=None,
                    duration_min=self._extract_duration_minutes(option),
                    flight_number=first.get("flight_number") or None,
                ))

        if not return_candidates:
            raise RuntimeError("SerpAPI round-trip search returned no return options")

        # Duration cap on return leg (soft).
        ret_cap = _duration_cap_minutes(arr, dep)
        if ret_cap is not None:
            capped = [r for r in return_candidates if r.duration_min is None or r.duration_min <= ret_cap]
            if capped:
                return_candidates = capped
            elif on_progress:
                longest = max((r.duration_min or 0) for r in return_candidates)
                on_progress(
                    f"All return options exceed the {ret_cap}min cap "
                    f"(longest: {longest}min). Returning best available."
                )

        # 10.3 — Late-return preference with layered fallback: ≥14:00 → ≥11:00 → duration-first.
        best_return: _FlightOption
        if prefer_late_return:
            LATE, LATE_FB = 14 * 60, 11 * 60
            late = [r for r in return_candidates if (_dep_time_minutes(r.dep_time) or 0) >= LATE]
            if not late:
                late = [r for r in return_candidates if (_dep_time_minutes(r.dep_time) or 0) >= LATE_FB]
            best_return = (
                max(late, key=lambda r: (_dep_time_minutes(r.dep_time) or 0))
                if late else return_candidates[0]
            )
        else:
            return_candidates.sort(key=lambda r: (r.duration_min or 10**9))
            best_return = return_candidates[0]

        # ── Format output ─────────────────────────────────────────────────────
        out_strs: list[str] = []
        for o in top_outbounds:
            fn = f" {o.flight_number}" if o.flight_number else ""
            seg = f"{o.airline}{fn} {o.dep_time}->{o.arr_time}"
            dur = self._format_duration(o.duration_min)
            if dur:
                seg += f" Duration: {dur}"
            if o.price is not None:
                seg += f" Cost: ${o.price}"
            out_strs.append(seg)

        # Emit up to max_results return options: best_return at index 0, rest as runners-up.
        # No cost on return legs — round-trip total is attributed to outbound (Day 1) only.
        ordered_returns = [best_return] + [r for r in return_candidates if r is not best_return]
        ret_strs: list[str] = []
        for r in ordered_returns[:max_results]:
            fn = f" {r.flight_number}" if r.flight_number else ""
            seg = f"{r.airline}{fn} {r.dep_time}->{r.arr_time}"
            dur = self._format_duration(r.duration_min)
            if dur:
                seg += f" Duration: {dur}"
            ret_strs.append(seg)

        if on_progress:
            on_progress(
                f"Round-trip: {len(out_strs)} outbound + {len(ret_strs)} return option(s) selected."
            )

        return out_strs, ret_strs

    def _resolve_with_fallback(
        self, location: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> tuple[str, str | None, str | None, float | None, float | None]:
        """Resolve a city to an IATA code, with ≤200 mi nearest-airport fallback.

        Returns (iata, fallback_city, drive_leg_summary, lat, lon).
        fallback_city and drive_leg_summary are None when location resolves directly.
        lat/lon are the resolved location's coordinates (airport for direct IATA;
        city geocode for fallback) — used by search_lodging_near.
        Raises RuntimeError if no airport found within 200 mi.
        """
        # Fast path: direct IATA resolution.
        try:
            iata = self._resolve_airport_code(location)
            if on_progress:
                on_progress(f"Airport for '{location}': {iata}")
            # Look up airport lat/lon for lodging geo search.
            ap_lat: float | None = None
            ap_lon: float | None = None
            if airportsdata is not None:
                airports = airportsdata.load("IATA")
                meta = airports.get(iata.upper(), {})
                try:
                    ap_lat = float(meta["lat"])
                    ap_lon = float(meta["lon"])
                except (KeyError, TypeError, ValueError):
                    pass
            return iata, None, None, ap_lat, ap_lon
        except RuntimeError:
            pass

        if airportsdata is None:
            raise RuntimeError("airportsdata package not installed. Run: pip install airportsdata")

        # Geocode to lat/lon.
        try:
            geocode = self._client().geocode(location)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot geocode '{location}' for airport fallback: {exc}"
            ) from exc

        if not geocode:
            raise RuntimeError(f"Geocode returned no results for '{location}'")

        loc = geocode[0]["geometry"]["location"]
        loc_lat, loc_lon = float(loc["lat"]), float(loc["lng"])

        # Find nearest US airport within 200 mi.
        airports = airportsdata.load("IATA")
        best_code: str | None = None
        best_city: str | None = None
        best_dist = float("inf")

        for code, meta in airports.items():
            if not code or not _IATA_RE.match(code):
                continue
            if meta.get("country") != "US":
                continue
            # Skip seaplane bases, heliports, and private airparks — no scheduled service.
            name_lower = str(meta.get("name", "")).lower()
            if any(tag in name_lower for tag in _NON_COMMERCIAL_TAGS):
                continue
            try:
                ap_lat = float(meta["lat"])
                ap_lon = float(meta["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            dist = _haversine_miles(loc_lat, loc_lon, ap_lat, ap_lon)
            if dist < best_dist:
                best_dist = dist
                best_code = code
                best_city = str(meta.get("city") or meta.get("name") or code)

        if best_code is None or best_dist > 200:
            raise RuntimeError(
                f"No airport within 200 mi of '{location}' "
                f"(nearest: {best_dist:.0f} mi)"
            )

        # Drive leg from location to the fallback airport city.
        try:
            drive_route = self.route_summary(location, best_city, "driving") or ""
        except Exception:
            drive_route = f"Distance: {int(best_dist)} mi"

        drive_leg = f"Drive {int(best_dist)} mi to {best_code}; {drive_route}"
        if on_progress:
            on_progress(
                f"No direct airport for '{location}'; nearest is {best_code} "
                f"({int(best_dist)} mi away) — drive transfer included."
            )
        # Return city lat/lon (geocoded destination) for lodging search radius.
        return best_code, best_city, drive_leg, loc_lat, loc_lon

    def _resolve_airport_code(self, location: str) -> str:
        loc = (location or "").strip()
        if _IATA_RE.match(loc.upper()):
            return loc.upper()
        if airportsdata is None:
            raise RuntimeError("airportsdata package not installed. Run: pip install airportsdata")

        airports = airportsdata.load("IATA")
        target = self._norm(loc)

        # Alias map: resolves metros whose airportsdata city field doesn't match
        # the colloquial name, and prevents seaplane-base false positives.
        if target in _CITY_AIRPORT_ALIASES:
            return _CITY_AIRPORT_ALIASES[target][0]

        matches: list[tuple[str, str, str]] = []
        for code, meta in airports.items():
            if not code or not _IATA_RE.match(code):
                continue
            city = self._norm(str(meta.get("city", "")))
            name = self._norm(str(meta.get("name", "")))
            if city == target or target in name:
                matches.append((code, city, str(meta.get("name", ""))))

        if not matches:
            raise RuntimeError(f"Could not resolve airport code for '{location}'")

        def rank(item: tuple[str, str, str]) -> tuple[int, int, int, int, str]:
            code, city, name_raw = item
            name = self._norm(name_raw)
            exact_city = city == target
            is_international = "international" in name
            starts_with_city = name.startswith(target)
            return (
                0 if exact_city else 1,
                0 if is_international else 1,
                0 if starts_with_city else 1,
                len(name),
                code,
            )

        matches.sort(key=rank)
        return matches[0][0]

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _estimate_taxi_cost(distance_text: str) -> int | None:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mi", distance_text or "", flags=re.IGNORECASE)
        if not m:
            return None
        miles = float(m.group(1))
        return max(12, int(round(miles * 2.5)))

    @staticmethod
    def _normalize_outbound_date(raw_date: str) -> str:
        today = datetime.utcnow().date()
        txt = (raw_date or "").strip()
        if not txt:
            return today.strftime("%Y-%m-%d")
        try:
            dt = datetime.strptime(txt[:10], "%Y-%m-%d").date()
        except ValueError:
            return today.strftime("%Y-%m-%d")
        if dt < today:
            return today.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def default_live_apis() -> LiveTravelAPIs:
    return LiveTravelAPIs.from_env()
