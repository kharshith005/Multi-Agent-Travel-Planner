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
from typing import Any

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


@dataclass
class _RoundTripPair:
    outbound: _FlightOption
    inbound: _FlightOption
    total_price: int | None


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

    # -------------------- Flight options (SerpAPI Google Flights) --------------------
    def flight_search(
        self,
        origin: str,
        destination: str,
        date: str,
        *,
        max_results: int = 5,
        budget: int | None = None,
        prefer_late_departure: bool = False,
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

        try:
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            if resp.status_code == 400 and "cannot be in the past" in resp.text.lower():
                params["outbound_date"] = datetime.utcnow().strftime("%Y-%m-%d")
                resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
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
                    )
                )

        options = self._dedupe_flight_options(options)
        ranked = self._rank_flight_options(
            options, budget=budget,
            dep_id=dep_id, arr_id=arr_id,
            prefer_late_departure=prefer_late_departure,
        )

        out: list[str] = []
        for o in ranked[:max_results]:
            duration_txt = self._format_duration(o.duration_min)
            seg = f"{o.airline} {o.dep_time}->{o.arr_time}"
            if duration_txt:
                seg += f" Duration: {duration_txt}"
            if o.price is not None:
                seg += f" Cost: ${o.price}"
            out.append(seg)

        if not out:
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
    ) -> list[_FlightOption]:
        if not options:
            return []

        # 9.1 — Distance-aware duration cap (applied before budget filter so a
        # very cheap but very long option doesn't slip through the budget check).
        cap_min = _duration_cap_minutes(dep_id, arr_id)
        if cap_min is not None:
            capped = [o for o in options if o.duration_min is None or o.duration_min <= cap_min]
            if capped:
                options = capped
            # else: skip cap — better to return something than nothing

        # Hard budget filter: a single leg costing more than 50% of the trip
        # budget leaves nothing for the return leg, let alone lodging and
        # dining. Drop those options entirely (only when a budget is known and
        # the filter leaves at least one option standing).
        if budget and budget > 0:
            hard_cap = int(0.50 * budget)
            within_cap = [
                o for o in options
                if o.price is None or (o.price is not None and o.price <= hard_cap)
            ]
            if within_cap:
                options = within_cap

        priced = [o for o in options if o.price is not None]
        if not priced:
            return sorted(options, key=lambda o: (o.duration_min or 10**9, o.dep_time, o.arr_time))

        cheapest = min(o.price for o in priced if o.price is not None)
        # Prefer shortest-duration flights; a traveler values their time.
        # Accept up to 40% (min $100) price premium for a significantly shorter flight.
        tolerance = max(100, int(0.40 * cheapest))

        near_price = [o for o in priced if o.price is not None and o.price <= cheapest + tolerance]

        soft_per_leg = int(budget * 0.25) if budget and budget > 0 else None
        if soft_per_leg:
            within_soft = [o for o in near_price if o.price is not None and o.price <= soft_per_leg]
            candidate_pool = within_soft if within_soft else near_price
        else:
            candidate_pool = near_price

        # 9.2 — Late-return preference: bias toward ≥14:00 departures within
        # a 1.3× price ceiling vs. the cheapest candidate. Falls back to
        # duration-first when no late option fits the ceiling.
        if prefer_late_departure:
            LATE_THRESHOLD = 14 * 60
            late_pool = [
                o for o in candidate_pool
                if (_dep_time_minutes(o.dep_time) or 0) >= LATE_THRESHOLD
            ]
            if late_pool:
                cheapest_pool_price = min(
                    (o.price for o in candidate_pool if o.price is not None), default=None
                )
                best_late = min(late_pool, key=lambda o: (o.price or 10**9, o.dep_time))
                if (
                    cheapest_pool_price is None
                    or best_late.price is None
                    or best_late.price <= 1.3 * cheapest_pool_price
                ):
                    selected = best_late
                else:
                    selected = min(
                        candidate_pool,
                        key=lambda o: (o.duration_min or 10**9, o.price or 10**9, o.dep_time),
                    )
            else:
                selected = min(
                    candidate_pool,
                    key=lambda o: (o.duration_min or 10**9, o.price or 10**9, o.dep_time),
                )
        else:
            selected = min(
                candidate_pool,
                key=lambda o: (o.duration_min or 10**9, o.price or 10**9, o.dep_time),
            )

        remainder = [o for o in options if o is not selected]
        remainder_sorted = sorted(
            remainder,
            key=lambda o: (
                0 if o.price is not None else 1,
                o.price or 10**9,
                o.duration_min or 10**9,
                o.dep_time,
            ),
        )
        return [selected] + remainder_sorted

    def flight_search_round_trip(
        self,
        origin: str,
        dest: str,
        depart_date: str,
        return_date: str,
        *,
        max_results: int = 5,
        budget: int | None = None,
        prefer_late_return: bool = True,
        dep_id: str | None = None,
        arr_id: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Round-trip flight search using SerpAPI type=1.

        Returns (outbound_strings, return_strings). Outbound strings carry
        'Cost: $N' (the round-trip price). Return strings omit the cost
        fragment — cost is attributed to Day 1 only.

        Raises RuntimeError on SerpAPI failure or empty results; the coordinator
        falls back to two one-way searches on any exception.
        """
        if not self.serpapi_api_key:
            raise RuntimeError("Set SERPAPI_API_KEY for flight search")

        dep = (dep_id or origin).upper()
        arr = (arr_id or dest).upper()

        outbound_date = self._normalize_outbound_date(depart_date)
        params = {
            "engine": "google_flights",
            "departure_id": dep,
            "arrival_id": arr,
            "outbound_date": outbound_date,
            "return_date": self._normalize_outbound_date(return_date),
            "type": "1",
            "currency": "USD",
            "hl": "en",
            "adults": 1,
            "api_key": self.serpapi_api_key,
        }

        try:
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            if resp.status_code == 400 and "cannot be in the past" in resp.text.lower():
                params["outbound_date"] = datetime.utcnow().strftime("%Y-%m-%d")
                resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            raise RuntimeError(f"SerpAPI round-trip request failed: {e}") from e

        pairs: list[_RoundTripPair] = []
        for bucket in ("best_flights", "other_flights"):
            for option in payload.get(bucket, []) or []:
                out_flights = option.get("flights") or []
                if not out_flights:
                    continue

                out_first = out_flights[0]
                out_last = out_flights[-1]
                out_dep = (out_first.get("departure_airport") or {}).get("time", "")
                out_arr = (out_last.get("arrival_airport") or {}).get("time", "")
                out_airline = out_first.get("airline", "Flight")
                out_dur = self._extract_duration_minutes(option)
                total_price = self._parse_price(option.get("price"))

                # Parse embedded return leg
                ret_data = option.get("return_flights") or {}
                ret_flights = ret_data.get("flights") if isinstance(ret_data, dict) else None
                if not ret_flights and isinstance(ret_data, list) and ret_data:
                    ret_flights = ret_data[0].get("flights") if isinstance(ret_data[0], dict) else None
                if not ret_flights:
                    continue

                ret_first = ret_flights[0]
                ret_last = ret_flights[-1]
                ret_dep = (ret_first.get("departure_airport") or {}).get("time", "")
                ret_arr = (ret_last.get("arrival_airport") or {}).get("time", "")
                ret_airline = ret_first.get("airline", "Flight")
                ret_dur = self._extract_duration_minutes({"flights": ret_flights})

                out_opt = _FlightOption(
                    airline=out_airline, dep_time=out_dep, arr_time=out_arr,
                    price=total_price, duration_min=out_dur,
                )
                ret_opt = _FlightOption(
                    airline=ret_airline, dep_time=ret_dep, arr_time=ret_arr,
                    price=None, duration_min=ret_dur,
                )
                pairs.append(_RoundTripPair(outbound=out_opt, inbound=ret_opt, total_price=total_price))

        if not pairs:
            raise RuntimeError("SerpAPI round-trip search returned no paired options")

        # Duration cap on each leg
        out_cap = _duration_cap_minutes(dep, arr)
        if out_cap is not None:
            capped = [p for p in pairs if p.outbound.duration_min is None or p.outbound.duration_min <= out_cap]
            if capped:
                pairs = capped

        ret_cap = _duration_cap_minutes(arr, dep)
        if ret_cap is not None:
            capped = [p for p in pairs if p.inbound.duration_min is None or p.inbound.duration_min <= ret_cap]
            if capped:
                pairs = capped

        # Budget filter on total round-trip price
        if budget and budget > 0:
            within = [p for p in pairs if p.total_price is None or p.total_price <= budget]
            if within:
                pairs = within

        # Late-return preference on inbound leg (9.2)
        if prefer_late_return:
            LATE_THRESHOLD = 14 * 60
            late = [
                p for p in pairs
                if (_dep_time_minutes(p.inbound.dep_time) or 0) >= LATE_THRESHOLD
            ]
            if late:
                cheapest_total = min(
                    (p.total_price for p in pairs if p.total_price is not None), default=None
                )
                best_late = min(late, key=lambda p: (p.total_price or 10**9, p.inbound.dep_time))
                if (
                    cheapest_total is None
                    or best_late.total_price is None
                    or best_late.total_price <= 1.3 * cheapest_total
                ):
                    pairs = [best_late] + [p for p in pairs if p is not best_late]

        # Sort remaining pairs by total price + outbound duration
        if len(pairs) > 1:
            head, rest = pairs[0], pairs[1:]
            rest.sort(key=lambda p: (p.total_price or 10**9, p.outbound.duration_min or 10**9))
            pairs = [head] + rest

        out_strs: list[str] = []
        ret_strs: list[str] = []
        for pair in pairs[:max_results]:
            out = pair.outbound
            ret = pair.inbound

            out_dur = self._format_duration(out.duration_min)
            out_seg = f"{out.airline} {out.dep_time}->{out.arr_time}"
            if out_dur:
                out_seg += f" Duration: {out_dur}"
            if out.price is not None:
                out_seg += f" Cost: ${out.price}"
            out_strs.append(out_seg)

            ret_dur = self._format_duration(ret.duration_min)
            ret_seg = f"{ret.airline} {ret.dep_time}->{ret.arr_time}"
            if ret_dur:
                ret_seg += f" Duration: {ret_dur}"
            # No cost on return leg — round-trip cost is on outbound (Day 1) only
            ret_strs.append(ret_seg)

        return out_strs, ret_strs

    def _resolve_with_fallback(
        self, location: str
    ) -> tuple[str, str | None, str | None]:
        """Resolve a city to an IATA code, with ≤200 mi nearest-airport fallback.

        Returns (iata, fallback_city, drive_leg_summary).
        fallback_city and drive_leg_summary are None when location resolves directly.
        Raises RuntimeError if no airport found within 200 mi.
        """
        # Fast path: direct IATA resolution.
        try:
            iata = self._resolve_airport_code(location)
            return iata, None, None
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
        return best_code, best_city, drive_leg

    def _resolve_airport_code(self, location: str) -> str:
        loc = (location or "").strip()
        if _IATA_RE.match(loc.upper()):
            return loc.upper()
        if airportsdata is None:
            raise RuntimeError("airportsdata package not installed. Run: pip install airportsdata")

        airports = airportsdata.load("IATA")
        target = self._norm(loc)
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
