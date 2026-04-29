"""In-memory sandbox exposing the six TravelPlanner tools.

Loads `*_ref_info.jsonl` once and builds:
  - city -> attractions / restaurants / accommodations
  - (origin, dest, date) -> flights
  - (origin, dest) -> self-driving / taxi distance rows

The tools are deterministic and return structured records that agents can
cite verbatim. Cities outside the corpus return an empty list.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"
_STATE_INDEX_PATH = Path(__file__).resolve().parents[1] / "eval" / "state_city_index.json"


@lru_cache(maxsize=1)
def _load_state_index() -> dict[str, list[str]]:
    if _STATE_INDEX_PATH.exists():
        return json.loads(_STATE_INDEX_PATH.read_text(encoding="utf-8"))
    return {}


# ----- parsing helpers -----

_FLIGHT_KEY = re.compile(r"^Flight from (.+) to (.+) on (\d{4}-\d{2}-\d{2})$")
_DRIVE_KEY = re.compile(r"^Self-driving from (.+) to (.+)$")
_TAXI_KEY = re.compile(r"^Taxi from (.+) to (.+)$")
_ATTR_KEY = re.compile(r"^Attractions in (.+)$")
_REST_KEY = re.compile(r"^Restaurants in (.+)$")
_ACC_KEY = re.compile(r"^Accommodations in (.+)$")
_CITIES_KEY = re.compile(r"^Cities in (.+)$")


def _records(value: Any) -> list[dict]:
    """Normalize a field's value into a list of dict records."""
    if value is None:
        return []
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict)]
    return []


# ----- sandbox -----

@dataclass
class Sandbox:
    attractions: dict[str, list[dict]] = field(default_factory=dict)
    restaurants: dict[str, list[dict]] = field(default_factory=dict)
    accommodations: dict[str, list[dict]] = field(default_factory=dict)
    flights: dict[tuple[str, str, str], list[dict]] = field(default_factory=dict)
    driving: dict[tuple[str, str], str] = field(default_factory=dict)
    taxi: dict[tuple[str, str], str] = field(default_factory=dict)
    cities_by_state: dict[str, list[str]] = field(default_factory=dict)

    # ----- loaders -----
    @classmethod
    def load(cls, jsonl_paths: list[Path]) -> "Sandbox":
        sb = cls()
        for path in jsonl_paths:
            if not path.exists():
                continue
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sb._absorb(json.loads(line))
        return sb

    def _absorb(self, record: dict) -> None:
        for key, value in record.items():
            if m := _ATTR_KEY.match(key):
                self.attractions.setdefault(m.group(1), []).extend(_records(value))
            elif m := _REST_KEY.match(key):
                self.restaurants.setdefault(m.group(1), []).extend(_records(value))
            elif m := _ACC_KEY.match(key):
                self.accommodations.setdefault(m.group(1), []).extend(_records(value))
            elif m := _FLIGHT_KEY.match(key):
                self.flights.setdefault(
                    (m.group(1), m.group(2), m.group(3)), []
                ).extend(_records(value))
            elif m := _DRIVE_KEY.match(key):
                if isinstance(value, str) and value:
                    self.driving[(m.group(1), m.group(2))] = value
            elif m := _TAXI_KEY.match(key):
                if isinstance(value, str) and value:
                    self.taxi[(m.group(1), m.group(2))] = value
            elif m := _CITIES_KEY.match(key):
                self.cities_by_state.setdefault(m.group(1), []).extend(
                    v for v in (value or []) if isinstance(v, str)
                )

    # ----- the 6 tools -----

    def city_search(self, state: str) -> list[str]:
        """Return city names in a US state."""
        return sorted(set(self.cities_by_state.get(state, [])))

    def flight_search(self, origin: str, destination: str, date: str) -> list[dict]:
        return list(self.flights.get((origin, destination, date), []))

    def distance_matrix(self, origin: str, destination: str, mode: str = "driving") -> str | None:
        mode = mode.lower()
        if mode in ("driving", "self-driving", "car"):
            return self.driving.get((origin, destination))
        if mode == "taxi":
            return self.taxi.get((origin, destination))
        return None

    def attraction_search(self, city: str) -> list[dict]:
        return list(self.attractions.get(city, []))

    def restaurant_search(self, city: str) -> list[dict]:
        return list(self.restaurants.get(city, []))

    def accommodation_search(self, city: str) -> list[dict]:
        return list(self.accommodations.get(city, []))

    def cities_in(self, location: str) -> list[str]:
        """Return the destination cities for a TravelPlanner location string.

        If `location` is a known city (has sandbox data), returns [location].
        If it is a US state, returns the cities from the state index.
        Falls back to [location] when neither match is found.
        """
        if (location in self.attractions or location in self.restaurants
                or location in self.accommodations):
            return [location]
        state_cities = _load_state_index().get(location)
        if state_cities:
            return list(state_cities)
        return [location]

    # ----- introspection -----
    def stats(self) -> dict[str, int]:
        return {
            "cities_with_attractions": len(self.attractions),
            "cities_with_restaurants": len(self.restaurants),
            "cities_with_accommodations": len(self.accommodations),
            "flight_triples": len(self.flights),
            "driving_pairs": len(self.driving),
            "taxi_pairs": len(self.taxi),
        }


@lru_cache(maxsize=4)
def default_sandbox(kind: str = "frozen") -> Sandbox:
    """Cached sandbox loader. kind = 'frozen' or 'refreshed'."""
    if kind == "refreshed":
        paths = [DATASET_DIR / "validation_fresh_ref_info.jsonl"]
    else:
        paths = [
            DATASET_DIR / "train_ref_info.jsonl",
            DATASET_DIR / "validation_ref_info.jsonl",
        ]
    return Sandbox.load(paths)


if __name__ == "__main__":
    sb = default_sandbox("frozen")
    print(sb.stats())
