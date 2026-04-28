"""Pydantic schemas exchanged between agents and returned by the coordinator."""
from __future__ import annotations

import re
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


class Intent(BaseModel):
    """Parsed structured intent extracted from a user's chat message."""
    org: str
    dest: str
    days: int = Field(default=1, ge=1, le=14)
    dates: list[str] = Field(default_factory=list)  # YYYY-MM-DD strings
    people: int = Field(default=1, ge=1)
    budget: Optional[int] = None
    house_rule: Optional[str] = None
    cuisine: Optional[str] = None
    room_type: Optional[str] = None
    transportation: Optional[str] = None  # preferred/required mode

    @field_validator("days", "people", mode="before")
    @classmethod
    def _default_int_when_null(cls, v):
        return 1 if v is None else v

    @field_validator("dates", mode="before")
    @classmethod
    def _dates_none_to_empty(cls, v):
        return v or []


class DayTransport(BaseModel):
    day: int
    description: str = "-"  # human-readable, matches example_submission.jsonl style


class TransportPlan(BaseModel):
    days: list[DayTransport]

    @model_validator(mode="before")
    @classmethod
    def _normalize_days_payload(cls, value):
        if not isinstance(value, dict):
            return value
        if "days" in value and isinstance(value["days"], list):
            return value
        # LLM sometimes returns a single {day, description} dict without the
        # outer {"days": [...]} wrapper — recover by wrapping it.
        if "day" in value and "description" in value:
            return {"days": [value]}
        # Or returns {"plan": [...]} under an alternate key.
        alt = value.get("plan")
        if isinstance(alt, list):
            return {**value, "days": alt}
        return value


class DayLodging(BaseModel):
    day: int
    city: str
    description: str = "-"


class LodgingPlan(BaseModel):
    days: list[DayLodging]

    @model_validator(mode="before")
    @classmethod
    def _normalize_day_values(cls, value):
        if not isinstance(value, dict):
            return value
        rows = value.get("days")
        if not isinstance(rows, list):
            return value

        normalized: list[object] = []
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                normalized.append(row)
                continue

            raw_day = row.get("day")
            day_num: int | None = None
            if isinstance(raw_day, int):
                day_num = raw_day
            elif isinstance(raw_day, str):
                m = re.search(r"\d+", raw_day)
                if m:
                    day_num = int(m.group(0))

            if day_num is None or day_num <= 0:
                day_num = idx

            normalized.append({**row, "day": day_num})

        return {**value, "days": normalized}


class DayMeals(BaseModel):
    day: int
    city: str
    breakfast: str = "-"
    lunch: str = "-"
    dinner: str = "-"


class DiningPlan(BaseModel):
    days: list[DayMeals]

    @model_validator(mode="before")
    @classmethod
    def _normalize_days_payload(cls, value):
        if not isinstance(value, dict):
            return value

        rows = value.get("days")
        if not isinstance(rows, list):
            alt = value.get("plan")
            rows = alt if isinstance(alt, list) else None
        if rows is None:
            return value

        normalized: list[object] = []
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                normalized.append(row)
                continue

            raw_day = row.get("day")
            day_num: int | None = None
            if isinstance(raw_day, int):
                day_num = raw_day
            elif isinstance(raw_day, str):
                m = re.search(r"\d+", raw_day)
                if m:
                    day_num = int(m.group(0))

            if day_num is None or day_num <= 0:
                day_num = idx

            normalized.append({**row, "day": day_num})

        return {**value, "days": normalized}


class DayAttractions(BaseModel):
    day: int
    city: str
    description: str = "-"


class SightseeingPlan(BaseModel):
    days: list[DayAttractions]


class PlanDay(BaseModel):
    """One day of the assembled plan; keys match example_submission.jsonl."""
    days: int
    current_city: str
    transportation: str = "-"
    breakfast: str = "-"
    attraction: str = "-"
    lunch: str = "-"
    dinner: str = "-"
    accommodation: str = "-"


class FullPlan(BaseModel):
    query: str
    plan: list[PlanDay]


class Violation(BaseModel):
    rule: str
    detail: str
    responsible: Literal[
        "transport", "lodging", "dining", "sightseeing", "budget", "coordinator"
    ]


class VerifierReport(BaseModel):
    passed: bool
    violations: list[Violation] = Field(default_factory=list)


class BudgetReport(BaseModel):
    """Output of the Budget Agent (paper §4.3).

    Produced post-assembly; gives the coordinator a detailed per-category
    cost breakdown so repair requests target the correct Worker agent.
    """
    total_cost: int
    budget: Optional[int]
    category_costs: dict[str, int]   # keys: transport / dining / lodging
    passed: bool
    highest_category: str            # category with the highest spend


class AgentDecision(BaseModel):
    """Single attribution record produced by a Worker agent (paper §4.5)."""
    agent: str    # transport | lodging | dining | sightseeing | budget
    day: int      # 0 = trip-level summary (used by budget agent)
    field: str    # transportation | accommodation | breakfast | lunch | dinner | attraction | total_cost
    value: str    # the chosen text value


class AttributionLog(BaseModel):
    """Full attribution log for one planning query (paper §4.5).

    Each Worker agent records every decision it made so the coordinator can
    perform post-hoc analysis of which agents introduce the most constraint
    violations and route repair requests accurately.
    """
    decisions: list[AgentDecision] = Field(default_factory=list)


class TripWindows(TypedDict, total=False):
    """Derived from the transport specialist's output.

    Populated before the lodging/dining/sightseeing fan-out so those specialists
    know the real arrival and departure times and can plan meals, attractions,
    and accommodation around them without relying on post-hoc patching.
    """
    arrival_minutes_day1: int | None
    departure_minutes_last: int | None
    arrival_hhmm_day1: str | None
    departure_hhmm_last: str | None


class ToolContext(TypedDict, total=False):
    """Pre-fetched live API data shared across all parallel specialists via CoordinatorState.

    Populated once by the research node before specialists fan out, eliminating
    redundant API calls during parallel execution.
    """
    flights_outbound: list[str]        # formatted strings, org → dest
    flights_return: list[str]          # formatted strings, dest → org
    flights_round_trip: bool           # True when outbound cost is the round-trip total
    route_drive: str | None            # driving route summary
    route_taxi: str | None             # taxi route summary
    hotels: list[dict]                 # raw Places API rows
    restaurants: list[dict]            # raw Places API rows
    attractions: list[dict]            # raw Places API rows
    trip_windows: TripWindows          # populated after transport specialist runs
    category_caps: dict[str, int]      # per-category budget caps (paper §4.3)
    meal_cost_targets: dict[int, int]  # price_level → estimated meal cost
    lodging_cost_targets: dict[int, int]  # price_level → estimated nightly cost
    transport_one_way_target: int      # one-way flight budget target
