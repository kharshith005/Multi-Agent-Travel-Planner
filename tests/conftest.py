"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from agents.schemas import (
    DayAttractions,
    DayLodging,
    DayMeals,
    DayTransport,
    DiningPlan,
    FullPlan,
    Intent,
    LodgingPlan,
    PlanDay,
    SightseeingPlan,
    TransportPlan,
)


@pytest.fixture()
def basic_intent() -> Intent:
    return Intent(
        org="Washington DC",
        dest="Myrtle Beach",
        days=3,
        dates=["2026-06-01", "2026-06-02", "2026-06-03"],
        people=2,
        budget=1400,
    )


@pytest.fixture()
def three_day_plan() -> FullPlan:
    """Minimal valid 3-day plan with costs in every field."""
    return FullPlan(
        query="test",
        plan=[
            PlanDay(
                days=1,
                current_city="from Washington DC to Myrtle Beach",
                transportation="United Airlines 08:00->10:00 JFK-MYR; Cost: $200",
                breakfast="-",
                attraction="Broadway at the Beach; Cost: $0",
                lunch="Sea Captain's House, Myrtle Beach; Cost: $25",
                dinner="Croissants Bistro, Myrtle Beach; Cost: $30",
                accommodation="Marriott Resort, Myrtle Beach; Cost: $180 for 2 nights",
            ),
            PlanDay(
                days=2,
                current_city="Myrtle Beach",
                transportation="-",
                breakfast="Pancake House, Myrtle Beach; Cost: $15",
                attraction="Ripley's Aquarium, Myrtle Beach; Cost: $25",
                lunch="Bubba's Fish Shack, Myrtle Beach; Cost: $20",
                dinner="Flying Fish Public Market, Myrtle Beach; Cost: $35",
                accommodation="Marriott Resort, Myrtle Beach",
            ),
            PlanDay(
                days=3,
                current_city="from Myrtle Beach to Washington DC",
                transportation="United Airlines 14:00->16:00 MYR-JFK; Cost: $200",
                breakfast="Einstein Bros Bagels, Myrtle Beach; Cost: $10",
                attraction="-",
                lunch="-",
                dinner="-",
                accommodation="-",
            ),
        ],
    )


@pytest.fixture()
def transport_plan() -> TransportPlan:
    return TransportPlan(days=[
        DayTransport(day=1, description="United Airlines 08:00->10:00; Cost: $200"),
        DayTransport(day=2, description="-"),
        DayTransport(day=3, description="United Airlines 14:00->16:00; Cost: $200"),
    ])


@pytest.fixture()
def lodging_plan() -> LodgingPlan:
    return LodgingPlan(days=[
        DayLodging(day=1, city="Myrtle Beach",
                   description="Marriott Resort, Myrtle Beach; Cost: $180 for 2 nights"),
        DayLodging(day=2, city="Myrtle Beach", description="-"),
        DayLodging(day=3, city="Myrtle Beach", description="-"),
    ])


@pytest.fixture()
def dining_plan() -> DiningPlan:
    return DiningPlan(days=[
        DayMeals(day=1, city="Myrtle Beach",
                 breakfast="-", lunch="Sea Captain's House; Cost: $25", dinner="Croissants Bistro; Cost: $30"),
        DayMeals(day=2, city="Myrtle Beach",
                 breakfast="Pancake House; Cost: $15", lunch="Bubba's Fish Shack; Cost: $20",
                 dinner="Flying Fish; Cost: $35"),
        DayMeals(day=3, city="Myrtle Beach",
                 breakfast="Einstein Bros; Cost: $10", lunch="-", dinner="-"),
    ])


@pytest.fixture()
def sightseeing_plan() -> SightseeingPlan:
    return SightseeingPlan(days=[
        DayAttractions(day=1, city="Myrtle Beach", description="Broadway at the Beach"),
        DayAttractions(day=2, city="Myrtle Beach", description="Ripley's Aquarium"),
        DayAttractions(day=3, city="Myrtle Beach", description="-"),
    ])
