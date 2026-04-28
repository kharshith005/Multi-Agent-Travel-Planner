"""Rule-based verifier. Ports a pragmatic subset of the TravelPlanner evaluator.

Checks commonsense rules (diversity, completeness, within-sandbox) and the
hard constraints extracted into the Intent (budget, cuisine, room-type, etc.).
Each violation is tagged with the responsible specialist so the coordinator
can route a repair note.
"""
from __future__ import annotations

import re

from .budget import extract_cost as _extract_cost
from .rules import (
    RULE_ARRIVAL_ATTRACTION,
    RULE_ARRIVAL_DINNER,
    RULE_ARRIVAL_LUNCH,
    RULE_BUDGET,
    RULE_COMPLETE_INFORMATION,
    RULE_CUISINE,
    RULE_DEPARTURE_ATTRACTION,
    RULE_DEPARTURE_BREAKFAST,
    RULE_DEPARTURE_LUNCH,
    RULE_DIVERSE_ATTRACTIONS,
    RULE_DIVERSE_RESTAURANTS,
    RULE_MIN_NIGHTS,
    RULE_MISSING_COST,
    RULE_NON_CONFLICTING_TRANSPORT,
    RULE_ROOM_RULE,
    RULE_ROOM_TYPE,
    RULE_TRANSPORTATION,
    RULE_WITHIN_SANDBOX,
    THRESH_D1_ATTRACTION_NO,
    THRESH_D1_DINNER_NO,
    THRESH_D1_LUNCH_NO,
    THRESH_LAST_ATTRACTION_YES,
    THRESH_LAST_BREAKFAST_NO,
    THRESH_LAST_LUNCH_YES,
    cuisine_present as _cuisine_present,
)
from .schemas import FullPlan, Intent, TripWindows, VerifierReport, Violation
from tools.sandbox import Sandbox


_COST_FRAGMENT_RE = re.compile(r"Cost[:\s]*\$?\d+(?:\s+for\s+\d+\s+nights?)?", re.IGNORECASE)
_HAS_COST_RE = re.compile(r"Cost[:\s]*\$?\d+", re.IGNORECASE)


def _is_timing_sentinel(text: str) -> bool:
    """Return True for post-assembly timing placeholders that don't represent real meals."""
    return text.strip().lower().startswith(("in transit", "late arrival", "dinner before"))


def _names_from_csv_list(text: str) -> list[str]:
    """Split a semicolon-separated listing like 'A, City;B, City;' into names.
    Strips 'Cost: N' / 'Cost: N for D nights' fragments so they aren't mistaken for names.
    """
    if not text or text == "-":
        return []
    cleaned = _COST_FRAGMENT_RE.sub("", text)
    items = [s.strip() for s in cleaned.split(";") if s.strip()]
    names = []
    for item in items:
        name = item.split(",")[0].strip()
        if name:
            names.append(name)
    return names


def verify(
    plan: FullPlan,
    intent: Intent,
    sandbox: Sandbox | None = None,
    trip_windows: TripWindows | None = None,
    flights_round_trip: bool = False,
) -> VerifierReport:
    violations: list[Violation] = []

    # ---- completeness ----
    expected_days = intent.days
    if len(plan.plan) != expected_days:
        violations.append(Violation(
            rule=RULE_COMPLETE_INFORMATION,
            detail=f"Expected {expected_days} days, got {len(plan.plan)}",
            responsible="coordinator",
        ))

    # ---- diversity ----
    all_attractions: list[str] = []
    all_restaurants: list[str] = []
    for d in plan.plan:
        all_attractions.extend(_names_from_csv_list(d.attraction))
        for meal in (d.breakfast, d.lunch, d.dinner):
            names = _names_from_csv_list(meal)
            all_restaurants.extend(names)
    if len(all_attractions) != len(set(all_attractions)) and all_attractions:
        violations.append(Violation(
            rule=RULE_DIVERSE_ATTRACTIONS,
            detail=f"Duplicate attractions: {[a for a in all_attractions if all_attractions.count(a) > 1]}",
            responsible="sightseeing",
        ))
    if len(all_restaurants) != len(set(all_restaurants)) and all_restaurants:
        violations.append(Violation(
            rule=RULE_DIVERSE_RESTAURANTS,
            detail=f"Duplicate restaurants: {[r for r in all_restaurants if all_restaurants.count(r) > 1]}",
            responsible="dining",
        ))

    # ---- within sandbox (attractions & restaurants) ----
    # Only enforce this rule in dataset/sandbox mode.
    if sandbox is not None:
        known_attr = {r.get("Name") for rows in sandbox.attractions.values() for r in rows}
        known_rest = {r.get("Name") for rows in sandbox.restaurants.values() for r in rows}
        known_acc = {r.get("NAME") for rows in sandbox.accommodations.values() for r in rows}
        for name in all_attractions:
            if name not in known_attr:
                violations.append(Violation(
                    rule=RULE_WITHIN_SANDBOX,
                    detail=f"Attraction '{name}' not in sandbox",
                    responsible="sightseeing",
                ))
                break  # one report per category is enough
        for name in all_restaurants:
            if name not in known_rest:
                violations.append(Violation(
                    rule=RULE_WITHIN_SANDBOX,
                    detail=f"Restaurant '{name}' not in sandbox",
                    responsible="dining",
                ))
                break
        for d in plan.plan:
            name = _names_from_csv_list(d.accommodation)
            if name and name[0] not in known_acc:
                violations.append(Violation(
                    rule=RULE_WITHIN_SANDBOX,
                    detail=f"Accommodation '{name[0]}' not in sandbox",
                    responsible="lodging",
                ))
                break

    # ---- budget (hard constraint) ----
    if intent.budget:
        total_cost = 0
        for d in plan.plan:
            for field in (d.transportation, d.breakfast, d.lunch, d.dinner, d.accommodation):
                total_cost += _extract_cost(field)
        if total_cost > intent.budget:
            # attribute to the single highest-cost category
            cat_costs = {
                "transport": sum(_extract_cost(d.transportation) for d in plan.plan),
                "dining": sum(
                    _extract_cost(d.breakfast) + _extract_cost(d.lunch) + _extract_cost(d.dinner)
                    for d in plan.plan
                ),
                "lodging": sum(_extract_cost(d.accommodation) for d in plan.plan),
            }
            worst = max(cat_costs, key=cat_costs.get)
            violations.append(Violation(
                rule=RULE_BUDGET,
                detail=f"Total ${total_cost} exceeds budget ${intent.budget}; "
                       f"largest category: {worst} (${cat_costs[worst]})",
                responsible=worst,  # type: ignore[arg-type]
            ))

    # ---- cuisine (hard) ----
    if intent.cuisine:
        wanted = [c.strip().lower() for c in intent.cuisine.split(",") if c.strip()]
        all_meals_text = " ".join(
            (d.breakfast + " " + d.lunch + " " + d.dinner) for d in plan.plan
        )
        missing = [c for c in wanted if not _cuisine_present(c, all_meals_text)]
        if missing:
            violations.append(Violation(
                rule=RULE_CUISINE,
                detail=f"Missing required cuisine(s): {missing}",
                responsible="dining",
            ))

    # ---- transportation mode (hard) ----
    if intent.transportation:
        mode = intent.transportation.lower()
        transport_text = " ".join(d.transportation for d in plan.plan).lower()
        if "no " in mode or "avoid" in mode:
            forbidden = mode.replace("no ", "").replace("avoid", "").strip()
            if forbidden and forbidden in transport_text:
                violations.append(Violation(
                    rule=RULE_TRANSPORTATION,
                    detail=f"Forbidden mode '{forbidden}' present in transportation",
                    responsible="transport",
                ))
        else:
            if mode.split()[0] not in transport_text:
                violations.append(Violation(
                    rule=RULE_TRANSPORTATION,
                    detail=f"Required mode '{mode}' not found in transportation",
                    responsible="transport",
                ))

    # ---- min_nights (commonsense): every non-final day must have lodging ----
    for d in plan.plan:
        if d.days == intent.days:
            continue  # departure day may be "-"
        if not d.accommodation or d.accommodation.strip() == "-":
            violations.append(Violation(
                rule=RULE_MIN_NIGHTS,
                detail=f"Day {d.days} has no accommodation (non-final day)",
                responsible="lodging",
            ))
            break  # one report is enough to trigger repair

    # ---- non_conflicting_transport (commonsense): no mixed flight+drive on same day ----
    for d in plan.plan:
        t = d.transportation.lower()
        if "flight" in t and ("self-driving" in t or "self driving" in t):
            violations.append(Violation(
                rule=RULE_NON_CONFLICTING_TRANSPORT,
                detail=f"Day {d.days} mixes flight with self-driving",
                responsible="transport",
            ))
            break

    # ---- missing cost (commonsense): priced fields should carry "Cost: $N" ----
    # Transport: every non-"-" day must declare a cost (each leg is distinct).
    #   Exception: when flights_round_trip=True, the last day's return leg has
    #   no cost (round-trip cost is attributed to Day 1 only).
    # Meals: every non-"-" breakfast/lunch/dinner must carry a cost so budget
    #   math isn't silently under-counted.
    # Accommodation: the cost is declared ONCE on the first non-"-" day; later
    #   days forward-fill by name only. Flag only if NO accommodation carries cost.
    # Skip timing sentinels created post-assembly ("In transit ...", etc.).
    _missing_cost_days: list[int] = []
    for d in plan.plan:
        if d.transportation and d.transportation != "-" and not _HAS_COST_RE.search(d.transportation):
            # Round-trip return leg on the last day has no cost by design.
            if flights_round_trip and d.days == intent.days:
                pass
            else:
                _missing_cost_days.append(d.days)
        for meal in (d.breakfast, d.lunch, d.dinner):
            if (meal and meal != "-" and not _is_timing_sentinel(meal)
                    and not _HAS_COST_RE.search(meal)):
                _missing_cost_days.append(d.days)
                break
    acc_texts = [d.accommodation for d in plan.plan if d.accommodation and d.accommodation != "-"]
    if acc_texts and not any(_HAS_COST_RE.search(txt) for txt in acc_texts):
        _missing_cost_days.append(0)  # sentinel 0 → "accommodation has no cost anywhere"
    if _missing_cost_days:
        violations.append(Violation(
            rule=RULE_MISSING_COST,
            detail=f"Missing 'Cost: $N' on day(s) {sorted(set(_missing_cost_days))}",
            responsible="coordinator",
        ))

    # ---- room_type (hard, requires sandbox lookup) ----
    if intent.room_type and sandbox is not None:
        rt = intent.room_type.lower()
        acc_names = [_names_from_csv_list(d.accommodation)[0]
                     for d in plan.plan
                     if _names_from_csv_list(d.accommodation)]
        matched = False
        for name in acc_names:
            for rows in sandbox.accommodations.values():
                for r in rows:
                    if r.get("NAME") == name and rt in str(r.get("room type", "")).lower():
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if acc_names and not matched:
            violations.append(Violation(
                rule=RULE_ROOM_TYPE,
                detail=f"No booked accommodation matches room_type='{intent.room_type}'",
                responsible="lodging",
            ))

    # ---- room_rule / house_rule (hard, requires sandbox lookup) ----
    if intent.house_rule and sandbox is not None:
        rule_text = intent.house_rule.lower()
        acc_names = [_names_from_csv_list(d.accommodation)[0]
                     for d in plan.plan
                     if _names_from_csv_list(d.accommodation)]
        matched = False
        for name in acc_names:
            for rows in sandbox.accommodations.values():
                for r in rows:
                    if r.get("NAME") != name:
                        continue
                    rules_text = str(r.get("house_rules", "")).lower()
                    if rule_text in rules_text or rules_text in rule_text:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if acc_names and not matched:
            violations.append(Violation(
                rule=RULE_ROOM_RULE,
                detail=f"No booked accommodation matches house_rule='{intent.house_rule}'",
                responsible="lodging",
            ))

    # ---- timing rules (9.3): arrival/departure slot violations ----
    # Only checked when trip_windows is populated (transport specialist has run).
    # These rules fire before the safety clamp so the repair loop can fix them.
    if trip_windows:
        arr_min = trip_windows.get("arrival_minutes_day1")
        arr_hhmm = trip_windows.get("arrival_hhmm_day1") or ""
        dep_min = trip_windows.get("departure_minutes_last")
        dep_hhmm = trip_windows.get("departure_hhmm_last") or ""

        day1 = plan.plan[0] if plan.plan else None
        last_day = plan.plan[-1] if plan.plan else None

        if day1 is not None and arr_min is not None:
            if (arr_min >= THRESH_D1_LUNCH_NO
                    and day1.lunch and day1.lunch != "-"
                    and not _is_timing_sentinel(day1.lunch)):
                violations.append(Violation(
                    rule=RULE_ARRIVAL_LUNCH,
                    detail=(
                        f"Lunch planned but traveler arrives at {arr_hhmm} "
                        f"(≥14:00 — still in transit). Value: '{day1.lunch[:50]}'"
                    ),
                    responsible="dining",
                ))
            if (arr_min >= THRESH_D1_DINNER_NO
                    and day1.dinner and day1.dinner != "-"
                    and not _is_timing_sentinel(day1.dinner)):
                violations.append(Violation(
                    rule=RULE_ARRIVAL_DINNER,
                    detail=(
                        f"Dinner planned but traveler arrives at {arr_hhmm} "
                        f"(≥19:00 — too late for a restaurant). Value: '{day1.dinner[:50]}'"
                    ),
                    responsible="dining",
                ))
            if (arr_min >= THRESH_D1_ATTRACTION_NO
                    and day1.attraction and day1.attraction != "-"):
                violations.append(Violation(
                    rule=RULE_ARRIVAL_ATTRACTION,
                    detail=(
                        f"Attraction planned but traveler arrives at {arr_hhmm} "
                        f"(≥16:00 — no time for sightseeing). Value: '{day1.attraction[:50]}'"
                    ),
                    responsible="sightseeing",
                ))

        if last_day is not None and dep_min is not None:
            if (dep_min < THRESH_LAST_BREAKFAST_NO
                    and last_day.breakfast and last_day.breakfast != "-"):
                violations.append(Violation(
                    rule=RULE_DEPARTURE_BREAKFAST,
                    detail=(
                        f"Breakfast planned but departure is at {dep_hhmm} "
                        f"(<08:00 — very early). Value: '{last_day.breakfast[:50]}'"
                    ),
                    responsible="dining",
                ))
            if (dep_min < THRESH_LAST_LUNCH_YES
                    and last_day.lunch and last_day.lunch != "-"
                    and not _is_timing_sentinel(last_day.lunch)):
                violations.append(Violation(
                    rule=RULE_DEPARTURE_LUNCH,
                    detail=(
                        f"Lunch planned but departure is at {dep_hhmm} "
                        f"(<13:30 — no time). Value: '{last_day.lunch[:50]}'"
                    ),
                    responsible="dining",
                ))
            if (dep_min < THRESH_LAST_ATTRACTION_YES
                    and last_day.attraction and last_day.attraction != "-"):
                violations.append(Violation(
                    rule=RULE_DEPARTURE_ATTRACTION,
                    detail=(
                        f"Attraction planned but departure is at {dep_hhmm} "
                        f"(<16:00 — no time). Value: '{last_day.attraction[:50]}'"
                    ),
                    responsible="sightseeing",
                ))

    return VerifierReport(passed=not violations, violations=violations)
