"""Budget Agent: tracks cumulative spending across all planning domains.

Worker agent #5 (paper §4.3). Runs post-assembly to evaluate total trip cost
against the intent budget and produce a per-category breakdown.

Rule-based evaluation (no LLM call) is consistent with Chen et al. [7]:
rule-based checking outperforms LLM-based self-correction for constraint
satisfaction tasks.

The public `extract_cost` helper is imported by verifier.py and
eval/constraints.py to eliminate duplicated cost-parsing logic.
"""
from __future__ import annotations

import re

from .schemas import BudgetReport, FullPlan, Intent

_COST_RE = re.compile(r"Cost[:\s]*\$?(\d+)", re.IGNORECASE)

# Fixed cost tables used when intent.budget is None (fallback only).
_DEFAULT_MEAL_TABLE: dict[int, int] = {0: 8, 1: 15, 2: 28, 3: 45, 4: 70}
_DEFAULT_LODGING_TABLE: dict[int, int] = {0: 60, 1: 90, 2: 130, 3: 190, 4: 280}


def extract_cost(text: str) -> int:
    """Sum all 'Cost: $N' values found in a plan field string."""
    if not text or text == "-":
        return 0
    return sum(int(m.group(1)) for m in _COST_RE.finditer(text))


def allocate(intent: Intent) -> dict[str, int]:
    """Allocate a traveler's budget across transport / lodging / dining categories.

    Gives each specialist an explicit per-category cap so the first-pass plan is
    within-budget by construction, not only after rule-based repair. Attractions
    are rarely priced in the plan text, so no explicit cap is emitted for them.

    Scaling: ratios are fixed (transport 45%, lodging 35%, dining 20%). Lodging
    scales naturally with trip length because the specialist computes nightly
    rate × nights; transport is trip-level; dining scales with people * days.
    """
    if intent.budget is None or intent.budget <= 0:
        return {}
    b = int(intent.budget)
    return {
        "transport": max(50, int(b * 0.45)),
        "lodging":   max(50, int(b * 0.35)),
        "dining":    max(30, int(b * 0.20)),
    }


def derive_meal_targets(intent: Intent) -> dict[int, int]:
    """5-point cost curve indexed by Maps price_level (0..4).

    Scaled to intent.budget so specialists pick appropriately priced meals.
    Falls back to the fixed table when no dining budget cap is available.
    """
    caps = allocate(intent)
    if not caps.get("dining"):
        return _DEFAULT_MEAL_TABLE
    num_meals = max(1, intent.days * 3 - 2)  # no breakfast d1, no dinner dN
    avg = caps["dining"] // num_meals
    return {
        0: max(5,  int(avg * 0.5)),
        1: max(8,  int(avg * 0.7)),
        2: max(12, int(avg * 1.0)),
        3: max(20, int(avg * 1.5)),
        4: max(30, int(avg * 2.0)),
    }


def derive_lodging_targets(intent: Intent) -> dict[int, int]:
    """Per-night cost curve indexed by Maps price_level.

    Falls back to the fixed table when no lodging budget cap is available.
    """
    caps = allocate(intent)
    if not caps.get("lodging"):
        return _DEFAULT_LODGING_TABLE
    nights = max(1, intent.days - 1)
    avg = caps["lodging"] // nights
    return {
        0: max(40,  int(avg * 0.6)),
        1: max(60,  int(avg * 0.8)),
        2: max(80,  int(avg * 1.0)),
        3: max(100, int(avg * 1.3)),
        4: max(140, int(avg * 1.6)),
    }


def price_levels_for_nightly_target(intent: Intent) -> set[int]:
    """Return the set of Google Maps price_level values (0–4) within the per-night lodging cap.

    Used by live_apis.search_lodging_near to filter the candidate pool to
    hotels that are actually affordable given the trip budget. Levels above
    the target are excluded; the target itself and one level above are kept
    to handle light over-estimates (price_level is coarse).
    """
    targets = derive_lodging_targets(intent)
    if not targets:
        return {0, 1, 2, 3, 4}  # no budget constraint — accept all tiers

    avg = targets.get(2, 0)  # level-2 is the baseline "average" tier

    # Include all levels whose estimated cost is ≤ 1.5× the average per-night cap.
    ceiling = avg * 1.5
    accepted: set[int] = set()
    for level, cost in targets.items():
        if cost <= ceiling:
            accepted.add(level)

    return accepted or {0, 1, 2, 3, 4}


def derive_transport_target(intent: Intent) -> int:
    """One-way flight budget target: half of the total transport cap."""
    caps = allocate(intent)
    return caps.get("transport", 200) // 2


def evaluate(plan: FullPlan, intent: Intent) -> BudgetReport:
    """Evaluate cumulative spending across transport, dining, and lodging.

    Returns a BudgetReport with per-category cost breakdown and the
    highest-spend category — used by the coordinator to route targeted
    repair requests to the correct Worker agent when the budget is exceeded.
    """
    categories: dict[str, int] = {"transport": 0, "dining": 0, "lodging": 0}

    for d in plan.plan:
        categories["transport"] += extract_cost(d.transportation)
        categories["dining"] += (
            extract_cost(d.breakfast)
            + extract_cost(d.lunch)
            + extract_cost(d.dinner)
        )
        categories["lodging"] += extract_cost(d.accommodation)

    total = sum(categories.values())
    passed = intent.budget is None or total <= intent.budget
    highest = max(categories, key=lambda k: categories[k])
    return BudgetReport(
        total_cost=total,
        budget=intent.budget,
        category_costs=categories,
        passed=passed,
        highest_category=highest,
    )
