"""Single source of truth for planning rules.

Imported by specialist system prompts AND the verifier so that any rule
change propagates consistently to both layers without drift.
"""
from __future__ import annotations

import re
from typing import Any


# ── Verifier rule identifiers ──────────────────────────────────────────────────
# These strings appear in Violation.rule; keep them aligned with eval/constraints.py.
RULE_COMPLETE_INFORMATION = "complete_information"
RULE_DIVERSE_ATTRACTIONS  = "diverse_attractions"
RULE_DIVERSE_RESTAURANTS  = "diverse_restaurants"
RULE_WITHIN_SANDBOX       = "within_sandbox"
RULE_MIN_NIGHTS           = "min_nights"
RULE_NON_CONFLICTING_TRANSPORT = "non_conflicting_transport"
RULE_MISSING_COST         = "missing_cost"
RULE_BUDGET               = "budget"
RULE_CUISINE              = "cuisine"
RULE_ROOM_TYPE            = "room_type"
RULE_ROOM_RULE            = "room_rule"
RULE_TRANSPORTATION       = "transportation"

# ── Timing verifier rules (Phase 9.3) ─────────────────────────────────────────
RULE_ARRIVAL_LUNCH        = "arrival_lunch_after_14"
RULE_ARRIVAL_DINNER       = "arrival_dinner_after_19"
RULE_ARRIVAL_ATTRACTION   = "arrival_attraction_after_16"
RULE_DEPARTURE_BREAKFAST  = "departure_breakfast_before_08"
RULE_DEPARTURE_LUNCH      = "departure_lunch_before_1330"
RULE_DEPARTURE_ATTRACTION = "departure_attraction_before_16"


# ── Day classification helpers ─────────────────────────────────────────────────
def is_arrival_day(day_no: int) -> bool:
    return day_no == 1


def is_departure_day(day_no: int, total_days: int) -> bool:
    return day_no == total_days


# ── Prompt snippets imported into specialist system prompts ────────────────────
ARRIVAL_DAY_RULE = (
    "Day 1 is the ARRIVAL day: set breakfast to '-' because the traveler "
    "is still in transit that morning. Lunch, dinner, and attractions are "
    "planned normally for the destination."
)

DEPARTURE_DAY_RULE = (
    "The LAST day is the DEPARTURE day: set dinner to '-' (traveler departs "
    "in the evening) and accommodation to '-' (no overnight stay needed). "
    "Breakfast and a light lunch are planned normally."
)

NO_REPEAT_RESTAURANTS = (
    "Each restaurant name must appear AT MOST ONCE across the entire trip — "
    "never repeat a restaurant in any meal slot on any day."
)

NO_REPEAT_ATTRACTIONS = (
    "Each attraction name must appear AT MOST ONCE across the entire trip — "
    "never repeat an attraction across days."
)

SINGLE_ACCOMMODATION = (
    "Use ONE accommodation property for the entire stay. Emit an entry for "
    "every day 1..N; repeat the SAME hotel name on every non-departure day "
    "(days 1..N-1); the departure-day entry must be '-'. The Cost fragment "
    "should appear on day 1 only (covers the whole stay) to avoid being "
    "double-counted."
)

BUDGET_GUIDANCE = (
    "The trip budget is split by FIXED ratios: transport 45%, lodging 35%, "
    "dining 20% (see agents/budget.py). Your category cap below is computed "
    "from this split — stay at or under it. Favour lower-cost options when "
    "the total approaches your cap. Cite 'Cost: $N' for every priced item."
)


def format_cost(n: int, *, nights: int | None = None) -> str:
    """Return a canonical cost fragment matched by the verifier/evaluator regex.

    Use this instead of f-string literals to prevent regex-breaking typos.
    """
    if nights is not None:
        return f"Cost: ${n} for {nights} nights"
    return f"Cost: ${n}"


# ── Cuisine synonyms (shared by verifier + dining-substitute fallback) ────────
CUISINE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "japanese":      ("japanese", "sushi", "ramen", "izakaya"),
    "italian":       ("italian", "pasta", "pizzeria", "trattoria"),
    "chinese":       ("chinese", "dim sum", "szechuan", "cantonese"),
    "mexican":       ("mexican", "taqueria", "taco", "cantina"),
    "indian":        ("indian", "curry", "tandoori", "biryani"),
    "french":        ("french", "bistro", "brasserie"),
    "thai":          ("thai",),
    "korean":        ("korean", "bbq", "bulgogi"),
    "american":      ("american", "diner", "grill", "steakhouse"),
    "mediterranean": ("mediterranean", "greek", "kebab"),
    "vietnamese":    ("vietnamese", "pho", "banh mi"),
    "spanish":       ("spanish", "tapas", "paella"),
}


def cuisine_present(cuisine_token: str, blob: str) -> bool:
    """Return True if `cuisine_token` (or a synonym) appears as a word in blob."""
    token = cuisine_token.strip().lower()
    if not token:
        return True
    candidates = CUISINE_SYNONYMS.get(token, (token,))
    for c in candidates:
        if re.search(rf"\b{re.escape(c)}\b", blob, flags=re.IGNORECASE):
            return True
    return False


# ── Timing thresholds (shared by format_trip_windows grid + safety clamp) ────
# Day 1 thresholds (arrival_minutes)
THRESH_D1_LUNCH_NO    = 14 * 60   # lunch NO if arrival >= 14:00
THRESH_D1_ATTRACTION_NO = 16 * 60 # attraction NO if arrival >= 16:00
THRESH_D1_DINNER_NO   = 19 * 60   # dinner NO if arrival >= 19:00
# Last-day thresholds (departure_minutes)
THRESH_LAST_BREAKFAST_NO = 8 * 60       # breakfast NO if dep < 08:00
THRESH_LAST_LUNCH_YES    = 13 * 60 + 30 # lunch YES iff dep >= 13:30
THRESH_LAST_DINNER_YES   = 20 * 60      # dinner YES iff dep >= 20:00
THRESH_LAST_ATTRACTION_YES = 16 * 60    # attraction YES iff dep >= 16:00


# ── Shared prompt block: flight-timing summary for downstream specialists ─────
def format_trip_windows(ctx: Any) -> str:
    """Render arrival/departure timing and a YES/NO availability grid.

    Emits an explicit slot grid so dining and sightseeing specialists plan
    correctly upfront, rather than relying on post-hoc patching. Only emitted
    when trip_windows is populated (i.e. transport specialist has already run).
    """
    tw = (ctx or {}).get("trip_windows") or {}
    arr_hhmm = tw.get("arrival_hhmm_day1")
    dep_hhmm = tw.get("departure_hhmm_last")
    arr_min  = tw.get("arrival_minutes_day1")
    dep_min  = tw.get("departure_minutes_last")

    if not arr_hhmm and not dep_hhmm:
        return ""

    parts: list[str] = ["Flight timing:"]
    if arr_hhmm:
        parts.append(f"  - Day 1 arrival at destination: {arr_hhmm}")
    if dep_hhmm:
        parts.append(f"  - Last day departure from destination: {dep_hhmm}")

    # Day 1 (arrival) slot grid
    if arr_min is not None:
        parts.append("")
        parts.append("Day 1 (arrival):")
        parts.append("  - breakfast: NO  (still in transit)")
        if arr_min >= THRESH_D1_LUNCH_NO:
            parts.append(f"  - lunch:      NO  (still in transit, arrival {arr_hhmm})")
        else:
            parts.append(f"  - lunch:      YES (arrival before 14:00)")
        if arr_min >= THRESH_D1_DINNER_NO:
            parts.append(f"  - dinner:     NO  (arrival {arr_hhmm}, too late for restaurant)")
        else:
            parts.append(f"  - dinner:     YES (arrival before 19:00)")
        if arr_min >= THRESH_D1_ATTRACTION_NO:
            parts.append(f"  - attraction: NO  (too late after arrival {arr_hhmm})")
        else:
            parts.append(f"  - attraction: YES (arrives in time for sightseeing)")

    # Last day (departure) slot grid
    if dep_min is not None:
        parts.append("")
        parts.append("Last day (departure):")
        if dep_min < THRESH_LAST_BREAKFAST_NO:
            parts.append(f"  - breakfast: NO  (very early departure {dep_hhmm})")
        else:
            parts.append(f"  - breakfast: YES (window before {dep_hhmm})")
        if dep_min >= THRESH_LAST_LUNCH_YES:
            parts.append(f"  - lunch:      YES (departure {dep_hhmm} — enough time)")
        else:
            parts.append(f"  - lunch:      NO  (departs {dep_hhmm} — no time for lunch)")
        if dep_min >= THRESH_LAST_DINNER_YES:
            parts.append(f"  - dinner:     YES (late departure {dep_hhmm})")
        else:
            parts.append(f"  - dinner:     NO  (already departed or departing {dep_hhmm})")
        if dep_min >= THRESH_LAST_ATTRACTION_YES:
            parts.append(f"  - attraction: YES (time for sightseeing before {dep_hhmm} departure)")
        else:
            parts.append(f"  - attraction: NO  (no time before {dep_hhmm} departure)")

    return "\n".join(parts)
