"""Dining specialist: proposes breakfast/lunch/dinner per day."""
from __future__ import annotations

from .llm import call_json
from .rules import (
    ARRIVAL_DAY_RULE,
    BUDGET_GUIDANCE,
    CUISINE_SYNONYMS,
    DEPARTURE_DAY_RULE,
    NO_REPEAT_RESTAURANTS,
    format_trip_windows,
)
from .schemas import DiningPlan, Intent, ToolContext


SYSTEM = (
    "You are the Dining specialist in a multi-agent travel planner. "
    "Propose breakfast, lunch, and dinner for each day in the destination city.\n\n"
    "CRITICAL GROUNDING RULE: You MUST pick every restaurant name verbatim and "
    "EXACTLY as it appears in the numbered [N] list below. Do NOT invent new "
    "names, do NOT paraphrase, do NOT add 'Restaurant' or 'Cafe' suffixes. "
    "If a name is 'Joe's Pizza' in the list, write exactly 'Joe's Pizza' — "
    "never 'Joes Pizza', 'Joe Pizza', or 'Joe's Italian'.\n\n"
    "Rules:\n"
    f"- {ARRIVAL_DAY_RULE}\n"
    f"- {DEPARTURE_DAY_RULE}\n"
    f"- {NO_REPEAT_RESTAURANTS}\n"
    "- **CUISINE CONSTRAINT**: If `intent.cuisine` is set (may be comma-separated "
    "for multiple cuisines), at least one meal per requested cuisine must come from "
    "a restaurant whose `cuisines` tag contains that cuisine. Match against the "
    "explicit `cuisines` field in the tool results — do NOT guess from the "
    "restaurant name alone.\n"
    f"- {BUDGET_GUIDANCE}\n"
    "- Every non-'-' meal MUST carry a 'Cost: $N' fragment.\n"
    "- Follow the Flight timing YES/NO grid exactly when present — set any meal "
    "slot marked NO to '-'; do not invent meals during transit.\n\n"
    "Format each meal as 'Name, City; Cost: $N'.\n\n"
    "Return ONLY a real JSON object of the exact shape shown in the example "
    "below — fill in actual meal strings; never echo the placeholder text.\n\n"
    "Few-shot example (2-day trip to Austin, no cuisine constraint):\n"
    '{"days": ['
    '{"day": 1, "city": "Austin", "breakfast": "-", '
    '"lunch": "La Condesa, Austin; Cost: $28", "dinner": "Uchiko, Austin; Cost: $55"}, '
    '{"day": 2, "city": "Austin", "breakfast": "Tacodeli, Austin; Cost: $12", '
    '"lunch": "Franklin Barbecue, Austin; Cost: $20", "dinner": "-"}'
    "]}"
)


def run(intent: Intent, tool_context: ToolContext, repair_note: str = "") -> DiningPlan:
    tool_results = _format_tool_context(tool_context)
    if tool_results == "(none)":
        raise RuntimeError("No live restaurant results available in tool context")

    allowed = _allowed_names(tool_context)
    plan = _call(intent, tool_context, tool_results, repair_note)
    invalid = _invalid_names(plan, allowed)
    if invalid and allowed:
        plan = _substitute_restaurants(plan, allowed, intent, tool_context)
    return plan


def _substitute_restaurants(
    plan: DiningPlan, allowed: set[str], intent: Intent, ctx: ToolContext
) -> DiningPlan:
    """Deterministically replace invalid meal names with allowed ones.

    When `intent.cuisine` is set, the replacement pool is pre-sorted so that
    names matching the cuisine (via `CUISINE_SYNONYMS`) are picked first —
    preserves the `cuisine` hard constraint while also fixing `within_sandbox`.
    Also never reuses a name (preserves `diverse_restaurants`).
    """
    cuisine_tokens = _cuisine_tokens(intent.cuisine or "")
    pool = _prioritised_pool(allowed, cuisine_tokens, ctx)
    used: set[str] = set()
    for d in plan.days:
        for meal in (d.breakfast, d.lunch, d.dinner):
            if meal and meal != "-":
                for name in _extract_meal_names(meal):
                    if name in allowed:
                        used.add(name)

    def next_name() -> str | None:
        for n in pool:
            if n not in used:
                used.add(n)
                return n
        return None

    city = intent.dest
    for d in plan.days:
        for slot in ("breakfast", "lunch", "dinner"):
            meal = getattr(d, slot)
            if not meal or meal == "-":
                continue
            names = _extract_meal_names(meal)
            if not names:
                continue
            primary = names[0]
            if primary.lower().startswith(("in transit", "dinner before", "late arrival")):
                continue
            cost_idx = meal.lower().find("cost")
            cost_frag = meal[cost_idx:] if cost_idx >= 0 else ""
            if all(n in allowed for n in names):
                continue
            if primary in allowed:
                chosen = primary
                used.add(primary)
            else:
                chosen = next_name()
                if chosen is None:
                    continue
            new_val = f"{chosen}, {city}" + (f"; {cost_frag}" if cost_frag else "")
            setattr(d, slot, new_val)
    return plan


def _cuisine_tokens(cuisine: str) -> list[str]:
    """Expand 'italian,japanese' into a flat list of match keywords."""
    out: list[str] = []
    for c in cuisine.split(","):
        c = c.strip().lower()
        if not c:
            continue
        out.extend(CUISINE_SYNONYMS.get(c, (c,)))
    return out


def _prioritised_pool(
    allowed: set[str], cuisine_tokens: list[str], ctx: ToolContext | None = None
) -> list[str]:
    """Return the allowed pool sorted so cuisine-matching names come first.

    When `ctx` is provided, cuisine matching uses the explicit `cuisines` field
    from the restaurant record (more accurate). Falls back to name-keyword match.
    """
    if not cuisine_tokens:
        return sorted(allowed)

    # Build a name→cuisines_text lookup from ctx when available.
    cuisines_by_name: dict[str, str] = {}
    if ctx:
        for r in (ctx.get("restaurants") or []):
            name = (r.get("name") or "").strip()
            cuisines_field = (r.get("cuisines") or "").lower()
            if name:
                cuisines_by_name[name] = cuisines_field

    matching: list[str] = []
    other: list[str] = []
    for name in sorted(allowed):
        cuisines_text = cuisines_by_name.get(name, name.lower())
        if any(tok in cuisines_text for tok in cuisine_tokens):
            matching.append(name)
        else:
            other.append(name)
    return matching + other


def _multicity_hint(ctx: ToolContext, intent: Intent) -> str:
    cities = ctx.get("dest_cities") or []
    if len(cities) <= 1:
        return ""
    city_list = ", ".join(cities)
    days_each = max(1, intent.days // len(cities))
    return (
        f"MULTI-CITY TRIP: Visit cities in this order: {city_list}. "
        f"Spend roughly {days_each} day(s) per city. "
        f"Set `city` in each day's output to the city where that day is spent. "
        f"Each restaurant must be in the same city as the day it appears on."
    )


def _call(intent: Intent, ctx: ToolContext, tool_results: str, repair_note: str) -> DiningPlan:
    windows = format_trip_windows(ctx)
    cap = _format_dining_cap(ctx, intent)
    mc_hint = _multicity_hint(ctx, intent)
    dest_label = intent.dest if not ctx.get("dest_cities") else (
        f"{intent.dest} (cities: {', '.join(ctx['dest_cities'])})"
    )
    user = (
        f"Intent: {intent.for_specialist('dining')}\n\n"
        f"Restaurants in {dest_label}:\n{tool_results}\n\n"
        + (mc_hint + "\n\n" if mc_hint else "")
        + (windows + "\n\n" if windows else "")
        + (cap + "\n\n" if cap else "")
        + (f"Repair note: {repair_note}\n\n" if repair_note else "")
        + f"Produce a DiningPlan covering days 1..{intent.days}."
    )
    think = bool(repair_note)
    return call_json(SYSTEM, user, DiningPlan, max_tokens=4096, think_first=think)


def _format_tool_context(ctx: ToolContext) -> str:
    rows = ctx.get("restaurants") or []
    if not rows:
        return "(none)"
    cost_map: dict[int, int] = ctx.get("meal_cost_targets") or {0: 8, 1: 15, 2: 28, 3: 45, 4: 70}
    lines = []
    for i, r in enumerate(rows, start=1):
        try:
            level = int(r.get("price_level", 2))
        except (TypeError, ValueError):
            level = 2
        cost = cost_map.get(level, cost_map.get(2, 28))
        parts = [
            f"  [{i}] {r.get('name')}",
            f"rating:{r.get('rating')}",
            f"est_cost:${cost}",
        ]
        if r.get("cuisines"):
            parts.append(f"cuisines:{r['cuisines']}")
        if r.get("city"):
            parts.append(f"city:{r['city']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _format_dining_cap(ctx: ToolContext, intent: Intent) -> str:
    caps = ctx.get("category_caps") or {}
    dining_cap = caps.get("dining")
    if not dining_cap:
        return ""
    return (
        f"Per-category budget cap: dining spending across the whole trip "
        f"should stay under ${dining_cap}."
    )


def _allowed_names(ctx: ToolContext) -> set[str]:
    return {
        (r.get("name") or "").strip()
        for r in (ctx.get("restaurants") or [])
        if r.get("name")
    }


def _invalid_names(plan: DiningPlan, allowed: set[str]) -> set[str]:
    """Flag ANY restaurant name in a meal slot that isn't in the allowed pool.

    A meal slot can contain multiple venues separated by ';' (e.g. the LLM
    sometimes emits 'A, City; B, City; Cost: $15'). Split on ';' FIRST so
    every name is checked, not just the first — this mirrors eval's `_names`
    helper and prevents silent grounding failures on secondary venues.
    """
    bad: set[str] = set()
    if not allowed:
        return bad
    for d in plan.days:
        for meal in (d.breakfast, d.lunch, d.dinner):
            if not meal or meal == "-":
                continue
            for name in _extract_meal_names(meal):
                if name.lower().startswith(("in transit", "dinner before", "late arrival")):
                    continue
                if name not in allowed:
                    bad.add(name)
    return bad


def _extract_meal_names(meal: str) -> list[str]:
    names: list[str] = []
    for seg in meal.split(";"):
        chunk = seg.strip()
        if not chunk or chunk.lower().startswith("cost"):
            continue
        name = chunk.split(",")[0].strip()
        if name:
            names.append(name)
    return names


def _price_level_to_meal_cost(price_level: object) -> int:
    try:
        level = int(price_level)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        level = 2
    return {0: 8, 1: 15, 2: 28, 3: 45, 4: 70}.get(level, 28)
