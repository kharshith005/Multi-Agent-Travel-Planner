"""Lodging specialist: chooses one accommodation for the entire stay."""
from __future__ import annotations

from .llm import call_json
from .rules import (
    BUDGET_GUIDANCE,
    DEPARTURE_DAY_RULE,
    SINGLE_ACCOMMODATION,
    format_trip_windows,
)
from .schemas import Intent, LodgingPlan, ToolContext


SYSTEM = (
    "You are the Lodging specialist in a multi-agent travel planner. "
    "Pick ONE accommodation that satisfies the traveler's house_rule and "
    "room_type constraints when present. You MUST choose a name verbatim "
    "from the numbered list in the tool results — do NOT invent names.\n\n"
    "Rules:\n"
    f"- {SINGLE_ACCOMMODATION}\n"
    f"- {DEPARTURE_DAY_RULE}\n"
    f"- {BUDGET_GUIDANCE}\n"
    "- Every non-'-' entry MUST include a 'Cost: $N for D nights' fragment.\n"
    "- **ROOM TYPE**: If `intent.room_type` is set, you MUST pick a hotel whose "
    "`room` field matches the requested type (e.g. 'Entire home/apt', 'Private room'). "
    "Skip any candidate whose `room` field does NOT match.\n"
    "- **HOUSE RULES**: If `intent.house_rule` is set, you MUST pick a hotel whose "
    "`rules` field is compatible (e.g. if rule is 'pets allowed', skip hotels with "
    "'no_pets'; if rule is 'no smoking', require 'no_smoking' in rules). "
    "When in doubt, skip the hotel.\n"
    "- **MIN NIGHTS**: Reject any candidate whose `min_nights` value exceeds the "
    "trip length (number of nights = days - 1).\n"
    "- If the traveler's return flight departs after 18:00 on the last day, "
    "you may still mark the last-day description as '-' (no overnight needed) — "
    "dinner on that day is handled by the dining specialist.\n\n"
    "Emit ONE entry per day from day 1 through the last day.\n"
    "- Day 1: 'Name, City; Cost: $N for D nights' (cost declared ONCE here).\n"
    "- Days 2..N-1: 'Name, City' (same hotel, NO Cost fragment — the stay was "
    "already paid on day 1).\n"
    "- Last day: '-' (departure day, no overnight stay).\n\n"
    "Return ONLY a real JSON object of the exact shape shown in the example "
    "below — fill in actual hotel strings; never echo the placeholder text.\n\n"
    "Few-shot example (3-day trip to Austin; same hotel, cost declared on day 1 only):\n"
    '{"days": ['
    '{"day": 1, "city": "Austin", "description": "Hotel Van Zandt, Austin; Cost: $189 for 2 nights"}, '
    '{"day": 2, "city": "Austin", "description": "Hotel Van Zandt, Austin"}, '
    '{"day": 3, "city": "Austin", "description": "-"}'
    "]}"
)


def run(intent: Intent, tool_context: ToolContext, repair_note: str = "") -> LodgingPlan:
    tool_results = _format_tool_context(tool_context)
    if tool_results == "(none)":
        raise RuntimeError("No live hotel results available in tool context")

    allowed = _allowed_names(tool_context)
    plan = _call(intent, tool_context, tool_results, repair_note)
    invalid = _invalid_names(plan, allowed)
    if invalid:
        # Retry once only when constraint-aware selection is needed; otherwise
        # mechanical substitution is equally good and saves the LLM call.
        needs_constraint_retry = bool(intent.house_rule or intent.room_type)
        if needs_constraint_retry:
            retry_note = (
                f"{repair_note} "
                f"Your previous plan referenced hotels NOT in the allowed list: {sorted(invalid)}. "
                "Select ONE hotel name verbatim from the numbered tool results above."
            ).strip()
            plan = _call(intent, tool_context, tool_results, retry_note)
            still_invalid = _invalid_names(plan, allowed)
            if still_invalid and allowed:
                plan = _substitute_hotel(plan, next(iter(sorted(allowed))), intent.dest)
        elif allowed:
            plan = _substitute_hotel(plan, next(iter(sorted(allowed))), intent.dest)

    # Deterministic post-hoc constraint enforcement: if the LLM picked a hotel
    # that violates room_type, house_rules, or min_nights, swap to the best
    # candidate that satisfies all active constraints.
    plan = _enforce_constraints(plan, intent, tool_context)
    return plan


def _substitute_hotel(plan: LodgingPlan, hotel_name: str, city: str) -> LodgingPlan:
    for d in plan.days:
        if not d.description or d.description == "-":
            continue
        # Preserve the Cost fragment (if any) but replace the name+city prefix.
        cost_idx = d.description.lower().find("cost")
        cost_frag = d.description[cost_idx:] if cost_idx >= 0 else ""
        d.description = f"{hotel_name}, {city}" + (f"; {cost_frag}" if cost_frag else "")
    return plan


def _multicity_hint(ctx: ToolContext, intent: Intent) -> str:
    cities = ctx.get("dest_cities") or []
    if len(cities) <= 1:
        return ""
    city_list = ", ".join(cities)
    days_each = max(1, intent.days // len(cities))
    return (
        f"MULTI-CITY TRIP: Cities in order: {city_list}. "
        f"Spend roughly {days_each} day(s) per city. "
        f"Pick one hotel per city block of consecutive days. "
        f"The hotel for each night must be in the same city as that night's stay."
    )


def _call(intent: Intent, ctx: ToolContext, tool_results: str, repair_note: str) -> LodgingPlan:
    windows = format_trip_windows(ctx)
    cap = _format_lodging_cap(ctx, intent)
    mc_hint = _multicity_hint(ctx, intent)
    dest_label = intent.dest if not ctx.get("dest_cities") else (
        f"{intent.dest} (cities: {', '.join(ctx['dest_cities'])})"
    )
    user = (
        f"Intent: {intent.for_specialist('lodging')}\n\n"
        f"Accommodations in {dest_label}:\n{tool_results}\n\n"
        + (mc_hint + "\n\n" if mc_hint else "")
        + (windows + "\n\n" if windows else "")
        + (cap + "\n\n" if cap else "")
        + (f"Repair note: {repair_note}\n\n" if repair_note else "")
        + f"Produce a LodgingPlan covering days 1..{intent.days} "
          f"(last day description must be '-')."
    )
    think = bool(repair_note)
    return call_json(SYSTEM, user, LodgingPlan, max_tokens=600, think_first=think)


def _format_tool_context(ctx: ToolContext) -> str:
    rows = ctx.get("hotels") or []
    if not rows:
        return "(none)"
    cost_map: dict[int, int] = ctx.get("lodging_cost_targets") or {0: 60, 1: 90, 2: 130, 3: 190, 4: 280}
    lines = ["Live API hotels:"]
    for i, r in enumerate(rows, start=1):
        try:
            level = int(r.get("price_level", 2))
        except (TypeError, ValueError):
            level = 2
        est_price = cost_map.get(level, cost_map.get(2, 130))
        parts = [
            f"  [{i}] {r.get('name')}",
            f"rating:{r.get('rating')}",
            f"est_nightly:${est_price}",
        ]
        if r.get("city"):
            parts.append(f"city:{r['city']}")
        if r.get("room_type"):
            parts.append(f"room:{r['room_type']}")
        if r.get("house_rules"):
            parts.append(f"rules:{r['house_rules']}")
        if r.get("min_nights") and int(r.get("min_nights") or 0) > 1:
            parts.append(f"min_nights:{r['min_nights']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _format_lodging_cap(ctx: ToolContext, intent: Intent) -> str:
    caps = ctx.get("category_caps") or {}
    lodging_cap = caps.get("lodging")
    if not lodging_cap:
        return ""
    return (
        f"Per-category budget cap: total lodging cost for the whole stay "
        f"should stay under ${lodging_cap}."
    )


def _allowed_names(ctx: ToolContext) -> set[str]:
    return {
        (r.get("name") or "").strip()
        for r in (ctx.get("hotels") or [])
        if r.get("name")
    }


def _invalid_names(plan: LodgingPlan, allowed: set[str]) -> set[str]:
    bad: set[str] = set()
    if not allowed:
        return bad
    for d in plan.days:
        desc = d.description
        if not desc or desc == "-":
            continue
        name = desc.split(",")[0].split(";")[0].strip()
        if name and name not in allowed:
            bad.add(name)
    return bad


def _price_level_to_nightly(price_level: object) -> int:
    try:
        level = int(price_level)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        level = 2
    return {0: 60, 1: 90, 2: 130, 3: 190, 4: 280}.get(level, 130)


def _satisfies_constraints(record: dict, intent: Intent) -> bool:
    """Return True when a hotel record satisfies all active constraints."""
    nights = max(1, intent.days - 1)

    # min_nights check
    min_n = record.get("min_nights")
    if min_n is not None:
        try:
            if int(min_n) > nights:
                return False
        except (TypeError, ValueError):
            pass

    # room_type check
    if intent.room_type:
        rt = (record.get("room_type") or "").lower()
        if rt and intent.room_type.lower() not in rt and rt not in intent.room_type.lower():
            return False

    # house_rules check (basic: look for "no_X" in rules when intent says "X allowed")
    if intent.house_rule:
        rules = (record.get("house_rules") or "").lower()
        rule_req = intent.house_rule.lower()
        if rules:
            # "pets allowed" requirement conflicts with "no_pets" in rules
            if "pets" in rule_req and "allowed" in rule_req and "no_pets" in rules:
                return False
            # "no smoking" requirement is satisfied by "no_smoking" in rules
            if "smoking" in rule_req and "no_smoking" not in rules and "no smoking" not in rules:
                # only fail if hotel has an explicit smoking-allowed marker
                if "smoking_allowed" in rules or "smoking allowed" in rules:
                    return False

    return True


def _enforce_constraints(plan: LodgingPlan, intent: Intent, ctx: ToolContext) -> LodgingPlan:
    """Swap the chosen hotel to a constraint-satisfying candidate when needed."""
    if not (intent.room_type or intent.house_rule or intent.days > 1):
        return plan

    # Find the hotel name currently used in the plan.
    chosen_name: str | None = None
    for d in plan.days:
        if d.description and d.description != "-":
            chosen_name = d.description.split(",")[0].split(";")[0].strip()
            break
    if chosen_name is None:
        return plan

    # Look up the chosen hotel's record.
    hotels = ctx.get("hotels") or []
    record_map = {(r.get("name") or "").strip(): r for r in hotels}
    chosen_record = record_map.get(chosen_name)

    if chosen_record is not None and _satisfies_constraints(chosen_record, intent):
        return plan  # already fine

    # Find the best replacement candidate.
    for r in hotels:
        name = (r.get("name") or "").strip()
        if not name or name == chosen_name:
            continue
        if _satisfies_constraints(r, intent):
            return _substitute_hotel(plan, name, intent.dest)

    return plan  # no better candidate found — leave as-is
