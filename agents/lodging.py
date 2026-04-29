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
        retry_note = (
            f"{repair_note} "
            f"Your previous plan referenced hotels NOT in the allowed list: {sorted(invalid)}. "
            "Select ONE hotel name verbatim from the numbered tool results above."
        ).strip()
        plan = _call(intent, tool_context, tool_results, retry_note)
        # Deterministic fallback: if the LLM still hallucinates, substitute
        # the first allowed name. Better to ground than to fail within_sandbox.
        still_invalid = _invalid_names(plan, allowed)
        if still_invalid and allowed:
            plan = _substitute_hotel(plan, next(iter(sorted(allowed))), intent.dest)
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


def _call(intent: Intent, ctx: ToolContext, tool_results: str, repair_note: str) -> LodgingPlan:
    windows = format_trip_windows(ctx)
    cap = _format_lodging_cap(ctx, intent)
    user = (
        f"Intent: {intent.for_specialist('lodging')}\n\n"
        f"Accommodations in {intent.dest}:\n{tool_results}\n\n"
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
        lines.append(
            f"  [{i}] {r.get('name')} | rating:{r.get('rating')} | est_nightly:${est_price}"
        )
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
