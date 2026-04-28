"""Sightseeing specialist: proposes attractions per day."""
from __future__ import annotations

from .llm import call_json
from .rules import (
    ARRIVAL_DAY_RULE,
    DEPARTURE_DAY_RULE,
    NO_REPEAT_ATTRACTIONS,
    format_trip_windows,
)
from .schemas import Intent, SightseeingPlan, ToolContext


SYSTEM = (
    "You are the Sightseeing specialist in a multi-agent travel planner. "
    "Propose attractions for each day in the destination city.\n\n"
    "CRITICAL GROUNDING RULE: You MUST pick every attraction name verbatim and "
    "EXACTLY as it appears in the numbered [N] list. Do NOT invent names, do NOT "
    "paraphrase. If only a few attractions are listed and some days have no "
    "available option, use '-' for those days rather than inventing names.\n\n"
    "Rules:\n"
    f"- {ARRIVAL_DAY_RULE}\n"
    f"- {DEPARTURE_DAY_RULE}\n"
    f"- {NO_REPEAT_ATTRACTIONS}\n"
    "- On the departure day, keep the schedule light or use '-' to avoid "
    "missing the return journey.\n"
    "- Follow the Flight timing YES/NO grid exactly when present — set any "
    "attraction slot marked NO to '-'; do not plan sightseeing during transit.\n\n"
    "Format: 'Attraction Name, City; Another Name, City;' (semicolon-separated "
    "with trailing semicolon).\n\n"
    "Return ONLY a real JSON object of the exact shape shown in the example "
    "below — fill in actual attraction strings; never echo the placeholder text.\n\n"
    "Few-shot example (2-day trip to Austin):\n"
    '{"days": ['
    '{"day": 1, "city": "Austin", '
    '"description": "Barton Springs Pool, Austin; South Congress Avenue, Austin;"}, '
    '{"day": 2, "city": "Austin", "description": "-"}'
    "]}"
)


def run(intent: Intent, tool_context: ToolContext, repair_note: str = "") -> SightseeingPlan:
    tool_results = _format_tool_context(intent, tool_context)
    if tool_results == "(none)":
        raise RuntimeError("No live attraction results available in tool context")

    allowed = _allowed_names(tool_context)
    plan = _call(intent, tool_context, tool_results, repair_note)
    invalid = _invalid_names(plan, allowed)
    if invalid:
        retry_note = (
            f"{repair_note} "
            f"Your previous plan referenced attractions NOT in the allowed list: {sorted(invalid)}. "
            "Every attraction MUST be copied verbatim from the numbered tool results above."
        ).strip()
        plan = _call(intent, tool_context, tool_results, retry_note)
        still_invalid = _invalid_names(plan, allowed)
        if still_invalid and allowed:
            plan = _substitute_attractions(plan, allowed, intent.dest)
    return plan


def _substitute_attractions(plan: SightseeingPlan, allowed: set[str], city: str) -> SightseeingPlan:
    """Replace invalid attraction names with unused allowed ones (preserves diversity)."""
    pool = [n for n in sorted(allowed)]
    used: set[str] = set()
    for d in plan.days:
        if not d.description or d.description == "-":
            continue
        for seg in d.description.split(";"):
            name = seg.split(",")[0].strip()
            if name and name in allowed:
                used.add(name)

    def next_name() -> str | None:
        for n in pool:
            if n not in used:
                used.add(n)
                return n
        return None

    for d in plan.days:
        if not d.description or d.description == "-":
            continue
        new_segs: list[str] = []
        for seg in d.description.split(";"):
            seg_s = seg.strip()
            if not seg_s:
                continue
            name = seg_s.split(",")[0].strip()
            if name in allowed:
                new_segs.append(f"{name}, {city}")
                continue
            replacement = next_name()
            if replacement is None:
                continue
            new_segs.append(f"{replacement}, {city}")
        d.description = "; ".join(new_segs) + (";" if new_segs else "")
        if not new_segs:
            d.description = "-"
    return plan


def _call(intent: Intent, ctx: ToolContext, tool_results: str, repair_note: str) -> SightseeingPlan:
    windows = format_trip_windows(ctx)
    user = (
        f"Intent: {intent.model_dump_json()}\n\n"
        f"Attractions in {intent.dest}:\n{tool_results}\n\n"
        + (windows + "\n\n" if windows else "")
        + (f"Repair note: {repair_note}\n\n" if repair_note else "")
        + f"Produce a SightseeingPlan covering days 1..{intent.days}."
    )
    return call_json(SYSTEM, user, SightseeingPlan, think_first=True)


def _format_tool_context(intent: Intent, ctx: ToolContext) -> str:
    rows = ctx.get("attractions") or []
    if not rows:
        return "(none)"
    lines = [f"Live API attractions in {intent.dest}:"]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"  [{i}] {r.get('name')} | rating:{r.get('rating')} | address:{r.get('address')}"
        )
    return "\n".join(lines)


def _allowed_names(ctx: ToolContext) -> set[str]:
    return {
        (r.get("name") or "").strip()
        for r in (ctx.get("attractions") or [])
        if r.get("name")
    }


def _invalid_names(plan: SightseeingPlan, allowed: set[str]) -> set[str]:
    bad: set[str] = set()
    if not allowed:
        return bad
    for d in plan.days:
        desc = d.description
        if not desc or desc == "-":
            continue
        for seg in desc.split(";"):
            name = seg.split(",")[0].strip()
            if not name:
                continue
            if name not in allowed:
                bad.add(name)
    return bad
