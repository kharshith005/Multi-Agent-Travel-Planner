"""Transport specialist: proposes inbound, outbound, and inter-city legs."""
from __future__ import annotations

import re

from .llm import call_json
from .rules import ARRIVAL_DAY_RULE, BUDGET_GUIDANCE, DEPARTURE_DAY_RULE
from .schemas import Intent, ToolContext, TransportPlan


# A flight identifier begins with letters+digits ("F3001", "DL1234").
# Self-driving/taxi/train strings won't match and are skipped by grounding.
_FLIGHT_ID_RE = re.compile(r"^\s*([A-Z]{1,3}\s*\d{2,5})\b")


SYSTEM = (
    "You are the Transport specialist in a multi-agent travel planner. "
    "Propose transportation for every day of the itinerary using ONLY the "
    "flight/route options provided in the tool results.\n\n"
    "Rules:\n"
    f"- {ARRIVAL_DAY_RULE}\n"
    f"- {DEPARTURE_DAY_RULE}\n"
    "- Day 1 uses the outbound flight or driving/taxi route. "
    "Last day uses the return flight or route. "
    "Middle days with no city change: description is '-'.\n"
    f"- {BUDGET_GUIDANCE}\n"
    "- **FLIGHT SELECTION PRIORITY — DURATION FIRST**: The flight options are "
    "listed in recommended order (shortest total duration, budget-aware). "
    "**Always select the FIRST option listed** for the outbound and return legs. "
    "A flight that is 30–40% more expensive but 50%+ shorter in travel time is "
    "strongly preferred — traveler time is valuable. Never pick a flight with "
    "layovers or 2× the duration of a shorter available option. "
    "Preserve the EXACT departure and arrival times from the listed option "
    "(e.g., 'Airline HH:MM->HH:MM Duration: Xh Ym Cost: $N').\n"
    "- For self-driving or taxi use 'Self-driving; Cost: $N' or 'Taxi; Cost: $N'.\n"
    "- **DRIVE LEGS — VERBATIM**: A flight option may include drive segments before "
    "and/or after the flight: 'Drive {N} mi to {IATA}; {flight}; Drive {M} mi from "
    "{IATA} to {city}'. ALL drive segments are part of the day's transportation — "
    "include every segment verbatim. Drive costs are bundled into the flight cost.\n"
    "- **ROUND-TRIP COST RULE**: When the tool context notes 'ROUND-TRIP FARES', "
    "the Cost: $N shown on outbound options is the TOTAL round-trip price. "
    "Use it on Day 1 only. The last-day return description must NOT include any "
    "Cost: $N fragment — it is already captured on Day 1.\n\n"
    "Return ONLY a real JSON object of the exact shape shown in the example "
    "below — fill in actual flight strings; never echo the placeholder text.\n\n"
    "Few-shot example (2-day trip, Dallas → Austin, one-way fares):\n"
    '{"days": ['
    '{"day": 1, "description": "Southwest 1234 10:00->11:10 Duration: 1h 10m Cost: $89"}, '
    '{"day": 2, "description": "Southwest 5678 17:00->18:10 Duration: 1h 10m Cost: $89"}'
    "]}\n\n"
    "Few-shot example (2-day trip into a city without a direct airport — drive "
    "legs on BOTH ends of each flight, round-trip fare):\n"
    '{"days": ['
    '{"day": 1, "description": "Drive 50 mi to JFK; Delta 1234 09:00->11:30 Duration: 2h 30m Cost: $310; Drive 87 mi from PNS to Pensacola"}, '
    '{"day": 2, "description": "Drive 87 mi from Pensacola to PNS; Delta 5678 17:00->20:00 Duration: 3h; Drive 50 mi from JFK to New York"}'
    "]}"
)


def run(intent: Intent, tool_context: ToolContext, repair_note: str = "") -> TransportPlan:
    tool_results = _format_tool_context(intent, tool_context)
    if tool_results == "(no transport data available)":
        raise RuntimeError("No live transport results available in tool context")

    allowed = _allowed_flight_ids(tool_context)
    plan = _call(intent, tool_context, tool_results, repair_note)
    invalid = _invalid_flight_ids(plan, allowed)
    if invalid and allowed:
        retry_note = (
            f"{repair_note} "
            f"Your previous plan referenced flight IDs NOT in the allowed list: {sorted(invalid)}. "
            "Copy the flight identifier and HH:MM times verbatim from the numbered tool results above."
        ).strip()
        plan = _call(intent, tool_context, tool_results, retry_note)
        still_invalid = _invalid_flight_ids(plan, allowed)
        if still_invalid:
            # Deterministic fallback: overwrite any bad flight description with
            # the FIRST allowed option for the corresponding leg (outbound on
            # day 1, return on the last day).
            plan = _substitute_flights(plan, tool_context, intent.days)
    return plan


def _call(intent: Intent, ctx: ToolContext, tool_results: str, repair_note: str) -> TransportPlan:
    cap = _format_transport_cap(ctx)
    user = (
        f"Intent: {intent.for_specialist('transport')}\n\n"
        f"Tool results:\n{tool_results}\n\n"
        + (cap + "\n\n" if cap else "")
        + (f"Repair note from verifier: {repair_note}\n\n" if repair_note else "")
        + f"Produce a TransportPlan covering days 1..{intent.days}."
    )
    think = bool(repair_note)
    return call_json(SYSTEM, user, TransportPlan, max_tokens=800, think_first=think)


def _allowed_flight_ids(ctx: ToolContext) -> set[str]:
    pool: set[str] = set()
    for line in (ctx.get("flights_outbound") or []) + (ctx.get("flights_return") or []):
        m = _FLIGHT_ID_RE.match(line)
        if m:
            pool.add(m.group(1).replace(" ", ""))
    return pool


def _extract_flight_id(desc: str) -> str | None:
    m = _FLIGHT_ID_RE.match(desc or "")
    return m.group(1).replace(" ", "") if m else None


def _invalid_flight_ids(plan: TransportPlan, allowed: set[str]) -> set[str]:
    bad: set[str] = set()
    if not allowed:
        return bad
    for d in plan.days:
        fid = _extract_flight_id(d.description)
        if fid and fid not in allowed:
            bad.add(fid)
    return bad


def _substitute_flights(plan: TransportPlan, ctx: ToolContext, total_days: int) -> TransportPlan:
    outbound = (ctx.get("flights_outbound") or [])
    inbound = (ctx.get("flights_return") or [])
    out_first = outbound[0] if outbound else None
    ret_first = inbound[0] if inbound else None
    for d in plan.days:
        fid = _extract_flight_id(d.description)
        if fid is None:
            continue  # self-driving/taxi/train — leave alone
        if d.day == 1 and out_first:
            d.description = out_first
        elif d.day == total_days and ret_first:
            d.description = ret_first
    return plan


def _format_transport_cap(ctx: ToolContext) -> str:
    ctx = ctx or {}
    caps = ctx.get("category_caps") or {}
    t_cap = caps.get("transport")
    one_way = ctx.get("transport_one_way_target")
    parts: list[str] = []
    if t_cap:
        parts.append(
            f"Per-category budget cap: combined transport cost (outbound + return) "
            f"should stay under ${t_cap}."
        )
    if one_way:
        parts.append(f"Each one-way flight should ideally stay under ${one_way}.")
    if not parts:
        return ""
    return " ".join(parts) + " Prefer the shortest remaining option whose price fits this cap."


def _format_tool_context(intent: Intent, ctx: ToolContext) -> str:
    lines: list[str] = []

    outbound = ctx.get("flights_outbound") or []
    inbound = ctx.get("flights_return") or []
    drive = ctx.get("route_drive")
    taxi = ctx.get("route_taxi")

    is_rt = bool(ctx.get("flights_round_trip"))
    if outbound:
        fare_note = " [ROUND-TRIP FARES — Cost shown is the full round-trip price]" if is_rt else ""
        lines.append(f"Live flight options {intent.org} → {intent.dest}{fare_note}:")
        lines.extend(f"  - {f}" for f in outbound)
    if inbound:
        rt_note = " [ROUND-TRIP FARES — do NOT add Cost on this leg]" if is_rt else ""
        lines.append(f"Live flight options {intent.dest} → {intent.org}{rt_note}:")
        lines.extend(f"  - {f}" for f in inbound)
    if drive:
        lines.append(f"Self-driving: {drive}")
    if taxi:
        lines.append(f"Taxi: {taxi}")

    return "\n".join(lines) if lines else "(no transport data available)"
