"""Single-agent ReAct-style baseline.

One LLM call sees all six tool results at once (the closest faithful
rendition of the 'sole-planning' + ReAct setup from TravelPlanner) and
produces the entire plan. No specialists, no verifier, no repair.

Supports two data modes:
  - sandbox (eval): reads from TravelPlanner 2022 reference data
  - live (app):     reads from a pre-built ToolContext (same shape as coordinator)
"""
from __future__ import annotations

from typing import Callable

from agents.coordinator import parse_intent
from agents.llm import call_json
from agents.schemas import FullPlan, Intent, ToolContext
from agents.verifier import verify
from agents.schemas import VerifierReport
from tools.sandbox import Sandbox, default_sandbox


SYSTEM = (
    "You are a single travel-planning agent. You have full tool results "
    "for the requested destination and must produce a FullPlan JSON "
    "matching the schema: {query: str, plan: [{days, current_city, "
    "transportation, breakfast, attraction, lunch, dinner, accommodation}, "
    "...]}. Cite only entries present in the tool results. Keep restaurants "
    "and attractions unique across the trip. First day transportation is "
    "the inbound leg; last day is the outbound leg. Breakfast on day 1 and "
    "dinner on the last day are '-'. Respect all hard constraints.\n\n"
    "COST FORMAT — every non-'-' field MUST carry a cost fragment:\n"
    "- transportation: use only the flight info from the tool results (e.g. "
    "'Delta 09:00->11:30 Duration: 2h Cost: $210'). Do NOT add 'Self-driving' "
    "when a flight option is available. Only use 'Self-driving; Cost: $N' or "
    "'Taxi; Cost: $N' when the tool results contain no flights. "
    "If round-trip, put the full price on Day 1 only; Day N return leg gets no cost.\n"
    "- breakfast/lunch/dinner: append '; Cost: $N' (e.g. 'Joe's Pizza; Cost: $18').\n"
    "- accommodation: on the FIRST stay day write 'Name; Cost: $N for D nights'. "
    "  Subsequent days repeat the name only (no cost). Use the estimated nightly "
    "  rate from the tool results multiplied by total nights.\n"
    "Use the estimated costs provided in the tool results. If no price is listed, "
    "make a reasonable estimate that fits within the trip budget."
)


def _bundle_tools(intent: Intent, sandbox: Sandbox) -> str:
    parts: list[str] = []
    for city in [intent.dest]:
        parts.append(f"### Attractions in {city}")
        for r in sandbox.attraction_search(city)[:20]:
            parts.append(f"- {r.get('Name')}")
        parts.append(f"\n### Restaurants in {city}")
        for r in sandbox.restaurant_search(city)[:25]:
            parts.append(f"- {r.get('Name')} | cuisine:{r.get('Cuisines')} | ${r.get('Average Cost')}")
        parts.append(f"\n### Accommodations in {city}")
        for r in sandbox.accommodation_search(city)[:10]:
            parts.append(
                f"- {r.get('NAME')} | ${r.get('price')} | rules:{r.get('house_rules')} | "
                f"room:{r.get('room type')} | min:{r.get('minimum nights')}"
            )
    if intent.dates:
        parts.append(f"\n### Flights {intent.org} -> {intent.dest} on {intent.dates[0]}")
        for f in sandbox.flight_search(intent.org, intent.dest, intent.dates[0])[:5]:
            parts.append(
                f"- {f.get('Flight Number')} {f.get('DepTime')}->{f.get('ArrTime')} ${f.get('Price')}"
            )
        parts.append(f"\n### Flights {intent.dest} -> {intent.org} on {intent.dates[-1]}")
        for f in sandbox.flight_search(intent.dest, intent.org, intent.dates[-1])[:5]:
            parts.append(
                f"- {f.get('Flight Number')} {f.get('DepTime')}->{f.get('ArrTime')} ${f.get('Price')}"
            )
    drive = sandbox.distance_matrix(intent.org, intent.dest, "driving")
    taxi = sandbox.distance_matrix(intent.org, intent.dest, "taxi")
    if drive:
        parts.append(f"\n### Self-driving: {drive}")
    if taxi:
        parts.append(f"### Taxi: {taxi}")
    return "\n".join(parts)


_DEFAULT_MEAL_COSTS: dict[int, int]    = {0: 8,  1: 15, 2: 28, 3: 45, 4: 70}
_DEFAULT_LODGING_COSTS: dict[int, int] = {0: 60, 1: 90, 2: 130, 3: 190, 4: 280}


def _strip_drive_legs(flight_str: str) -> str:
    """Remove drive-to-airport prefix/suffix from a composed flight string.

    _compose_flight_segments joins drive legs and flight info with '; '.
    Drive leg segments start with 'Drive ' or 'Distance:' — drop them so
    the single-agent only sees the clean flight description.
    """
    parts = [p.strip() for p in flight_str.split("; ")]
    kept = [
        p for p in parts
        if p
        and not p.startswith("Drive ")
        and not p.startswith("Distance:")
    ]
    return "; ".join(kept) if kept else flight_str


def _bundle_live_context(intent: Intent, ctx: ToolContext) -> str:
    """Format a live ToolContext into a tool-dump string for the single-agent prompt."""
    meal_costs    = ctx.get("meal_cost_targets")    or _DEFAULT_MEAL_COSTS
    lodging_costs = ctx.get("lodging_cost_targets") or _DEFAULT_LODGING_COSTS
    nights        = max(1, intent.days - 1)

    parts: list[str] = []
    parts.append(f"### Attractions in {intent.dest}")
    for r in (ctx.get("attractions") or [])[:20]:
        parts.append(f"- {r.get('name')}")

    parts.append(f"\n### Restaurants in {intent.dest}  (estimated meal cost per person)")
    for r in (ctx.get("restaurants") or [])[:25]:
        level = int(r.get("price_level") or 2)
        est   = meal_costs.get(level, meal_costs.get(2, 28))
        parts.append(f"- {r.get('name')} | est. Cost: ${est}")

    parts.append(f"\n### Hotels in {intent.dest}  (estimated cost for {nights} night(s))")
    for r in (ctx.get("hotels") or [])[:10]:
        level     = int(r.get("price_level") or 2)
        nightly   = lodging_costs.get(level, lodging_costs.get(2, 130))
        total_est = nightly * nights
        parts.append(
            f"- {r.get('name')} | rating:{r.get('rating')} "
            f"| est. Cost: ${total_est} for {nights} nights (${nightly}/night)"
        )

    outbound = ctx.get("flights_outbound") or []
    ret      = ctx.get("flights_return") or []
    has_flights = bool(outbound or ret)

    for label, flights in (
        (f"Flights {intent.org} -> {intent.dest}", outbound),
        (f"Flights {intent.dest} -> {intent.org}", ret),
    ):
        if flights:
            parts.append(f"\n### {label}")
            for f in flights[:5]:
                parts.append(f"- {_strip_drive_legs(f)}")

    # Only show driving/taxi when no flights exist (they are alternative modes,
    # not combined with flights for the single-agent baseline).
    if not has_flights:
        if ctx.get("route_drive"):
            parts.append(f"\n### Self-driving option: {ctx['route_drive']}")
        if ctx.get("route_taxi"):
            parts.append(f"### Taxi option: {ctx['route_taxi']}")

    transport_target = ctx.get("transport_one_way_target")
    if transport_target:
        parts.append(f"\n### Budget guidance")
        parts.append(f"- Transport cap (one-way): ~${transport_target}")

    return "\n".join(parts)


def plan_trip_single(
    query: str,
    sandbox_kind: str = "frozen",
    on_progress: Callable[[str], None] | None = None,
    model: str | None = None,
    tool_context: ToolContext | None = None,
    parsed_intent: Intent | None = None,
    bypass_date_check: bool = False,
) -> tuple[FullPlan, VerifierReport]:
    """Plan a trip with a single LLM call.

    When tool_context is provided (live app mode), data comes from live APIs.
    Otherwise, falls back to sandbox data (eval mode).
    """
    from agents.runtime import use_model
    from agents.models import default_model_id
    model_id = model or default_model_id()

    sandbox = default_sandbox(sandbox_kind) if tool_context is None else None

    with use_model(model_id):
        if on_progress:
            on_progress("Single-agent parsing query...")
        intent = parsed_intent or parse_intent(query)

        if on_progress:
            on_progress("Single-agent composing plan with all tools...")
        if tool_context is not None:
            tool_dump = _bundle_live_context(intent, tool_context)
        else:
            tool_dump = _bundle_tools(intent, sandbox)  # type: ignore[arg-type]
        user = (
            f"Query: {query}\n\n"
            f"Intent: {intent.model_dump_json()}\n\n"
            f"All tool results:\n{tool_dump}\n\n"
            f"Produce the FullPlan for all {intent.days} days."
        )
        plan = call_json(SYSTEM, user, FullPlan, max_tokens=3500)

    rt = bool((tool_context or {}).get("flights_round_trip", False))
    report = verify(plan, intent, sandbox, flights_round_trip=rt)
    return plan, report
