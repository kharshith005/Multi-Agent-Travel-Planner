"""Multi-agent without specialization baseline (paper §4.7, Baseline 3).

Architecture: coordinator (parse) → research node (shared ToolContext) →
single generalist Worker → Budget Agent → verifier → (repair or done).

Replaces the four domain-specific Worker agents (transport, lodging, dining,
sightseeing) with ONE general-purpose LLM that receives the full ToolContext
and produces the entire FullPlan in a single call.

Comparing this baseline against the full multi-agent system isolates the
contribution of role specialization.
"""
from __future__ import annotations

from typing import Callable

from agents.budget import evaluate as budget_evaluate
from agents.coordinator import parse_intent
from agents.llm import call_json
from agents.schemas import FullPlan, Intent, ToolContext, VerifierReport
from agents.verifier import verify
from tools.live_apis import default_live_apis


GENERALIST_SYSTEM = (
    "You are a general-purpose travel planning agent with access to live data "
    "for flights, hotels, restaurants, and attractions. "
    "Produce a complete FullPlan JSON matching the schema:\n"
    '  {"query": str, "plan": [{"days": int, "current_city": str, '
    '"transportation": str, "breakfast": str, "attraction": str, '
    '"lunch": str, "dinner": str, "accommodation": str}, ...]}\n\n'
    "Strict rules:\n"
    "- Cite ONLY names that appear in the tool results.\n"
    "- Each restaurant and attraction must appear AT MOST ONCE across the trip.\n"
    "- Day 1 transportation = inbound leg; last day = outbound leg.\n"
    "- Day 1 breakfast = '-' (traveler still in transit).\n"
    "- Last day dinner = '-' and accommodation = '-' (departure day).\n"
    "- Use ONE accommodation for all overnight stays.\n"
    "- Respect all hard constraints: budget, cuisine, room type, transport mode.\n"
    "- Format costs as 'Cost: $N' so they can be parsed.\n\n"
    "Think step by step through the constraints before writing the JSON."
)


def _fetch_tool_context(intent: Intent) -> ToolContext:
    live = default_live_apis()
    depart = intent.dates[0] if intent.dates else ""
    ret = intent.dates[-1] if intent.dates else ""

    ctx: ToolContext = {}  # type: ignore[assignment]
    try:
        ctx["flights_outbound"] = live.flight_search(
            intent.org, intent.dest, depart, budget=intent.budget
        )
    except Exception:
        ctx["flights_outbound"] = []
    try:
        ctx["flights_return"] = live.flight_search(
            intent.dest, intent.org, ret, budget=intent.budget
        )
    except Exception:
        ctx["flights_return"] = []
    try:
        ctx["route_drive"] = live.route_summary(intent.org, intent.dest, "driving")
    except Exception:
        ctx["route_drive"] = None
    try:
        ctx["route_taxi"] = live.route_summary(intent.org, intent.dest, "taxi")
    except Exception:
        ctx["route_taxi"] = None

    ctx["hotels"] = live.search_places(intent.dest, "hotels", max_results=8)
    ctx["restaurants"] = live.search_places(intent.dest, "restaurants", max_results=12)
    ctx["attractions"] = live.search_places(
        intent.dest, "top tourist attractions", max_results=12
    )
    return ctx


def _format_tool_context(intent: Intent, ctx: ToolContext) -> str:
    parts: list[str] = []

    if ctx.get("flights_outbound"):
        parts.append(f"### Flights {intent.org} → {intent.dest}")
        parts.extend(f"  - {f}" for f in ctx["flights_outbound"][:5])

    if ctx.get("flights_return"):
        parts.append(f"\n### Flights {intent.dest} → {intent.org} (return)")
        parts.extend(f"  - {f}" for f in ctx["flights_return"][:5])

    if ctx.get("route_drive"):
        parts.append(f"\n### Self-driving: {ctx['route_drive']}")
    if ctx.get("route_taxi"):
        parts.append(f"### Taxi: {ctx['route_taxi']}")

    parts.append(f"\n### Hotels in {intent.dest}")
    for r in (ctx.get("hotels") or [])[:8]:
        parts.append(
            f"  - {r.get('name')} | rating:{r.get('rating')} "
            f"| address:{r.get('address')}"
        )

    parts.append(f"\n### Restaurants in {intent.dest}")
    for r in (ctx.get("restaurants") or [])[:12]:
        parts.append(
            f"  - {r.get('name')} | rating:{r.get('rating')} "
            f"| price_level:{r.get('price_level')}"
        )

    parts.append(f"\n### Attractions in {intent.dest}")
    for r in (ctx.get("attractions") or [])[:12]:
        parts.append(
            f"  - {r.get('name')} | rating:{r.get('rating')} "
            f"| address:{r.get('address')}"
        )

    return "\n".join(parts)


def plan_trip_no_specialization(
    query: str,
    on_progress: Callable[[str], None] | None = None,
    parsed_intent: Intent | None = None,
    tool_context: ToolContext | None = None,
    model: str | None = None,
) -> tuple[FullPlan, VerifierReport]:
    """Plan a trip using a single generalist worker instead of domain specialists.

    parsed_intent / tool_context can be supplied by the evaluator so that
    historical (2022) TravelPlanner queries work without live API calls.
    """
    from agents.runtime import use_model
    from agents.models import default_model_id
    model_id = model or default_model_id()

    with use_model(model_id):
        if parsed_intent is not None:
            intent = parsed_intent
        else:
            if on_progress:
                on_progress("No-specialization: parsing query...")
            intent = parse_intent(query)

        if tool_context is not None:
            ctx = tool_context
        else:
            if on_progress:
                on_progress(f"No-specialization: fetching live data for {intent.dest}...")
            ctx = _fetch_tool_context(intent)

        if on_progress:
            on_progress("No-specialization: generalist worker composing full plan...")
        tool_dump = _format_tool_context(intent, ctx)
        user = (
            f"Query: {query}\n\n"
            f"Intent: {intent.model_dump_json()}\n\n"
            f"Tool results:\n{tool_dump}\n\n"
            f"Produce the FullPlan for all {intent.days} days."
        )
        plan = call_json(GENERALIST_SYSTEM, user, FullPlan, max_tokens=3500, think_first=True)

    # Budget Agent evaluation (paper §4.3) — deterministic, no model needed
    budget_report = budget_evaluate(plan, intent)
    if on_progress:
        status = "within budget" if budget_report.passed else f"OVER by ${budget_report.total_cost - (budget_report.budget or 0)}"
        on_progress(f"Budget Agent: ${budget_report.total_cost} — {status}")

    report = verify(plan, intent, sandbox=None)
    return plan, report
