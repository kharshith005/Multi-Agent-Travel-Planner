"""Multi-agent without verification baseline (paper §4.7, Baseline 2).

Runs the full multi-agent coordinator pipeline — parse → research →
parallel specialists → assemble → budget — but skips the verifier
and repair loop entirely.

Isolates the contribution of constraint checking: comparing this baseline
against the full multi-agent system shows how much the verify-and-repair
loop improves final pass rate.
"""
from __future__ import annotations

from typing import Callable

from agents.coordinator import plan_trip
from agents.schemas import FullPlan, Intent, ToolContext, VerifierReport


def plan_trip_no_verify(
    query: str,
    on_progress: Callable[[str], None] | None = None,
    parsed_intent: Intent | None = None,
    tool_context: ToolContext | None = None,
    bypass_date_check: bool = False,
    model: str | None = None,
) -> tuple[FullPlan, VerifierReport]:
    """Plan a trip using multi-agent coordination without the verify-and-repair loop.

    `parsed_intent` / `tool_context` / `bypass_date_check` pass through so the
    evaluator can run this baseline against the frozen TravelPlanner sandbox
    (historical dates, pre-built ToolContext) exactly like the full system.
    """
    return plan_trip(
        query,
        on_progress=on_progress,
        skip_verify=True,
        parsed_intent=parsed_intent,
        tool_context=tool_context,
        bypass_date_check=bypass_date_check,
        model=model,
    )
