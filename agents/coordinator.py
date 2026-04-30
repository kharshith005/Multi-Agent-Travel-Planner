"""Live-only coordinator powered by LangGraph.

Flow:
- Parse intent
- Run domain specialists (parallel or sequential)
- Assemble final plan
- Apply timing adjustments (arrival/departure time-aware meal/attraction fixes)
- Verify constraints
- Up to REPAIR_MAX_ROUNDS repair passes, with repeat-violation escalation
- Finalize response
"""
from __future__ import annotations

import re as _re
from datetime import date, datetime
from typing import Annotated, Any, Callable, Iterable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import budget, dining, lodging, sightseeing, transport, verifier
from .rules import (
    THRESH_D1_ATTRACTION_NO,
    THRESH_D1_DINNER_NO,
    THRESH_D1_LUNCH_NO,
    THRESH_LAST_BREAKFAST_NO,
    THRESH_LAST_DINNER_YES,
    THRESH_LAST_LUNCH_YES,
)
from .llm import call_json
from .perf import Timeline, merge_timelines
from .runtime import current_model, use_model
from .schemas import (
    AgentDecision,
    AttributionLog,
    BudgetReport,
    DiningPlan,
    FullPlan,
    Intent,
    LodgingPlan,
    PlanDay,
    SightseeingPlan,
    ToolContext,
    TransportPlan,
    VerifierReport,
)
from tools.live_apis import default_live_apis


def _call_live(fn, *args, skip_key: str = "", **kwargs):
    """Call a live API function and return a 3-tuple (status, result, error_msg).

    status values:
      "ok"      — call succeeded; result is the data
      "skipped" — key env-var is absent; result is the empty default
      "error"   — key is present but call failed; result is the empty default

    This replaces bare try/except that silently swallows API failures,
    so callers can surface error state in on_progress without leaking key names.
    """
    import os
    if skip_key and not os.environ.get(skip_key, "").strip():
        return "skipped", [] if not kwargs.get("default") else kwargs["default"], ""
    try:
        return "ok", fn(*args), ""
    except Exception as exc:
        return "error", [] if not kwargs.get("default") else kwargs["default"], str(exc)


INTENT_SYSTEM = (
    "You extract a structured travel intent from a user's chat message. "
    "Return JSON matching the Intent schema with fields: org, dest, days, "
    "dates (YYYY-MM-DD strings), people, budget (int or null), house_rule, "
    "cuisine, room_type, transportation. Use null when a field is not mentioned. "
    "The trip is always to a single destination city.\n"
    "IMPORTANT: If the user states a date range (e.g. 'June 1 to June 3'), include "
    "BOTH the start date and end date in `dates` (e.g. ['2026-06-01', '2026-06-03']) "
    "and set `days` to the number of days in that range (inclusive). "
    "If only a start date is given, include just that one date in `dates` and set "
    "`days` from any explicit duration mentioned; if no duration, set `days` to 1.\n"
    "IMPORTANT: If the user did NOT state an unambiguous trip START date, return "
    "an empty `dates` list. Do not guess a start date. Do not use today's date.\n"
    "IMPORTANT: When multiple cuisines, room rules, or transportation modes are "
    "mentioned, return them as a single comma-separated string — never as an array."
)


def parse_intent(query: str) -> Intent:
    return call_json(INTENT_SYSTEM, query, Intent)


def intent_constraint_violations(intent: Intent) -> list[str]:
    violations: list[str] = []
    if not (intent.org or "").strip():
        violations.append("missing trip origin city (from)")
    if not (intent.dest or "").strip():
        violations.append("missing trip destination city (to)")
    if intent.budget is None:
        violations.append("missing budget")
    if not any((d or "").strip() for d in intent.dates):
        violations.append("missing trip start date (provide a start date, e.g. 'March 15')")
    return violations


def _past_dates(dates: list[str]) -> list[str]:
    today = date.today()
    out: list[str] = []
    for raw in dates:
        txt = (raw or "").strip()
        if not txt:
            continue
        try:
            d = datetime.strptime(txt[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < today:
            out.append(d.isoformat())
    return out


REPAIR_MAX_ROUNDS = 2


class CoordinatorState(TypedDict, total=False):
    query: str
    execution_mode: str
    model_id: str                    # active LLM model for this run (set by plan_trip)
    intent: Intent
    tool_context: ToolContext        # pre-fetched live data shared across all specialists
    repair_notes: dict[str, str]
    rerun_specialists: list[str]
    repair_round: int
    prev_violation_keys: list[str]   # (rule, responsible) pairs from last round, for escalation
    transport_plan: TransportPlan
    lodging_plan: LodgingPlan
    dining_plan: DiningPlan
    sightseeing_plan: SightseeingPlan
    plan: FullPlan
    budget_plan: BudgetReport        # Budget Agent output (paper §4.3)
    attribution_log: AttributionLog  # per-decision agent attribution (paper §4.5)
    report: VerifierReport
    timeline: Annotated[Timeline, merge_timelines]  # parallel nodes each return a fresh local Timeline


def assemble(
    intent: Intent,
    t: TransportPlan,
    l: LodgingPlan,
    d: DiningPlan,
    s: SightseeingPlan,
    query: str,
) -> FullPlan:
    def by_day(items: Iterable, key: str = "day") -> dict:
        return {getattr(x, key): x for x in items}

    t_by = by_day(t.days)
    d_by = by_day(d.days)
    s_by = by_day(s.days)

    # Forward-fill accommodation across non-departure days. The first non-"-"
    # description carries the full "Name, City; Cost: $N for D nights" string
    # and counts toward the budget; subsequent days carry only the name so
    # min_nights passes without double-counting lodging cost (external eval sums
    # Cost fragments across every day).
    stay_full = "-"
    stay_name_only = "-"
    for day_obj in sorted(l.days, key=lambda x: x.day):
        desc = (day_obj.description or "").strip()
        if desc and desc != "-":
            stay_full = day_obj.description
            # Strip any "; Cost: ..." segment; keep "Name, City".
            stay_name_only = _COST_SEGMENT_RE.sub("", stay_full).rstrip(" ;").strip()
            break

    days: list[PlanDay] = []
    first_stay_day_assigned = False
    for i in range(1, intent.days + 1):
        current_city = (
            f"from {intent.org} to {intent.dest}" if i == 1
            else (f"from {intent.dest} to {intent.org}" if i == intent.days else intent.dest)
        )
        meals = d_by.get(i)

        if i == intent.days:
            accommodation = "-"
        else:
            if not first_stay_day_assigned and stay_full != "-":
                accommodation = stay_full
                first_stay_day_assigned = True
            else:
                accommodation = stay_name_only

        days.append(PlanDay(
            days=i,
            current_city=current_city,
            transportation=(t_by[i].description if i in t_by else "-"),
            breakfast=(meals.breakfast if meals else "-"),
            attraction=(s_by[i].description if i in s_by else "-"),
            lunch=(meals.lunch if meals else "-"),
            dinner=(meals.dinner if meals else "-"),
            accommodation=accommodation,
        ))
    return FullPlan(query=query, plan=days)


_HHMM_AFTER_ARROW = _re.compile(r"->\s*(\d{1,2}):(\d{2})")
_HHMM_BEFORE_ARROW = _re.compile(r"(\d{1,2}):(\d{2})\s*->")
_PRICE_LEVEL_MAP = {0: 8, 1: 15, 2: 28, 3: 45, 4: 70}
_COST_SEGMENT_RE = _re.compile(r";\s*Cost[:\s][^;]*", flags=_re.IGNORECASE)
_DRIVE_DISTANCE_RE = _re.compile(r"Distance:\s*([0-9]+(?:\.[0-9]+)?)\s*mi", _re.IGNORECASE)
_DRIVE_LEG_HEAD_RE = _re.compile(r"^Drive\s+(\d+)\s+mi\s+to\s+([A-Z]{3})", _re.IGNORECASE)


def _reverse_drive_leg(drive_leg: str | None, target_city: str) -> str | None:
    """Turn 'Drive N mi to IATA; route' into 'Drive N mi from IATA to city'.

    Used to surface the drive-from-airport leg after the flight on the same
    day, mirroring how the drive-to-airport leg is shown before the flight.
    """
    if not drive_leg:
        return None
    m = _DRIVE_LEG_HEAD_RE.match(drive_leg)
    if not m:
        return None
    return f"Drive {m.group(1)} mi from {m.group(2)} to {target_city}"


def _compose_flight_segments(*segments: str | None) -> str:
    """Join non-empty drive/flight/drive segments with '; ' separators."""
    return "; ".join(s for s in segments if s)


def _parse_drive_distance(route_drive: str | None) -> float | None:
    """Parse miles from Maps route summary 'Distance: 87.3 mi, Duration: ...'."""
    if not route_drive:
        return None
    m = _DRIVE_DISTANCE_RE.search(route_drive)
    return float(m.group(1)) if m else None


def _to_minutes(h: str, m: str) -> int:
    return int(h) * 60 + int(m)


def _extract_arrival_minutes(transport_desc: str) -> int | None:
    """Parse destination arrival time from 'Airline HH:MM->HH:MM ...' string."""
    match = _HHMM_AFTER_ARROW.search(transport_desc)
    return _to_minutes(match.group(1), match.group(2)) if match else None


def _extract_departure_minutes(transport_desc: str) -> int | None:
    """Parse origin departure time from 'Airline HH:MM->...' string."""
    match = _HHMM_BEFORE_ARROW.search(transport_desc)
    return _to_minutes(match.group(1), match.group(2)) if match else None


def _fmt_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def derive_trip_windows(
    flights_outbound: list[str],
    flights_return: list[str],
) -> dict:
    """Derive TripWindows from pre-fetched flight strings without an LLM call.

    Parses the first outbound flight's arrival time and the first return
    flight's departure time — the same values transport specialist produces
    by picking ranked[0]. Called by the research node so all four specialists
    can run in parallel with timing context already populated.
    """
    arr = _extract_arrival_minutes(flights_outbound[0]) if flights_outbound else None
    dep = _extract_departure_minutes(flights_return[0]) if flights_return else None
    return {
        "arrival_minutes_day1": arr,
        "departure_minutes_last": dep,
        "arrival_hhmm_day1": _fmt_time(arr) if arr is not None else None,
        "departure_hhmm_last": _fmt_time(dep) if dep is not None else None,
    }


def _merge_trip_windows(
    ctx: ToolContext | None, transport_plan: TransportPlan | None
) -> ToolContext | None:
    """Compute trip_windows from a transport plan and return an updated ToolContext.

    Gives downstream lodging/dining/sightseeing specialists real arrival and
    departure times BEFORE they plan, so meal/attraction/lodging choices can
    account for actual flight timing instead of needing post-hoc patching.
    """
    if transport_plan is None or not transport_plan.days:
        return None
    days = sorted(transport_plan.days, key=lambda d: d.day)
    first_desc = days[0].description or ""
    last_desc = days[-1].description if len(days) > 1 else ""

    arr = _extract_arrival_minutes(first_desc)
    dep = _extract_departure_minutes(last_desc) if last_desc else None

    windows = {
        "arrival_minutes_day1": arr,
        "departure_minutes_last": dep,
        "arrival_hhmm_day1": _fmt_time(arr) if arr is not None else None,
        "departure_hhmm_last": _fmt_time(dep) if dep is not None else None,
    }
    new_ctx: dict = dict(ctx) if ctx else {}
    new_ctx["trip_windows"] = windows
    return new_ctx  # type: ignore[return-value]


def _pick_unused_restaurant(tool_context: ToolContext | None, used: set[str]) -> str | None:
    """Return a formatted restaurant string for an unused restaurant in ToolContext."""
    if not tool_context:
        return None
    for r in (tool_context.get("restaurants") or []):
        name = (r.get("name") or "").strip()
        if not name or name.lower() in used:
            continue
        try:
            cost = _PRICE_LEVEL_MAP.get(int(r.get("price_level", 2)), 28)
        except (TypeError, ValueError):
            cost = 28
        addr = r.get("address", "") or ""
        city = addr.split(",")[-1].strip() if addr else ""
        return f"{name}{', ' + city if city else ''}; Cost: ${cost}"
    return None


def _apply_timing_adjustments(plan: FullPlan, tool_context: ToolContext | None) -> FullPlan:
    """Post-process assembled plan to account for actual flight arrival/departure times.

    Day 1 (arrival): Clear meals and attractions that fall before the traveler
    reaches the destination city.
    Last day (departure): Restore dinner when the return flight departs late
    enough (≥ 18:00) for a proper meal before heading to the airport.
    """
    days = list(plan.plan)
    if not days:
        return plan

    # Collect restaurant names already placed in the plan to avoid duplicates.
    used_restaurants: set[str] = set()
    for day in days:
        for meal in (day.breakfast, day.lunch, day.dinner):
            if meal and meal != "-":
                name = meal.split(",")[0].split(";")[0].strip()
                if name:
                    used_restaurants.add(name.lower())

    # ── Day 1: arrival-day timing safety clamp ────────────────────────────────
    # Specialists should have obeyed the YES/NO grid; this clamp fires only
    # when a specialist disregarded it (defense-in-depth). Clamp values are
    # always "-" so the transportation field stays the single source of truth
    # for arrival/departure times (no echoed HH:MM in meal/attraction fields).
    day1 = days[0]
    arrival_min = _extract_arrival_minutes(day1.transportation)
    if arrival_min is not None:
        updates: dict = {}
        if arrival_min >= THRESH_D1_LUNCH_NO and day1.lunch and day1.lunch != "-":
            updates["lunch"] = "-"
        if arrival_min >= THRESH_D1_ATTRACTION_NO and day1.attraction and day1.attraction != "-":
            updates["attraction"] = "-"
        if arrival_min >= THRESH_D1_DINNER_NO and day1.dinner and day1.dinner != "-":
            updates["dinner"] = "-"
        if updates:
            days[0] = day1.model_copy(update=updates)

    # ── Last day: departure-day timing safety clamp ───────────────────────────
    if len(days) > 1:
        last = days[-1]
        dep_min = _extract_departure_minutes(last.transportation)
        if dep_min is not None:
            updates = {}
            if dep_min < THRESH_LAST_BREAKFAST_NO:
                updates["breakfast"] = "-"
                updates["lunch"] = "-"
            elif dep_min < THRESH_LAST_LUNCH_YES:
                updates["lunch"] = "-"

            # Very late departure — restore dinner only when a real restaurant
            # is available; otherwise leave as "-" (no echoed departure time).
            if dep_min >= THRESH_LAST_DINNER_YES and (not last.dinner or last.dinner == "-"):
                choice = _pick_unused_restaurant(tool_context, used_restaurants)
                if choice:
                    updates["dinner"] = choice

            if updates:
                days[-1] = last.model_copy(update=updates)

    return plan.model_copy(update={"plan": days})


def _run_specialist(
    name: str,
    fn: Callable[[Intent, ToolContext, str], Any],
    state: CoordinatorState,
    on_progress: Callable[[str], None] | None,
) -> Any:
    intent = state["intent"]
    tool_context: ToolContext = state.get("tool_context", {})  # type: ignore[assignment]
    repair_note = state.get("repair_notes", {}).get(name, "")
    model_id = state.get("model_id") or current_model()
    if on_progress:
        on_progress(f"{name} working...")
    with use_model(model_id):
        return fn(intent, tool_context, repair_note)


def _should_run_specialist(state: CoordinatorState, name: str) -> bool:
    rerun = state.get("rerun_specialists")
    # None = unset (first run) → run all specialists.
    # [] = mechanical pre-repair resolved everything → skip LLM re-runs.
    # [...] = explicit list → run only those named.
    if rerun is None:
        return True
    return name in rerun


def _mech_repair_dining(
    dining_plan: "DiningPlan",
    tool_context: "ToolContext",
) -> "tuple[DiningPlan, bool]":
    """Replace duplicate restaurant entries with unused ones from ToolContext.

    Returns (patched_plan, True) when at least one duplicate was replaced;
    (original_plan, False) when the plan is already duplicate-free or no
    unused restaurants are available.
    """
    restaurants = (tool_context or {}).get("restaurants") or []
    if not restaurants:
        return dining_plan, False

    meal_fields = ("breakfast", "lunch", "dinner")
    assigned: set[str] = set()
    seen_duplicates: list[tuple[int, str]] = []  # (day_no, field)

    for day in sorted(dining_plan.days, key=lambda d: d.day):
        for field in meal_fields:
            text = getattr(day, field, "-") or "-"
            if text == "-":
                continue
            name = text.split(",")[0].split(";")[0].strip().lower()
            if not name:
                continue
            if name in assigned:
                seen_duplicates.append((day.day, field))
            else:
                assigned.add(name)

    if not seen_duplicates:
        return dining_plan, False

    unused = [r for r in restaurants if r.get("name") and r["name"].strip().lower() not in assigned]
    if not unused:
        return dining_plan, False

    days_by_no = {d.day: d for d in dining_plan.days}
    unused_iter = iter(unused)
    patched = False

    for day_no, field in seen_duplicates:
        try:
            r = next(unused_iter)
        except StopIteration:
            break
        day = days_by_no.get(day_no)
        if day is None:
            continue
        r_name = (r.get("name") or "").strip()
        r_addr = r.get("address", "") or ""
        r_city = r_addr.split(",")[-1].strip() if r_addr else ""
        try:
            r_cost = _PRICE_LEVEL_MAP.get(int(r.get("price_level", 2)), 28)
        except (TypeError, ValueError):
            r_cost = 28
        new_text = f"{r_name}{', ' + r_city if r_city else ''}; Cost: ${r_cost}"
        days_by_no[day_no] = day.model_copy(update={field: new_text})
        assigned.add(r_name.lower())
        patched = True

    if not patched:
        return dining_plan, False

    return DiningPlan(days=[days_by_no[k] for k in sorted(days_by_no)]), True


def _mech_repair_sightseeing(
    sightseeing_plan: "SightseeingPlan",
    tool_context: "ToolContext",
) -> "tuple[SightseeingPlan, bool]":
    """Replace duplicate attraction entries with unused ones from ToolContext."""
    attractions = (tool_context or {}).get("attractions") or []
    if not attractions:
        return sightseeing_plan, False

    assigned: set[str] = set()
    seen_duplicates: list[int] = []  # day numbers with duplicate attractions

    for day in sorted(sightseeing_plan.days, key=lambda d: d.day):
        text = (day.description or "").strip()
        if text == "-":
            continue
        name = text.split(",")[0].split(";")[0].strip().lower()
        if not name:
            continue
        if name in assigned:
            seen_duplicates.append(day.day)
        else:
            assigned.add(name)

    if not seen_duplicates:
        return sightseeing_plan, False

    unused = [a for a in attractions if a.get("name") and a["name"].strip().lower() not in assigned]
    if not unused:
        return sightseeing_plan, False

    days_by_no = {d.day: d for d in sightseeing_plan.days}
    unused_iter = iter(unused)
    patched = False

    for day_no in seen_duplicates:
        try:
            a = next(unused_iter)
        except StopIteration:
            break
        day = days_by_no.get(day_no)
        if day is None:
            continue
        a_name = (a.get("name") or "").strip()
        a_addr = a.get("address", "") or ""
        a_city = a_addr.split(",")[-1].strip() if a_addr else ""
        new_text = f"{a_name}{', ' + a_city if a_city else ''}"
        days_by_no[day_no] = day.model_copy(update={"description": new_text})
        assigned.add(a_name.lower())
        patched = True

    if not patched:
        return sightseeing_plan, False

    return SightseeingPlan(days=[days_by_no[k] for k in sorted(days_by_no)]), True


def _build_graph(
    execution_mode: str,
    on_progress: Callable[[str], None] | None = None,
    skip_verify: bool = False,
    bypass_date_check: bool = False,
):
    def parse_node(state: CoordinatorState) -> CoordinatorState:
        local_tl = Timeline()
        model_id = state.get("model_id") or current_model()
        intent = state.get("intent")
        with local_tl.phase("parse") as ph:
            if intent is None:
                if on_progress:
                    on_progress("Coordinator parsing query...")
                try:
                    with use_model(model_id):
                        intent = parse_intent(state["query"])
                    ph.cache_misses += 1
                except Exception as e:
                    raise RuntimeError(
                        "Cannot plan trip because required constraints are missing or unclear: "
                        "trip origin city (from); trip destination city (to); budget; travel dates. "
                        "Please provide these details and retry."
                    ) from e
            else:
                if on_progress:
                    on_progress("Coordinator using pre-parsed intent...")
                ph.cache_hits += 1

        violations = intent_constraint_violations(intent)
        if violations:
            raise RuntimeError(
                "Cannot plan trip because required constraints are missing: "
                + "; ".join(violations)
                + ". Please provide these details and retry."
            )

        if not bypass_date_check:
            stale_dates = _past_dates(intent.dates)
            if stale_dates:
                raise RuntimeError(
                    "Trip dates cannot be in the past. "
                    f"Found past date(s): {', '.join(stale_dates)}. "
                    "Please provide dates on or after today."
                )
        if on_progress:
            on_progress(f"Parsed: {intent.org} -> {intent.dest}, {intent.days} days")
        return {
            "intent": intent,
            "timeline": local_tl,
            "repair_notes": {},
            # rerun_specialists is intentionally absent here: None means "run all"
            # in _should_run_specialist. repair_prepare_node sets an explicit list.
            "repair_round": state.get("repair_round", 0),
        }

    def research_node(state: CoordinatorState) -> CoordinatorState:
        """Fetch all live API data once and store in state.

        Runs after parse and before the specialist fan-out so every specialist
        reads from the same pre-fetched ToolContext instead of making
        independent (redundant) API calls in parallel.

        Skipped when tool_context is already populated in state (e.g., pre-built
        from sandbox data for evaluation on historical TravelPlanner queries).
        """
        import os as _os
        local_tl = Timeline()
        intent = state["intent"]
        caps = budget.allocate(intent)

        # If tool_context was injected (e.g., from sandbox for eval), reuse it
        # but still attach per-category budget caps and cost targets (paper §4.3).
        if state.get("tool_context"):
            if on_progress:
                on_progress("Research: using pre-built tool context (sandbox eval mode)")
            with local_tl.phase("research"):
                pass  # no live API calls; phase recorded for timing completeness
            merged: dict = dict(state["tool_context"])
            if caps:
                merged["category_caps"] = caps
                merged["meal_cost_targets"] = budget.derive_meal_targets(intent)
                merged["lodging_cost_targets"] = budget.derive_lodging_targets(intent)
                merged["transport_one_way_target"] = budget.derive_transport_target(intent)
            if not merged.get("trip_windows"):
                tw = derive_trip_windows(
                    merged.get("flights_outbound") or [],
                    merged.get("flights_return") or [],
                )
                if any(v is not None for v in tw.values()):
                    merged["trip_windows"] = tw
            return {"tool_context": merged, "timeline": local_tl}

        if on_progress:
            on_progress(f"Research: fetching live data for {intent.dest}...")
        live = default_live_apis()
        depart = intent.dates[0] if intent.dates else ""
        ret = intent.dates[-1] if intent.dates else ""

        ctx: ToolContext = {}  # type: ignore[assignment]

        # ── Routes first — needed for the 100 mi drive-only threshold ─────────
        try:
            ctx["route_drive"] = live.route_summary(intent.org, intent.dest, "driving")
        except Exception:
            ctx["route_drive"] = None

        try:
            ctx["route_taxi"] = live.route_summary(intent.org, intent.dest, "taxi")
        except Exception:
            ctx["route_taxi"] = None

        # ── Flight search: drive threshold + nearest-airport fallback ─────────
        flight_warnings: list[str] = []
        serpapi_key = _os.environ.get("SERPAPI_API_KEY", "").strip()
        drive_dist_mi = _parse_drive_distance(ctx.get("route_drive"))
        dest_lat: float | None = None
        dest_lon: float | None = None

        if drive_dist_mi is not None and drive_dist_mi <= 100:
            # Short trip — drive only; skip flight search entirely.
            ctx["flights_outbound"] = []
            ctx["flights_return"] = []
            ctx["flights_round_trip"] = False
            if on_progress:
                on_progress(
                    f"Trip is {int(drive_dist_mi)} mi — "
                    "using drive-only transport (no flight search)."
                )
        elif not serpapi_key:
            # SERPAPI not configured — skip silently.
            ctx["flights_outbound"] = []
            ctx["flights_return"] = []
            ctx["flights_round_trip"] = False
        else:
            # Resolve airport codes with ≤200 mi nearest-airport fallback.
            try:
                org_iata, _org_fc, org_drive_leg, _org_lat, _org_lon = live._resolve_with_fallback(
                    intent.org, on_progress=on_progress
                )
                _dest_res = live._resolve_with_fallback(
                    intent.dest, on_progress=on_progress
                )
                dest_iata, dest_fc, dest_drive_leg, dest_lat, dest_lon = _dest_res

                # Drive-from-airport segments (after the flight) so both Day 1
                # and last-day transportation strings show the full chain when
                # the user's city has no direct airport.
                out_arrival_drive = _reverse_drive_leg(dest_drive_leg, intent.dest)
                ret_arrival_drive = _reverse_drive_leg(org_drive_leg, intent.org)

                # Try round-trip search first (9.4); fall back to two one-ways on error.
                rt_ok = False
                if depart and ret and ret != depart:
                    try:
                        out_strs, ret_strs = live.flight_search_round_trip(
                            org_iata, dest_iata, depart, ret,
                            dep_id=org_iata, arr_id=dest_iata,
                            on_progress=on_progress,
                        )
                        ctx["flights_outbound"] = [
                            _compose_flight_segments(org_drive_leg, f, out_arrival_drive)
                            for f in out_strs
                        ]
                        ctx["flights_return"] = [
                            _compose_flight_segments(dest_drive_leg, f, ret_arrival_drive)
                            for f in ret_strs
                        ]
                        ctx["flights_round_trip"] = True
                        rt_ok = True
                    except RuntimeError as e:
                        flight_warnings.append(f"round-trip search failed, using one-ways: {e}")

                if not rt_ok:
                    try:
                        out_strs = live.flight_search(
                            org_iata, dest_iata, depart,
                            prefer_early_departure=True,
                            on_progress=on_progress,
                        )
                        ctx["flights_outbound"] = [
                            _compose_flight_segments(org_drive_leg, f, out_arrival_drive)
                            for f in out_strs
                        ]
                    except RuntimeError as e:
                        ctx["flights_outbound"] = []
                        flight_warnings.append(f"outbound flight search failed: {e}")

                    try:
                        ret_strs = live.flight_search(
                            dest_iata, org_iata, ret,
                            prefer_late_departure=True,
                            on_progress=on_progress,
                        )
                        ctx["flights_return"] = [
                            _compose_flight_segments(dest_drive_leg, f, ret_arrival_drive)
                            for f in ret_strs
                        ]
                    except RuntimeError as e:
                        ctx["flights_return"] = []
                        flight_warnings.append(f"return flight search failed: {e}")

                    ctx["flights_round_trip"] = False

            except RuntimeError as e:
                ctx["flights_outbound"] = []
                ctx["flights_return"] = []
                ctx["flights_round_trip"] = False
                flight_warnings.append(f"airport resolution failed: {e}")

        if flight_warnings and on_progress:
            on_progress(
                "Flight search: " + "; ".join(flight_warnings)
                + " — transport specialist will use driving/taxi fallback."
            )

        # ── Candidate pool scaled to trip length and constraint specificity ──
        cuisine_count = len([c for c in (intent.cuisine or "").split(",") if c.strip()])
        n_hotels      = min(25, max(15, intent.days * 3))
        n_restaurants = min(50, max(16, intent.days * 4 + cuisine_count * 4))
        n_attractions = min(35, max(16, intent.days * 4))

        # Destination data — live only. The live planner never falls back to
        # sandbox: a missing live result must surface as an error so the user
        # knows live data is unavailable, instead of silently sourcing from
        # the historical TravelPlanner sandbox.
        # Hotels: prefer geo-radius search near destination coordinates when
        # available from airport resolution; fall back to text search otherwise.
        if dest_lat is not None and dest_lon is not None:
            try:
                budget_lvls = budget.price_levels_for_nightly_target(intent)
                hotels_geo = live.search_lodging_near(
                    (dest_lat, dest_lon),
                    budget_levels=budget_lvls,
                    max_results=n_hotels,
                )
            except Exception:
                hotels_geo = []
            ctx["hotels"] = hotels_geo or live.search_places(
                intent.dest, "hotels", max_results=n_hotels
            )
        else:
            ctx["hotels"] = live.search_places(intent.dest, "hotels", max_results=n_hotels)

        ctx["restaurants"] = live.search_places(
            intent.dest, "restaurants", max_results=n_restaurants
        )
        ctx["attractions"] = live.search_places(
            intent.dest, "top tourist attractions", max_results=n_attractions
        )

        if not ctx.get("hotels"):
            raise RuntimeError(
                f"No hotel results returned for '{intent.dest}'. "
                "Check GOOGLE_MAPS_API_KEY and destination spelling."
            )
        if not ctx.get("restaurants"):
            raise RuntimeError(f"No restaurant results returned for '{intent.dest}'.")
        if not ctx.get("attractions"):
            raise RuntimeError(f"No attraction results returned for '{intent.dest}'.")

        # ── Budget caps and per-category cost targets ─────────────────────────
        if caps:
            ctx["category_caps"] = caps
            ctx["meal_cost_targets"] = budget.derive_meal_targets(intent)
            ctx["lodging_cost_targets"] = budget.derive_lodging_targets(intent)
            ctx["transport_one_way_target"] = budget.derive_transport_target(intent)

        # Derive trip_windows from ranked flight lists so all specialists
        # receive timing context even when running in parallel with transport.
        tw = derive_trip_windows(
            ctx.get("flights_outbound") or [],
            ctx.get("flights_return") or [],
        )
        if any(v is not None for v in tw.values()):
            ctx["trip_windows"] = tw

        if on_progress:
            on_progress(
                f"Research complete: {len(ctx['hotels'])} hotels, "
                f"{len(ctx['restaurants'])} restaurants, "
                f"{len(ctx['attractions'])} attractions fetched."
            )
        return {"tool_context": ctx, "timeline": local_tl}

    def transport_node(state: CoordinatorState) -> CoordinatorState:
        round_no = state.get("repair_round", 0)
        phase_name = "transport" if round_no == 0 else f"transport (repair r{round_no})"
        if not _should_run_specialist(state, "transport"):
            if on_progress:
                on_progress("transport reused previous result")
            tp = state.get("transport_plan")
            if tp is not None:
                merged_ctx = _merge_trip_windows(state.get("tool_context"), tp)
                return {"tool_context": merged_ctx} if merged_ctx is not None else {}
            return {}
        local_tl = Timeline()
        with local_tl.phase(phase_name):
            tp = _run_specialist("transport", transport.run, state, on_progress)
        merged_ctx = _merge_trip_windows(state.get("tool_context"), tp)
        out: dict = {"transport_plan": tp, "timeline": local_tl}
        if merged_ctx is not None:
            out["tool_context"] = merged_ctx
        return out

    def lodging_node(state: CoordinatorState) -> CoordinatorState:
        round_no = state.get("repair_round", 0)
        phase_name = "lodging" if round_no == 0 else f"lodging (repair r{round_no})"
        if not _should_run_specialist(state, "lodging"):
            if on_progress:
                on_progress("lodging reused previous result")
            return {}
        local_tl = Timeline()
        with local_tl.phase(phase_name):
            plan = _run_specialist("lodging", lodging.run, state, on_progress)
        return {"lodging_plan": plan, "timeline": local_tl}

    def dining_node(state: CoordinatorState) -> CoordinatorState:
        round_no = state.get("repair_round", 0)
        phase_name = "dining" if round_no == 0 else f"dining (repair r{round_no})"
        if not _should_run_specialist(state, "dining"):
            if on_progress:
                on_progress("dining reused previous result")
            return {}
        local_tl = Timeline()
        with local_tl.phase(phase_name):
            plan = _run_specialist("dining", dining.run, state, on_progress)
        return {"dining_plan": plan, "timeline": local_tl}

    def sightseeing_node(state: CoordinatorState) -> CoordinatorState:
        round_no = state.get("repair_round", 0)
        phase_name = "sightseeing" if round_no == 0 else f"sightseeing (repair r{round_no})"
        if not _should_run_specialist(state, "sightseeing"):
            if on_progress:
                on_progress("sightseeing reused previous result")
            return {}
        local_tl = Timeline()
        with local_tl.phase(phase_name):
            plan = _run_specialist("sightseeing", sightseeing.run, state, on_progress)
        return {"sightseeing_plan": plan, "timeline": local_tl}

    def assemble_node(state: CoordinatorState) -> CoordinatorState:
        local_tl = Timeline()
        t = state["transport_plan"]
        l = state["lodging_plan"]
        d = state["dining_plan"]
        s = state["sightseeing_plan"]
        with local_tl.phase("assemble"):
            plan = assemble(state["intent"], t, l, d, s, state["query"])
            plan = _apply_timing_adjustments(plan, state.get("tool_context"))

        # Build attribution log (paper §4.5): tag every decision with its agent.
        decisions: list[AgentDecision] = []
        for day in t.days:
            decisions.append(AgentDecision(
                agent="transport", day=day.day,
                field="transportation", value=day.description,
            ))
        for day in l.days:
            decisions.append(AgentDecision(
                agent="lodging", day=day.day,
                field="accommodation", value=day.description,
            ))
        for day in d.days:
            decisions.append(AgentDecision(agent="dining", day=day.day, field="breakfast", value=day.breakfast))
            decisions.append(AgentDecision(agent="dining", day=day.day, field="lunch",     value=day.lunch))
            decisions.append(AgentDecision(agent="dining", day=day.day, field="dinner",    value=day.dinner))
        for day in s.days:
            decisions.append(AgentDecision(
                agent="sightseeing", day=day.day,
                field="attraction", value=day.description,
            ))

        return {"plan": plan, "attribution_log": AttributionLog(decisions=decisions), "timeline": local_tl}

    def budget_node(state: CoordinatorState) -> CoordinatorState:
        """Budget Agent (paper §4.3): evaluate cumulative spending post-assembly.

        Runs after all Worker specialists have produced their partial plans and
        the coordinator has assembled them. Stores a BudgetReport for attribution
        logging and appends a trip-level summary decision to the AttributionLog.
        """
        local_tl = Timeline()
        if on_progress:
            on_progress("Budget Agent evaluating cumulative costs...")
        with local_tl.phase("budget"):
            budget_report = budget.evaluate(state["plan"], state["intent"])

        existing_log: AttributionLog = state.get("attribution_log") or AttributionLog()
        over = budget_report.total_cost - (budget_report.budget or 0)
        summary = (
            f"${budget_report.total_cost} total "
            f"(transport:${budget_report.category_costs['transport']}, "
            f"dining:${budget_report.category_costs['dining']}, "
            f"lodging:${budget_report.category_costs['lodging']}) "
            f"/ budget:${budget_report.budget or 'none'}"
            + (f" — OVER by ${over}" if not budget_report.passed else " — within budget")
        )
        existing_log.decisions.append(AgentDecision(
            agent="budget", day=0, field="total_cost", value=summary,
        ))

        if on_progress:
            status = "within budget" if budget_report.passed else f"OVER by ${over}"
            on_progress(
                f"Budget Agent: ${budget_report.total_cost} total — {status} "
                f"(highest: {budget_report.highest_category})"
            )
        return {"budget_plan": budget_report, "attribution_log": existing_log, "timeline": local_tl}

    def verify_node(state: CoordinatorState) -> CoordinatorState:
        local_tl = Timeline()
        if on_progress:
            on_progress("Verifier checking constraints...")
        tool_ctx = state.get("tool_context") or {}
        trip_windows = tool_ctx.get("trip_windows")
        flights_round_trip = bool(tool_ctx.get("flights_round_trip", False))
        with local_tl.phase("verify"):
            report = verifier.verify(
                state["plan"], state["intent"], None,
                trip_windows=trip_windows,
                flights_round_trip=flights_round_trip,
            )
        return {"report": report, "timeline": local_tl}

    def repair_or_finalize(state: CoordinatorState) -> str:
        report = state["report"]
        if not report.passed and state.get("repair_round", 0) < REPAIR_MAX_ROUNDS:
            return "repair"
        return "finalize"

    def repair_prepare_node(state: CoordinatorState) -> CoordinatorState:
        report = state["report"]
        repair_notes: dict[str, str] = {}
        rerun: set[str] = set()
        budget_report: BudgetReport | None = state.get("budget_plan")
        tool_context: ToolContext = state.get("tool_context") or {}  # type: ignore[assignment]
        prev_keys = set(state.get("prev_violation_keys") or [])
        cur_keys: list[str] = []

        # ── Deterministic pre-repair: fix mechanical violations without LLM ────
        # Diversity rules (duplicate venues) are fixed by substituting unused
        # venues from ToolContext. Only skip LLM re-run when ALL violations for
        # that specialist are mechanical — if there's also a budget or cuisine
        # violation, the specialist must still re-run to address it.
        _MECH_RULES = {"diverse_restaurants", "diverse_attractions"}
        specialist_rules: dict[str, set[str]] = {
            "dining": set(), "sightseeing": set(),
        }
        for v in report.violations:
            if v.responsible in specialist_rules:
                specialist_rules[v.responsible].add(v.rule)

        mech_updates: dict = {}
        mech_fixed: set[str] = set()

        dp = state.get("dining_plan")
        if (
            dp is not None
            and specialist_rules["dining"]
            and specialist_rules["dining"].issubset(_MECH_RULES)
        ):
            new_dp, fixed = _mech_repair_dining(dp, tool_context)
            if fixed:
                mech_updates["dining_plan"] = new_dp
                mech_fixed.add("dining")
                if on_progress:
                    on_progress("Pre-repair: fixed duplicate restaurants deterministically.")

        sp = state.get("sightseeing_plan")
        if (
            sp is not None
            and specialist_rules["sightseeing"]
            and specialist_rules["sightseeing"].issubset(_MECH_RULES)
        ):
            new_sp, fixed = _mech_repair_sightseeing(sp, tool_context)
            if fixed:
                mech_updates["sightseeing_plan"] = new_sp
                mech_fixed.add("sightseeing")
                if on_progress:
                    on_progress("Pre-repair: fixed duplicate attractions deterministically.")

        for v in report.violations:
            key = f"{v.rule}|{v.responsible}"
            cur_keys.append(key)

            # Skip LLM routing for mechanically-fixed violations.
            if v.responsible in mech_fixed and v.rule in _MECH_RULES:
                continue

            is_repeat = key in prev_keys
            escalate_prefix = "[URGENT — prior repair did NOT resolve this] " if is_repeat else ""

            detail = v.detail
            # Budget-specific: attach explicit overage target so specialist can act on it.
            if v.rule == "budget" and budget_report is not None and budget_report.budget:
                over = budget_report.total_cost - budget_report.budget
                cat = budget_report.highest_category
                detail = (
                    f"{detail}. Target: reduce {cat} spending by at least ${over}. "
                    f"Category costs: {budget_report.category_costs}."
                )

            # On repeat failures, instruct specialists to pick a different option.
            if is_repeat and v.responsible in {"transport", "lodging", "dining", "sightseeing"}:
                detail += " Previous choice was rejected — pick a DIFFERENT option this time."

            prev = repair_notes.get(v.responsible, "")
            repair_notes[v.responsible] = (
                prev + f" {escalate_prefix}[{v.rule}] {detail}"
            ).strip()

            # Targeted routing — budget violations go only to worst category's specialist.
            if v.rule == "budget" and budget_report is not None:
                worst = budget_report.highest_category
                mapping = {"transport": "transport", "dining": "dining", "lodging": "lodging"}
                target = mapping.get(worst)
                if target:
                    rerun.add(target)
                    continue

            if v.responsible == "coordinator":
                # coordinator-level violations (complete_information, missing_cost)
                # are mechanical — no single specialist owns them and an LLM re-run
                # of all 4 rarely fixes them while wasting 3-4 calls per round.
                # Skip the LLM rerun; the mechanical pre-repair pass above handles
                # diversity; cost/completeness issues surface in the repair note.
                pass
            elif v.responsible in {"transport", "lodging", "dining", "sightseeing"}:
                rerun.add(v.responsible)

        # Remove mechanically-fixed specialists from the LLM rerun list so they
        # preserve their patched plans through the next assemble pass.
        rerun -= mech_fixed

        rerun_list = sorted(rerun)  # may be [] if all violations were mechanical
        cur_round = state.get("repair_round", 0) + 1
        if on_progress:
            mech_note = (
                f"; {len(mech_fixed)} fixed deterministically"
                if mech_fixed else ""
            )
            on_progress(
                f"Repair pass {cur_round}/{REPAIR_MAX_ROUNDS} "
                f"({len(report.violations)} violations; rerun: "
                f"{', '.join(rerun_list) or 'none (all mechanical)'}{mech_note})..."
            )
        return {
            "repair_notes": repair_notes,
            "rerun_specialists": rerun_list,
            "repair_round": cur_round,
            "prev_violation_keys": cur_keys,
            **mech_updates,
        }

    def finalize_node(state: CoordinatorState) -> CoordinatorState:
        if on_progress:
            on_progress("Final agent compiling the response...")
        return {}

    graph = StateGraph(CoordinatorState)
    graph.add_node("parse", parse_node)
    graph.add_node("research", research_node)
    graph.add_node("transport", transport_node)
    graph.add_node("lodging", lodging_node)
    graph.add_node("dining", dining_node)
    graph.add_node("sightseeing", sightseeing_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("budget", budget_node)   # Budget Agent (paper §4.3)
    graph.add_node("verify", verify_node)
    graph.add_node("repair_prepare", repair_prepare_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "parse")
    # Research always runs once after parse; repair skips it (tool_context stays in state).
    graph.add_edge("parse", "research")

    if execution_mode == "sequential":
        graph.add_edge("research", "transport")
        graph.add_edge("transport", "lodging")
        graph.add_edge("lodging", "dining")
        graph.add_edge("dining", "sightseeing")
        graph.add_edge("sightseeing", "assemble")

        graph.add_edge("repair_prepare", "transport")
    else:
        # Full parallel fan-out: research_node pre-populates trip_windows from
        # flights_outbound[0]/flights_return[0] so all four specialists receive
        # timing context (arrival/departure grid) without waiting for transport.
        # transport_node still calls _merge_trip_windows to update the context
        # with the specialist's actual flight pick (used by the verifier).
        for node in ("transport", "lodging", "dining", "sightseeing"):
            graph.add_edge("research", node)
            graph.add_edge(node, "assemble")
            graph.add_edge("repair_prepare", node)

    # Budget Agent always runs after assembly (paper §4.3).
    graph.add_edge("assemble", "budget")

    if skip_verify:
        # Baseline 2: multi-agent without verification (paper §4.7).
        graph.add_edge("budget", "finalize")
    else:
        graph.add_edge("budget", "verify")
        graph.add_conditional_edges(
            "verify",
            repair_or_finalize,
            {
                "repair": "repair_prepare",
                "finalize": "finalize",
            },
        )

    graph.add_edge("finalize", END)
    return graph.compile()


def build_live_tool_context(
    intent: Intent,
    on_progress: Callable[[str], None] | None = None,
) -> ToolContext:
    """Fetch live API data for *intent* and return a populated ToolContext.

    Shared by both the multi-agent coordinator (via research_node) and the
    single-agent baseline so all app-path executions use live data.
    """
    import os

    if on_progress:
        on_progress(f"Research: fetching live data for {intent.dest}...")
    live = default_live_apis()
    depart = intent.dates[0] if intent.dates else ""
    ret    = intent.dates[-1] if intent.dates else ""

    ctx: ToolContext = {}  # type: ignore[assignment]

    try:
        ctx["route_drive"] = live.route_summary(intent.org, intent.dest, "driving")
    except Exception:
        ctx["route_drive"] = None
    try:
        ctx["route_taxi"] = live.route_summary(intent.org, intent.dest, "taxi")
    except Exception:
        ctx["route_taxi"] = None

    flight_warnings: list[str] = []
    serpapi_key  = os.environ.get("SERPAPI_API_KEY", "").strip()
    drive_dist_mi = _parse_drive_distance(ctx.get("route_drive"))
    dest_lat: float | None = None
    dest_lon: float | None = None

    if drive_dist_mi is not None and drive_dist_mi <= 100:
        ctx["flights_outbound"] = []
        ctx["flights_return"]   = []
        ctx["flights_round_trip"] = False
        if on_progress:
            on_progress(
                f"Trip is {int(drive_dist_mi)} mi — "
                "using drive-only transport (no flight search)."
            )
    elif not serpapi_key:
        ctx["flights_outbound"] = []
        ctx["flights_return"]   = []
        ctx["flights_round_trip"] = False
    else:
        try:
            org_iata, _, org_drive_leg, _, _ = live._resolve_with_fallback(
                intent.org, on_progress=on_progress
            )
            dest_iata, _, dest_drive_leg, dest_lat, dest_lon = live._resolve_with_fallback(
                intent.dest, on_progress=on_progress
            )
            out_arrival_drive = _reverse_drive_leg(dest_drive_leg, intent.dest)
            ret_arrival_drive = _reverse_drive_leg(org_drive_leg, intent.org)

            rt_ok = False
            if depart and ret and ret != depart:
                try:
                    out_strs, ret_strs = live.flight_search_round_trip(
                        org_iata, dest_iata, depart, ret,
                        dep_id=org_iata, arr_id=dest_iata,
                        on_progress=on_progress,
                    )
                    ctx["flights_outbound"] = [
                        _compose_flight_segments(org_drive_leg, f, out_arrival_drive)
                        for f in out_strs
                    ]
                    ctx["flights_return"] = [
                        _compose_flight_segments(dest_drive_leg, f, ret_arrival_drive)
                        for f in ret_strs
                    ]
                    ctx["flights_round_trip"] = True
                    rt_ok = True
                except RuntimeError as e:
                    flight_warnings.append(f"round-trip search failed, using one-ways: {e}")

            if not rt_ok:
                try:
                    out_strs = live.flight_search(
                        org_iata, dest_iata, depart,
                        prefer_early_departure=True, on_progress=on_progress,
                    )
                    ctx["flights_outbound"] = [
                        _compose_flight_segments(org_drive_leg, f, out_arrival_drive)
                        for f in out_strs
                    ]
                except RuntimeError as e:
                    ctx["flights_outbound"] = []
                    flight_warnings.append(f"outbound flight search failed: {e}")
                try:
                    ret_strs = live.flight_search(
                        dest_iata, org_iata, ret,
                        prefer_late_departure=True, on_progress=on_progress,
                    )
                    ctx["flights_return"] = [
                        _compose_flight_segments(dest_drive_leg, f, ret_arrival_drive)
                        for f in ret_strs
                    ]
                except RuntimeError as e:
                    ctx["flights_return"] = []
                    flight_warnings.append(f"return flight search failed: {e}")
                ctx["flights_round_trip"] = False

        except RuntimeError as e:
            ctx["flights_outbound"] = []
            ctx["flights_return"]   = []
            ctx["flights_round_trip"] = False
            flight_warnings.append(f"airport resolution failed: {e}")

    if flight_warnings and on_progress:
        on_progress(
            "Flight search: " + "; ".join(flight_warnings)
            + " — transport specialist will use driving/taxi fallback."
        )

    cuisine_count = len([c for c in (intent.cuisine or "").split(",") if c.strip()])
    n_hotels      = min(25, max(15, intent.days * 3))
    n_restaurants = min(50, max(16, intent.days * 4 + cuisine_count * 4))
    n_attractions = min(35, max(16, intent.days * 4))

    if dest_lat is not None and dest_lon is not None:
        try:
            budget_lvls = budget.price_levels_for_nightly_target(intent)
            hotels_geo  = live.search_lodging_near(
                (dest_lat, dest_lon),
                budget_levels=budget_lvls,
                max_results=n_hotels,
            )
        except Exception:
            hotels_geo = []
        ctx["hotels"] = hotels_geo or live.search_places(
            intent.dest, "hotels", max_results=n_hotels
        )
    else:
        ctx["hotels"] = live.search_places(intent.dest, "hotels", max_results=n_hotels)

    ctx["restaurants"] = live.search_places(
        intent.dest, "restaurants", max_results=n_restaurants
    )
    ctx["attractions"] = live.search_places(
        intent.dest, "top tourist attractions", max_results=n_attractions
    )

    if not ctx.get("hotels"):
        raise RuntimeError(
            f"No hotel results returned for '{intent.dest}'. "
            "Check GOOGLE_MAPS_API_KEY and destination spelling."
        )
    if not ctx.get("restaurants"):
        raise RuntimeError(f"No restaurant results returned for '{intent.dest}'.")
    if not ctx.get("attractions"):
        raise RuntimeError(f"No attraction results returned for '{intent.dest}'.")

    caps = budget.allocate(intent)
    if caps:
        ctx["category_caps"]          = caps
        ctx["meal_cost_targets"]       = budget.derive_meal_targets(intent)
        ctx["lodging_cost_targets"]    = budget.derive_lodging_targets(intent)
        ctx["transport_one_way_target"] = budget.derive_transport_target(intent)

    tw = derive_trip_windows(
        ctx.get("flights_outbound") or [],
        ctx.get("flights_return") or [],
    )
    if any(v is not None for v in tw.values()):
        ctx["trip_windows"] = tw

    if on_progress:
        on_progress(
            f"Research complete: {len(ctx['hotels'])} hotels, "
            f"{len(ctx['restaurants'])} restaurants, "
            f"{len(ctx['attractions'])} attractions fetched."
        )
    return ctx


def plan_trip(
    query: str,
    on_progress: Callable[[str], None] | None = None,
    execution_mode: str = "parallel",
    parsed_intent: Intent | None = None,
    skip_verify: bool = False,
    bypass_date_check: bool = False,
    tool_context: ToolContext | None = None,
    model: str | None = None,
) -> tuple[FullPlan, VerifierReport]:
    """Main entry point. Returns (plan, final verifier report).

    skip_verify=True       — disables verify-and-repair loop (paper §4.7 Baseline 2).
    bypass_date_check=True — skips past-date rejection for historical datasets.
    tool_context           — pre-built ToolContext (e.g., from sandbox); research skipped.
    model                  — override the active model (falls back to LLM_MODEL env or default).
    """
    from agents.models import default_model_id, get_model
    model_id = model or current_model() or default_model_id()

    # Claude Vertex quota (TPM/RPM) is tight; parallel fan-out spikes usage and
    # triggers 429s. Force sequential execution so specialists queue their calls.
    try:
        _provider = get_model(model_id).provider
    except KeyError:
        _provider = "unknown"
    if _provider == "claude" and execution_mode == "parallel":
        execution_mode = "sequential"

    mode = "sequential" if execution_mode == "sequential" else "parallel"
    app = _build_graph(
        mode, on_progress,
        skip_verify=skip_verify,
        bypass_date_check=bypass_date_check,
    )
    init_state: CoordinatorState = {
        "query": query,
        "execution_mode": mode,
        "model_id": model_id,
        "timeline": Timeline(),
    }
    if parsed_intent is not None:
        init_state["intent"] = parsed_intent
    if tool_context is not None:
        init_state["tool_context"] = tool_context

    with use_model(model_id):
        final_state = app.invoke(init_state)

    report = final_state.get("report") or VerifierReport(passed=True, violations=[])
    return final_state["plan"], report
