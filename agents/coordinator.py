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
from tools.sandbox import default_sandbox


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
    "IMPORTANT: If the user did NOT state an unambiguous trip START date, return "
    "an empty `dates` list. Do not guess a start date. Do not use today's date."
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


REPAIR_MAX_ROUNDS = 3


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


def _sandbox_fallback(city: str, kind: str, limit: int = 12) -> list[dict]:
    """Return sandbox rows for `city` shaped like Google Places results.

    Used when the live Google Places API returns nothing for a valid city so
    planning can still proceed with reasonable venue data instead of failing
    the whole query (which would collapse delivery_rate).
    """
    try:
        sb = default_sandbox("frozen")
    except Exception:
        return []

    if kind == "hotels":
        rows = sb.accommodations.get(city, [])
        out: list[dict] = []
        for r in rows[:limit]:
            out.append({
                "name": r.get("NAME"),
                "rating": r.get("review rate number"),
                "address": f"{city}",
                "price_level": 2,
            })
        return [r for r in out if r["name"]]

    if kind == "restaurants":
        rows = sb.restaurants.get(city, [])
        out = []
        for r in rows[:limit]:
            out.append({
                "name": r.get("Name"),
                "rating": r.get("Aggregate Rating"),
                "address": f"{city}",
                "price_level": 2,
            })
        return [r for r in out if r["name"]]

    if kind == "attractions":
        rows = sb.attractions.get(city, [])
        out = []
        for r in rows[:limit]:
            out.append({
                "name": r.get("Name"),
                "rating": None,
                "address": f"{r.get('City') or city}",
            })
        return [r for r in out if r["name"]]

    return []


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
    # when a specialist disregarded it (defense-in-depth).
    day1 = days[0]
    arrival_min = _extract_arrival_minutes(day1.transportation)
    if arrival_min is not None:
        updates: dict = {}
        arr = _fmt_time(arrival_min)
        if arrival_min >= THRESH_D1_LUNCH_NO:
            if day1.lunch and day1.lunch != "-":
                updates["lunch"] = f"In transit (arrives {arr})"
        if arrival_min >= THRESH_D1_ATTRACTION_NO:
            if day1.attraction and day1.attraction != "-":
                updates["attraction"] = "-"
        if arrival_min >= THRESH_D1_DINNER_NO:
            if day1.dinner and day1.dinner != "-":
                updates["dinner"] = f"Late arrival {arr} — hotel dinner / room service recommended"
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

            # Very late departure — restore dinner if specialist left it blank.
            if dep_min >= THRESH_LAST_DINNER_YES and (not last.dinner or last.dinner == "-"):
                choice = _pick_unused_restaurant(tool_context, used_restaurants)
                updates["dinner"] = choice or f"Dinner before {_fmt_time(dep_min)} departure"

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
    rerun = state.get("rerun_specialists") or []
    return not rerun or name in rerun


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
            "rerun_specialists": ["transport", "lodging", "dining", "sightseeing"],
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
            if caps:
                merged: dict = dict(state["tool_context"])
                merged["category_caps"] = caps
                merged["meal_cost_targets"] = budget.derive_meal_targets(intent)
                merged["lodging_cost_targets"] = budget.derive_lodging_targets(intent)
                merged["transport_one_way_target"] = budget.derive_transport_target(intent)
                return {"tool_context": merged, "timeline": local_tl}
            return {"timeline": local_tl}

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
                org_iata, _org_fc, org_drive_leg = live._resolve_with_fallback(intent.org)
                dest_iata, dest_fc, dest_drive_leg = live._resolve_with_fallback(intent.dest)

                if dest_fc and on_progress:
                    on_progress(
                        f"No airport in {intent.dest}; nearest is {dest_iata} "
                        "— flights include drive transfer."
                    )

                # Try round-trip search first (9.4); fall back to two one-ways on error.
                rt_ok = False
                if depart and ret and ret != depart:
                    try:
                        out_strs, ret_strs = live.flight_search_round_trip(
                            org_iata, dest_iata, depart, ret,
                            dep_id=org_iata, arr_id=dest_iata,
                        )
                        ctx["flights_outbound"] = [
                            f"{org_drive_leg}; {f}" for f in out_strs
                        ] if org_drive_leg else out_strs
                        ctx["flights_return"] = [
                            f"{dest_drive_leg}; {f}" for f in ret_strs
                        ] if dest_drive_leg else ret_strs
                        ctx["flights_round_trip"] = True
                        rt_ok = True
                        if on_progress:
                            on_progress("Round-trip flight search succeeded.")
                    except RuntimeError as e:
                        flight_warnings.append(f"round-trip search failed, using one-ways: {e}")

                if not rt_ok:
                    try:
                        ctx["flights_outbound"] = live.flight_search(
                            org_iata, dest_iata, depart,
                        )
                        if org_drive_leg:
                            ctx["flights_outbound"] = [
                                f"{org_drive_leg}; {f}" for f in ctx["flights_outbound"]
                            ]
                    except RuntimeError as e:
                        ctx["flights_outbound"] = []
                        flight_warnings.append(f"outbound flight search failed: {e}")

                    try:
                        ctx["flights_return"] = live.flight_search(
                            dest_iata, org_iata, ret,
                            prefer_late_departure=True,
                        )
                        if dest_drive_leg:
                            ctx["flights_return"] = [
                                f"{dest_drive_leg}; {f}" for f in ctx["flights_return"]
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

        # ── Candidate pool scaled to trip length ─────────────────────────────
        n_hotels      = min(20, max(8,  intent.days * 2))
        n_restaurants = min(40, max(12, intent.days * 4))
        n_attractions = min(30, max(12, intent.days * 3))

        # Destination data — live first, sandbox fallback if empty so delivery_rate
        # does not collapse when Google Maps returns nothing for a valid city.
        ctx["hotels"] = live.search_places(intent.dest, "hotels", max_results=n_hotels)
        ctx["restaurants"] = live.search_places(
            intent.dest, "restaurants", max_results=n_restaurants
        )
        ctx["attractions"] = live.search_places(
            intent.dest, "top tourist attractions", max_results=n_attractions
        )

        degraded: list[str] = []
        if not ctx.get("hotels"):
            fb = _sandbox_fallback(intent.dest, "hotels", limit=n_hotels)
            if fb:
                ctx["hotels"] = fb
                degraded.append("hotels")
        if not ctx.get("restaurants"):
            fb = _sandbox_fallback(intent.dest, "restaurants", limit=n_restaurants)
            if fb:
                ctx["restaurants"] = fb
                degraded.append("restaurants")
        if not ctx.get("attractions"):
            fb = _sandbox_fallback(intent.dest, "attractions", limit=n_attractions)
            if fb:
                ctx["attractions"] = fb
                degraded.append("attractions")

        if not ctx.get("hotels"):
            raise RuntimeError(
                f"No hotel results returned for '{intent.dest}' (live + sandbox both empty). "
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

        if on_progress:
            note = f" (sandbox fallback for: {', '.join(degraded)})" if degraded else ""
            on_progress(
                f"Research complete: {len(ctx['hotels'])} hotels, "
                f"{len(ctx['restaurants'])} restaurants, "
                f"{len(ctx['attractions'])} attractions fetched{note}."
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
        prev_keys = set(state.get("prev_violation_keys") or [])
        cur_keys: list[str] = []

        for v in report.violations:
            key = f"{v.rule}|{v.responsible}"
            cur_keys.append(key)
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
                # coordinator-level violations (e.g., missing_cost, completeness) touch
                # whichever specialist owns the missing field; rerun all as a safe default.
                rerun.update({"transport", "lodging", "dining", "sightseeing"})
            elif v.responsible in {"transport", "lodging", "dining", "sightseeing"}:
                rerun.add(v.responsible)

        # `rerun` is guaranteed non-empty here: verify only routes here when
        # violations exist, and every Violation.responsible maps to at least
        # one specialist (budget → worst category, coordinator → all four).
        rerun_list = sorted(rerun)
        cur_round = state.get("repair_round", 0) + 1
        if on_progress:
            on_progress(
                f"Repair pass {cur_round}/{REPAIR_MAX_ROUNDS} "
                f"({len(report.violations)} violations; rerun: {', '.join(rerun_list)})..."
            )
        return {
            "repair_notes": repair_notes,
            "rerun_specialists": rerun_list,
            "repair_round": cur_round,
            "prev_violation_keys": cur_keys,
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
        # Transport-first fan-out: downstream specialists see arrival/departure
        # windows in tool_context before they plan meals/attractions/lodging.
        graph.add_edge("research", "transport")
        graph.add_edge("repair_prepare", "transport")  # repair also starts with transport
        for node in ("lodging", "dining", "sightseeing"):
            graph.add_edge("transport", node)
            graph.add_edge(node, "assemble")

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
    from agents.models import default_model_id
    model_id = model or current_model() or default_model_id()

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
