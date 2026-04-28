"""Streamlit chat UI for the Multi-Agent Travel Itinerary Planner.

Run:  streamlit run app.py
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents import budget as _budget_agent
from agents.coordinator import intent_constraint_violations, parse_intent, plan_trip
from agents.models import available_models, default_model_id
from agents.runtime_status import flights_available, llm_available, places_available
from agents.schemas import BudgetReport, FullPlan, Intent, VerifierReport
from baseline.single_agent import plan_trip_single


load_dotenv(Path(__file__).resolve().parent / ".env")


# --------- page ---------
st.set_page_config(page_title="Multi-Agent Travel Planner", layout="wide")
st.title("Multi-Agent Travel Itinerary Planner")
st.caption("CSE 572 Spring 2026 — live API planner with coordinator + specialist sub-agents.")

# --------- sidebar ---------
with st.sidebar:
    st.header("Settings")
    planning_mode = st.radio(
        "Planning mode",
        ["Multi-agent (coordinator + specialists)", "Single-agent (one prompt)"],
        index=0,
        help="Multi-agent: coordinator fans out to 4 specialist sub-agents with verify+repair. "
             "Single-agent: one LLM call with all tool data — useful as a baseline comparison.",
    )
    is_multi_agent = planning_mode.startswith("Multi")

    if is_multi_agent:
        execution_mode = st.radio(
            "Sub-agent execution",
            ["single agent at a time (sequential)", "multiple agents in parallel"],
            index=0,
        )
        execution_mode_arg = "sequential" if execution_mode.startswith("single") else "parallel"
    else:
        execution_mode_arg = "sequential"

    st.divider()
    _avail_models = available_models()
    _default_id = default_model_id()
    if _avail_models:
        _model_options = {m.display_name: m.id for m in _avail_models}
        _default_name = next(
            (m.display_name for m in _avail_models if m.id == _default_id),
            list(_model_options)[0],
        )
        _selected_name = st.selectbox(
            "LLM model",
            options=list(_model_options),
            index=list(_model_options).index(_default_name),
            help="Only models whose auth is configured are shown. "
                 "Claude entries require GOOGLE_CLOUD_PROJECT + CLAUDE_VERTEX_REGION. "
                 "GPT/OpenAI is excluded: Vertex AI does not host OpenAI models.",
        )
        selected_model_id = _model_options[_selected_name]
        # Reset cached plan state when model changes to avoid cross-model contamination.
        if st.session_state.get("_last_model_id") != selected_model_id:
            st.session_state["_last_model_id"] = selected_model_id
    else:
        st.warning("No models available. Check VERTEX_AI_API_KEY in .env.")
        selected_model_id = _default_id

    st.divider()
    st.subheader("Live data sources")
    _ok = ":green[✓ active]"
    _missing = ":orange[✗ not configured]"
    st.markdown(f"- **LLM planning:** {_ok if llm_available() else _missing}")
    st.markdown(f"- **Flight search:** {_ok if flights_available() else _missing}")
    st.markdown(f"- **Places & maps:** {_ok if places_available() else _missing}")
    st.caption("Run `scripts/validate_env.py` to check configuration.")


# --------- chat state ---------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None
if "awaiting_details" not in st.session_state:
    st.session_state["awaiting_details"] = False

# Replay history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# --------- render helpers ---------
_EMPTY_TOKENS = {"", "-", "none", "null", "n/a", "unknown", "unspecified"}


def _clean_text(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text or "-"


def _is_missing_text(value: object) -> bool:
    return _clean_text(value).lower() in _EMPTY_TOKENS


def _empty_reason(field: str, day_no: int, total_days: int) -> str:
    if field == "transportation":
        if day_no in {1, total_days}:
            return "No reliable live transfer option matched the final constraints for this travel leg."
        return "No inter-city transfer is required for an in-city day."
    if field == "breakfast":
        if day_no == 1:
            return "Arrival-day morning is typically before destination check-in window."
        return "No restaurant candidate met constraints with high confidence for this slot."
    if field == "lunch":
        return "No restaurant candidate met constraints with high confidence for this slot."
    if field == "dinner":
        if day_no == total_days:
            return "Departure-day evening is reserved for return travel."
        return "No restaurant candidate met constraints with high confidence for this slot."
    if field == "attraction":
        if day_no == total_days:
            return "Departure day is kept lighter to reduce missed-return risk."
        return "No attraction option met timing/constraint checks strongly enough."
    if field == "accommodation":
        if day_no == total_days:
            return "Final day returns home, so no overnight stay is planned."
        return "No lodging option met all constraints strongly enough for this night."
    return "No value was produced for this field."


def _empty_display_text(field: str, day_no: int, total_days: int) -> str:
    if field == "transportation":
        if day_no in {1, total_days}:
            return "Transfer details unavailable"
        return "No inter-city transfer needed"
    if field == "breakfast":
        if day_no == 1:
            return "Flexible breakfast (arrival day)"
        return "No confirmed breakfast option"
    if field == "lunch":
        return "No confirmed lunch option"
    if field == "dinner":
        if day_no == total_days:
            return "Return-travel evening"
        return "No confirmed dinner option"
    if field == "attraction":
        if day_no == total_days:
            return "Light schedule for departure day"
        return "No confirmed attraction option"
    if field == "accommodation":
        if day_no == total_days:
            return "No stay required (return day)"
        return "No confirmed accommodation option"
    return "Details pending"


def _render_field(label: str, field_key: str, value: object, day_no: int, total_days: int) -> None:
    clean = _clean_text(value)
    if _is_missing_text(clean):
        reason = _empty_reason(field_key, day_no, total_days)
        display_text = _empty_display_text(field_key, day_no, total_days)
        st.markdown(f"**{label}:** {display_text}")
        st.caption(f"Reason: {reason}")
        return
    st.markdown(f"**{label}:** {clean}")


def _val(text: object) -> str:
    """Clean a plan field value, returning '-' for empty/missing."""
    return _clean_text(text)


def render_plan(
    plan: FullPlan,
    title: str,
    intent: "Intent | None" = None,
    budget_report: "BudgetReport | None" = None,
) -> None:
    st.subheader(title)
    total_days = len(plan.plan)
    org = (intent.org if intent else None) or "Origin"
    dest = (intent.dest if intent else None) or "Destination"
    budget = (intent.budget if intent else None)

    # ── (A) Trip Summary card ─────────────────────────────────────────────────
    with st.container(border=True):
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Trip", f"{org} → {dest}")
        r1c2.metric("Days", total_days)
        r1c3.metric("People", intent.people if intent else 1)

        if budget_report:
            total = budget_report.total_cost
            r2c1, r2c2, r2c3 = st.columns(3)
            delta = total - (budget or 0) if budget else None
            r2c1.metric(
                "💰 Total cost",
                f"${total:,}",
                delta=f"${delta:+,} vs budget" if delta is not None else None,
                delta_color="inverse",
            )
            r2c2.metric("Budget", f"${budget:,}" if budget else "—")
            r2c3.metric(
                "Highest category",
                budget_report.highest_category.title(),
                f"${budget_report.category_costs.get(budget_report.highest_category, 0):,}",
            )

            # Budget gauge
            if budget and budget > 0:
                ratio = min(1.0, total / budget)
                if ratio <= 0.9:
                    color_label = ":green[Within budget]"
                elif ratio <= 1.0:
                    color_label = ":orange[Near budget limit]"
                else:
                    color_label = ":red[**OVER BUDGET**]"
                st.caption(color_label)
                st.progress(ratio)

    # ── (B) Cost breakdown bar ────────────────────────────────────────────────
    if budget_report and budget_report.category_costs:
        cats = budget_report.category_costs
        total_cat = sum(cats.values()) or 1
        b1, b2, b3 = st.columns(3)
        b1.metric("✈️ Transport", f"${cats.get('transport', 0):,}",
                  f"{cats.get('transport', 0) / total_cat * 100:.0f}%")
        b2.metric("🏨 Lodging", f"${cats.get('lodging', 0):,}",
                  f"{cats.get('lodging', 0) / total_cat * 100:.0f}%")
        b3.metric("🍽️ Dining", f"${cats.get('dining', 0):,}",
                  f"{cats.get('dining', 0) / total_cat * 100:.0f}%")

    st.divider()

    # ── (D) Per-day timeline ──────────────────────────────────────────────────
    for day in plan.plan:
        is_travel_day = day.days in {1, total_days}
        header = f"**Day {day.days}** — {day.current_city}"
        with st.container(border=True):
            st.markdown(header)
            if is_travel_day and not _is_missing_text(day.transportation):
                st.caption(f"🚀 {_val(day.transportation)}")

            col_t, col_s, col_stay = st.columns([1, 2, 1])

            with col_t:
                st.markdown("**✈️ Transport**")
                if _is_missing_text(day.transportation):
                    st.caption(_empty_display_text("transportation", day.days, total_days))
                else:
                    st.markdown(_val(day.transportation))

            with col_s:
                st.markdown("**📅 Schedule**")

                def _sched_row(icon: str, label: str, field: str, value: object) -> None:
                    clean = _val(value)
                    if _is_missing_text(clean):
                        st.markdown(
                            f"{icon} **{label}:** "
                            f":gray[{_empty_display_text(field, day.days, total_days)}]"
                        )
                    else:
                        st.markdown(f"{icon} **{label}:** {clean}")

                _sched_row("🍳", "Breakfast", "breakfast", day.breakfast)
                _sched_row("🥗", "Lunch", "lunch", day.lunch)
                _sched_row("🎟️", "Attraction", "attraction", day.attraction)
                _sched_row("🍽️", "Dinner", "dinner", day.dinner)

            with col_stay:
                st.markdown("**🏨 Stay**")
                if _is_missing_text(day.accommodation):
                    st.caption(_empty_display_text("accommodation", day.days, total_days))
                else:
                    st.markdown(_val(day.accommodation))


def _missing_details(query: str) -> tuple[list[str], Intent | None]:
    """Parse with coordinator LLM and return missing required constraints."""
    try:
        intent = parse_intent(query)
    except Exception:
        return [
            "trip from (origin city)",
            "trip to (destination city)",
            "budget",
            "trip start date (e.g. 'March 15')",
        ], None

    return intent_constraint_violations(intent), intent


def render_verifier(report: VerifierReport) -> None:
    if report.passed:
        st.success(f"✅ All {len(report.violations) == 0 and 'constraints' or '?'} satisfied"
                   if report.violations == [] else "✅ All constraints satisfied")
    else:
        st.error(f"❌ {len(report.violations)} violation(s) found")
        by_agent: dict[str, list] = {}
        for v in report.violations:
            by_agent.setdefault(v.responsible, []).append(v)
        with st.expander("Violations (grouped by specialist)", expanded=True):
            for agent, vlist in by_agent.items():
                st.markdown(f"**{agent.title()} specialist**")
                for v in vlist:
                    st.markdown(f"  - `{v.rule}`: {v.detail}")


# --------- chat input ---------
user_query = st.chat_input("Plan a 3-day trip from Washington to Myrtle Beach under $1400…")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    run_query: str | None = None
    parsed_intent: Intent | None = None
    if st.session_state["awaiting_details"] and st.session_state["pending_query"]:
        candidate = (
            f"{st.session_state['pending_query']}\n\n"
            f"Additional details from user: {user_query}"
        )
        still_missing, candidate_intent = _missing_details(candidate)
        if still_missing:
            ask = (
                "I still need a few details before I can run all specialists:\n\n"
                + "\n".join(f"- {m}" for m in still_missing)
                + "\n\nPlease reply with these details and I will continue immediately."
            )
            with st.chat_message("assistant"):
                st.info(ask)
            st.session_state["messages"].append({"role": "assistant", "content": ask})
            st.session_state["pending_query"] = candidate
        else:
            run_query = candidate
            parsed_intent = candidate_intent
            st.session_state["awaiting_details"] = False
            st.session_state["pending_query"] = None
    else:
        missing, user_intent = _missing_details(user_query)
        if missing:
            ask = (
                "Before I run the multi-agent pipeline, I need:\n\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\n\nPlease provide these in one message and I will continue."
            )
            with st.chat_message("assistant"):
                st.info(ask)
            st.session_state["messages"].append({"role": "assistant", "content": ask})
            st.session_state["awaiting_details"] = True
            st.session_state["pending_query"] = user_query
        else:
            run_query = user_query
            parsed_intent = user_intent

    if run_query:
        progress_box = st.empty()
        log: list[str] = []
        log_lock = threading.Lock()
        result: dict[str, object] = {}
        done = threading.Event()

        def on_progress(msg: str) -> None:
            with log_lock:
                log.append(msg)

        def _worker() -> None:
            try:
                if is_multi_agent:
                    plan, report = plan_trip(
                        run_query,
                        on_progress=on_progress,
                        execution_mode=execution_mode_arg,
                        parsed_intent=parsed_intent,
                        model=selected_model_id,
                    )
                else:
                    plan, report = plan_trip_single(
                        run_query,
                        on_progress=on_progress,
                        parsed_intent=parsed_intent,
                        model=selected_model_id,
                    )
                result["plan"] = plan
                result["report"] = report
            except Exception as e:
                result["error"] = e
            finally:
                done.set()

        with st.chat_message("assistant"):
            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            while not done.is_set():
                with log_lock:
                    snapshot = list(log)
                if snapshot:
                    progress_box.info("\n\n".join(f"• {m}" for m in snapshot[-10:]))
                else:
                    progress_box.info("• Initializing coordinator and specialists...")
                time.sleep(0.15)

            thread.join()
            with log_lock:
                final_snapshot = list(log)
            if final_snapshot:
                progress_box.info("\n\n".join(f"• {m}" for m in final_snapshot[-12:]))

            err = result.get("error")
            if err is not None:
                st.error(f"Planning failed: {err.__class__.__name__}: {err}")
                st.session_state["messages"].append({"role": "assistant", "content": f"Error: {err}"})
            else:
                plan = result["plan"]
                report = result["report"]
                cur_intent: Intent | None = parsed_intent
                budget_report: BudgetReport | None = None
                if cur_intent is not None:
                    try:
                        budget_report = _budget_agent.evaluate(plan, cur_intent)
                    except Exception:
                        budget_report = None

                mode_label = f"multi-agent {execution_mode_arg}" if is_multi_agent else "single-agent"
                render_plan(
                    plan,
                    f"Your Itinerary ({mode_label} · {selected_model_id})",
                    intent=cur_intent,
                    budget_report=budget_report,
                )
                render_verifier(report)

                from agents.llm import get_call_stats
                llm_stats = get_call_stats()
                st.divider()
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("LLM calls", llm_stats.get("llm_calls", 0))
                fc2.metric(
                    "Tokens",
                    f"{llm_stats.get('input_tokens', 0)}↑ {llm_stats.get('output_tokens', 0)}↓",
                )
                fc3.metric(
                    "Cache",
                    f"{llm_stats.get('cache_hits', 0)} hits / "
                    f"{llm_stats.get('cache_misses', 0)} misses",
                )
                ok = ":green[active]"
                na = ":orange[not configured]"
                st.caption(
                    f"Model: `{selected_model_id}` · "
                    f"LLM {ok if llm_available() else na} · "
                    f"Flights {ok if flights_available() else na} · "
                    f"Places {ok if places_available() else na}"
                )
                progress_box.empty()
                st.session_state["messages"].append({"role": "assistant", "content": "Plan rendered above."})
