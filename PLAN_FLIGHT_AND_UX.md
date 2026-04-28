# Plan — Flight selection, UX, and timing-aware repair (Phase 9)

Continues `PLAN_PLANNING_FIXES.md`. Five quality issues observed during live
app testing after Phase 8 landed. Tracked here so they can be sequenced,
gated, and shipped without disturbing the multi-model infrastructure.

---

## 1. Objectives

| ID    | Goal                                                                                                                                           |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| O9-1  | Cap flight duration relative to great-circle distance so a 300–500 mi route never picks a 10+ hr layover option.                                |
| O9-2  | Bias the **return** flight selection toward late-afternoon / evening departures so the last day can include attractions and a real lunch.       |
| O9-3  | Promote arrival/departure timing checks into the **verifier** so the existing repair loop fixes specialist drift instead of a silent post-clamp.|
| O9-4  | Search **round-trip** fares first; fall back to two one-ways. Round-trip is typically 10–25 % cheaper.                                          |
| O9-5  | Replace the flat-expander Streamlit output with a **Trip Summary card + cost-breakdown bar + per-day timeline** layout.                         |

**Out of scope (deferred):** ground transport between airport and city center,
multi-city itineraries, hotel ↔ attraction route optimization, multi-stop
flight scoring beyond the duration cap.

---

## 2. Sequencing & validation gates

Each step is gated by:

1. `pytest tests/` keeps the **179/179** baseline (target ≥ 200 after this phase).
2. For steps that touch ranking or repair (9.1, 9.2, 9.3, 9.4):
   `python -m eval.run_eval --split validation --system multi --limit 20` —
   macro-hard pass rate must stay within ±1 pp of the pre-Phase-9 baseline.
3. For 9.5 (UI only): manual visual inspection. No eval delta expected.

| Step | Issue | Effort | Files                                                                                                                                         | New tests |
| ---- | ----- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| 9.1  | 4     | S      | `tools/live_apis.py`, `tests/unit/test_flight_duration_cap.py`                                                                               | 5–6       |
| 9.2  | 1     | S      | `tools/live_apis.py`, `agents/coordinator.py`, `tests/unit/test_late_return_preference.py`                                                   | 4–5       |
| 9.3  | 5     | M      | `agents/verifier.py`, `agents/rules.py`, `agents/coordinator.py`, `tests/unit/test_timing_violations.py`                                     | 6–8       |
| 9.4  | 3     | M      | `tools/live_apis.py`, `agents/coordinator.py`, `agents/schemas.py`, `agents/transport.py`, `tests/unit/test_round_trip_search.py`            | 5–7       |
| 9.5  | 2     | M      | `app.py` (optional: new `ui_components.py`)                                                                                                   | manual    |

After 9.5, capture the canonical numbers for the report:

```bash
python -m eval.run_eval --split validation --system all --models all
```

This becomes the post-Phase-9 row in the paper-comparison table.

---

## 3. Step-by-step plans

### 9.1 — Distance-aware flight duration cap

**Symptom.** SerpAPI Google Flights returns multi-stop options that take
10+ hours for short-haul routes (300–500 mi). The current
`_rank_flight_options` in `tools/live_apis.py` filters by hard budget cap and
sorts by duration with a 40 % price tolerance — but if a 10 hr 1-stop is the
cheapest option, it can win and force the LLM to keep it.

**Changes** (`tools/live_apis.py:_rank_flight_options`):

1. Compute great-circle distance between `dep_id` / `arr_id` using
   `airportsdata` lat/lon + `_haversine_miles` (already added in Phase 8.5).
2. Bracket-based duration cap (minutes):

   | Distance (mi) | Max duration |
   | ------------- | ------------ |
   | < 500         | 180          |
   | < 1000        | 240          |
   | < 2000        | 360          |
   | ≥ 2000        | 600          |

3. Apply cap **before** the budget filter; keep options where
   `duration_min is None or duration_min <= cap`.
4. If the cap empties the pool, skip it (better to suggest a long flight than
   raise — the user can always pick driving).
5. When `airportsdata` lacks coords for either airport, log via `on_progress`
   and skip the cap.

Plumbing:
- `flight_search()` already has access to `dep_id` / `arr_id`; pass them into
  `_rank_flight_options(... dep_id=..., arr_id=...)`.

**Tests** (`tests/unit/test_flight_duration_cap.py`):

- `test_short_haul_filters_10hr_option`
- `test_short_haul_keeps_2hr_option`
- `test_cross_country_keeps_6hr_option`
- `test_when_all_options_exceed_cap_keeps_shortest`
- `test_distance_unknown_falls_back_to_no_cap`
- `test_cap_applied_before_budget_filter`

**Validation gate.** Run validation eval `--limit 20`. Macro-hard within ±1 pp.
Spot-check a TravelPlanner short-haul query (e.g. Washington → Myrtle Beach):
no flight option ≥ 4 hr should appear in the top result.

**Risks.** Low. The cap is loose (3× the realistic best for the bracket) so
real-world delays still pass. Cap is bypassed when `airportsdata` doesn't
have coords for a code.

---

### 9.2 — Late-return preference for last-day flexibility

**Symptom.** The current ranker picks the shortest-duration return flight,
which is frequently a morning departure (8–10 AM). This kills last-day
attractions and lunch even when a 16:00–19:00 flight is available within
budget.

**Changes.**

1. Add `prefer_late_departure: bool = False` to `flight_search()` and
   `_rank_flight_options()`.
2. In `agents/coordinator.py:research_node`, call return-leg search with
   `prefer_late_departure=True`. Outbound stays unchanged.
3. In `_rank_flight_options` when the flag is set:
   - Build the price/duration tolerance pool as today.
   - Split pool into `late = [o for o in pool if dep_time_min(o) >= 14*60]`
     and `early = [...]`.
   - If `late` is non-empty AND its cheapest entry's price ≤
     `1.3 × cheapest(early or pool)`: pick the cheapest late option.
   - Else: fall back to current rule (duration-first within budget).
4. Use `_HHMM_BEFORE_ARROW`-style parsing on `dep_time` to compare to
   `14*60` minutes.

Output: return flight list ordered with the chosen late option first when
feasible; the rest sorted as today.

**Tests** (`tests/unit/test_late_return_preference.py`):

- `test_late_return_picks_evening_when_available`
- `test_late_return_falls_back_when_only_morning_options`
- `test_late_return_respects_budget_cap`
- `test_outbound_unaffected_by_late_return_flag`
- `test_late_return_within_duration_cap` (interaction with 9.1)

**Validation gate.** Run `--limit 20`. The eval CSV's per-rule pass-rate for
`departure_attraction_*` (added in 9.3) should improve; macro-hard within
±1 pp.

**Risks.** Could push return-leg cost up by ≤ 30 %. Mitigation: 1.3× cap vs.
cheapest pool entry. Outbound leg untouched.

---

### 9.3 — Promote timing checks into the verifier

**Symptom.** Specialists sometimes plan day-1 lunch when the traveler arrives
at 17:00, or last-day breakfast when departure is 06:00. Today the safety
clamp in `agents/coordinator.py:_apply_timing_adjustments` rewrites these
fields silently after assembly — but the clamp runs *after* the verifier, so
the LLM never gets a chance to fix it, and the rendered string sometimes
reflects an uneconomical clamp ("In transit (arrives 17:00)") instead of a
real meal that *was* possible.

**Decision.** Keep the safety clamp as a final defense but add real verifier
rules so timing failures route through the existing repair loop (3 rounds).

**Changes.**

`agents/rules.py` — add rule names:

```
RULE_ARRIVAL_LUNCH      = "arrival_lunch_after_14"
RULE_ARRIVAL_DINNER     = "arrival_dinner_after_19"
RULE_ARRIVAL_ATTRACTION = "arrival_attraction_after_16"
RULE_DEPARTURE_BREAKFAST = "departure_breakfast_before_08"
RULE_DEPARTURE_LUNCH    = "departure_lunch_before_1330"
RULE_DEPARTURE_ATTRACTION = "departure_attraction_before_16"
```

`agents/verifier.py:verify` — extend signature to take `trip_windows`
(extracted from the same `tool_context` already populated post-transport):

```
def verify(plan, intent, sandbox=None, trip_windows: TripWindows | None = None) -> VerifierReport
```

For each new rule, append a `Violation` with the right `responsible`:
- arrival_* lunch/dinner → `dining`
- arrival_attraction → `sightseeing`
- departure_breakfast / lunch → `dining`
- departure_attraction → `sightseeing`

Detail string includes the offending field value AND the arrival/departure
HH:MM so the specialist's repair note is actionable.

`agents/coordinator.py:verify_node` — pass `state["tool_context"].get("trip_windows")`
into `verify()`.

`agents/coordinator.py:_apply_timing_adjustments` — keep behavior as-is. It
runs in `assemble_node` (before verify). Its job becomes "make the rendered
plan internally consistent if the LLM still drifts after 3 repair rounds".
Document this with a one-line comment.

**Tests** (`tests/unit/test_timing_violations.py`):

- `test_lunch_after_14_flagged_as_dining`
- `test_dinner_after_19_flagged_as_dining`
- `test_attraction_after_16_flagged_as_sightseeing`
- `test_no_violation_when_grid_obeyed`
- `test_violation_routes_to_correct_specialist` (round-trips through `repair_prepare`)
- `test_safety_clamp_still_runs_after_repair_exhausted`
- `test_no_trip_windows_skips_timing_rules`
- `test_breakfast_before_08_flagged`

**Validation gate.** Run `--limit 20`. Track `safety_clamp_fire_rate`
(count of plans where `_apply_timing_adjustments` actually changed something);
expected to drop because the LLM now fixes it via repair. Macro-hard ±1 pp;
commonsense rate expected to improve.

**Risks.** Low. New rules only add violations (never silence existing ones).
3-round repair cap prevents infinite loops. Safety clamp stays as fallback.

---

### 9.4 — Round-trip flight search

**Symptom.** Two independent one-way searches priced separately. Real-world
round-trip fares are typically 10–25 % cheaper because airlines bundle.

**Changes.**

`tools/live_apis.py`:

```python
@dataclass
class _RoundTripPair:
    outbound: _FlightOption
    inbound:  _FlightOption
    total_price: int | None

def flight_search_round_trip(
    self, origin: str, dest: str,
    depart_date: str, return_date: str,
    *, max_results: int = 5, budget: int | None = None,
    prefer_late_return: bool = True,
) -> tuple[list[str], list[str]]:
    """Returns (outbound_strings, return_strings) drawn from the SAME
    round-trip pairs (parallel lists ordered identically). Cost: $N appears
    on the outbound leg ONLY; the return string omits the Cost fragment."""
```

Implementation:
1. SerpAPI `type=1` (default round-trip) with `outbound_date` AND `return_date`.
2. SerpAPI returns `best_flights`/`other_flights` where each option contains
   both an outbound and a return segment. Parse both halves into
   `_FlightOption`.
3. Apply duration cap (9.1) on both legs separately.
4. Apply late-return preference (9.2) on the return half only.
5. Combined price filter: drop pairs where `total_price > budget` (when known).
6. Format output: outbound carries `Cost: $N` (the full round-trip price);
   return carries airline + times only (no Cost fragment).

`agents/schemas.py:ToolContext` — add `flights_round_trip: bool = False`.

`agents/coordinator.py:research_node`:
- Try `flight_search_round_trip(org_iata, dest_iata, depart, ret)` first.
- If results ≥ 1: set `ctx["flights_outbound"]` and `ctx["flights_return"]`
  from the pair lists; `ctx["flights_round_trip"] = True`.
- On error or empty results: fall back to two one-way searches (current
  Phase 8.5 path).
- Drive-leg prepending (Phase 8.5) still applies in either path.

`agents/transport.py`:
- System prompt: when `ctx.get("flights_round_trip")` is true, declare
  `Cost: $N` on day 1 only; the last-day return description has no Cost.
- Update the few-shot example to show this.

**Tests** (`tests/unit/test_round_trip_search.py`):

- `test_round_trip_search_returns_paired_options` (mock SerpAPI payload)
- `test_round_trip_falls_back_on_serpapi_error`
- `test_round_trip_falls_back_when_empty_results`
- `test_round_trip_cost_attributed_to_day_1_only`
- `test_round_trip_return_leg_has_no_cost_fragment`
- `test_one_way_path_unchanged_when_round_trip_disabled`
- `test_round_trip_respects_duration_cap_on_both_legs`

**Validation gate.** Run `--limit 20`. Verify via `eval/results.csv` that
mean transport spend per query drops (round-trip discount). Macro-hard
within ±1 pp.

**Risks.**
- SerpAPI quota: round-trip is one call instead of two — neutral or better.
- Verifier sums all `Cost: $N` fragments per day; round-trip cost on day 1
  only still passes the budget rule. Confirm with an explicit test.
- Drive-leg prefix interaction: drive prefix is prepended to each formatted
  string regardless of round-trip vs one-way path.

---

### 9.5 — UI redesign

**Symptom.** Current `app.py:render_plan()` stacks `st.expander` blocks per
day. Cost vs. budget isn't visually obvious. Verifier result is one colored
line. The Performance & data sources block is at the bottom in a collapsed
expander.

**Changes** (`app.py` — optional split into `ui_components.py`):

**(A) Trip Summary card** (top of result block):

- `st.container(border=True)` containing two `st.columns([1,1,1])` rows:
  - Row 1 (`st.metric`): Trip (`{org} → {dest}`), Days, Travelers
  - Row 2 (`st.metric`): Total cost (with `delta=total - budget` colored
    red if over, green if under), Cache hits, LLM calls
- A budget gauge underneath: `st.progress(min(1.0, total/budget))` colored:
  - green if `total ≤ 0.9 × budget`
  - amber if `0.9 < total/budget ≤ 1.0`
  - red if `total > budget` (paired with explicit "OVER BUDGET" badge)

**(B) Verifier badge:**

- If `report.passed`: `st.success("All N constraints satisfied")` as a
  large chip directly under the summary card.
- If failed: `st.error(...)` chip + `st.expander("Violations", expanded=True)`
  grouping by `Violation.responsible`.

**(C) Cost breakdown:**

- Single horizontal stacked bar (`st.bar_chart` with a one-row dataframe or
  a small inline `plotly.graph_objects.Bar` with `orientation='h'`).
- Categories: transport / lodging / dining (sourced from `BudgetReport.category_costs`).
- Below: a 3-column `st.metric` row showing $ + % per category.

**(D) Per-day timeline** (replaces current expanders):

- For each day, a `st.container(border=True)`:
  - Header: `**Day {N}** — {current_city}` plus a small caption with the
    transportation summary on travel days.
  - Three columns: `[Transport, Schedule, Stay]`
    - Transport (col 1): airline + times + cost OR "No transfer needed"
      caption with reason.
    - Schedule (col 2): four stacked sub-rows (Breakfast / Lunch / Attraction
      / Dinner), each on its own `st.markdown` line; empty slots greyed
      with the existing `_empty_display_text` + reason.
    - Stay (col 3): hotel name + nightly cost OR `"No stay (return day)"`.
- This keeps every field the verifier reads (the `PlanDay` shape is
  unchanged) — only the *render* changes.

**(E) Performance section:**

- Move out of the bottom expander into a small footer strip with three
  `st.metric` widgets: LLM calls, tokens (in↑ / out↓), cache hit rate.
- The "Live data sources" line stays as-is.

**(F) Emoji icons (always on, no toggle):**

- Day-card column headers: `✈️ Transport`, `📅 Schedule`, `🏨 Stay`.
- Schedule rows: `🍳 Breakfast`, `🥗 Lunch`, `🎟️ Attraction`, `🍽️ Dinner`.
- Verifier badges: `✅ All N constraints satisfied` / `❌ N violation(s)`.
- Budget gauge label: `💰 Budget`.
- Icons are part of the layout, not gated behind a sidebar toggle.

**Tests.** No automated tests; `app.py` is not import-safe under pytest
(Streamlit context). Manual checks:
1. Cross-country trip (NYC → LA): budget gauge red if over, green if under.
2. 3-day Austin trip with 19:00 return: timeline shows attraction + dinner
   on last day.
3. Drive-only trip (Washington → Richmond ~110 mi): transport column shows
   "Self-driving" or driving route summary, no flight string.
4. Verifier-failure case: violations panel groups by specialist.

Capture screenshots for `PROJECT_REPORT_PLAN.md` (paper figure budget).

**Risks.** None functional. Streamlit re-render performance unaffected
(still uses session state). Bigger surface area to review for visual
regressions, but the underlying `FullPlan` schema doesn't change.

---

## 4. Cross-cutting concerns

### 4.1 Interactions between steps

- 9.1 + 9.2 + 9.4: the duration cap and late-return preference both apply
  inside `_rank_flight_options`. The round-trip path uses the same ranker
  on each leg.
- 9.3 + 9.4: when round-trip declares cost on day 1 only, the verifier's
  existing `missing_cost` rule must NOT fire on the last-day return string.
  Check `agents/verifier.py:_check_missing_cost` to confirm it only flags
  when transport is non-`-` AND the round-trip flag is unset (or treat the
  round-trip return as "cost-bundled" via a new dedicated check).
- 9.3 + 9.5: the new violations grouping in the UI (4.B) reads the same
  `Violation.responsible` field today's verifier produces — no extra wiring.

### 4.2 Architecture invariants (append to `ARCHITECTURE.md` §11)

- Invariant #17: round-trip cost is attributed to day 1 only when
  `tool_context["flights_round_trip"]` is true; the verifier and budget
  agent both treat this as a single cost fragment for the trip.
- Invariant #18: timing violations are first-class verifier rules; the
  `_apply_timing_adjustments` clamp is a final-pass safety net only,
  guaranteed idempotent when the grid is obeyed.

### 4.3 Eval CSV columns

No new columns required. Existing `latency_seconds`, `input_tokens`,
`output_tokens`, hard/commonsense rule breakdowns, and per-rule pass rates
will surface the impact of every step.

### 4.4 Paper comparability

- 9.1 (duration cap) doesn't change plan format → verifier rules unaffected.
- 9.2 (late return) only re-orders ranked flight options → format unchanged.
- 9.3 (timing rules) adds violations the LLM is asked to fix; final
  rendered plan shape unchanged.
- 9.4 (round-trip) shifts cost attribution to day 1 only — verifier rule
  `total ≤ budget` unchanged because it sums across all days.
- 9.5 (UI) is presentation-only, not seen by the verifier or eval.

### 4.5 Rollback strategy

Each step lands as its own commit. Kill-switches:
- 9.1: `FLIGHT_DURATION_CAP=0` env var → ranker skips the cap.
- 9.2: `prefer_late_departure=False` (don't pass the flag from coordinator).
- 9.3: `TIMING_VERIFIER_RULES=0` env var → verifier skips the new block.
- 9.4: env var `FLIGHT_ROUND_TRIP=0` → coordinator goes straight to two
  one-way searches.
- 9.5: keep the old `render_plan` behind a sidebar toggle for one release
  cycle.

---

## 5. Done criteria

Phase 9 is "done" when:

- All five steps merged with passing tests.
- `pytest tests/` reports ≥ **205** tests passing (179 + ~25 new).
- One full ablation matrix run captured in `eval/results.csv`:
  ```bash
  python -m eval.run_eval --split validation --system all --models all
  ```
- Manual app smoke checks pass (see 9.5 test list).
- Updated row in `PROJECT_REPORT_PLAN.md` table with post-Phase-9 numbers.
- Updated `ARCHITECTURE.md` §11 with invariants #17–#18.

---

## 6. Risk register

| Risk                                                                              | Likelihood | Mitigation                                                                                       |
| --------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------ |
| Late-return preference forces an over-budget flight                               | Low        | 1.3× cap vs. cheapest pool entry; falls back to current rule when no late option fits.            |
| Duration cap filters out the only available option                                | Low        | Skip cap whenever it would empty the pool.                                                       |
| Round-trip cost-attribution change breaks verifier `missing_cost` on return leg   | Medium     | Add `flights_round_trip` flag to `_check_missing_cost`; guard with explicit unit test.            |
| Timing verifier rules cause infinite repair loops                                 | Low        | 3-round cap; safety clamp is the last word.                                                      |
| Streamlit redesign hides info eval reviewers expect                               | Low        | Sidebar toggle to fall back to the old flat-expander view for one release cycle.                  |
| SerpAPI returns empty results for round-trip query                                | Medium     | Try-then-fallback to two one-way searches; mirrors current `_call_live` pattern.                  |
| `airportsdata` lacks lat/lon for some IATA codes (cap can't apply)                | Medium     | Skip the cap and log via `on_progress`; ranker still produces a valid result.                     |
| Phase 9 prompt changes invalidate LLM disk cache → eval reruns expensive          | High       | Run the full ablation matrix once after all five steps land, not per-step.                       |

---

## 7. Resolved defaults

Confirmed by user before coding:

- **Late-return preference (9.2):** soft preference with a `1.3×` ceiling vs.
  the cheapest pool entry. When the budget is tight (no late option fits the
  ceiling), automatically falls back to the duration-first rule.
- **Round-trip search (9.4):** try `flight_search_round_trip` first; on
  SerpAPI error or empty results, fall back to two one-way searches (the
  existing Phase 8.5 path). Cost attributed to day 1 only.
- **UI emoji icons (9.5):** always on (no sidebar toggle). Icons are part of
  the layout — `✈️ Transport`, `🏨 Stay`, `🍳 Breakfast`, `🥗 Lunch`,
  `🎟️ Attraction`, `🍽️ Dinner`, `✅`/`❌` for verifier status.

## 8. Open questions remaining

- Duration cap brackets (9.1): are 180 / 240 / 360 / 600 minutes the right
  knobs, or do we tune them after the first eval run?
