# Plan — Planning-quality fixes (Phase 8)

Continues `PLAN_MULTI_MODEL.md`. Five planning-quality issues identified during
app testing. Tracked separately because these are correctness/UX fixes, not
multi-model infrastructure.

---

## 1. Objectives

| ID    | Goal                                                                                                                                                  |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| O8-1  | Reject queries that lack an unambiguous trip **start date** before any specialist runs.                                                                |
| O8-2  | Scale the live candidate-data pool (hotels / restaurants / attractions) with trip length so the no-repeat rules can actually be satisfied.            |
| O8-3  | Replace fixed cost defaults (level-2 = $28 meals, $130 hotel, etc.) with **budget-derived per-category cost targets** that scale with intent.budget.  |
| O8-4  | Move arrival/return-day timing logic out of the post-hoc `_apply_timing_adjustments` patch and into the **specialist prompts** themselves.            |
| O8-5  | Handle destinations whose city has no direct airport: find nearest IATA airport (≤200 mi); prepend a self-drive transfer to flight strings. Use **drive-only** when total drive ≤ **100 mi**. |

**Out of scope (deliberately deferred):** cross-category budget redistribution
(slack passing). Requires sequential execution to forward each specialist's
under-spend to the next; parallel mode cannot pass slack within one superstep
without doubling LLM cost. Defer until a sequential path is required for some
other reason.

---

## 2. Sequencing & validation gates

Each step is gated by:

1. `pytest tests/` keeps **100/100** pass rate.
2. `python -m eval.run_eval --split validation --system multi --limit 20`
   macro-hard pass rate must not regress vs. the prior step's baseline (track
   delta in PR descriptions; >1pp drop blocks merge).

| Step | Issue | Effort | Files                                                                                                                                                                                                                       | New tests |
| ---- | ----- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| 8.1  | 2     | S      | `agents/coordinator.py`, `app.py`, `tests/unit/test_intent_validation.py`                                                                                                                                                   | 4–5       |
| 8.2  | 4     | S      | `agents/coordinator.py`, `tests/unit/test_research_pool.py`                                                                                                                                                                  | 3         |
| 8.3  | 5     | M      | `agents/budget.py`, `agents/dining.py`, `agents/lodging.py`, `agents/transport.py`, `agents/coordinator.py`, `agents/schemas.py`, `tests/unit/test_cost_targets.py`                                                          | 6–8       |
| 8.4  | 3     | M      | `agents/rules.py`, `agents/dining.py`, `agents/sightseeing.py`, `agents/coordinator.py`, `tests/unit/test_trip_windows.py`                                                                                                  | 5–7       |
| 8.5  | 1     | L      | `tools/live_apis.py`, `agents/coordinator.py`, `agents/transport.py`, `tests/unit/test_airport_fallback.py`, `tests/unit/test_drive_threshold.py`                                                                            | 6–8       |

After 8.5, capture canonical numbers for the report:

```bash
python -m eval.run_eval --split validation --system all --models all
```

This becomes the paper-comparison table in `PROJECT_REPORT_PLAN.md`.

---

## 3. Step-by-step plans

### 8.1 — Reject missing trip start date

**Symptom.** A query like *"Plan a 3-day trip to Myrtle Beach under $1400"*
slips through `intent_constraint_violations` if the LLM hallucinates a start
date during `parse_intent`. Result: planner runs against fabricated dates
without warning the user.

**Changes.**

- `agents/coordinator.py:INTENT_SYSTEM` — add a clause: *"If the user did NOT
  state an unambiguous trip START date, return an empty `dates` list. Do not
  guess. Do not use today's date."*
- `agents/coordinator.py:intent_constraint_violations` — replace the existing
  `if not any(...)` check with:
  - Required: `intent.dates and intent.dates[0].strip()` (start date present).
  - If only the *end* date is given (single date but `intent.days > 1`), reject
    with **"missing trip start date (from)"**.
- `app.py` — when the new violation surfaces, the chat re-prompt copy should
  call out the start date specifically, not the generic "missing travel dates".

**Tests** (`tests/unit/test_intent_validation.py`):

- `test_intent_no_dates_rejected`
- `test_intent_only_end_date_rejected`
- `test_intent_one_date_one_day_trip_accepted` (single-day trip is unambiguous)
- `test_intent_full_date_range_accepted`
- `test_constraint_message_mentions_start_date`

**Validation gate.** Eval uses TravelPlanner queries which always have full
date ranges; this should not regress macro-hard pass rate. Manual app-side
smoke: empty-date query produces a clear re-prompt within 1 turn.

**Risks.** Low. Eval workload is unaffected; only app-time queries change.

---

### 8.2 — Scale candidate pool by trip length

**Symptom.** `research_node` fetches a fixed 8 hotels / 12 restaurants / 12
attractions regardless of trip length. A 7-day trip needs ≥21 unique
restaurants for the no-repeat rule but only sees 12 → forces hallucinations
or repeat violations.

**Changes** (`agents/coordinator.py:research_node`):

```
n_hotels       = min(20, max(8,  intent.days * 2))    # 3d→8,  5d→10, 7d→14, 10d→20
n_restaurants  = min(40, max(12, intent.days * 4))    # 3d→12, 5d→20, 7d→28
n_attractions  = min(30, max(12, intent.days * 3))    # 3d→12, 5d→15, 7d→21
```

- Hard upper caps (20/40/30) bound prompt size for very long trips.
- Lower bounds match current behavior so short-trip benchmark scores are
  unchanged.

**Tests** (`tests/unit/test_research_pool.py`):

- `test_pool_3day_matches_current_baseline` (regression guard)
- `test_pool_7day_scales_up`
- `test_pool_capped_at_upper_bound_for_long_trips`

(Use `monkeypatch` on `live.search_places` to assert the `max_results=` arg.)

**Validation gate.** Run `--system multi --limit 20` on validation. Track
token-count delta in `eval/results.csv` (the existing `input_tokens` column);
expected +20–30% on long trips, baseline unchanged on 3-day trips.

**Risks.** Higher Maps API quota usage (still within free tier for typical
eval workloads). Larger specialist prompts → +tokens/+latency on long trips
only.

---

### 8.3 — Budget-derived cost targets

**Symptom.** `_PRICE_LEVEL_MAP[2] = 28` is the "default" meal cost emitted in
specialist prompts regardless of whether the budget is $400 or $4000. Same
issue for `_price_level_to_nightly`. Specialists pick options around $28
meals when they should be picking $50+ on luxury trips, or $15 on tight
trips.

**Changes.**

New helpers in `agents/budget.py`:

```
def derive_meal_targets(intent: Intent) -> dict[int, int]:
    """5-point cost curve indexed by Maps price_level (0..4)."""
    caps = allocate(intent)
    if not caps.get("dining"):
        return _DEFAULT_MEAL_TABLE              # fall back to fixed table
    num_meals = max(1, intent.days * 3 - 2)     # no breakfast d1, no dinner dN
    avg = caps["dining"] // num_meals
    return {
        0: max(5,  int(avg * 0.5)),
        1: max(8,  int(avg * 0.7)),
        2: max(12, int(avg * 1.0)),
        3: max(20, int(avg * 1.5)),
        4: max(30, int(avg * 2.0)),
    }

def derive_lodging_targets(intent: Intent) -> dict[int, int]:
    """Per-night cost curve indexed by Maps price_level."""
    caps = allocate(intent)
    if not caps.get("lodging"):
        return _DEFAULT_LODGING_TABLE
    nights = max(1, intent.days - 1)
    avg = caps["lodging"] // nights
    return {
        0: max(40, int(avg * 0.6)),
        1: max(60, int(avg * 0.8)),
        2: max(80, int(avg * 1.0)),
        3: max(100, int(avg * 1.3)),
        4: max(140, int(avg * 1.6)),
    }

def derive_transport_target(intent: Intent) -> int:
    """One-way leg target (caps['transport'] / 2)."""
    caps = allocate(intent)
    return caps.get("transport", 200) // 2
```

Wire-up:

- `ToolContext` (in `agents/schemas.py`) — add optional keys `meal_cost_targets`,
  `lodging_cost_targets`, `transport_one_way_target`.
- `coordinator.py:research_node` — populate these from `derive_*` helpers and
  attach to `ctx`.
- `dining.py:_format_tool_context` — replace `_price_level_to_meal_cost(price_level)`
  call with a lookup against `ctx.get("meal_cost_targets")` falling back to the
  current fixed table.
- `lodging.py:_format_tool_context` — same swap against `ctx.get("lodging_cost_targets")`.
- `transport.py` — when `ctx.get("transport_one_way_target")` is set, include
  it in the prompt as: *"Each one-way flight should ideally stay under $X."*

Existing `_PRICE_LEVEL_MAP` constants stay as the **`_DEFAULT_*_TABLE`** fallbacks
(`intent.budget is None` path).

**Tests** (`tests/unit/test_cost_targets.py`):

- `test_meal_targets_sum_under_dining_cap`
- `test_lodging_targets_per_night_under_lodging_cap`
- `test_transport_target_is_half_of_transport_cap`
- `test_no_budget_falls_back_to_fixed_table`
- `test_targets_present_in_tool_context`
- `test_dining_prompt_uses_dynamic_costs` (mock `call_json`, capture user prompt)

**Validation gate.** This is the highest-risk step for paper comparability.
Before merging:

1. Capture baseline `multi` macro-hard pass rate on 30 validation queries.
2. Run with new cost targets; compare.
3. Macro-hard must stay within ±1pp; commonsense pass rate is expected to
   improve (better budget adherence).

If macro-hard drops more than 1pp, investigate before continuing — likely the
target curve needs tuning (e.g., raise level-3 multiplier from 1.5x → 1.4x).

**Risks.** TravelPlanner's verifier checks `total ≤ budget`; our 45/35/20
allocation is conservative and should not push any plan over budget. The risk
is *under*-spending on critical categories. Curve multipliers are tunable.

---

### 8.4 — Time-aware return/arrival schedule (push into specialist prompts)

**Symptom.** `coordinator.py:_apply_timing_adjustments` post-processes the
assembled plan with hardcoded thresholds (`dep_min < 12*60` → drop lunch).
Specialists don't see real flight timing when they plan, so the LLM keeps
producing a fixed last-day pattern that gets rewritten downstream.

**Changes.**

Extend `agents/rules.py:format_trip_windows()` to emit an explicit
**available-slots grid** per critical day, e.g.:

```
Flight timing:
  - Day 1 arrival at destination: 18:30
  - Last day departure from destination: 11:30

Day 1 (arrival):
  - breakfast: NO  (still in transit)
  - lunch:     NO  (still in transit, arrival 18:30)
  - dinner:    YES (arrival before 19:00)
  - attraction: NO (too late after arrival 18:30)

Last day (departure):
  - breakfast: YES (window 07:00-09:00)
  - lunch:     NO  (departs 11:30)
  - dinner:    NO  (already departed)
  - attraction: NO (no time before 11:30 departure)
```

Specialists (`dining.py`, `sightseeing.py`) consume the YES/NO grid and plan
correctly upfront. Reduce `_apply_timing_adjustments` to a thin **safety clamp**
that fires only when the specialist disregarded the grid.

Threshold rules (kept consistent across grid generator and clamp):

- Day 1 breakfast: always NO.
- Day 1 lunch: NO if `arrival ≥ 14:00`.
- Day 1 dinner: NO if `arrival ≥ 19:00`.
- Day 1 attraction: NO if `arrival ≥ 16:00`.
- Last day breakfast: NO if `departure < 08:00`.
- Last day lunch: YES iff `departure ≥ 13:30`.
- Last day dinner: YES iff `departure ≥ 20:00`.
- Last day attraction: YES iff `departure ≥ 16:00` (must finish ≥1h before flight).

**Tests** (`tests/unit/test_trip_windows.py`):

- `test_grid_emits_yes_no_for_arrival_18_30`
- `test_grid_omits_late_dinner_when_arrival_after_19`
- `test_grid_omits_lunch_when_departure_before_13_30`
- `test_safety_clamp_idempotent_when_grid_obeyed`
- `test_safety_clamp_fixes_specialist_drift`
- Optional: `test_dining_prompt_includes_grid` (mock + assertion on user prompt)

**Validation gate.** Run `--limit 20`; check that the safety-clamp fire rate
(count plans where `_apply_timing_adjustments` actually changed something)
drops to <10% (currently fires on most plans).

**Risks.** Bigger prompts → more tokens. Mitigation: only emit grid when
`trip_windows` is populated (already gated). Clamp keeps current behavior as
a fallback in case the LLM misreads the grid.

---

### 8.5 — Nearest-airport fallback + 100 mi drive cutoff

**Symptom.** `tools/live_apis.py:_resolve_airport_code` raises `RuntimeError`
when a destination has no direct IATA airport. Coordinator's bare `try/except`
in research silently zeros out flights → transport falls back to driving even
for cross-country trips.

**Decision tree** (in `research_node`, gated by `route_drive`):

```
1. Compute drive_distance from route_drive (Maps already returns this).
2. If drive_distance is None or unparseable → treat as "long" (try airports).
3. If drive_distance ≤ 100 mi:
     mode = "drive_only"
     skip flight search entirely; transport specialist gets route_drive + route_taxi.
4. Else:
     try direct IATA resolution for both endpoints.
     If origin OR destination has no direct airport:
       call _resolve_with_fallback(missing_endpoint) within 200 mi search radius.
       Prepend self-drive transfer leg to each formatted flight string.
```

**Changes in `tools/live_apis.py`:**

```
def _resolve_with_fallback(self, location: str) -> tuple[str, str | None, str | None]:
    """Returns (iata, fallback_city, drive_leg_summary).
    fallback_city/drive_leg_summary are None when location resolves directly.
    Search radius for fallback is 200 mi; raises RuntimeError if none found.
    """
```

Implementation:
1. Direct IATA match → return `(iata, None, None)`.
2. Geocode `location` via Maps to lat/lon.
3. Iterate `airportsdata.load("IATA")`, filter to entries with valid lat/lon
   AND `country == "US"`, compute haversine distance, sort ascending.
4. Pick the first within 200 mi. If none → raise.
5. Compute `route_summary(location, f"{airport_city}", "driving")` to get the
   drive leg ("Distance: 95 mi, Duration: 1h 50m").
6. Return `(airport_iata, airport_city, "Drive 95 mi to {iata} (1h 50m)")`.

`flight_search()` updates: when fallback is used for either endpoint, prepend
`drive_leg_summary` to each formatted output string. Format remains parseable
by transport's existing flight-ID regex (it matches the airline portion which
is unchanged).

**Changes in `agents/coordinator.py:research_node`:**

- Compute `drive_distance` from `route_drive` (parse Maps' "Distance: X mi"
  string).
- Branch on the 100 mi threshold; emit `on_progress` for both branches:
  - `"Trip is 87 mi — using drive-only transport (no flight search)."`
  - `"No airport in {dest}; nearest is {iata} {dist} mi away — flights include drive transfer."`

**Changes in `agents/transport.py`:**

System prompt addendum: *"When a flight option begins with 'Drive {N} mi to
{IATA} (...);', that drive leg is part of the day's transportation — keep
it verbatim in your description. The drive cost is bundled into the flight cost."*

**Tests** (`tests/unit/test_airport_fallback.py`, `tests/unit/test_drive_threshold.py`):

- `test_resolve_direct_iata` (passthrough)
- `test_resolve_fallback_picks_nearest_within_200mi`
- `test_resolve_fallback_raises_when_no_airport_in_radius`
- `test_flight_string_includes_drive_leg`
- `test_drive_threshold_skips_flight_search_under_100mi`
- `test_drive_threshold_uses_flights_at_120mi`
- `test_drive_threshold_when_route_drive_unparseable_falls_back_to_flights`
- `test_transport_specialist_preserves_drive_prefix`

**Validation gate.** Run `--limit 20` validation. Two checks:

1. Macro-hard pass rate not down >1pp.
2. Spot-check: a TravelPlanner query whose destination is a small city —
   confirm via `on_progress` log that fallback fires and produces transport
   strings the verifier accepts.

**Risks.**

- `airportsdata` lat/lon completeness — guard with `if not (lat and lon): skip`.
- Extra Maps geocode + distance call per fallback (~+1.5s latency); cache
  results by (location, airport_iata) within a single eval run.
- Fallback could pick a tiny regional airport with no actual flights → SerpAPI
  returns empty → transport falls back to drive-only. Acceptable: log it.

---

## 4. Cross-cutting concerns

### 4.1 Architecture & invariants

After 8.5, append to `ARCHITECTURE.md` §11:

- Invariant #14: cost targets in specialist prompts are budget-derived when
  `intent.budget` is set; fixed-table fallback only when budget is `None`.
- Invariant #15: arrival/departure timing is communicated to specialists via
  the `format_trip_windows` grid; `_apply_timing_adjustments` is a
  defense-in-depth clamp, not the primary mechanism.
- Invariant #16: airport fallback ≤200 mi, drive-only ≤100 mi — both cutoffs
  pinned in a constants module so eval reproducibility is guaranteed.

### 4.2 Eval CSV columns (no change needed)

Existing columns (`model`, `provider`, `latency_seconds`, `cache_hits`,
`cache_misses`, `input_tokens`, `output_tokens`, hard/commonsense rule
breakdowns) remain. Phase 8 changes are observable in token/latency deltas
without new columns.

### 4.3 Paper comparability

All Phase 8 changes are compatible with TravelPlanner's evaluator:

- Issue 2 only affects rejection (queries that wouldn't make a valid plan).
- Issue 4 only changes prompt context size, never plan output format.
- Issue 5 leaves `verify(... budget)` rule unchanged.
- Issue 4 changes the *content* of last-day fields but verifier's
  `missing_cost`/`min_nights` rules check format, not whether the LLM had time
  to plan dinner.
- Issue 1 changes flight string format (prepended drive leg). The verifier
  already only parses `Cost: $N` fragments out of these strings — no rule
  parses the airline ID.

### 4.4 Rollback strategy

- Each step lands in its own commit/PR.
- A single env-var kill-switch per step (e.g., `BUDGET_DERIVED_COSTS=0`)
  reverts to old behavior without code changes — useful if eval regression
  surfaces post-merge.
- Alternative: keep the fallback codepaths in `agents/budget.py` and
  `coordinator.py` permanently, controlled by `intent.budget is None`.

---

## 5. Done criteria

Phase 8 is "done" when:

- All five steps merged with passing tests.
- `pytest tests/` reports ≥120 tests passing (current 100 + ~25 new).
- One full ablation matrix run captured in `eval/results.csv`:
  ```
  python -m eval.run_eval --split validation --system all --models all
  ```
- Updated row in `PROJECT_REPORT_PLAN.md` table comparing our `multi` numbers
  to TravelPlanner's published results, on the same `validation` split.
- Updated `ARCHITECTURE.md` §11 with invariants #14–16.

## 6. Risk register

| Risk                                                                             | Likelihood | Mitigation                                                                          |
| -------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------- |
| Budget-derived costs regress macro-hard pass rate                                | Medium     | A/B test; tunable curve multipliers; fallback to fixed table when budget is None.   |
| Time-aware grid confuses LLM (long prompt + new format)                          | Low        | Safety clamp keeps current behavior as fallback; grid only added when timing known. |
| Airport fallback picks regional airport with no SerpAPI coverage                 | Medium     | Falls back to drive-only with clear `on_progress` notice.                           |
| Issue 4 token growth pushes specialist past Gemini Flash output budget           | Low        | Hard upper caps (20/40/30); prompt + response stays under 8k tokens.                |
| Phase 8 changes prompts → cache invalidation → eval reruns expensive             | High       | Run eval once after all five steps land, not per-step (steps validated by limit=20). |
| User-time delta on app side (extra Maps calls)                                   | Low        | +2–3s on first plan; cached within session.                                         |

---

## 7. Open questions for confirmation before coding

None blocking. Defaults captured above:
- 100 mi drive cutoff (confirmed by user).
- 200 mi airport fallback radius.
- Curve multipliers `[0.5, 0.7, 1.0, 1.5, 2.0]` for meals; `[0.6, 0.8, 1.0, 1.3, 1.6]` for lodging.
- US-only airport fallback (TravelPlanner is US-domestic).
