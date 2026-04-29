# Fix Plan — Airport Resolution, Model Registry, Eval Resilience

Three independent issues, each surgical. Plan-only — no code changes yet.

---

## Issue 1 — "new york city" picks NYS (Skyports Seaplane Base) instead of JFK/LGA/EWR

### Root cause

`tools/live_apis.py::_resolve_airport_code` (lines 766–802) does substring matching of the destination against the `airportsdata` package's `city` and `name` fields:

- For input `"new york city"`, `target = "new york city"`.
- Airports for JFK/LGA/EWR have `city="New York"` (not `"New York City"`).
- `city == target` → False; `target in name` → False (none of those names contain the literal "new york city").
- `_resolve_airport_code` raises `RuntimeError`.

Control falls through to `_resolve_with_fallback` (lines 723–749), which geocodes the city (~40.71, -74.0) and picks the **single nearest IATA-coded airport** in the airportsdata DB. That database includes non-commercial fields like **NYS** (Skyports Seaplane Base, ~2 mi from Manhattan), which wins by raw distance.

The Tempe→PHX case in the same log is *not* a bug — PHX is the right answer; the "drive transfer included" message is just informational.

### Fix (layered, smallest change first)

**Layer A — alias normalization (fixes 90% of US-city inputs without any network call).** In `_resolve_airport_code`, before scanning `airportsdata`:

1. Lowercase + collapse whitespace (already done in `_norm`).
2. Strip trailing tokens: `" city"`, `" metro"`, `" metropolitan area"`, `", usa"`, `", us"`.
3. Apply a small alias table for collisions the dataset is known to have. Minimum needed (curated, ~10 entries):
   - `nyc | new york city | manhattan` → `new york`
   - `dc | washington dc | washington d.c.` → `washington`
   - `la | l.a.` → `los angeles`
   - `sf | san fran` → `san francisco`
   - `bay area` → `san francisco`
   - `the big apple` → `new york`
4. Re-run the existing `city == target or target in name` match against the normalized form.

This alone resolves "new york city" → matches city="new york" → ranks JFK/LGA/EWR via the existing `rank()` function (which prefers names containing "international").

**Layer B — restrict the nearest-airport fallback to commercial airports.** In `_resolve_with_fallback` (lines 723–749), the brute-force `for code, meta in airports.items()` scan currently considers every IATA code. Fix:

- Filter by name keyword: skip entries whose name does **not** contain at least one of `"International"`, `"Regional"`, `"Municipal"`, `"Field"`, `"Airport"`. (`airportsdata` exposes `name` per entry.)
- Negative filter: skip entries whose name contains `"Seaplane"`, `"Heliport"`, `"Air Base"`, `"AAF"`, `"Naval"`, `"Air Park"`. NYS is filtered by "Seaplane".
- Optionally bias scoring: when two airports are within 25 mi of each other, prefer the one whose name contains "International".

**Layer C — Places API fallback (the user's suggestion; only triggered if A+B still find nothing).** Use the existing `googlemaps.Client.places_nearby` (already used at `live_apis.py:152`) with `type="airport"` around the geocoded city center, radius 80 km. Filter results by name keywords as in Layer B. Pick the result whose name best matches "International" + has the highest `user_ratings_total` (proxy for size). Map result to an IATA code by cross-referencing the `airportsdata` table (match by name + country + lat/lon within 5 mi).

Layer C is the safety net — it costs one Maps API call per session (cache the result on `self`), so cost is bounded.

### Files

- `tools/live_apis.py` — extend `_resolve_airport_code` (Layer A), `_resolve_with_fallback` (Layer B), add new `_places_airport_lookup` (Layer C).
- New constants module for the alias table + name-filter keywords (or inline a module-level dict in `live_apis.py`).

### Validation

1. Unit test: `pytest tests/unit/test_airport_resolution.py` (new) — covers `"new york city"` → JFK/LGA/EWR, `"nyc"` → ditto, `"washington dc"` → DCA/IAD, `"tempe"` → PHX, `"l.a."` → LAX, plus the no-direct-airport case (`"flagstaff"` → FLG).
2. Manual: re-run the user's NYC query in `app.py`; confirm the progress log shows JFK/LGA/EWR rather than NYS.
3. No regression on `python -m eval.run_eval --split validation --system multi --limit 20` (eval doesn't hit live airport resolution, so this is a smoke check that nothing else broke).

### Risk

Low. Layers A+B are pure string/data-filter changes. Layer C is gated behind A+B and only fires for unusual destinations.

---

## Issue 2 — `KeyError: 'claude-haiku-4-5@20251001' not in registry`

### Root cause

`agents/models.py:113`:

```python
_REGISTRY_BY_ID: dict[str, ModelEntry] = {m.id.split("@", 1)[0]: m for m in REGISTRY}
```

The dict is keyed by the **bare** model name (`claude-haiku-4-5`), but `ModelEntry.id` retains the `@revision` suffix (`claude-haiku-4-5@20251001`). When the UI displays models via `available_models()` and the user picks one, the selected `entry.id` (full, with `@`) gets passed back through some path into `get_model()` → KeyError.

`agents/providers/claude.py::_vertex_model_id` already strips `@` for the actual API call, but `get_model()` does the lookup first, so it crashes earlier.

### Fix

Single change in `agents/models.py::get_model` (lines 116–123):

```python
def get_model(model_id: str) -> ModelEntry:
    bare = model_id.split("@", 1)[0]
    entry = _REGISTRY_BY_ID.get(bare)
    if entry is None:
        raise KeyError(
            f"Model {model_id!r} not in registry. Available: {sorted(_REGISTRY_BY_ID)}"
        )
    return entry
```

That's it — the registry keys are already bare; just normalize the input the same way.

### Files

- `agents/models.py` — 2-line change inside `get_model`.

### Validation

1. Unit test: `pytest tests/unit/test_models.py::test_get_model_strips_revision` (new, ~4 lines).
2. Manual: launch `streamlit run app.py`, pick "Claude Haiku 4.5" in the model dropdown, run any query — KeyError should not surface.
3. CLI: `python -m eval.run_eval --split validation --system multi --models claude-haiku-4-5@20251001 --limit 1` — should resolve identically to `--models claude-haiku-4-5`.

### Risk

None — strictly more permissive; bare IDs continue to work.

---

## Issue 3 — Eval row failures

Two separate failure modes were lumped together in the user's report. They need different fixes.

### Issue 3a — `RuntimeError: No live X results available in tool context` (rows 180, 171, 179, 177)

#### Root cause

Each specialist (`agents/{transport,lodging,dining,sightseeing}.py`, lines 46/51/63/45) raises this when its `_format_tool_context` returns `"(none)"`/`"(no transport data available)"`. In the **eval path**, this happens when `sandbox.{flight,accommodation,restaurant,attraction}_search(intent.dest)` returns an empty list — i.e., a destination row exists in `validation.csv` but has no matching reference rows in TravelPlanner's `*_ref_info.jsonl`.

The guard exists for the live path, where empty == silent API failure (per the `ARCHITECTURE.md` §11 strict-runtime invariant). In the sandbox path, empty == legitimate data gap.

#### Fix (do both)

**A. Patch the sandbox upstream (preferred — heals data, not symptoms).** In `eval/run_eval.py::_sandbox_tool_context` (lines 82–140), after each search:

- If `sandbox.accommodation_search(intent.dest)` returns `[]`, retry with the destination's parent **state** (look up via `Dataset/cities.json` if present, or split the dest on common patterns).
- If still empty, fall back to the dest's nearest large city (use the same `airportsdata` nearest-by-distance with the Layer-B commercial-airport filter from Issue 1).
- Log every such expansion in the per-row `WARN` stream so the report can quote it.

**B. Add a `data_gap` exit path that doesn't crash.** In `_run_one` (lines 195–), after `_sandbox_tool_context`, validate:

```python
if not (ctx.get("hotels") and ctx.get("restaurants") and ctx.get("attractions")):
    return _empty_plan(intent), ConstraintReport.data_gap(missing=[...]), zero_stats
```

Track `data_gap` rows separately in the final summary table so they're visible but don't masquerade as system bugs. The macro denominator stays honest (the row counts as a fail), but the failure mode is labeled.

#### Files

- `eval/run_eval.py` — add `_expand_sandbox_search` helper, validation guard in `_run_one`, `data_gap` row tag.
- `eval/constraints.py` — add `ConstraintReport.data_gap(...)` factory or a sentinel field.
- `eval/format_report.py` — render the `data_gap` count as a separate column.

#### Validation

1. Re-run the four specific failing rows with `--limit 1 --row-id <id>` (add the flag if not present, ~3-line addition).
2. Confirm `python -m eval.run_eval --split validation --system multi` no longer crashes on any row; the summary lists `data_gap` count.
3. The pre-existing pass-rate for *non*-data-gap rows must not move (cache stays warm; only error handling changed).

### Issue 3b — `LLM response was truncated before the JSON object was closed (got 632 chars)` (row 161)

#### Root cause

`agents/llm.py::_extract_json` (lines 348–353) raises a 503-tagged retryable RuntimeError. The retry loop (`_with_retry`, line 169) retries up to `_MAX_ATTEMPTS=3` times — but **with the same `max_tokens`**. If the prompt+model combo deterministically truncates at 632 chars, all 3 retries truncate identically.

632 chars is small — this is a specialist call where `max_tokens` was set too low for the actual output (likely lodging or sightseeing on a long trip with many candidate hotels/attractions to enumerate).

#### Fix (do both)

**A. Token-budget escalation on truncation.** Distinguish `TruncationError` from generic 503s:

1. Subclass / sentinel `TruncationError(RuntimeError)` raised by `_extract_json`.
2. In `_with_retry` (or in `do_call`), when retry is triggered by a `TruncationError`, double `effective_max` (cap at, say, 16384) for the next attempt. Other 503-class retries (rate limit, overload) keep the same token budget.
3. After the call completes, the next call resets to the original `max_tokens` (this is per-call state, not persistent).

**B. Audit per-specialist `max_tokens` defaults.** Cross-check the documented values from the previous latency audit:
- transport: 800
- lodging: 600
- dining: 1500
- sightseeing: 800

The dining call for a 7-day trip emits `~7 * 3 * ~70 chars = 1500` chars JSON in the *best* case — leaving zero headroom. Bump dining to **2400** and sightseeing to **1200** (raises the floor for long trips; truncation escalation in (A) handles the long tail).

#### Files

- `agents/llm.py` — `TruncationError` class, retry-loop branch, `_extract_json` raise type.
- `agents/dining.py` (line 164), `agents/sightseeing.py` (line 113) — bump `max_tokens` argument.
- `tests/unit/test_llm_retry.py` — new test that mocks a truncated response and confirms the second attempt is called with `2 × max_tokens`.

#### Validation

1. `pytest tests/` — must stay 100%.
2. Re-run row 161 specifically; confirm it now succeeds with one retry that doubled tokens.
3. `python -m eval.run_eval --split validation --system multi --llm-cache-dir .cache/llm` — no truncation errors in the log; tokens-per-row in the summary may rise slightly for the long-trip tail, which is expected.

#### Risk

Low. Token escalation is bounded by the cap; cache is keyed on `effective_max`, so a successful escalation creates a separate cache entry rather than poisoning the original.

---

## Issue 4 — Eval data gaps: Kaggle 2022 flight backfill + Places API venue backfill (with deflation)

### Goal

Stop `_sandbox_tool_context` from returning empty fields for rows whose `(org, dest, date)` triple is missing from `*_ref_info.jsonl`. Preserve **paper-comparability** by keeping the canonical 2022 sandbox as primary; only fill in gaps from sources that produce 2022-dollar-equivalent prices.

### Sources, in priority order (per record type)

**Flights**

1. `Dataset/<split>_ref_info.jsonl` (current canonical sandbox).
2. `Dataset/kaggle_flights_2022.csv` — Dilwong's *Flight Prices Dataset* on Kaggle, April–October 2022. Same source TravelPlanner used; prices are 2022-native, no scaling needed.
3. None → row marked `data_gap=flights`; transport specialist falls back to driving (existing path).

**Hotels / restaurants / attractions**

1. `*_ref_info.jsonl`.
2. Google Places API (`googlemaps.Client.places_nearby` for hotels with geo, `search_places` for restaurants/attractions).
3. None → row marked `data_gap=<category>`; row counted as fail with a labeled tag (per Issue 3a).

### Cost-deflation rule for Places API gap-fills

The sandbox's price levels were calibrated against 2022 dollars (the cost map `{0: 60, 1: 90, 2: 130, 3: 190, 4: 280}` in `agents/lodging.py:97` is mid-2020s nominal; the verifier and Budget Agent compare against TravelPlanner-era 2022 budgets in `validation.csv`).

Define a single deflation factor **D = 1 / 1.20 ≈ 0.833** (mirrors the `--inflation 1.20` in `scripts/refresh_sandbox.py`, applied in the opposite direction). Apply it whenever a Places API record fills a sandbox gap:

```python
# pseudo
def _deflate_cost(modern_cost: int) -> int:
    return int(round(modern_cost * 0.833))
```

Per category:

- **Hotels.** When backfilled from Places, derive nightly cost from `price_level` using the existing cost map, then multiply by `D` and write the deflated value into `est_nightly` on the record. Don't overwrite the `price_level` itself — only the dollar mapping that the dining/lodging prompts surface.
- **Restaurants.** Same: derive `est_cost` from `price_level`, deflate, surface deflated.
- **Attractions.** Free in TravelPlanner; no cost field to deflate.

This keeps the `budget` evaluator (`eval/constraints.py:204`) on the same dollar scale as `validation.csv`'s `budget` column, regardless of which source filled the row.

### Implementation

**A. Kaggle 2022 flight loader.** New `tools/kaggle_flights.py`:

- One-time download script (`scripts/fetch_kaggle_flights.py`) — uses `kaggle` CLI or a direct URL; writes `Dataset/kaggle_flights_2022.csv`. Document in README that this is a one-time setup step for evaluators reproducing our numbers.
- At import, parse the CSV into `dict[(origin, dest, date_iso), list[FlightRecord]]`. Match the same record shape the sandbox emits (`Flight Number`, `DepTime`, `ArrTime`, `ActualElapsedTime`, `Price`, `FlightDate`).
- Public function `kaggle_flight_search(origin, dest, date) -> list[dict]` that mirrors `Sandbox.flight_search` so the eval can call it as a fallback with no shape changes.
- Cache the parsed dict in an `lru_cache(maxsize=1)` so the 100 MB CSV is loaded once per run.

**B. Wire it into eval.** In `eval/run_eval.py::_sandbox_tool_context`:

```python
raw_out = sandbox.flight_search(intent.org, intent.dest, intent.dates[0])
if not raw_out:
    raw_out = kaggle_flight_search(intent.org, intent.dest, intent.dates[0])
    if raw_out:
        progress(f"Flights filled from Kaggle 2022: {len(raw_out)} options")
```

Same fallback pattern for `accommodation_search`, `restaurant_search`, `attraction_search` → if empty, call Places API (gated behind `--allow-live-fallback` CLI flag), apply deflation, log the substitution.

**C. Telemetry.** Add three counters to the per-row report:

- `sandbox_hits`: how many of the 4 categories came from `*_ref_info.jsonl`.
- `kaggle_hits`: flights came from Kaggle.
- `live_hits`: any category came from Places.

The summary table at the end prints `sandbox/kaggle/live` row counts so the report can quote: "of 180 rows, 168 fully sandbox, 8 used Kaggle flight backfill, 4 used Places API backfill."

### Files

- `tools/kaggle_flights.py` (new) — loader + lookup.
- `scripts/fetch_kaggle_flights.py` (new) — one-time download.
- `eval/run_eval.py` — backfill chain in `_sandbox_tool_context`, `--allow-live-fallback` flag, summary counters.
- `tools/live_apis.py` — small `_deflate_cost` helper exposed for the eval path.
- `Project_Code/.gitignore` — add `Dataset/kaggle_flights_2022.csv` (raw data, ~100 MB).
- `README.md` — document the one-time Kaggle fetch step.

### Validation

1. With `kaggle_flights_2022.csv` present, re-run rows that previously failed with empty flights. Confirm transport specialist now produces a flight string instead of falling through to driving.
2. `--allow-live-fallback` off → identical numbers to current sandbox-only run (regression gate).
3. `--allow-live-fallback` on → strictly ≥ pass rate of off; new `kaggle_hits` and `live_hits` counters non-zero on the previously-broken rows.
4. Spot-check 3 backfilled hotel rows: confirm `est_nightly` after deflation is within ±5% of comparable sandbox hotel prices in the same city.

### Risk

Medium. The Kaggle CSV is large (~100 MB) and one-time provisioning is required for any new evaluator. Document loudly in README. Deflation factor is defensible (CPI 2022→2026 ≈ +18–20%) but pick **one** number (0.833) and put a short note in the report's appendix justifying it.

---

## Issue 5 — Per-rule recovery: surface missing constraint fields (cuisine, room_type, room_rule, min_nights)

### Diagnosis (from the per-rule pass-rate table)

| Rule | Pass rate | Field needed | Currently in ToolContext? |
|------|-----------|--------------|---------------------------|
| `cuisine` | **0/17** | restaurant `Cuisines` tag | ❌ stripped at `run_eval.py:120` |
| `room_type` | **3/22** | hotel `room type` field | ❌ stripped at `run_eval.py:108` |
| `room_rule` | **7/25** | hotel `house_rules` field | ❌ stripped at `run_eval.py:108` |
| `min_nights` | **38/61** | hotel `minimum nights` field | ❌ stripped at `run_eval.py:108` |
| `budget` | 58/61 | (cost target) | ✅ surfaced; minor specialist drift only |

The `*_ref_info.jsonl` records contain every needed field (verified by inspection of `validation_ref_info.jsonl`):

```
Restaurants → keys: ['Name', 'Average Cost', 'Cuisines', 'Aggregate Rating', 'City']
Accommodations → keys: ['NAME', 'price', 'room type', 'house_rules', 'minimum nights',
                        'maximum occupancy', 'review rate number', 'city']
```

The eval flattens these to `name | rating | price_level` and `name | rating | address | price_level` — losing the exact attributes the verifier scores. The LLM is being graded blind.

### Fix (eval-path; live-app caveat below)

**A. Pass the fields through ToolContext** (`eval/run_eval.py::_sandbox_tool_context`):

```python
ctx["hotels"] = [
    {
        "name": r.get("NAME"),
        "rating": r.get("review rate number"),
        "address": r.get("city", intent.dest),
        "price_level": _price_to_level(r.get("price")),
        "room_type": r.get("room type"),         # NEW
        "house_rules": r.get("house_rules"),     # NEW
        "min_nights": int(r.get("minimum nights") or 1),  # NEW
        "max_occupancy": int(r.get("maximum occupancy") or 0),  # NEW
    }
    for r in sandbox.accommodation_search(intent.dest)
]

ctx["restaurants"] = [
    {
        "name": r.get("Name"),
        "rating": r.get("Aggregate Rating"),
        "price_level": _avg_cost_to_level(r.get("Average Cost")),
        "cuisines": r.get("Cuisines"),           # NEW
    }
    for r in sandbox.restaurant_search(intent.dest)
]
```

**B. Render the new fields in specialist prompts.**

`agents/lodging.py::_format_tool_context` (line 93) — change to:

```
[1] Hotel Van Zandt | rating:4.6 | est_nightly:$189 | room:Entire home/apt | rules:no_smoking,no_pets | min_nights:2
[2] ...
```

`agents/dining.py::_format_tool_context` (line 168) — change to:

```
[1] Joe's Pizza | rating:4.5 | est_cost:$28 | cuisines:Italian,Pizza
[2] ...
```

**C. Tighten the SYSTEM prompts.**

`agents/lodging.py::SYSTEM` — add rules:

- "If `intent.house_rule` is set, you MUST pick a hotel whose `rules` field contains a compatible match (e.g. `house_rule='pets allowed'` requires `rules` does NOT contain `no_pets`)."
- "If `intent.room_type` is set, you MUST pick a hotel whose `room` field equals the requested type."
- "Reject any candidate whose `min_nights` exceeds the trip length."

`agents/dining.py::SYSTEM` — replace the existing "respect the cuisine constraint" bullet with:

- "If `intent.cuisine` is set (it may be comma-separated for multiple cuisines), at least one meal **per requested cuisine** must come from a restaurant whose `cuisines` field contains that cuisine. Match on the explicit `cuisines` tag — never guess from the restaurant name."

**D. Add a deterministic post-hoc filter (insurance against LLM drift).**

`agents/lodging.py::run` — after the LLM returns a plan but before validation:
- If `intent.house_rule` is set and the picked hotel's `house_rules` would violate it, deterministically swap to the first sandbox hotel that **does** satisfy. (Pattern already exists for `_substitute_hotel`.)
- Same for `room_type` and `min_nights`.

`agents/dining.py::_substitute_restaurants` already does cuisine-aware substitution, but it currently uses name-keyword matching (`CUISINE_SYNONYMS`). Update it to read `ctx["restaurants"][i]["cuisines"]` directly when present, falling back to the name-keyword match only when the field is missing (live-app path).

### Live-app caveat (explicit, document this)

Google Maps Places API does **not** return `room_type`, `house_rules`, or `minimum_nights`. For the live `app.py` path:

- `cuisines` can be partially recovered from Places `types` field (`["restaurant", "italian_restaurant"]`) — fix is best-effort; record it as `cuisines: italian` when the type contains a cuisine token.
- `room_type` and `house_rules` cannot be recovered from Places. The live-app path remains best-effort; the report should note that the eval gains on these rules don't transfer to the live UX.

### Files

- `eval/run_eval.py` — extend `_sandbox_tool_context` (8 new fields total).
- `agents/schemas.py::ToolContext` — extend the `hotels` and `restaurants` `TypedDict` shapes with the new fields. **Keep them all `Optional`** so the live path can omit them without type errors.
- `agents/lodging.py` — `_format_tool_context`, `SYSTEM`, post-hoc filter in `run`.
- `agents/dining.py` — `_format_tool_context`, `SYSTEM`, update `_substitute_restaurants`.
- `tools/live_apis.py::search_places` — populate `cuisines` from Places `types` when available; leave `room_type`/`house_rules` as `None` for hotels.

### Targets

| Rule | Current | Target | Stretch |
|------|---------|--------|---------|
| `cuisine` | 0/17 | **14/17** | 16/17 |
| `room_type` | 3/22 | **18/22** | 21/22 |
| `room_rule` | 7/25 | **20/25** | 24/25 |
| `min_nights` | 38/61 | **56/61** | 60/61 |

If targets aren't hit on the first re-run, escalate the relevant SYSTEM-prompt rule from advisory to imperative and re-run with `think_first=True` (cheap because cache stays warm for the unaffected rules).

### Validation

1. Per-rule deltas in the `Per-rule pass rates` block of the eval output. Track each rule's pre/post number explicitly in the PR description.
2. No regression on the rules currently passing (`complete_info`, `diverse_*`, `non_conflicting_transport`, `reasonable_city_route`, `transportation`, `within_current_city`).
3. `within_sandbox` must stay ≥ current 59/61 — surfacing more fields shouldn't change which names the LLM picks.

### Risk

Low for the prompt + ToolContext changes (no schema-breaking moves). Medium for the deterministic post-hoc swap — make sure the swap preserves all other constraints (price, min_nights itself, max_occupancy).

---

## Issue 6 — Flight selection: N+N pairing (top-K outbound × top-K return)

### Current behavior

`tools/live_apis.py::flight_search_round_trip` (lines 481–672):
- Call 1: fetch up to `max_results=10` outbounds, sort by `(duration, price)`.
- Call 2: for the **single** best outbound (`top_outbounds[0]`), fetch return options.
- Output: `(out_strs[:10], ret_strs=[best_return])` — 10 outbound + **1** return.

The specialist sees 10 outbound options but only 1 return. If it picks `out_strs[3]`, the return paired with `out_strs[0]` may not be feasible (different airline, different connection city, different return-day departure window).

### Redesign (N+N with paired scoring)

**A. Top-K outbound selection (Pareto + budget filter).** Same as today but expose `K=5`:

- Filter by duration cap (already done at lines 575–582).
- Filter by `budget` if set (already done at 584–587).
- Sort by `(duration_min, price)` — Pareto-optimal first 5.

**B. Top-K return selection per outbound.** For each of the top-K outbounds, call SerpAPI with that `departure_token` to fetch its returns. Keep top 3 returns per outbound, ranked by:

- Existing late-return preference (≥14:00 → ≥11:00 → no constraint).
- Duration cap (existing).
- For round-trip total price: lowest blended `out.price + ret.price_delta` first.

This is K=5 outbound × 3 return = 15 SerpAPI Call-2 invocations worst case. Each is one HTTP roundtrip, so ~5–10 s added to a flight search. Acceptable for app UX; gated by cache for eval reproducibility.

**C. Specialist sees both lists, picks the pair.** Update `_compose_flight_segments` and the transport specialist prompt to render:

```
Outbound options:
  [O1] Delta 1234 09:00→11:30 Duration: 2h30m Cost: $310
  [O2] United 5678 06:15→09:00 Duration: 2h45m Cost: $295
  [O3] American 9012 12:00→14:30 Duration: 2h30m Cost: $340
  [O4] ...
  [O5] ...

Return options:
  [R1] Delta 4321 17:00→20:00 Duration: 3h Cost: $0  (paired with O1; total $310)
  [R2] United 8765 18:00→21:00 Duration: 3h Cost: $20 (paired with O2; total $315)
  [R3] American 2109 16:00→19:00 Duration: 3h Cost: $30 (paired with O3; total $370)
  [R4] ...
  [R5] ...

Pick ONE outbound and ONE return. The (Oi, Ri) pair must be from the same row above —
do NOT mix across rows. Day 1 transportation = the picked outbound; last day transportation = the picked return.
```

The "must be from the same row" wording prevents incompatible pairings (different airlines, different connection cities). Round-trip total cost is declared on Day 1 only (existing convention).

**D. Schema additions.** `agents/schemas.py::ToolContext`:

```python
flights_outbound: list[str]            # already exists
flights_return: list[str]              # already exists
flights_pairs: list[tuple[int, int]]   # NEW — (out_index, ret_index) of compatible pairs
flights_round_trip: bool               # already exists
```

Eval and live paths both populate `flights_pairs`. When `flights_round_trip=True`, only pairs in `flights_pairs` are valid; the specialist's `_invalid_flight_ids` check (already in `agents/transport.py:65`) extends to reject mismatched pairs.

**E. One-way fallback (when round-trip search fails).** Same Pareto-rank logic, but the two lists are independent. No `flights_pairs` constraint. Specialist picks one of each.

### Files

- `tools/live_apis.py::flight_search_round_trip` — return `(out_strs, ret_strs, pairs)`; coordinator unpacks.
- `agents/coordinator.py:693` — adapt the unpacker; populate `ctx["flights_pairs"]`.
- `agents/transport.py` — render the paired layout in `_format_tool_context`; tighten `_invalid_flight_ids` to also enforce pair compatibility.
- `agents/schemas.py::ToolContext` — add `flights_pairs`.

### Validation

1. Existing `non_conflicting_transport` rule (`61/61`) must stay perfect.
2. New unit test in `tests/unit/test_flight_pairing.py`: assert that `flights_outbound[2]` paired with `flights_return[0]` (mismatched index) raises in `_invalid_flight_ids`.
3. Manual app run: confirm the rendered prompt shows the paired layout; confirm cost on Day 1 == round-trip total, last day cost == "-".
4. Latency budget: the K=5 × 3 expansion adds ~5 s to round-trip search. Confirm app P95 stays under the existing 60s ceiling.

### Risk

Medium-high. SerpAPI's departure_token return-call per outbound is the cost driver. Add a hard cap of 5 outbound × 3 return = 15 calls per flight search, with circuit breaker (if 3 consecutive return-calls fail, fall back to one-way). Cache aggressively at the LLM-cache layer.

---

## Issue 7 — Candidate pool sizing scaled to constraints

### Diagnosis

`agents/coordinator.py:755-757`:

```python
n_hotels      = min(20, max(8,  intent.days * 2))
n_restaurants = min(40, max(12, intent.days * 4))
n_attractions = min(30, max(12, intent.days * 3))
```

For the most common test row shape (3-day trip), all three saturate at the floor: **8 / 12 / 12** — exactly what the user observed. The pool sizes don't react to the *number* or *specificity* of constraints, so a 3-day trip with `cuisine=Indian, room_type=Entire home/apt, house_rule=pets allowed` gets the same pool as an unconstrained trip.

For multi-cuisine trips, the issue compounds: 12 random restaurants in a city like Myrtle Beach typically include 0–2 Indian, 0 BBQ, 0 Sushi. The dining specialist can't satisfy a `cuisine=Indian, BBQ, Sushi` constraint regardless of prompt quality.

### Redesign

**New floors and per-constraint scaling:**

```python
# Hotels — must satisfy the 3-way conjunction (price ∧ room_type ∧ house_rule ∧ min_nights).
# 15 candidates is the empirical minimum to cover all four with non-trivial probability.
n_hotels = min(25, max(15, intent.days * 3))

# Restaurants — scaled by cuisine specificity.
cuisine_count = len([c for c in (intent.cuisine or "").split(",") if c.strip()])
n_restaurants = min(50, max(16, intent.days * 4 + cuisine_count * 4))

# Attractions — paper-style ~3/day; +4 buffer for diverse_attractions slack.
n_attractions = min(35, max(16, intent.days * 4))
```

Examples:
- 3-day trip, no cuisine: `15 / 16 / 16` (was 8/12/12).
- 3-day trip, cuisine=Indian: `15 / 20 / 16`.
- 3-day trip, cuisine=Indian,BBQ,Sushi: `15 / 28 / 16` — actually has a chance.
- 7-day trip, cuisine=Italian,Japanese: `21 / 36 / 28`.

**Apply the same scaling in the eval path** (`_sandbox_tool_context` currently doesn't subset — it returns whatever the sandbox has). For sandbox parity, **truncate** the sandbox lists to the same `n_*` values *after* prioritising by relevance (rating, then cuisine match for restaurants). This keeps eval and live behaviour identical at the prompt-input level.

**Cuisine-aware ranking for restaurants** (eval and live):

When `intent.cuisine` is set, sort the candidate pool so that cuisine-matching restaurants come first, then by rating. This guarantees the top `n_restaurants` includes every cuisine match the source has — the LLM doesn't have to hunt through 40 random rows for the 2 Indian places.

```python
def _rank_restaurants(rows, cuisine_tokens, max_results):
    def score(r):
        cuisines = (r.get("cuisines") or "").lower()
        cuisine_hit = any(t in cuisines for t in cuisine_tokens)
        rating = float(r.get("rating") or 0)
        return (0 if cuisine_hit else 1, -rating)
    return sorted(rows, key=score)[:max_results]
```

Same idea for hotels when `intent.room_type` or `intent.house_rule` is set: pre-filter, *then* truncate.

### Files

- `agents/coordinator.py:755-757` — new formula.
- `eval/run_eval.py::_sandbox_tool_context` — add ranking + truncation for parity.
- `agents/coordinator.py` — small ranking helpers (or a new `tools/candidate_ranker.py` if it grows).

### Validation

1. Cuisine pass rate (Issue 5) — Issues 5 and 7 are coupled; Issue 7 *enables* Issue 5's targets to be hit.
2. Re-run all current rules; expect:
   - `cuisine`: large jump (combined effect with Issue 5).
   - `diverse_restaurants`: stays at 61/61 (more candidates can't hurt diversity).
   - `room_type`, `room_rule`, `min_nights`: medium jump (more candidates means at least one likely satisfies).
3. Latency: bigger pools = bigger prompts. Token budget per specialist may need a small bump (already on the table for Issue 3b).
4. Token-cost regression: the per-row `total_in_tok` will rise ~30% on cuisine-constrained rows. Acceptable for the per-rule-pass-rate gains; re-baseline in the report.

### Risk

Low. Pool sizing is a tuning change; everything downstream consumes it identically.

---

## Issue 8 — Reduce LLM call count (~40% per-row savings without accuracy loss)

### Diagnosis (per-row LLM call inventory)

| Path | Source | Calls |
|------|--------|-------|
| Parse intent | `coordinator.py:82` (skipped in eval; intent passed in) | 0–1 |
| 4 specialists, 1st pass | `transport.py`, `lodging.py`, `dining.py`, `sightseeing.py` | 4 |
| In-specialist retry on `_invalid_names` (hallucinated name) | `dining.py:63`, `lodging.py:57`, `transport.py:74` | up to 3 |
| Repair loop, `REPAIR_MAX_ROUNDS=3` rounds × up to 4 specialists | `coordinator.py:115, 1061` | up to 12 |
| **Worst case** | | **20** |
| **Observed in `eval/results.csv` (3-row sample)** | | 6, 6, 11 (avg 7.7 net of cache; p95=11) |

### Four leak points

**L1. In-specialist retry rarely succeeds.** When the LLM hallucinates a restaurant name once, the second call (with a "you got these wrong: [...]" note) usually hallucinates again. Mechanical substitution (`_substitute_restaurants`, `_substitute_hotel`) runs **after** the retry fails — so we pay for an extra LLM call whose typical outcome is "still wrong, fall through to mech substitution anyway."

**L2. `responsible="coordinator"` re-runs ALL 4 specialists.** `coordinator.py:1061`:

```python
elif v.responsible == "coordinator":
    rerun.update({"transport", "lodging", "dining", "sightseeing"})
```

When a coordinator-attributed violation fires (e.g., `complete_info` because one field is `None`), the repair re-runs every specialist. Most don't fix the violation; they just re-emit identical plans. Wastes 3 calls on the typical coordinator-violation row.

**L3. Round 3 of the repair loop is dead weight.** `REPAIR_MAX_ROUNDS=3`. Empirically (per the per-rule pass-rate table), rules that fail at round 2 fail at round 3 too — the LLM doesn't have new information to act on. Each round-3 specialist call burns ~600 input tokens to confirm a foregone conclusion.

**L4. Repair calls re-send the full candidate pool.** A `cuisine` repair on dining shouldn't need all 12 restaurants — it needs the 2–4 cuisine-matching ones. But the prompt is regenerated identically by `_format_tool_context`. Doesn't change call *count*, but inflates per-call cost ~3×.

### Fixes

**A. Drop in-specialist retry on hallucinated names; go straight to mechanical substitution.** (`dining.py:54-66`, `lodging.py:50-62`, `transport.py:67-80`)

```python
# Before:
plan = _call(intent, ctx, tool_results, repair_note)
invalid = _invalid_names(plan, allowed)
if invalid:
    plan = _call(intent, ctx, tool_results, retry_note)   # ← LLM call 2
    if _invalid_names(plan, allowed) and allowed:
        plan = _substitute_restaurants(plan, allowed, intent, ctx)  # ← mech anyway

# After:
plan = _call(intent, ctx, tool_results, repair_note)
invalid = _invalid_names(plan, allowed)
if invalid and allowed:
    plan = _substitute_restaurants(plan, allowed, intent, ctx)  # straight to mech
```

Quality impact: minimal. Mechanical substitution is grounded by construction (always picks an allowed name). The retry's supposed advantage was "maybe the LLM picks a *better* allowed name on round 2"; in practice the cuisine-aware ranking in `_prioritised_pool` (Issue 5+7) already produces equal-or-better picks. **Saves up to 3 calls/row.**

**Exception:** keep the retry **only** for `lodging` when `intent.house_rule` or `intent.room_type` is set, because mechanical substitution can't tell which hotel satisfies those constraints from name alone (Issue 5 surfaces them, but the deterministic swap may pick wrong if multiple hotels qualify). Skip retry otherwise.

**B. Stop "rerun all 4" on coordinator violations.** (`coordinator.py:1058-1063`) Walk the 12 verifier rules in `agents/rules.py` and classify each into one of:

- A specific specialist (`transport` / `lodging` / `dining` / `sightseeing`).
- `mechanical-only` — fixable without an LLM call (e.g., adding missing `current_city` strings, recomputing `Cost: $N` totals from existing entries).

Eliminate the `responsible="coordinator" → rerun all 4` branch. If a violation truly can't be assigned to one specialist, fix it mechanically in `_mech_repair_*` and skip the LLM rerun entirely. **Saves ~3 calls/row on ~20% of repair-triggering rows.**

Concrete rule-to-specialist mapping (proposed):

| Rule | Responsible | Currently |
|------|-------------|-----------|
| `complete_info` | mechanical (fill `-` for missing fields) | coordinator → all 4 |
| `within_current_city` | mechanical (rewrite `current_city` field) | coordinator → all 4 |
| `reasonable_city_route` | transport | coordinator → all 4 |
| `non_conflicting_transport` | transport | already transport |
| `min_nights` | lodging | already lodging |
| `room_type` / `room_rule` | lodging | already lodging |
| `cuisine` | dining | already dining |
| `diverse_restaurants` | mechanical (`_mech_repair_dining` exists) | already mech |
| `diverse_attractions` | mechanical (analogous to dining) | sightseeing |
| `within_sandbox` | mechanical (`_substitute_*` exists) | per-specialist |
| `budget` | mechanical (downscale prices in cost map) | budget agent + lodging |
| `transportation` | transport | already transport |

Result: the only coordinator-routed violation left is `complete_info`, and it's mechanical. Zero remaining all-specialist re-runs.

**C. `REPAIR_MAX_ROUNDS = 2`** (was 3). One-line change at `coordinator.py:115`. **Saves up to 4 calls/row in the long tail.**

Validation gate: if pre/post macro pass rate diverges by > 0.5 pp on `--system multi --limit 60`, revert to 3. Empirically the round-3 contribution to pass rate is statistical noise; this is a low-risk cut.

**D. Repair-pass tool_context narrowing.** `coordinator.py::_run_specialist` (line 388) currently passes the full `tool_context` regardless of whether it's a first pass or a repair. Add a repair-only path that filters the candidate lists down to the relevant subset:

- `cuisine` repair → restaurants filtered to cuisine matches only.
- `min_nights` repair → hotels filtered to those whose `min_nights ≤ trip_days`.
- `room_type` / `room_rule` repair → hotels filtered to attribute matches.
- `diverse_*` repair → drop names already used in the previous attempt.

Implementation: add an optional `repair_filter` arg to `_run_specialist`; the repair_prepare_node populates it from the violation list. Doesn't reduce call count; reduces input tokens **~70% on repair calls**, which translates to ~30% wall-clock and cost.

**E. Skip transport specialist when transportation is fully determined.** When (a) `intent.transportation == "self-driving"` AND `flights_outbound == []` (because driving threshold was met), or (b) all flight options were filtered out and only driving remains, the transport plan is deterministic: copy the drive-leg string for arrival/departure days, `-` for in-city days. Replace the LLM call with a deterministic builder in `agents/transport.py`. **Saves 1 call/row on ~25% of trips** (those under the 100-mile drive threshold or with `transportation="no flight"`).

**F. Tighten parse_intent caching.** Confirm `agents/llm.py::_cache_key` for parse_intent only includes the **query string**, not transient state. (Currently it includes schema name + max_tokens; should be deterministic for identical queries.) Live app only — eval bypasses parse_intent. Likely a 1-line check.

### Projected savings (per row, eval path)

| Scenario | Frequency | Current calls | After A+B+C+E | Δ |
|----------|-----------|---------------|---------------|---|
| No-repair, no hallucination | ~60% | 4 | 3 (E saves 1 on ~quarter; A doesn't fire) | ~−0.3 avg |
| Hallucinated name, 1 retry | ~15% | 5 | 4 (A saves the retry) | −1 |
| 1 repair round | ~15% | 6–8 | 4–5 (B narrows scope) | ~−2 avg |
| 2+ repair rounds | ~10% | 9–17 | 5–7 (B+C compound) | ~−6 avg |
| **Weighted average** | | **~7.7** | **~4.5** | **~−40%** |

Token-cost savings from D add roughly **+30% on top** of call-count savings, because tail repair calls are also the longest prompts.

### What this does NOT change

- Specialist count in first pass stays at 4 — required for the paper §4.7 ablation comparison (`no_specialization` baseline).
- Combining specialists into one batched call — would defeat the multi-agent design the report rests on.
- Verifier and Budget Agent stay deterministic (already 0 LLM calls).
- `parse_intent` stays in the live-app path — needed for free-text input.

### Files

- `agents/dining.py:54-66` — drop retry path; go to mech.
- `agents/lodging.py:50-62` — drop retry except when `house_rule`/`room_type` set.
- `agents/transport.py:67-80` — drop retry path.
- `agents/coordinator.py:115` — `REPAIR_MAX_ROUNDS = 2`.
- `agents/coordinator.py:1058-1063` — remove "coordinator → all 4" branch; route all rules through a single classified mapping.
- `agents/coordinator.py::_run_specialist` (line 388) — add `repair_filter` param.
- `agents/coordinator.py::repair_prepare_node` (line 970+) — populate `repair_filter` from violations.
- `agents/transport.py` — new `_deterministic_drive_plan(intent, ctx)` helper for fully-determined cases.
- `agents/coordinator.py::transport_node` — branch on the determined-transport check before calling the LLM.
- `agents/rules.py` — rule → responsible classification table (single source of truth).

### Validation

1. **Call-count regression gate.** Add a per-row `llm_calls` upper-bound check in `tests/integration/`: assert that a fixed reference query produces ≤ 4 calls when no repair is needed.
2. **Macro pass-rate gate.** `python -m eval.run_eval --split validation --system multi --limit 60` macro-hard pass rate must stay within −0.5 pp of pre-Issue-8 baseline. Larger drop reverts step C (rounds back to 3).
3. **Per-rule pass-rate gate.** No rule may regress by > 1 pp absolute. The post-Issue-5 targets stay valid.
4. **Wall-clock.** P50 latency should drop ~30%; P95 should drop ~50%. If P95 doesn't drop, step D didn't actually narrow the prompts.
5. **Cost.** Total token spend across `--system multi --models all` should fall ~40% on a re-run vs. the current `eval/results.csv` baseline.

### Risk

- **A (drop retry):** Low. Mechanical substitution already exists and runs as the eventual fallback.
- **B (route fixing):** Low-medium. Requires careful rule classification; if a rule is misrouted, the violation persists. Mitigated by keeping `_mech_repair_*` as the universal safety net.
- **C (max rounds 2):** Low. One-line revert if numbers move.
- **D (filtered context):** Low. Repair scope is already known per-violation; just plumbing.
- **E (deterministic transport):** Low-medium. Need to be careful about edge cases (multi-leg trips, mixed-mode days). Keep behind a flag and audit on app rows before turning on by default.
- **F (parse cache):** Trivial.

---

## Verified — `baseline/` is fully required (no cleanup)

Audited 2026-04-29. All three files in `baseline/` are imported and exercised:

| File | Imported by | Role | Paper §4.7 |
|------|-------------|------|-----------|
| `single_agent.py` | `app.py:18`, `eval/run_eval.py:44` | Streamlit "Single-agent" mode + `--system single` | Baseline 1 |
| `no_verify.py` | `eval/run_eval.py:43` | `--system no_verify` (pipeline minus verify-and-repair) | Baseline 2 |
| `no_specialization.py` | `eval/run_eval.py:42` | `--system no_specialization` (coordinator + one generalist worker) | Baseline 3 |

These four columns (`multi` + the three baselines) are the headline ablation table the report rests on. Do not delete or merge any of them.

---

## Sequencing (revised again)

| Order | Issue | Effort | Why this order |
|-------|-------|--------|----------------|
| 1 | **2** (registry KeyError) | XS (~5 min) | Strictly blocking; trivial fix; unblocks Claude testing |
| 2 | **3b** (truncation retry) | S (~30 min) | Self-contained in `llm.py`; unblocks long-prompt rows |
| 3 | **8A+8C** (drop retry, cap rounds) | S (~30 min) | Quick wins on call count; revert is one-line if eval regresses |
| 4 | **5** (per-rule field surfacing) | M (~2 h) | Biggest pass-rate lift; couples with 7 |
| 5 | **7** (pool sizing) | S (~45 min) | Cheap; required to make 5 land at target numbers |
| 6 | **8B+8D** (route fixing + filtered repair context) | M (~1.5 h) | Needs Issue 5's `responsible` mapping correct first |
| 7 | **3a** (sandbox data gaps + telemetry) | M (~1–2 h) | Cleaner logs after 5+7+8 |
| 8 | **8E** (deterministic transport) | M (~1 h) | Needs Issue 1's airport-resolution refactor stable |
| 9 | **4** (Kaggle + Places backfill with deflation) | L (~3–4 h) | Honest "no row crashes" claim for the report |
| 10 | **6** (flight N+N pairing) | M (~2 h) | UX polish; doesn't move eval rule numbers (already pass) |
| 11 | **1** (airport resolution) | M (~1–2 h) | App-only; do last |
| 12 | **8F** (parse_intent cache check) | XS (~10 min) | Trivial app-only verification |

After the chain lands: regenerate the canonical results with:

```bash
python -m eval.run_eval --split validation --system all --models all \
  --llm-cache-dir .cache/llm
```

Crashes should be zero. The per-rule pass-rate table should show the targets in Issue 5. The summary table should report `sandbox/kaggle/live_fallback/data_gap` row counts so the report can quote provenance honestly.

The report's headline TravelPlanner Table-3-analog row uses **frozen 2022 sandbox + Kaggle 2022 flight backfill only** — no live fallback. Live-fallback numbers go in an appendix column to demonstrate graceful degradation, not in the headline.
