# Plan — Latency, Accuracy, UI Cleanup, Cache & Tokens

Five asks, addressed in order. Each section lists the concrete files/lines that change, the rationale, and the validation step. **Do not start implementation yet — this is plan-only.**

The validation eval sweep (`python -m eval.run_eval --split validation --system all --llm-cache-dir .cache/llm`) is currently running in the background. None of these changes should land before that sweep completes — they would invalidate the report numbers mid-run.

---

## Item 1 — LLM response time + accuracy

### Diagnosis

Every specialist (`transport.py:98`, `lodging.py:89`, `dining.py:164`, `sightseeing.py:113`) calls `call_json(SYSTEM, user, ..., think_first=True)`. The `think_first=True` flag in `agents/llm.py:213, 238–248` does three things that are **silently expensive**:

1. **Floors `max_tokens` to 6000** (`llm.py:213`: `effective_max = max(max_tokens, 6000) if think_first else max_tokens`). Even a small lodging plan (one hotel, 3 days) is now generating up to 6000 output tokens — at typical decoding speeds that alone is 10–25 s per specialist call.
2. **Disables JSON mode** (`llm.py:245–248`). The model emits prose+JSON, then we run `_extract_json` (`llm.py:323–356`) to scrape the first `{...}` block. JSON mode cuts decoding overhead and removes a class of "no JSON found / truncated JSON" retry failures.
3. **Adds the "Think step by step…" preamble** to system. This biases the model toward chain-of-thought prose, lengthening the generation further.

On top of that, each specialist:
- Dumps the full `intent.model_dump_json()` into the user message even when most fields don't apply to that specialist (e.g., `cuisine` is irrelevant to lodging).
- Prints the entire candidate pool with rating + address + est_price for every hotel/restaurant/attraction.
- Carries 1–3 few-shot examples in the SYSTEM prompt (transport has 3, ~120 tokens each).

The retry path in `_with_retry` (`llm.py:169–180`) tries up to **8 attempts** with exponential backoff capped at 45s + jitter. A single failing call can therefore take ~3 minutes before surfacing an error.

The repair loop runs up to `REPAIR_MAX_ROUNDS=3` rounds; each round can re-call up to 4 specialists. A failing query is currently `4 specialists × 3 rounds × up to 6000 tokens × think_first` of output, plus retries.

### Changes

**A. Drop `think_first=True` to false-by-default; use it only on retries when the verifier finds the same rule violated twice.**

Files:
- `agents/transport.py:98`, `agents/lodging.py:89`, `agents/dining.py:164`, `agents/sightseeing.py:113` — change `think_first=True` to `think_first=False` on the **first** call.
- `agents/coordinator.py` repair_prepare_node — when the same `(rule, responsible)` key appears in both `prev_violation_keys` and the current round's violations, set a per-specialist flag that the next call should pass `think_first=True`. Pipe that through to specialists' `run()`. Only escalates after a failed first attempt; cheap path stays cheap.

**B. Lower per-specialist `max_tokens` budgets** (passed in the new `call_json(..., max_tokens=N)` call):
- transport: 800 (one to two short flight strings, max 14 days)
- lodging: 600 (one hotel × N days)
- dining: 1500 (3 meals × up to 14 days)
- sightseeing: 800 (a few attractions × N days)

The 6000-token floor only kicks in if `think_first=True`, which is now the retry path.

**C. Trim the dumped Intent JSON for each specialist.**

Add a small helper `agents/schemas.py::Intent.for_specialist(name)` that returns a dict with only fields relevant to that specialist:
- transport: `org, dest, days, dates, transportation`
- lodging: `dest, days, dates, people, house_rule, room_type, budget`
- dining: `dest, days, dates, people, cuisine, budget`
- sightseeing: `dest, days, dates, people`

Each specialist replaces `intent.model_dump_json()` in its `_call` user message with `json.dumps(intent.for_specialist("dining"))`. Rough savings: 80–150 input tokens per call.

**D. Trim the candidate pool string.**

The current `_format_tool_context` printers include `rating | address | est_nightly` per row. For a 28-restaurant pool this is ~1.5 KB of input. Drop fields the LLM doesn't actually use for selection:
- Lodging: keep `name | rating | est_nightly` — drop `address` (the LLM doesn't use it; coords are the geo filter's job).
- Dining: already minimal — keep as-is.
- Sightseeing: drop `address`. Keep `name | rating`.

Rough savings: 200–400 input tokens per specialist call on long-trip pools.

**E. Cap `_MAX_ATTEMPTS` from 8 → 3** in `agents/llm.py:132`, and shrink the backoff cap from 45 s → 12 s in `_backoff` (`llm.py:165–166`). Eight attempts × 45 s = 6-minute worst case for a single LLM call. Three attempts × 12 s caps a single failing call at ~36 s.

**F. Few-shot trimming.** `transport.py:46–62` carries 3 few-shot examples (~600 tokens). Keep example 1 (one-way) and example 3 (drive legs). Drop example 2 (round-trip without drive legs) — example 3 already covers the round-trip + drive-leg case which is more general. Apply equivalent trimming to other specialists if they carry redundant examples.

### Validation

- Run `python -m eval.run_eval --split validation --system multi --limit 30 --llm-cache-dir .cache/llm` before and after. Targets:
  - `latency_seconds` per row down by **40–60%** (think_first off cuts 6000-token decode → ~600-token decode).
  - `final_pass` should be flat or marginally higher (escalation to think_first on retry preserves the safety net).
  - `avg_llm_calls` flat (same logical structure, fewer extra tokens per call).
- Spot-check 3 verifier-failing queries manually: confirm the second-round repair call now uses `think_first=True` (look at the timeline log).

### Risk

`think_first=True` was added because some specialists were dropping verifier rules silently. Removing it from the first call could reduce first-pass accuracy. The escalation-on-repeat-violation path in (A) is the mitigation. If the eval shows a regression > 2 pp on `final_pass`, fall back to `think_first=True` for transport (the spec with the strictest grounding).

### Time

~1.5 h, gated on the running eval finishing.

---

## Item 2 — UI: truncated `From → To` and duplicate rocket-icon line

### Diagnosis

`app.py:212` renders the trip with `st.metric("Trip", f"{org} → {dest}")`. `st.metric` truncates long values to fit its narrow column — "Washington DC → Myrtle Beach" overflows.

`app.py:265–266` renders `st.caption(f"🚀 {_val(day.transportation)}")` on travel days. Then `app.py:270–275` renders the **same** transportation string under "**✈️ Transport**" inside `col_t`. Same fact, two places.

### Changes

**A. Replace the truncating metric with a markdown header.**

In `app.py:209–214`, replace the 3-column row containing the "Trip" metric with:
```
st.markdown(f"### {org} → {dest}")
c1, c2 = st.columns(2)
c1.metric("Days", total_days)
c2.metric("People", intent.people if intent else 1)
```
Wider header, no truncation, less visual repetition.

**B. Remove the rocket-icon caption.**

Delete `app.py:265–266`:
```python
if is_travel_day and not _is_missing_text(day.transportation):
    st.caption(f"🚀 {_val(day.transportation)}")
```
The transport detail is already shown in `col_t` below.

### Validation

Local visual check via `streamlit run app.py` with two queries:
- Long city names (e.g., "San Francisco → Charleston, SC") — header should wrap, not truncate.
- Day-1 / last-day rows — should show transportation only once, under "✈️ Transport".

### Time

~10 min.

---

## Item 3 — Remove sidebar "Live data sources" block, the Claude tooltip, and the bottom status caption

### Diagnosis

Three places leak runtime-config noise into the UI:

1. **Sidebar block** at `app.py:90–97`:
   ```python
   st.subheader("Live data sources")
   st.markdown(f"- **LLM planning:** {_ok if llm_available() else _missing}")
   st.markdown(f"- **Flight search:** {_ok if flights_available() else _missing}")
   st.markdown(f"- **Places & maps:** {_ok if places_available() else _missing}")
   st.caption("Run `scripts/validate_env.py` to check configuration.")
   ```

2. **Selectbox tooltip** at `app.py:80–82`:
   ```python
   help="⚡ Fast & low-cost  |  🧠 Premium reasoning. "
        "Claude entries require GOOGLE_CLOUD_PROJECT + CLAUDE_VERTEX_REGION."
   ```

3. **Bottom status caption** at `app.py:473–480` (rendered after the plan is shown):
   ```python
   ok = ":green[active]"
   na = ":orange[not configured]"
   st.caption(
       f"Model: `{selected_model_id}` · "
       f"LLM {ok if llm_available() else na} · "
       f"Flights {ok if flights_available() else na} · "
       f"Places {ok if places_available() else na}"
   )
   ```

### Changes

- **Delete** `app.py:90–97` entirely (the `Live data sources` sidebar block).
- **Trim** the help text on `app.py:80–82` to: `help="⚡ Fast & low-cost  |  🧠 Premium reasoning."` (drop the Claude/GCP requirement line).
- **Replace** `app.py:473–480` with a single caption: `st.caption(f"Model: \`{selected_model_id}\`")`.
- The `flights_available`/`places_available`/`llm_available` imports at `app.py:17` may become dead code. Check for other call sites; if none, drop the import. If still used (e.g., in error paths), leave the import alone.

The user-facing message remains clean: when something is misconfigured, the planner already raises a clear `RuntimeError` (per `CLAUDE.md` strict-runtime invariant), and the existing `st.error(f"Planning failed: ...")` at `app.py:437` surfaces it. We don't need a sidebar dashboard repeating that.

### Validation

Visual check after `streamlit run app.py` — sidebar shows only the planning mode + sub-agent execution + model picker. After the plan renders, only the model name appears below.

### Time

~5 min.

---

## Item 4 — What the cache does, and whether to keep it

### Diagnosis

`agents/llm.py:57–127` implements a two-tier cache:

- **In-memory LRU** (`_json_cache`, size from `LLM_CACHE_SIZE`, default 256). Keyed by `sha256(provider|model|schema|max_tokens|think_first|system|user)`. Stores the raw JSON string the model returned, re-validated against the schema on each hit.
- **Disk cache** (`LLM_CACHE_DIR` env var, opt-in). Same key → JSON file on disk. Read at call time so the eval CLI can set the env var post-import.

Two distinct workloads:

1. **Live `app.py` use.** Each user query is unique; the system prompt + tool context + intent dump produce a cache key that almost never repeats. The in-memory cache only ever hits inside the same process when a specialist is re-called with **identical** input — which only happens if a verifier round produces the exact same repair note. Empirically, this is a small percentage.
2. **Eval batch use.** `eval/run_eval.py` runs the same 180 queries across 4 systems, and reruns are common during debugging / paper iteration. With `LLM_CACHE_DIR=.cache/llm`, every prompt/response pair is persisted; rerunning the same `--system multi` is essentially free and deterministic. **This is load-bearing for paper §5 efficiency-table reproducibility.**

### Verdict and changes

**Keep the cache.** Removing it would invalidate eval reproducibility, slow every rerun by ~30 minutes, and make the report numbers dependent on the real-time API state at the moment we ran the sweep — that's exactly what we don't want.

But the cache **stats in the live UI are misleading**: they show `0 hits / N misses` for every user query, suggesting something is broken when nothing is. Two clean-up changes:

- **Hide the cache metric from the live UI** in `app.py:468–472`. The `LLM calls` and `Tokens` metrics stay; the `Cache` metric is removed (it's confusing for a live single-query workflow).
- Leave `agents/llm.py` cache logic untouched. Eval continues to use it.

Optionally (deferred): expose the cache toggle via an env var the user can flip before running `streamlit run app.py` for offline demos. Not required for the report.

### Validation

- Eval rerun sanity: `python -m eval.run_eval --split validation --system multi --limit 5 --llm-cache-dir .cache/llm` twice in a row. Second run should be near-instant (cache hits >> misses) and produce identical numbers to the first.
- Live app: verify the `Cache` metric is gone and `LLM calls` / `Tokens` still render.

### Time

~5 min UI removal.

---

## Item 5 — What "tokens increasing / decreasing" means

### Definitions (tighten, then surface in the UI)

- **Input tokens** (`_stats["input_tokens"]`) — sum of every system + user prompt sent to the LLM during this query, across all specialists and repair rounds. Larger when: longer Intent JSON, larger candidate pools, more few-shot examples in SYSTEM, more repair rounds. **You pay for this on the way in.**
- **Output tokens** (`_stats["output_tokens"]`) — sum of every model-generated response. Larger when: `think_first=True` (forces up to 6000 generated tokens of reasoning + JSON), longer plans (more days = more meal/attraction strings), retries. **You pay for this on the way out — usually billed at a higher rate per token than input.**

Token count going **up** between two runs of the same query usually means:
- More repair rounds were needed (verifier flagged more rules).
- A different model is in use (some models verbalize more).
- `think_first` got triggered (long reasoning preamble).

Token count going **down** usually means:
- Cache hit (no LLM call → input/output tokens for that call counted as 0).
- Smaller pool (shorter trip → fewer hotels/restaurants/attractions to print).
- Item 1's optimizations landed (smaller `max_tokens`, no `think_first`, JSON mode).

### Changes (UI clarity, not behavior)

In `app.py:464–467`, replace:
```python
fc2.metric(
    "Tokens",
    f"{llm_stats.get('input_tokens', 0)}↑ {llm_stats.get('output_tokens', 0)}↓",
)
```
with two separate metrics:
```python
fc2.metric("Input tokens", f"{llm_stats.get('input_tokens', 0):,}")
fc3.metric("Output tokens", f"{llm_stats.get('output_tokens', 0):,}")
```
…and add a one-line caption below the metrics:
```python
st.caption(
    "Input = prompt tokens we sent. Output = response tokens the model generated. "
    "Lower = faster + cheaper. Repair rounds and `think_first` reasoning increase both."
)
```

### Validation

Visual check — labels are unambiguous; user can correlate token totals with repair-round counts in the timeline.

### Time

~5 min.

---

## Execution order

| # | Item | Time | Blocks report? |
|---|---|---|---|
| 1 | Item 3 — UI cleanup (sidebar + tooltip + bottom caption) | 5 min | No |
| 2 | Item 4 — hide cache metric in live UI | 5 min | No |
| 3 | Item 5 — tokens UI clarity | 5 min | No |
| 4 | Item 2 — `From → To` header + remove rocket dupe | 10 min | No |
| 5 | Item 1 — LLM speed/accuracy (drop think_first, trim contexts, lower retry cap) | 1.5 h | **Yes — must wait for the running eval to finish first** |

Items 2–5 can land immediately; they only touch `app.py` and don't affect eval numbers. Item 1 must land **after** the current `eval/results.csv` is captured for the report — otherwise the "before" baseline is lost.

## Done criteria

- `streamlit run app.py` shows: clean sidebar (no `Live data sources`), no Claude tooltip, no bottom status caption, `From → To` not truncated, no duplicate rocket caption, no `Cache` metric, separate `Input tokens` / `Output tokens` metrics with explanatory caption.
- After Item 1: `python -m eval.run_eval --split validation --system multi --limit 30` runs ~40–60% faster per row with `final_pass` flat or higher.
- ARCHITECTURE.md §11 invariants list still holds; only the cache section's user-facing surface changes (the cache itself is unchanged).

## Out of scope

- Model swap or provider change (Item 1 keeps Gemini 2.5 Flash-Lite as default).
- Cache key versioning (`PLAN_REPORT_BLOCKERS.md` Phase 13 finding 7 already tracks this).
- Sequential-mode dead-code removal (Phase 13 finding 9).
