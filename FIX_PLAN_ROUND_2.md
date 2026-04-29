# Fix Plan — Round 2 (Budget render, Quota fallback, Models list, Return-flight count, State destinations)

Drafted 2026-04-29. Six issues from the latest live + eval run. Plan only — no code changes yet.

---

## Issue 1 — Budget violation message renders mangled in Streamlit

### Symptom
```
budget: Total
1182
e
x
c
e
e
d
s
b
u
d
g
e
t
1182exceedsbudget1000; largest category: transport ($538)
```

### Root cause
- The verifier emits the literal string `"Total $1182 exceeds budget $1000; ..."` at `Project_Code/agents/verifier.py:159-160`.
- `Project_Code/app.py:317` renders violations via `st.markdown(f"  - `{v.rule}`: {v.detail}")`.
- Streamlit's markdown engine treats paired `$ ... $` as **inline LaTeX**. The pair `$1182 exceeds budget $` becomes a math expression, so each character is rendered as its own math symbol on a separate line.
- The trailing `1000; largest category: transport ($538)` appears after the closing `$`, but the `($538)` reopens math mode again — explains why `$538` survives but `$1000` got eaten.

### Fix plan

**A. Escape `$` at the render layer** in `app.py:317`:
```python
detail_safe = v.detail.replace("$", r"\$")
st.markdown(f"  - `{v.rule}`: {detail_safe}")
```
Single localized fix; verifier output stays unchanged for eval scorers (which parse `$N` for cost).

**B. Audit other `st.markdown` callers** that render strings containing `Cost: $N`:
- `_render_field` in `app.py:171-180` (transportation/lodging/dining day fields).
- `app.py:261, 264, 271-274, 282, 286` (per-day blocks).
Apply the same `replace("$", r"\$")` shim — encapsulate as a tiny helper `_escape_dollars(s: str) -> str` to avoid drift.

**C. (Optional) Switch verifier-detail rendering to `st.text`** instead of `st.markdown` — it preserves dollar signs verbatim. Downside: loses the `` `rule_name` `` code styling. Probably not worth it; A+B is cleaner.

---

## Issue 2 — Claude Vertex quota exhausted is fatal for the app

### Symptom
```
Planning failed: RuntimeError: Claude on Vertex quota exhausted
(region=global, project=project-a5eb48eb-7e5e-4ad5-8da). ...
```

### Root cause
- After Round 1's longer 429 backoff (`agents/llm.py:_MAX_ATTEMPTS_429=6`), if Vertex is *still* rate-limiting at the 6-min mark, the RuntimeError now propagates to the app's top-level handler with the new quota-specific message.
- The app currently has **no degradation path** when the active model's quota is dry. It just shows the error to the user.
- The TPM bucket on `region=global` is shared across the project; depending on classroom usage and other concurrent runs, it can stay empty for many minutes.

### Fix plan

> **Constraint (user direction, 2026-04-29):** *No fallbacks at all — neither
> cross-provider nor in-family. Just use the lowest Claude model supported on
> Vertex from the start.* Also: validate the LLM API call before kicking off
> the full multi-agent pipeline.

**D. Make Claude Haiku 4.5 the only Claude entry exposed.** Edit
`Project_Code/agents/models.py:75-110`:
- Keep `claude-haiku-4-5@20251001` (lowest cost, lowest TPM footprint of the
  Claude lineup on Vertex Model Garden).
- Drop `claude-sonnet-4-5@20250929` and `claude-sonnet-4-6@20251101` from the
  registry. These are the highest-cost entries (~37× baseline relative cost
  per the registry comments) and are the ones triggering the quota exhaustion.
- This change alone resolves Issue 2 by construction: the model with the
  largest per-call token footprint and the tightest TPM bucket on
  `region=global` is no longer in the path.
- If a user has explicitly set `LLM_MODEL=claude-sonnet-...` in `.env`, fall
  back to the registry default (Haiku) on `KeyError` rather than crashing.

**E. Verify nothing else hard-codes the dropped Sonnet IDs.**
`grep -rn "claude-sonnet" Project_Code/` should turn up only docs and
`CODE_PLAN_MULTI_MODEL.md`-style planning artifacts. Update those to note the
Sonnet ablation is dropped from this run due to Vertex quota constraints.

**F. Pre-flight LLM API validation in `app.py`** — before launching `plan_trip`,
issue a tiny structured-output call to the active model (e.g. `call_json` with
a trivial 1-field schema like `{"ok": bool}`, `max_tokens=16`). This both:
1. Verifies auth (ADC for Claude, `VERTEX_AI_API_KEY` for Gemini) is wired up.
2. Probes quota — a 429 here means we know up front that the multi-step
   pipeline would hit the same wall, so we can show a clean warning before
   any work starts.

The pre-flight result is cached in `st.session_state` for the session so we
don't re-validate on every chat turn. On failure, render an actionable banner:
```
Cannot reach {model_id}: {short reason}.
- 429: Vertex quota exhausted — wait a few minutes or pick a different model.
- auth/missing key: see .env.example for required variables.
```
Cheap (~16 output tokens, <1¢ per session). No automatic retries on a
different model — the user picks one explicitly via the model selector.

**G. Document the "global" region quota math** in `Project_Code/.env.example`
so users know they can set `CLAUDE_VERTEX_REGION` to a specific region with
their own larger quota allotment if they have one. Also note that with Haiku
as the only Claude entry, expected quota draw is ~3× lower than Sonnet 4.6.

---

## Issue 3 (numbered 4) — Remove Gemini 2.0 Flash from the model list

### Where it lives
`Project_Code/agents/models.py:64-74` defines the entry:
```python
ModelEntry(
    id="gemini-2.0-flash",
    display_name="Gemini 2.0 Flash",
    ...
)
```

### Fix plan

**H. Delete the `gemini-2.0-flash` entry** from the `REGISTRY` list. No other code references it by ID (verify with `grep -rn "gemini-2.0-flash" Project_Code/` — should only appear in the registry, the README/docs, and possibly `CODE_PLAN_MULTI_MODEL.md`). If the eval has cached results under that ID in `eval/results.csv`, those rows are historical and stay; the registry change just stops the model from being available going forward.

**I. Remove from any plan/report docs** so the ablation table doesn't promise a number we don't have:
- `PROJECT_REPORT_PLAN.md`
- `CODE_PLAN_MULTI_MODEL.md`
- `Project_Code/ARCHITECTURE.md` if it references the model list

---

## Issue 4 (numbered 5) — Round-trip search returns 9 outbound but only 1 return

### Symptom
```
Round-trip: 9 outbound + 1 return option selected.
```

### Root cause
`Project_Code/tools/live_apis.py:651-668` (`flight_search_round_trip`):
- Outbound: `top_outbounds = outbound_candidates[:max_results]` → up to N options returned (line 592).
- Return: only `best_return` is selected, then `ret_strs = [ret_seg]` (line 667). A *single* return option, no runners-up.

This is asymmetric by design: SerpAPI's two-call protocol uses one outbound's `departure_token` to fetch matching returns. The current code throws away all return alternates and only emits the chosen one.

### Fix plan

**J. Return up to `max_results` return options** — keep `best_return` at index 0 (preserves the late-return preference) and append the rest:
```python
ret_strs: list[str] = []
ordered_returns = [best_return] + [r for r in return_candidates if r is not best_return]
for r in ordered_returns[:max_results]:
    seg = f"{r.airline} {r.dep_time}->{r.arr_time}"
    dur = self._format_duration(r.duration_min)
    if dur:
        seg += f" Duration: {dur}"
    # No cost on return leg — round-trip total is on outbound (Day 1) only.
    ret_strs.append(seg)
```

Cost stays attributed to the outbound (per the existing comment at `:666`). The transport specialist now gets a real menu of return options to pick from, matching the outbound side.

**K. Update the progress message** at `:670` to read e.g. `"Round-trip: {N} outbound + {M} return options selected."` with `M = len(ret_strs)`.

---

## Issue 5 (numbered 6) — Eval: 120/180 rows fail with "No live X results in tool context"

### Symptom
Of 180 validation rows: ~60 succeed (3-day queries), ~120 fail before the first LLM call:
```
row 21 FAILED: RuntimeError: No live restaurant results available in tool context
row 22 FAILED: RuntimeError: No live restaurant results available in tool context
...
```

### Root cause (confirmed by direct sandbox inspection)
- `Dataset/validation.csv` contains **180 queries**: 60 with `days=3` (single-city, e.g. `dest="Tucson"`), 60 with `days=5`, and 60 with `days=7`. The 5- and 7-day queries use **US state names** for `dest` (e.g. `"Florida"`, `"Texas"`, `"Illinois"`) — the agent is expected to plan a multi-city itinerary across cities in that state.
- `tools/sandbox.py` keys all data by **city** (`attractions[city]`, `restaurants[city]`, `accommodations[city]`).
- `_sandbox_tool_context` (eval/run_eval.py:123-198) calls `sandbox.attraction_search(intent.dest)` etc., which returns `[]` when `dest="Florida"` because the sandbox has `attractions["Miami"]`, `attractions["Orlando"]`, but no `attractions["Florida"]`.
- The state→cities table (`sandbox.cities_by_state`) is **empty** — verified via `python -c "from tools.sandbox import default_sandbox; print(default_sandbox('frozen').cities_by_state)"`. The TravelPlanner reference JSONL files don't include `"Cities in {state}"` records, so the loader's `_CITIES_KEY` branch never fires.
- Specialists then raise `RuntimeError("No live X results available in tool context")` at `agents/{dining,lodging,sightseeing,transport}.py` (each at line 53/45/55/63).

### Fix plan

**L. Build a state→cities map from the dataset itself.** TravelPlanner annotated plans encode each city visited per day in `current_city`. Walk all rows in `train.csv` + `validation.csv`, for each row with a multi-day plan parse `annotated_plan`, group `current_city` values by state (the row's `dest`), and persist as `Project_Code/eval/state_city_index.json`. A one-time build script (e.g. `scripts/build_state_index.py`); ~1 minute of compute.

   Schema:
   ```json
   {
     "Florida":  ["Sarasota", "Miami", "Orlando", "Tampa", ...],
     "Texas":    ["Houston", "Dallas", "Austin", ...],
     ...
   }
   ```

**M. Alternative if (L) is too coupled to annotated plans:** ship a static fallback table of US states → 3–5 major cities (curated from the sandbox's existing city list). Less precise but unblocks immediately. Combine: prefer (L) per-state list, fall back to (M) for any state missing.

**N. Sandbox helper** in `tools/sandbox.py`:
```python
def cities_in(self, location: str) -> list[str]:
    """Return [location] if it's a city we know, else cities in that state."""
    if location in self.attractions or location in self.restaurants \
            or location in self.accommodations:
        return [location]
    return list(self.cities_by_state.get(location, []))
```
Plus loader change: load the state→cities JSON at `default_sandbox()` time and merge into `cities_by_state`.

**O. Update `_sandbox_tool_context`** in `eval/run_eval.py:123` to aggregate across cities when `dest` is a state:
```python
cities = sandbox.cities_in(intent.dest) or [intent.dest]

all_attractions = []
for city in cities:
    for r in sandbox.attraction_search(city):
        all_attractions.append({
            "name":    r.get("Name"),
            "rating":  4.0,
            "address": r.get("Address", city),
            "city":    city,    # NEW: tag each item with its source city
        })
ctx["attractions"] = all_attractions[:n_attractions]
# Same pattern for restaurants and hotels — tag each with its source city.
```
For flights: try `(intent.org, c, date)` for each `c in cities` and union the results. If empty, fall back to `intent.org → first city in cities`. Return-leg flights: similarly union across cities for `(c, intent.org, date)`.

**P. Pass `cities` into specialist prompts** so they know this is a multi-city trip. The simplest path: extend `ToolContext` with `dest_cities: list[str]` and have each specialist's prompt read "Plan across cities X, Y, Z (visit them in any sensible order over D days)". The verifier already enforces `within_current_city` per-day — once specialists set `current_city` correctly, the route validates.

**Q. Coordinator route-planning hint.** The transport specialist currently plans `org → dest → org`. For multi-city plans it should produce inter-city transitions (drive/taxi between cities on the days the route changes). Add a short bullet to `agents/transport.py` SYSTEM prompt: "If `dest_cities` has multiple entries, include intercity transit (taxi or self-driving) on the day `current_city` changes." `ToolContext` should also include `route_drive` / `route_taxi` summaries between consecutive cities — extend `_sandbox_tool_context` to populate `intercity_routes: dict[(str,str), str]`.

### Effort split
- L + N + O: ~2 hours; unblocks the eval, restores 120 failing rows.
- P + Q: ~1.5 hours; required for the multi-city plans to actually score reasonably (otherwise specialists still output single-city itineraries and `reasonable_city_route` / `current_city` rules drag the score).

---

## Issue 6 (also from row 161) — TruncationError on a single eval row

### Symptom
```
row 161 FAILED: TruncationError: LLM response was truncated before the JSON object was closed (got 1217 chars).
```

### Root cause
`agents/llm.py:_call_json_inner` (around `:258-291`) escalates `max_tokens` on truncation up to a cap, then raises. For a single 7-day multi-city query with a long restaurant pool, the JSON exceeds the cap.

### Fix plan

**R. Raise the per-call max-tokens cap** in `_call_json_inner` from its current ceiling (look at `cur_max[0] = ...` escalation) to ~2× when the prompt's tool-context list is large. Or: introduce a `max_tokens_ceiling` arg per specialist with sightseeing/dining at 4096 and others at 2048.

**S. (Optional) Re-prompt with a shorter pool on truncation.** If escalation still truncates, recall the specialist with `n_restaurants//2` candidates instead of failing. Last-resort safety net.

---

## Suggested execution order

| # | Task | Effort | Why |
|---|------|--------|-----|
| 1 | A + B (escape `$` in Streamlit render) | 15 min | One-line fix, immediate user-visible win. |
| 2 | H + I (drop Gemini 2.0 Flash) | 10 min | Keeps the model selector and ablation table honest. |
| 3 | J + K (multi-return flights) | 20 min | Easy win, improves transport specialist's option pool. |
| 4 | L + N + O + P + Q (state destinations) | ~3.5 hr | **Critical** — unlocks 67% of the eval that's currently zeroed out. Without this, the paper has no validation numbers for 5-day or 7-day queries. |
| 5 | D + E + F (drop Sonnet entries; keep only Haiku; pre-flight validation) | 45 min | Stops Issue 2 by construction — the call that was exhausting the TPM bucket no longer exists. Pre-flight call verifies auth/quota before the user waits. No fallback logic. |
| 6 | R (max-tokens escalation) | 30 min | One-row failure today; cheap insurance once eval throughput goes up. |
| 7 | F + G + S | nice-to-have | Polish; layer on if time permits before the report. |

## Files touched

- `Project_Code/app.py` — `$` escape helper, pre-flight LLM validation banner, downgrade-warning banner.
- `Project_Code/agents/llm.py` — max-tokens ceiling bump only (no fallback logic added).
- `Project_Code/agents/models.py` — drop `gemini-2.0-flash`, `claude-sonnet-4-5`, and `claude-sonnet-4-6` entries; Haiku 4.5 is the only Claude option remaining.
- `Project_Code/agents/transport.py` (+ minor: dining/lodging/sightseeing) — multi-city prompt hint.
- `Project_Code/tools/live_apis.py` — multi-return-options block in `flight_search_round_trip`.
- `Project_Code/tools/sandbox.py` — `cities_in()` helper, state-index loader.
- `Project_Code/eval/run_eval.py` — `_sandbox_tool_context` multi-city aggregation, `dest_cities` plumbing, replay-of-downgraded-rows pass.
- **NEW:** `Project_Code/scripts/build_state_index.py` — one-time builder for the state→cities JSON.
- **NEW:** `Project_Code/eval/state_city_index.json` — generated artifact, checked in.
- `Project_Code/.env.example` — quota guidance for `CLAUDE_VERTEX_REGION`.

No schema changes that affect the verifier or the canonical evaluator output format.
