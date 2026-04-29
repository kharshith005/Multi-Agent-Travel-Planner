# Code Plan — Multi-Model + Latency + Per-Model TravelPlanner Comparison

Engineering execution plan, scoped to the `Project_Code/` tree. Companion to `ARCHITECTURE.md` (current + planned architecture).

Companion documents at the workspace root:
- `../PROJECT_REPORT_PLAN.md` — paper-aligned *what / why* (Paper 1: TravelPlanner).
- `../CODE_PLAN_MULTI_MODEL.md` — full *how / why* including provider/auth context, risk register, and report-output specs.

This file is the in-repo execution plan: clear objectives, file-level changes, phased ordering, and per-phase validation gates. If you're only looking at `Project_Code/`, this is the document you act on.

---

## 1. Objectives

The work has six coupled objectives. Every code change should be traceable to one of them. **O1–O3 are landed (Phase 0–3 complete).** **O4–O6 are the next batch (Phase 4–6).**

### O1. Multi-model support on Vertex AI (provider-aware)  *(landed)*
- Let the Streamlit user pick the model from a curated list of **low-cost / fast-response** Vertex-hosted models.
- Cover **two providers** that Vertex AI hosts: **Gemini** (existing) and **Claude** (new).
- **GPT/OpenAI is excluded** — Vertex AI does not host OpenAI models. Document this in code comments and the README so it's not asked again.
- Same architecture for every model — no per-agent model routing in v1.

### O2. Validate and improve application response time  *(landed)*
- Instrument the pipeline before changing it.
- Establish a frozen 10-query latency benchmark.
- Land only changes that hold against the baseline (no P50 regression; P95 improves or holds).

### O3. Per-model TravelPlanner comparison  *(landed)*
- Extend `eval/run_eval.py` to produce a `(system × model)` results matrix.
- Match Paper 1's metric set exactly so the report's main table is a direct analog of TravelPlanner's Table 3.
- Generate report-ready Markdown tables and a Pareto chart from the results CSV.

### O4. Test infrastructure  *(new — Phase 4)*
- Establish a pytest suite that exercises deterministic logic (budget, constraints, verifier, assemble, timing, runtime, models, llm cache, perf) without spending API quota.
- Add integration tests that drive the coordinator end-to-end with a mocked LLM provider so refactors stay green.
- Make `pytest tests/unit` the fast feedback loop (target: < 30s); keep `run_eval` as the slow integration test for absolute pass rates.
- Lightweight CI: GitHub Actions runs unit tests on push.

### O5. Single-agent baseline UX + parity  *(new — Phase 5)*
- The single-prompt baseline becomes a first-class user-facing option, not an eval-only construct.
- Single-agent mode runs against the SAME `ToolContext` the multi-agent path uses (live or sandbox) so the comparison isolates orchestration, not data quality.
- Streamlit sidebar gets a "Mode: Multi-agent (default) | Single-agent (baseline)" toggle with the same Performance and verifier panels.
- Lets the report's headline argument ("multi-agent beats single-prompt") be experienced live, not just read in a table.

### O6. Specialist prompt audit + targeted improvements  *(new — Phase 6)*
- Audit each specialist's prompt (4 specialists + `single_agent` + `no_specialization`) against the full constraint list it can violate.
- Use per-rule pass rates from the per-model CSV (Phase 3) as the failure-mode signal, not intuition.
- Ship measurable improvements: at least one CS rule moves up per audited prompt; no aggregate regression.
- Deliverable: `Project_Code/PROMPT_AUDIT.md` checklist matrix mapping rules → prompt lines, plus before/after eval numbers committed alongside each prompt commit.

### O7. Live-API observability + strict-runtime fix  *(new — Phase 7)*
- Stop the live runtime from silently swallowing SerpAPI and Google Maps errors. Today, `agents/coordinator.py` (lines 468–481) and `baseline/no_specialization.py` (lines 51–62) wrap `live.flight_search(...)` in `try / except Exception: pass`, turning every failure — missing key, transient 5xx, rate limit, parse error — into `flights_outbound = []`. The user's symptom: "SerpAPI seems not being executed" even when `SERPAPI_API_KEY` is set correctly.
- Distinguish three states cleanly: **OK** (key set, call succeeded, results returned), **skipped** (key not set; intentional), **error** (key set, call failed; surface to UI with actionable message).
- Document explicitly that the **eval path does not invoke SerpAPI** — it builds flights from `tools/sandbox.py` (TravelPlanner 2022 reference data). This is by design (historical dataset compatibility), not a bug, and should be called out so future reviewers don't re-discover it as a defect.
- Add a per-API status panel in the Streamlit Performance expander (`serpapi_outbound`, `serpapi_return`, `maps_places_hotels`, `maps_route_drive` → `ok | skipped (no key) | error: <message>`).
- Ship a preflight `scripts/validate_env.py` that pings each configured API once and reports actionable failures before the user runs `app.py`.

### Non-goals
- OpenAI/GPT models (not hosted on Vertex).
- Per-agent model routing (deferred; would explode the eval matrix).
- BYO-API-key in UI / multi-tenant sessions.
- Test-set numbers (labels hidden in TravelPlanner; we run validation only).
- Refactoring `agents/coordinator.py`'s LangGraph topology (out of scope; only thread `model` through it).
- Full coverage in tests — pragmatic, high-value tests only (no chasing 100%).
- Replacing the deterministic verifier with an LLM-based critic (paper §4.6 finding; out of scope).

---

## 2. Provider decisions (recap)

| Provider | Vertex hosting | Existing? | SDK | Auth |
|---|---|---|---|---|
| Gemini | Native | Yes | `google-genai` | `VERTEX_AI_API_KEY` env (already wired) |
| Claude | Via Anthropic-Vertex partnership / Model Garden | No | `anthropic[vertex]` (`AnthropicVertex`) | ADC + `GOOGLE_CLOUD_PROJECT` + region (e.g. `us-east5`) |
| GPT/OpenAI | **Not hosted** | n/a | n/a | n/a |

**Curated model registry (target — exact IDs pinned in Phase 0):**

| Display name | Model ID (placeholder) | Provider | Tier | Use |
|---|---|---|---|---|
| Gemini 2.5 Flash-Lite *(default)* | `gemini-2.5-flash-lite` | gemini | lite | Current default |
| Gemini 2.5 Flash | `gemini-2.5-flash` | gemini | standard | Quality ceiling, same family |
| Gemini 2.0 Flash-Lite | `gemini-2.0-flash-lite` | gemini | lite | Generational comparison |
| Gemini 2.0 Flash | `gemini-2.0-flash` | gemini | standard | Generational comparison |
| Claude Haiku (latest) | `claude-haiku-*@<rev>` | claude | lite | Cross-provider lite tier |
| Claude Sonnet (latest) | `claude-sonnet-*@<rev>` | claude | standard | Cross-provider standard tier |

Drop the Claude rows automatically if Claude isn't enabled in the GCP project (graceful degradation).

---

## 3. File-level changes (the diff inventory)

### 3.1 New files
- `agents/models.py` — model registry: `id`, `display_name`, `provider`, `tier`, `family`, `default_temperature`, `est_relative_cost`, `auth_mode`. Pure data + filter helpers (`available_models()` returns only models whose auth is configured).
- `agents/providers/__init__.py` — provider-router: `get_backend(provider) -> Backend`.
- `agents/providers/gemini.py` — lifts existing `google-genai` logic out of `agents/llm.py`. No behavior change for Gemini.
- `agents/providers/claude.py` — `AnthropicVertex` client; structured output via **forced tool-use** (declare a single tool whose schema is the Pydantic JSON schema, force the model to call it, parse the tool input).
- `agents/runtime.py` — ContextVar holding the current model, plus `use_model(model_id)` context manager so internal helpers don't have to forward `model=` everywhere.
- `agents/perf.py` — `Timeline` object: phase entries with `name`, `start`, `end`; emit summary dict for UI/eval.
- `eval/perf_queries.txt` — frozen 10-query benchmark set, one row per difficulty/day combination.
- `eval/perf_baseline.csv` — recorded baseline timings; checked in.
- `eval/format_report.py` — generate the two report Markdown tables from `eval/results.csv`; optional Pareto chart.

### 3.2 Modified files
- `agents/llm.py` — becomes a thin facade. Per-call `model=` arg; provider+model-aware disk cache key (`sha256(provider + model + system + user + schema_name)`); per-model stats; per-call latency; remove import-time env capture for `LLM_MODEL`.
- `agents/coordinator.py` — `plan_trip(query, *, model=None, ...)`; add `model` to `CoordinatorState`; thread to `parse_intent` and every specialist invocation. Populate `Timeline` for each phase.
- `agents/transport.py`, `agents/lodging.py`, `agents/dining.py`, `agents/sightseeing.py` — accept `model` from state, pass to `call_json`. Mechanical change.
- `agents/budget.py`, `agents/verifier.py` — verifier is deterministic (no LLM call), but if any LLM-driven repair triggers from these paths, thread `model` through.
- `baseline/single_agent.py`, `baseline/no_specialization.py`, `baseline/no_verify.py` — accept `model=` kwarg, forward.
- `eval/run_eval.py` — new flags (`--model`, `--models`, `--cooldown-seconds`); rotate over models sequentially; new CSV columns (`model`, `provider`, `latency_seconds`, `cache_hits`, `cache_misses`); remove the `importlib.reload(agents.llm)` hack (no longer needed once import-time env capture is gone).
- `app.py` — sidebar `selectbox` populated from `agents/models.available_models()`; pass `model=` into `plan_trip`; add a "Performance" expander after the plan renders that consumes the `Timeline` summary.
- `requirements.txt` — add `anthropic[vertex]` (only when Claude work begins; don't add it speculatively).
- `.env.example` — add `GOOGLE_CLOUD_PROJECT`, `CLAUDE_VERTEX_REGION`, `GOOGLE_APPLICATION_CREDENTIALS` (all optional; missing = Gemini-only).
- `README.md` — already updated to reference this plan, `ARCHITECTURE.md`, and the GPT-exclusion note. Re-touch only if the model registry changes shape.
- `ARCHITECTURE.md` — already includes the planned multi-provider LLM layer (§4.2). Update §11 invariant 7 once Phase 1 lands so the cache-key invariant is no longer aspirational.

### 3.3 Deleted code
- The `importlib.reload(agents.llm)` block in `eval/run_eval.py` (after import-time env capture is removed).

### 3.4 Smoke checks  *(landed in Phase 1)*
- `scripts/smoke_models.py` — for each `available_models()` entry, call `parse_intent` on one fixed query, print model + latency + token counts. Lets us verify cache-key separation by running it twice and watching cache hits land only after the second run for the *same* model.

### 3.5 New files for Phase 4–7  *(planned)*

**Phase 4 — tests (deterministic logic, mocked LLM):**
- `tests/conftest.py` — shared fixtures: `sandbox` (frozen sandbox singleton), `intent_factory`, `tool_context_factory`, `fake_backend` (canned-response provider), `tmp_disk_cache`.
- `tests/unit/test_budget.py` — `extract_cost` regex coverage; `allocate` splits sum to budget; `evaluate` over hand-built `FullPlan`.
- `tests/unit/test_constraints.py` — every rule in `eval/constraints.py` against crafted plans (one passing + one failing per rule).
- `tests/unit/test_verifier.py` — every rule in `agents/verifier.py`; assert `responsible` tagging matches `agents/rules.py`.
- `tests/unit/test_assemble.py` — accommodation forward-fill; departure-day `"-"`; day-string format invariants (`Cost: $N`, `Cost: $N for D nights`).
- `tests/unit/test_timing.py` — `_apply_timing_adjustments` for arrival ≥ 19:00, departure < 8:00, late-dinner restoration when departure ≥ 18:00.
- `tests/unit/test_models.py` — registry IDs unique; `available_models()` filtering matches env var presence; `default_model_id()` env fallback.
- `tests/unit/test_runtime.py` — `use_model` nesting; ContextVar reset on exception; thread-propagation contract (skip if not propagated by the caller).
- `tests/unit/test_llm_cache.py` — provider+model+schema in cache key; same prompt/different model = miss; same prompt/same model = hit; ValidationError purges both memory and disk.
- `tests/unit/test_perf.py` — `Timeline.phase()` records start/end; `summary()` shape; cache hit/miss counts roll up correctly.
- `tests/integration/test_coordinator_e2e.py` — fake LLM returns canned JSON; pipeline runs to verifier; final state includes plan, report, attribution_log, timeline.
- `tests/integration/test_repair_routing.py` — inject specific `Violation` kinds; assert only the responsible specialist re-runs (not all four).
- `tests/integration/test_multi_provider.py` — stub `AnthropicVertex.messages.create` to return a tool-use block; verify Claude path produces a valid plan; verify provider-key separation via cache miss/hit pattern.
- `tests/fixtures/llm_responses/*.json` — canned responses for the integration tests, keyed by `(schema_name, query_hash)`.
- `tests/fixtures/cache/` *(optional)* — real responses captured once, checked in for end-to-end determinism.
- `Project_Code/pytest.ini` — `testpaths = tests`, `addopts = -ra -q`, marker definitions (`unit`, `integration`, `slow`, `requires_api_key`).
- `.github/workflows/test.yml` — runs `pytest -m unit` on push (no API keys needed); integration tests gated behind a manual workflow.

**Phase 4 — tests (live-API contracts, mocked transport):**
*Tests live in `tests/unit/` (mocked) and `tests/integration/` (opt-in real-key). Live-API code paths are critical: when broken, the user-facing app silently degrades to driving-only plans.*
- `tests/unit/test_live_apis_serpapi.py` — uses `responses` (or `unittest.mock.patch("requests.get")`) to mock SerpAPI HTTP layer:
  - Missing `SERPAPI_API_KEY` raises `RuntimeError("Set SERPAPI_API_KEY for flight search")` *before* any network call.
  - HTTP 400 with `"cannot be in the past"` body triggers exactly ONE retry with `outbound_date=today` (no infinite loop).
  - HTTP 5xx surfaces as `RuntimeError("SerpAPI flights request failed: ...")` — no silent swallow.
  - HTTP 401/403 (auth failure) surfaces with `RuntimeError` and the response status code attached.
  - JSON-decode error surfaces as `RuntimeError`, not `JSONDecodeError`.
  - `best_flights` + `other_flights` parsing extracts `airline / dep_time / arr_time / price / duration_min` correctly.
  - Empty `best_flights` AND empty `other_flights` raises `"SerpAPI flights returned no options"`.
  - `_dedupe_flight_options` collapses identical (airline, dep_time, arr_time) tuples.
  - `_rank_flight_options` orders by duration first, then price, then budget filter (verify with a curated 5-option set).
  - `_resolve_airport_code` returns IATA for known cities, raises for unknown — uses `airportsdata` mock.
  - The `api_key` value never appears in any raised exception's message (test asserts on the formatted exception).
- `tests/unit/test_live_apis_maps.py` — patches `googlemaps.Client` with a `MagicMock`:
  - Missing `GOOGLE_MAPS_API_KEY` raises `RuntimeError("Set GOOGLE_MAPS_API_KEY")` before calling the SDK.
  - `googlemaps` import-time absence raises with install instruction.
  - `search_places` returns name/address/rating/price_level for each result; truncates at `max_results`.
  - `search_places` SDK exception surfaces as `RuntimeError("Google Maps places API failed: ...")`.
  - `route_summary` extracts distance/duration when status == "OK"; returns None when status != "OK".
  - `route_summary` with `mode="taxi"` adds taxi cost estimate via `_estimate_taxi_cost` (matrix mode stays "driving").
  - `route_summary` SDK exception surfaces as `RuntimeError("Google Maps distance matrix API failed: ...")`.
- `tests/unit/test_env_validation.py` — env-var contracts shared across providers and APIs:
  - `agents.providers.gemini._vertex_api_key()` raises with the exact installation/configuration hint when `VERTEX_AI_API_KEY` is empty or whitespace-only.
  - `agents.providers.claude._project()` and `_region()` each raise with their own actionable message.
  - `agents.models.available_models()` filters Claude entries when *any* of `GOOGLE_CLOUD_PROJECT` / `CLAUDE_VERTEX_REGION` is missing (parametrized: 4 boolean combinations).
  - `tools.live_apis.LiveTravelAPIs.from_env()` strips whitespace from both keys; treats `"   "` as missing.
  - `monkeypatch.delenv` per test; never leaks into other tests.
- `tests/unit/test_no_secret_logging.py` — guards against leakage:
  - For each known API key env var, set it to `"SECRET-MARKER-12345"` and trigger every `RuntimeError` in `tools/live_apis.py` and `agents/providers/`. Assert the marker string never appears in `str(exc)` or the formatted traceback.
- `tests/integration/test_live_apis_e2e.py` — `@pytest.mark.requires_api_key`; opt-in:
  - Skipped by default; runs only when `SERPAPI_API_KEY` and `GOOGLE_MAPS_API_KEY` are present.
  - Hits SerpAPI for a known route (e.g. `JFK → LAX` next-week date) and asserts at least one parsed flight result.
  - Hits Google Maps `places("hotels in San Francisco")` and asserts ≥ 3 results.
  - Hits Google Maps distance matrix for `San Francisco ↔ Los Angeles` and asserts a non-None route summary.
  - Designed for manual / CI-with-secrets runs; never blocks the default `pytest -m unit` path.

**Phase 6 — prompt audit:**
- `Project_Code/PROMPT_AUDIT.md` — checklist matrix per specialist (one section each for `transport`, `lodging`, `dining`, `sightseeing`, `single_agent`, `no_specialization`): rule → prompt line(s) → status (covered / weak / missing) → planned edit → before/after pass rate.

**Phase 7 — live-API observability:**
- `Project_Code/LIVE_API_AUDIT.md` — per-call-site decision matrix for every `live.*` invocation in `agents/coordinator.py` and `baseline/no_specialization.py`: call site → today's behavior (swallow / raise / log) → desired behavior → key-set vs key-unset semantics.
- `scripts/validate_env.py` — preflight that pings each configured API once and reports actionable failures:
  - Vertex API key (Gemini) — issue a 1-token prompt; surface 4xx/5xx with mapped diagnostics ("API not enabled in project", "key restricted to wrong endpoint", etc.).
  - Google Maps API — call `client.places("test")` once; surface auth/quota errors with the `error_message` from the response.
  - SerpAPI — fetch a small known-good route ping (e.g. JFK→LAX next-week date); surface auth/credit-out errors.
  - Optional: GCP ADC for Claude — call `AnthropicVertex.messages.create` with a minimal stub payload; surface 401/403 vs region availability.
  - Exits with non-zero status code if any *configured* key fails. Skips silently if a key is unset (intentional disable).
- `tests/unit/test_validate_env.py` — unit tests for the preflight helpers (mock `requests.get` and `googlemaps.Client`); verifies status mapping and exit codes.

### 3.5b Hotfix-block files  *(must land before Phase 4)*

**New files:**
- `agents/runtime_status.py` — small adapter exposing feature-name booleans (`flights_available()`, `places_available()`, etc.) that wrap env checks. Never returns env-var names. Used by the Streamlit sidebar; replaces the inline `os.environ.get(...)` calls in `app.py`.
- `tests/unit/test_no_env_leak.py` *(coordinated with Phase 4)* — regression test that `grep`s `app.py` for any of the protected env-var names; fails if any appears.

**Modified files:**
- `agents/perf.py` — add `merge_timelines(left, right) -> Timeline` reducer; sort `Timeline.summary()` phases by `start`; add `(repair rN)` suffix to phase names when called inside the repair loop (helper `phase_with_round(name, round_idx)`).
- `agents/coordinator.py` — `CoordinatorState.timeline` becomes `Annotated[Timeline, merge_timelines]`; every node body returns `{"timeline": local_tl}` where `local_tl = Timeline()` is fresh per call (no shared mutation); repair-loop nodes pass `round_idx` so phase names disambiguate.
- `app.py` — remove the env-var-listing block from the sidebar; replace with the feature-level "Live data sources" panel sourced from `agents/runtime_status.py`. Caption points users to `scripts/validate_env.py` for diagnostics.
- `ARCHITECTURE.md` — §11 invariants gain a new entry: *"User-facing UI never displays env-var names; configuration diagnostics live in `scripts/validate_env.py` only."*

### 3.6 Modified files for Phase 4–7  *(planned)*
- `baseline/single_agent.py` — accept `tool_context: ToolContext | None = None`; when omitted and `sandbox_kind="live"`, build live tool_context from `default_live_apis()` matching coordinator's research_node. Today it's hardcoded to sandbox.
- `app.py` — sidebar adds **Mode** radio: `Multi-agent (default)` / `Single-agent (baseline)`. Single-agent mode calls `plan_trip_single` instead of `plan_trip`; render plan with mode indicator in the title; reuse the same Performance expander. **Phase 7 also:** show a per-API status table in the Performance expander (`serpapi_outbound`, `serpapi_return`, `maps_places_*`, `maps_route_*`); show a top-of-page banner when any key-set live API errored on this run.
- `agents/transport.py`, `agents/lodging.py`, `agents/dining.py`, `agents/sightseeing.py` — prompt edits per Phase 6.2 (rule coverage, missing format hints, stronger few-shot for observed failure modes).
- `baseline/single_agent.py`, `baseline/no_specialization.py` — prompts rebuilt to cover the union of specialist rules; same-format invariants; chain-of-thought via `think_first=True` if not already set. **Phase 7 also:** `no_specialization.py` lines 51–62 stop swallowing `live.flight_search` errors when `SERPAPI_API_KEY` is set (mirror the coordinator fix).
- `agents/rules.py` — possibly factor out small helpers if multiple specialists need the same rule statement (avoid duplicated drift).
- `agents/coordinator.py` — **Phase 7:** replace bare `try / except Exception: pass` blocks around `live.flight_search`, `live.route_summary`, and `live.search_places` with a `_call_live(name, fn, key_set) -> tuple[result, status_str]` helper. Stash status strings in `state["live_api_status"]` (new key) and propagate to `Timeline` for surfacing in the UI. Behavior split: key unset → `"skipped (no key)"` (no exception); key set + exception → log the *type* + status code, populate `"error: <short message>"`, do NOT replace ctx field with `[]` silently — instead leave it absent so downstream specialists know there was a failure.
- `agents/perf.py` — **Phase 7:** extend `PhaseEntry` with optional `live_api_status: dict[str, str]`. `Timeline.summary()` adds a top-level `live_api_status` aggregate for UI consumption.
- `tools/live_apis.py` — **Phase 7:** strip API-key strings from any `RuntimeError` message (defense-in-depth — already done, but enforce via `tests/unit/test_no_secret_logging.py`).
- `ARCHITECTURE.md` — new §13 entry pointing at `tests/` layout; §10 Baselines table updated to note that single-agent is now usable in the live app. **Phase 7:** §9.1 (Live API path) gets an explicit paragraph: *"SerpAPI is invoked only by the live coordinator path (`app.py`). The eval path (`eval/run_eval.py`) builds flights from `tools/sandbox.py` reference data and never calls SerpAPI; this is intentional because TravelPlanner uses 2022 historical dates that no live API will return."*
- `README.md` — **Phase 7:** add a "Validate your environment" subsection right after the `streamlit run app.py` line: tell the user to run `python scripts/validate_env.py` after editing `.env` and link to `LIVE_API_AUDIT.md` for failure-mode reference.

---

## 4. Phased work plan

Each phase is a commit-sized unit. Don't merge a phase without passing its validation gate.

### Phase 0 — Provider readiness (do this BEFORE writing code)
**Goal:** confirm Claude on Vertex is actually accessible in the GCP project. If not, scope back to Gemini-only with no code changes lost.

Checklist:
- [ ] Confirm GCP project has billing enabled.
- [ ] In Vertex AI Model Garden, accept terms for the Claude models we plan to use.
- [ ] Pick a region with the chosen Claude models GA (typical: `us-east5`).
- [ ] `gcloud auth application-default login` works on the dev box.
- [ ] Service-account JSON available for the eval machine (`GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`).
- [ ] Pin exact GA Claude model IDs (e.g. `claude-haiku-4-5@20251001`) — these go into `agents/models.py`.

**Gate:** if any item fails, the registry ships Gemini-only and Claude work is documented as future work in the report. No code is wasted.

### Phase 1 — Mechanical plumbing (multi-model support, no eval impact)
**Goal:** the user can pick from up to 6 models in the UI; backend routes correctly; cache and stats are model-aware.

Steps:
1. `agents/models.py` registry (Phase 0 IDs pinned).
2. `agents/providers/{__init__,gemini,claude}.py` — gemini is a lift-and-shift; claude is new with forced-tool-use structured output.
3. Refactor `agents/llm.py` into the facade. Provider+model-aware cache key. Per-model stats. Per-call latency.
4. `agents/runtime.py` ContextVar.
5. Thread `model` through coordinator, four specialists, and three baselines.
6. Streamlit sidebar selectbox + pass-through; auth-aware filtering.
7. `.env.example` updates.
8. `requirements.txt` adds `anthropic[vertex]`.
9. Run `scripts/smoke_models.py`.

**Validation gate:**
- Each model in `available_models()` returns a parsed `Intent` for the same fixed query.
- Running the same query twice on the same model = cache hit on second run.
- Running the same query on different models = cache miss on each first run (key separation works).
- Existing `LLM_MODEL` env workflow still works (no UI selection → Gemini default).
- `eval/run_eval.py --split train --system annotated --limit 10` still hits ~100% (evaluator unchanged).

### Phase 2 — Latency instrumentation + validation
**Goal:** measure where time goes; record a baseline; ship only changes that beat it.

Steps:
1. `agents/perf.py` `Timeline`; populate phases in `coordinator.plan_trip`.
2. Surface in app.py via a "Performance" expander (collapsed by default).
3. `eval/perf_queries.txt` — frozen 10-query set covering all 9 difficulty/day groups + one Hard×7-day.
4. `eval/perf_run.py` (or a flag on `run_eval.py`) — runs the perf set, records P50/P95 wall-clock, P50/P95 LLM-latency-sum, cache hits/misses per query, per phase. Writes `eval/perf_baseline.csv`.
5. Run baseline with `gemini-2.5-flash-lite`, `--llm-cache-dir .cache/llm`, cache cleared. Commit `perf_baseline.csv`.
6. Apply candidate optimizations one at a time (see §5 below); after each, rerun perf set; commit only if P50 doesn't regress and P95 is ≤ baseline.

**Validation gate:**
- `perf_baseline.csv` exists and is committed.
- Perf-panel shows phase breakdown in UI for at least one end-to-end query.
- Each merged optimization commit links to a perf-run output that beats baseline.

### Phase 3 — Eval matrix expansion + report tables
**Goal:** produce the per-model TravelPlanner numbers the report needs.

Steps:
1. `--model` and `--models` flags in `run_eval.py`. `--models all` enumerates from registry.
2. `--cooldown-seconds` between models.
3. CSV schema: add `model`, `provider`, `latency_seconds`, `cache_hits`, `cache_misses`. Regenerate `eval/results.csv` (gitignored, no migration needed).
4. `eval/format_report.py`:
   - Per-model TravelPlanner Table-3 analog (Markdown).
   - Per-rule pass-rate breakdown (paper Table-4 analog), full multi-agent only.
   - Optional Pareto chart `final_pass_rate vs avg_latency_seconds` colored by provider.
5. Run main sweep (see §6).

**Validation gate:**
- `eval/results.csv` rows have `model` and `provider` columns.
- `format_report.py` emits two Markdown tables that paste cleanly into the report.
- The annotated-train sanity row in the per-model table reads ~100% on every metric (proves the evaluator is independent of the model dimension).

### Hotfix block — Phase 2 follow-ups *(must land before Phase 4)*
**Goal:** fix two bugs introduced by already-landed Phase 1/2 code that block live use of the app. Small in scope, but they must close before any new feature work, because Phase 4 tests need a working baseline and Phase 7 builds on the same UI surfaces.

#### Bug H1 — `InvalidUpdateError: At key 'timeline'`

**Symptom:** Live `streamlit run app.py` (parallel mode) fails with:
```
InvalidUpdateError: At key 'timeline': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

**Root cause:** Phase 2 added `timeline: Timeline` to `CoordinatorState` and made every node return `{"timeline": tl, ...}`. In parallel mode, after the `transport` node, the graph fans out to `lodging`, `dining`, `sightseeing` running in the same LangGraph superstep. All three nodes return a `timeline` key simultaneously. LangGraph requires either:
- A reducer (`Annotated[T, reducer_fn]`) that knows how to merge multiple writes, OR
- Single-writer ownership (only one node ever writes the key per superstep).

Phase 2 satisfied neither, so the runtime errors out the first time parallel-fanout writes the key.

A second, latent issue hides behind the same code: even if LangGraph accepted the writes, three threads concurrently mutating the same `Timeline.phases` list (via `tl.phase("...")` → `phases.append`) is a data race. The reducer fix incidentally addresses this by giving each parallel node its own Timeline instance.

**Fix:**
1. Add a reducer to `agents/perf.py`:
   ```python
   def merge_timelines(left: Timeline | None, right: Timeline | None) -> Timeline:
       if left is None:
           return right or Timeline()
       if right is None:
           return left
       if left is right:
           return left  # same instance — idempotent merge
       return Timeline(phases=left.phases + right.phases)
   ```
   The reducer is associative and idempotent on identical inputs; tests will assert both properties.
2. Mark the state key with `Annotated` in `agents/coordinator.py`:
   ```python
   from typing import Annotated
   from .perf import Timeline, merge_timelines

   class CoordinatorState(TypedDict, total=False):
       ...
       timeline: Annotated[Timeline, merge_timelines]
   ```
3. **Each parallel node returns a fresh, single-phase Timeline** (not the shared one). The reducer concatenates them onto the running state Timeline:
   ```python
   def lodging_node(state: CoordinatorState) -> CoordinatorState:
       local_tl = Timeline()                         # new instance per node
       with local_tl.phase("lodging"):
           plan = _run_specialist("lodging", lodging.run, state, on_progress)
       return {"lodging_plan": plan, "timeline": local_tl}
   ```
   Apply the same pattern to `dining_node`, `sightseeing_node`. Sequential nodes (`parse`, `research`, `transport`, `assemble`, `budget`, `verify`, `repair_prepare`) keep returning a fresh Timeline with just their own phase — uniform across the whole graph.
4. `Timeline.summary()` sorts `phases` by `start` (perf_counter) before returning, so the UI shows phases in execution order regardless of merge ordering.
5. `repair_round_N` phases (when the loop reruns) get unique names like `"transport (repair r1)"` to disambiguate them in the merged timeline.

**Validation:**
- `streamlit run app.py` succeeds end-to-end in **parallel** mode for a real query (manual smoke).
- Same in **sequential** mode (no regression).
- Performance expander shows phases in start-time order.
- Repair-loop case (verifier fails once, repair triggers transport rerun): merged timeline shows both the original `transport` phase and the rerun, named distinctly.
- Phase 4 `tests/unit/test_perf.py` extends to cover:
  - `merge_timelines(None, t) == t`, `merge_timelines(t, None) == t`, `merge_timelines(t, t) is t` (identity case).
  - `merge_timelines(t1, t2)` concatenates phases preserving relative order within each input.
  - Reducer is associative: `merge(merge(a, b), c) == merge(a, merge(b, c))` (modulo phase ordering).
- Phase 4 `tests/integration/test_coordinator_e2e.py` runs in parallel mode and asserts the final state has all expected phase names exactly once each.

**Risks:**
- **Order non-determinism in merged timeline.** Mitigation: `summary()` sorts by `start_perf_counter`. Within-phase ties are stable (insertion order).
- **Reducer called with mutated left side after right was created.** Could surface if a node mutates the existing `state["timeline"].phases` instead of creating a fresh local one. Mitigation: code review checklist + grep for `state.get("timeline")` after the fix; only the reducer should touch the existing list.
- **Phase 4 mocked-LLM tests assert specific timeline shapes.** Mitigation: write the tests against `summary()` (sorted, stable), not raw `phases` ordering.
- **Repair-loop disambiguation increases phase-name cardinality.** Mitigation: UI groups by base name with a "(repair rN)" suffix; expandable in the Performance panel if visual clutter shows up.
- **Existing Phase 7 plan refers to `Timeline.summary()` shape.** Mitigation: this hotfix lands first, so Phase 7 builds on the merged-timeline shape from day one.

#### Bug H2 — Env-var names visible in Streamlit sidebar

**Symptom:** Streamlit sidebar's "Live API status" panel currently lists raw env-var names with their set/missing status:
```
- VERTEX_AI_API_KEY: set
- LLM_MODEL: gemini-2.5-flash-lite (env default)
- GOOGLE_MAPS_API_KEY: missing
- SERPAPI_API_KEY: set
```
This leaks internal configuration into a user-facing UI. End users don't need to know which env var controls which feature; demoers / screen-share contexts expose names that look like secrets even when only "set/missing" is shown.

**Root cause:** Phase 1 lifted the original sidebar layout without separating dev-facing diagnostics from user-facing UI. There's no boundary between the two.

**Fix:**
1. **Remove** the `Live API status` block from `app.py` sidebar entirely.
2. **Replace** with a feature-level status indicator that uses plain English labels — never env var names:
   ```
   Live data sources
     • Flights:           live  /  disabled
     • Places & routes:   live  /  disabled
     • LLM:               <selected model display name>
   ```
   `Flights: live` requires SerpAPI configured; `Places & routes: live` requires Google Maps configured; the indicator reads from a small adapter (`agents/runtime_status.py`) that wraps the env checks but exposes feature-name booleans only. The adapter NEVER returns the env var name itself.
3. **Move all env-var diagnostics** to `scripts/validate_env.py` (Phase 7). Devs run it locally; never shipped in the UI.
4. **Sidebar caption** under the new status block:
   > If a feature shows *disabled*, run `python scripts/validate_env.py` to diagnose configuration.
   This is the only place the user sees a script name, and it's an actionable next step, not internal naming.
5. **Phase 7's per-API status table** (Performance expander) uses the same feature-level names: `Flights API: ok` / `Places API: error: HTTP 403`. No env var names there either.
6. **Error banners (Phase 7)** classify failures as feature errors, not env errors: "Flights unavailable — check API configuration" rather than "SERPAPI_API_KEY invalid".

**Validation:**
- `grep -nE 'VERTEX_AI_API_KEY|GOOGLE_MAPS_API_KEY|SERPAPI_API_KEY|GOOGLE_CLOUD_PROJECT|CLAUDE_VERTEX_REGION|GOOGLE_APPLICATION_CREDENTIALS|LLM_MODEL' app.py` returns nothing (after the fix).
- Sidebar renders the new feature-level status correctly with each combination of keys present/absent (manual run with various `.env` configs).
- README's setup section continues to list the env vars (developer-facing), but the running app does not.
- Phase 4 `tests/unit/test_app_no_env_leak.py` does the same grep on the imported `app.py` source as a regression test.

**Risks:**
- **Users confused about which env var to set when a feature shows "disabled".** Mitigation: the validate_env caption gives a single command with diagnostic output; that's the contract.
- **Reintroducing env var names elsewhere** (error toast, performance expander, debug logs). Mitigation: `tests/unit/test_no_env_leak.py` runs the same grep across `app.py`, `agents/coordinator.py` UI surfaces, and any log lines emitted by `on_progress`. Combined with `test_no_secret_logging.py` (Phase 7) for the key-value leak case.
- **Future contributors adding new features may copy the old pattern.** Mitigation: a one-line note in `ARCHITECTURE.md` §11 invariants — "User-facing UI never displays env var names; that's a developer-tools concern."

**Done criteria for the hotfix block:**
- [ ] `merge_timelines` reducer added to `agents/perf.py` with unit tests for null/identity/associativity cases.
- [ ] `CoordinatorState.timeline` is `Annotated[Timeline, merge_timelines]`.
- [ ] Every coordinator/baseline node returns a single-phase fresh Timeline; no node mutates a shared `state["timeline"]`.
- [ ] Live `streamlit run app.py` succeeds in parallel mode, sequential mode, and the repair-loop case.
- [ ] `Timeline.summary()` returns phases sorted by start time.
- [ ] Sidebar in `app.py` no longer mentions any env var name; feature-level "Live data sources" panel renders correctly.
- [ ] `grep` regression check passes.
- [ ] `ARCHITECTURE.md` §11 gains the "no env-var names in user-facing UI" invariant.

### Phase 4 — Test infrastructure
**Goal:** establish a pytest suite that gives fast feedback on deterministic logic and lets us refactor without paying API quota or running the 180-query eval. Annotated-train sanity remains the single source of truth for evaluator correctness; tests cover everything *around* it.

Why now: Phase 1–3 added `agents/models.py`, `agents/runtime.py`, `agents/providers/`, `agents/perf.py`, plus per-model CSV plumbing and Timeline instrumentation. The codebase has more moving parts. The eval is a 180-query integration test, but it's expensive and slow — we need cheaper signals when changing internals.

Steps:
1. Create `tests/` directory with `unit/` and `integration/` subdirs (see §3.5 file list).
2. Add `tests/conftest.py` with shared fixtures:
   - `sandbox` — frozen sandbox singleton, scope="session".
   - `intent_factory` — builds an `Intent` with sensible defaults; overridable per-test.
   - `tool_context_factory` — builds a `ToolContext` from sandbox or hand-crafted dicts.
   - `fake_backend` — replaces `agents/providers/call_provider` via monkeypatch; takes a dict mapping `(schema_name, query_substring) → canned_json`.
   - `tmp_disk_cache` — temp dir fixture, sets `LLM_CACHE_DIR` for the test.
3. Write unit tests in priority order: budget → constraints → verifier → assemble → timing → llm_cache → models → runtime → perf.
4. Write integration tests: coordinator end-to-end, repair routing, multi-provider parity (with stubbed `AnthropicVertex`).
5. Add `pytest.ini` with `unit`/`integration`/`slow`/`requires_api_key` markers.
6. Add `.github/workflows/test.yml` running `pytest -m unit -q`.
7. Wire a `make test` target (or a `tasks.py` invoke entry) for ergonomics.

LLM mock strategy:
- The `FakeBackend` is selected by monkeypatching `agents.providers.call_provider`. For each `(schema_cls.__name__, query_substring)` key, it returns a canned `(text, in_tok, out_tok)` triple.
- Canned responses live in `tests/fixtures/llm_responses/<schema_name>__<id>.json`.
- For end-to-end determinism, integration tests can alternatively set `LLM_CACHE_DIR=tests/fixtures/cache` and use real responses captured once.
- Critically, the fake backend respects the cache key, so cache-hit tests still work.

Validation gate:
- `pytest tests/unit -m unit` passes in < 30s.
- `pytest tests/integration -m integration` passes with no real API calls (verified by `responses`/`pytest-monitor` or just a sentinel that asserts `_genai_client is None`).
- Annotated-train sanity check (`run_eval --split train --system annotated --limit 10`) still ~100% on within_sandbox.
- Coverage report (informational only — do not block on a number) shows the new modules from Phase 1–3 are exercised.

### Phase 5 — Single-agent baseline UX + parity
**Goal:** the single-prompt baseline is a first-class option in both eval and the Streamlit app. The user can experience, side-by-side, what a "no orchestration" approach produces, making the report's headline argument tangible.

Why now: `baseline/single_agent.py` is sandbox-only today, accessed only through eval. The report's claim ("multi-agent beats single-prompt") is much more credible when the gap can be reproduced live.

What "single agent" means in this codebase:
- One LLM call sees the bundled tool results: transport options + hotels + restaurants + attractions + routes for the destination.
- Returns a complete `FullPlan` JSON spanning all four domains in one pass.
- No specialist fan-out, no Budget Agent, no verifier-repair loop. (The verifier *runs* at the end for scoring, but its failures are not repaired.)
- Uses the SAME `ToolContext` the multi-agent path uses (live or sandbox) — only the decomposition differs, not the data.

Steps:
1. Modify `baseline/single_agent.py`:
   - Accept `tool_context: ToolContext | None = None`.
   - When `None` and not given a sandbox, build live tool_context from `default_live_apis()` exactly like coordinator's research_node (extract a `_build_live_tool_context(intent)` helper to share both code paths).
   - Reformat `_bundle_tools` to consume a `ToolContext` dict, not a sandbox object — sandbox path becomes "build sandbox-derived ToolContext, then call the unified bundler".
   - Default `bypass_date_check=False` for the live path.
2. Modify `app.py`:
   - Sidebar adds `Mode` radio: `Multi-agent (default)` / `Single-agent (baseline)`.
   - When Single-agent: call `plan_trip_single(query, model=..., on_progress=...)` instead of `plan_trip`.
   - Plan title gets a mode label so screenshots in the report are unambiguous.
   - Performance expander shows the same LLM stats; Timeline is omitted for single-agent (only one phase) but cache hit/miss still surface.
3. Optional Phase 5b: "Compare modes" button that runs both in parallel and renders side-by-side. Defer if time-constrained.

Validation gate:
- `plan_trip_single(query)` works end-to-end against live APIs for a real query (manual smoke).
- App's single-agent mode renders cleanly in the same UI components.
- Eval row count in the matrix: single-agent runs across the same model variants as multi (apples-to-apples cells).
- Annotated-train eval is unaffected (single-agent baseline doesn't touch the evaluator).
- Live single-agent on a 7-day query does not hit `max_tokens` truncation (bump from 3500 → 4500 if observed, with a guard log).

### Phase 6 — Specialist prompt audit + targeted improvements
**Goal:** each specialist's prompt is reviewed against the constraints it must satisfy and tightened based on observed failure modes from the per-rule pass-rate diagnostics. Output: measurable improvement on at least one CS rule per audited prompt, no aggregate regression.

Why now: with per-model CSV (Phase 3) we can finally see *which* rules drag *which* models down. Targeted prompt fixes will move the needle. Without instrumentation, prompt changes were guesses.

Audit method (per specialist):
For each of `transport`, `lodging`, `dining`, `sightseeing`, `single_agent`, `no_specialization`:
1. Enumerate every commonsense + hard rule that this specialist's output can violate.
2. Map each rule to the prompt line(s) that address it today (or note "missing").
3. Pull the failure modes from `eval/results.csv` per-rule pass rates and from the per-violation log.
4. Propose minimal prompt edits + a few-shot example for the most common failure mode.
5. Record the audit in `Project_Code/PROMPT_AUDIT.md` as a checklist matrix.

Suspected gaps (initial hypotheses to validate during the audit, NOT yet acted on):

**transport.py**
- Rules owned: arrival_day, departure_day, non_conflicting_transport, within_sandbox (transport), budget (transport portion), `transportation` hard constraint (mode preference).
- Strengths today: explicit "FIRST option listed", flight-ID format, time-window preservation.
- Suspected gaps:
  - Self-driving / taxi format isn't shown in the few-shot example (only flights are illustrated).
  - When user requests "no flights", the prompt doesn't redirect to driving/taxi explicitly.
  - One-way trip handling ambiguous (only Day 1 transportation, no return leg).
- Validation: per-rule pass rate on rows with a `transportation` constraint set.

**lodging.py**
- Rules owned: single_accommodation, room_rule, room_type, min_nights, within_sandbox (lodging), budget (lodging portion).
- Strengths today: SAME hotel name on every non-departure day, "Cost: $N for D nights" format, "verbatim from numbered list".
- Suspected gaps:
  - room_rule (e.g. "no parties allowed") and room_type (e.g. "entire home") aren't surfaced in the prompt — relies on `intent.model_dump_json()` context being parsed correctly by the model.
  - min_nights vs trip-length conflict not addressed (e.g. trip is 2 nights but only 5+ night rentals available — the prompt should pick a flexible option, not pretend the conflict away).
- Validation: per-rule pass rate on `room_rule` and `room_type`.

**dining.py**
- Rules owned: diverse_restaurants, cuisine, no_repeat across breakfast/lunch/dinner, within_sandbox (dining), arrival/departure day "-".
- Strengths today: cuisine synonyms, no-repeat reinforcement.
- Suspected gaps:
  - "Diverse" vs "no repeat" — clarify to "every restaurant unique across the entire trip, not just within a day".
  - Day-1 breakfast handling (often "-" because traveler arrives later) isn't pre-empted in the prompt; it's fixed up post-hoc by `_apply_timing_adjustments`.
  - Cuisine constraint can be soft when the sandbox is sparse — the prompt should say "match cuisine when possible; otherwise pick within_sandbox restaurants and surface in repair note".
- Validation: per-rule pass rate on `diverse_restaurants` and `cuisine`.

**sightseeing.py**
- Rules owned: diverse_attractions, within_sandbox (attractions), within_current_city, reasonable_city_route, arrival/departure day rules.
- Strengths today: verbatim names, no-repeat reinforcement.
- Suspected gaps:
  - departure-day attraction often ends as "-"; prompt should make this explicit so it's not relying on post-hoc patching.
  - Multi-day trips with heavy travel days could over-pack attractions; the prompt could cap at 1–2 attractions on travel days.
- Validation: per-rule pass rate on `diverse_attractions`.

**single_agent.py and no_specialization.py**
- These prompts cover ALL rules in one shot — they're the most likely to underperform on long-tail rules.
- Plan: rebuild these prompts using the union of specialist constraints, with a single-pass few-shot example demonstrating the format.
- These are also the prompts most exposed in Phase 5 (they ship in the Streamlit single-agent mode), so prompt quality directly affects user perception of the baseline.

Steps:
1. **Phase 6.1 — Audit (no code, only `PROMPT_AUDIT.md`).** For each of the six prompts, write the rule-coverage matrix and pull observed pass rates from the latest `eval/results.csv` at the default model.
2. **Phase 6.2 — Targeted edits (one prompt per commit).** For each prompt:
   - Apply the minimal edit identified in 6.1.
   - Run `python -m eval.run_eval --split validation --system multi --model gemini-2.5-flash-lite --llm-cache-dir .cache/llm` and diff per-rule pass rates against the pre-edit run.
   - Commit only if the targeted rule moved up ≥ 5pp and no other rule regressed > 2pp.
3. **Phase 6.3 — Cross-model regression.** Re-run the full sweep (`--models all`) once the four-specialist edits are in. Confirm the per-model TravelPlanner table doesn't show new regressions on Claude.

Validation gate:
- `PROMPT_AUDIT.md` exists with one section per prompt, each with a rule-coverage matrix and observed-failure analysis.
- Each merged 6.2 commit links to a per-rule diff showing ≥ 5pp improvement on at least one targeted rule, no rule regressing > 2pp.
- After 6.3, the per-model table in the report shows neutral-or-better numbers on every cell relative to pre-Phase-6 baseline.

Risks:
- Edits that improve one rule may hurt another (e.g. stronger budget enforcement → more "Cost: $0 — within constraint" hallucinations).
  *Mitigation:* one-prompt-per-commit; revert on net regression.
- Cross-model interaction — a prompt tuned for Gemini may underperform on Claude (forced tool-use changes the dynamics).
  *Mitigation:* track per-model pass rates separately; if divergent, document instead of force-merging.
- Few-shot bloat — adding examples grows token count and latency.
  *Mitigation:* perf benchmark (Phase 2 set) re-run after Phase 6 to ensure no P50/P95 regression.

### Phase 7 — Live-API observability + strict-runtime fix
**Goal:** the live runtime stops silently swallowing SerpAPI / Google Maps errors. When the user provides keys, they should see whether the calls actually succeeded — not get an empty `flights_outbound` and a plan that pretends flights aren't real.

#### Why now (root cause)

`tools/live_apis.py` is **already strict** — every method raises a clear `RuntimeError` with diagnostic context:
- `flight_search`: missing key (line 131), HTTP failure (line 157), empty results (line 196).
- `search_places`: missing key (line 61), SDK exception (line 73).
- `route_summary`: missing key (line 90), SDK exception (line 100).

But the **callers swallow these errors**:

```python
# agents/coordinator.py lines 468–481 (research_node)
try:
    ctx["flights_outbound"] = live.flight_search(intent.org, intent.dest, depart, budget=intent.budget)
except Exception:
    ctx["flights_outbound"] = []  # ← silently turns ANY error into empty list

try:
    ctx["flights_return"] = live.flight_search(intent.dest, intent.org, ret, budget=intent.budget)
except Exception:
    ctx["flights_return"] = []
```

`baseline/no_specialization.py` lines 51–62 has the same pattern. **Result:** when the user reports "SerpAPI seems not being executed" with `SERPAPI_API_KEY` set correctly, the actual cause could be a transient 5xx, a rate limit, an expired key, or a parser regression — but they all look identical from the UI side (empty `flights_outbound`, transport specialist falls back to driving/taxi or hallucinates).

This pattern violates the strict-runtime invariant in `ARCHITECTURE.md` §11.1: *"No deterministic / provider-switch fallback paths in the live runtime. Failures are errors, not silent stubs."* The invariant was honored when `tools/live_apis.py` was written but eroded over time as callers added defensive `try/except`.

A second, separate, source of confusion: **the eval path never calls SerpAPI at all.** `eval/run_eval.py` → `_sandbox_tool_context()` populates `flights_outbound/return` from `tools/sandbox.py` (TravelPlanner 2022 reference data). This is intentional — historical dates won't return live results — but is undocumented, so it surprises anyone instrumenting "where does SerpAPI fire?". Phase 7 documents this explicitly.

#### Three failure modes to disambiguate

| State | Today's behavior | Desired behavior |
|---|---|---|
| `SERPAPI_API_KEY` unset (live mode) | Bare `except` swallows `RuntimeError("Set SERPAPI_API_KEY")`; ctx flights = []. User has no idea flights were skipped intentionally vs failed. | Status `"skipped (no key)"`. Logged once at INFO. UI shows banner: "Live flights disabled — set SERPAPI_API_KEY to enable." Plan continues with driving/taxi. |
| `SERPAPI_API_KEY` set, transient error (5xx, rate limit, parse) | Bare `except` swallows; ctx flights = []. Specialist falls back; plan completes; user sees no error. | Status `"error: <short message>"` (e.g. `"error: HTTP 503"`). UI banner: "Live flights unavailable — see Performance panel for details." Plan continues with driving/taxi but the failure is visible. |
| `SERPAPI_API_KEY` set, success | ctx flights populated; transport specialist uses them. | Status `"ok"` with a count: `"ok (5 options)"`. Visible in Performance panel. |

The same three-state model applies to `search_places` (hotels/restaurants/attractions) and `route_summary` (driving/taxi). Eval path stays sandbox-only and reports nothing for live-API status.

#### Steps

1. **Audit pass (no code).** Walk every `try: live.<method>(...) except Exception: ...` site in the repo. Record in `LIVE_API_AUDIT.md` (one row per call site): file:line, method called, current swallow behavior, key-set semantics, key-unset semantics, planned new behavior. Sites known today:
   - `agents/coordinator.py:468-474` — `flight_search` outbound
   - `agents/coordinator.py:476-481` — `flight_search` return
   - `agents/coordinator.py:483-486` — `route_summary` driving
   - `agents/coordinator.py:488-491` — `route_summary` taxi
   - `agents/coordinator.py:495-499` — `search_places` (hotels/restaurants/attractions; currently NOT wrapped — sandbox fallback handles empties; verify this stays raise-on-key-error after the audit)
   - `baseline/no_specialization.py:51-62` — `flight_search` outbound + return
   - `baseline/no_specialization.py:63-70` — `route_summary` driving/taxi
2. **Helper.** Add `_call_live(name: str, fn: Callable, *, key_set: bool, on_status: Callable[[str, str], None]) -> Any | None` to `agents/coordinator.py` (or a new `tools/live_call.py` if it's reused by no_specialization). Behavior:
   - `key_set=False` → don't call; report status `"skipped (no key)"`; return `None`.
   - `key_set=True` and `fn()` succeeds → return result; report `"ok"` (with count when applicable).
   - `key_set=True` and `fn()` raises → catch, classify status (`"error: HTTP <code>"` for `RuntimeError` with `status_code` attr; `"error: <type>"` otherwise), report status, return `None`. Re-raise only when `LIVE_API_STRICT=1` env is set (used by tests and the future preflight).
3. **Status threading.** `coordinator.research_node` populates `state["live_api_status"]` (new typed key on `CoordinatorState`). `agents/perf.py` `Timeline` gets an optional top-level `live_api_status` field surfaced via `summary()`. The `eval/run_eval.py` path already skips research; no CSV columns added (eval doesn't exercise live APIs).
4. **UI surface.** `app.py`:
   - Performance expander: render the status table.
   - Top-of-plan banner: when any key-set live API errored on this run, show a one-liner with the *first* error and a "Show details" link to the Performance expander.
   - Sidebar Live API status panel: extend with three dots (green/yellow/red) reflecting the most recent run's status per API.
5. **Preflight script.** `scripts/validate_env.py` (per §3.5): runs the same `_call_live` helper with `LIVE_API_STRICT=1` against tiny known-good payloads. Exits non-zero on any *configured* failure. Documented in README.
6. **Tests.** Per §3.5 Phase 4 entries: `test_live_apis_serpapi.py`, `test_live_apis_maps.py`, `test_env_validation.py`, `test_no_secret_logging.py`, `test_validate_env.py`. Plus an integration test `tests/integration/test_research_node_status.py` that monkeypatches `tools.live_apis` and asserts the status dict is populated correctly across all three failure modes.
7. **Documentation.** `ARCHITECTURE.md` §9.1 + §11; `README.md` setup section; `LIVE_API_AUDIT.md` published as part of the Phase 7 commit.

#### Validation gate

- `python scripts/validate_env.py` exits 0 when all keys configured correctly; exits 1 with actionable message when any *configured* key fails.
- Live `app.py` run with valid `SERPAPI_API_KEY`: Performance expander shows `serpapi_outbound: ok (N options)`; flights specialist receives non-empty list.
- Live `app.py` run with `SERPAPI_API_KEY` unset: Performance expander shows `serpapi_outbound: skipped (no key)`; sidebar status dot is yellow; UI banner notes "Flights disabled".
- Live `app.py` run with deliberately INVALID `SERPAPI_API_KEY`: Performance expander shows `serpapi_outbound: error: HTTP 401`; sidebar dot is red; UI banner shows actionable error; plan still completes via driving/taxi.
- `pytest -m unit tests/unit/test_live_apis_*.py tests/unit/test_env_validation.py tests/unit/test_no_secret_logging.py` passes.
- Annotated-train sanity (eval, sandbox-only) unaffected — eval CSV schema unchanged.
- `LIVE_API_AUDIT.md` exists with one row per call site listed above.

#### Risks (Phase 7-specific)

- **Surfacing transient errors confuses users on quota hits.** Mitigation: classify `RuntimeError` by status code; show retry-with-delay guidance for 429/5xx; show key/billing guidance for 401/403.
- **Extending `Timeline` schema breaks Phase 4 fixtures.** Mitigation: land schema change before fixture commit; regen fixtures via `scripts/regen_fixtures.py` (Phase 4) once.
- **API keys leak into log lines or error messages.** Mitigation: `tests/unit/test_no_secret_logging.py` injects `SECRET-MARKER-12345` into every key env var, triggers every error path, and asserts the marker never appears in the formatted exception or any log call.
- **`LIVE_API_STRICT=1` mode used in production.** Mitigation: documented in README only as a debugging knob; preflight script sets it itself; `app.py` does NOT consult it.
- **Sandbox fallback in `_sandbox_fallback` (existing behavior) interacts oddly with new status reporting.** Mitigation: when `search_places` succeeds with empty results AND sandbox fallback fires, status is `"ok (sandbox fallback)"` not `"error"` — fallback is by design (line 502–516 of coordinator.py), the audit confirms no behavior change there.

---

## 5. Latency-improvement candidates (for Phase 2)

Ranked by impact × risk. Each is a separate commit, gated by §4 Phase 2 validation.

1. Default `eval/run_eval.py` to **parallel** specialist execution; reserve sequential for ablation.
2. Cache hit/miss instrumentation; warm cache before any benchmark run.
3. Verify research node executes **once** before fan-out (not redundantly per specialist) — confirm via `Timeline` phase order.
4. Confirm the UI clarification flow doesn't double-parse intent — `parsed_intent=` should already be threaded; verify with a Timeline test.
5. Confirm structured-output binding in both providers (no free-text-then-parse retries).
6. Memoize Pydantic schema → JSON schema conversion in `call_json` (one-line `functools.lru_cache`).
7. A/B test default temperature per provider — only ship if measurable.

Do not optimize anything that the perf panel doesn't show as a hot phase.

---

## 6. Main eval sweep (Phase 3)

Run order: Gemini first (safe, well-tested path), Claude second (region/quota risk).

```text
# Full multi-agent on all registered models — load-bearing rows for the report
python -m eval.run_eval --split validation --system multi --models all --llm-cache-dir .cache/llm

# Architectural ablations on 2 Gemini representatives only
python -m eval.run_eval --split validation --system no_verify        --models gemini-2.5-flash-lite,gemini-2.5-flash --llm-cache-dir .cache/llm
python -m eval.run_eval --split validation --system no_specialization --models gemini-2.5-flash-lite,gemini-2.5-flash --llm-cache-dir .cache/llm
python -m eval.run_eval --split validation --system single            --models gemini-2.5-flash-lite,gemini-2.5-flash --llm-cache-dir .cache/llm
```

Cells covered: 6 multi-parallel rows + 1 multi-sequential row + 6 baseline rows (3 baselines × 2 Gemini models) = **13 cells × 180 queries = 2,340 plans**. With disk cache, only the first run pays full price.

---

## 7. Comparison axes the report needs

For each model row in the results table:

1. **Vs paper Table 3.** Final pass rate compared to GPT-4-Turbo Direct (4.4%) and ReAct (0.6%). Paper has no Gemini-2.x or Claude rows — those are net-new data points.
2. **Cross-tier (within provider).** Lite vs standard at fixed family.
3. **Cross-generation (within Gemini).** 2.0 vs 2.5 at fixed tier.
4. **Cross-provider (Gemini vs Claude).** Two pairs at matched tiers: lite-vs-haiku, flash-vs-sonnet.
5. **Per-rule failure profile.** Hypothesis: Budget pass rate is identical across models (deterministic Budget Agent); Within-Sandbox tracks model size; Min-Nights tracks model reasoning quality.
6. **Sequential vs parallel.** Latency-only story on the default model; quality should match.

---

## 8. Done criteria (engineering)

### Phase 1–3 (landed)
- [x] User can pick any registered model in the sidebar; each runs end-to-end.
- [x] Claude entries auto-hidden when GCP project / ADC are not configured.
- [x] `LLM_MODEL` env var still works as a default for non-UI invocations.
- [x] Disk cache keys include provider + model — verified by running same query on two models, observing two cache misses then two cache hits.
- [x] UI Performance expander shows phase breakdown after every plan.
- [ ] `eval/perf_baseline.csv` committed; subsequent perf runs diff against it; no merged change regresses P50.
- [x] `eval/results.csv` has `model` and `provider` columns; sweep across registered models completes on validation split.
- [x] `eval/format_report.py` outputs the two Markdown tables.
- [ ] `README.md` mentions GPT is excluded because Vertex AI does not host OpenAI models.

### Hotfix block (must land before Phase 4)
- [x] H1: `merge_timelines` reducer added to `agents/perf.py`; unit-tested for null/identity/associativity.
- [x] H1: `CoordinatorState.timeline` is `Annotated[Timeline, merge_timelines]`.
- [x] H1: Every coordinator/baseline node returns a single-phase fresh `Timeline`; no shared mutation.
- [x] H1: `streamlit run app.py` in parallel mode succeeds end-to-end on a real query (no `InvalidUpdateError`).
- [x] H1: Sequential mode + repair-loop case both produce a clean phase list (sorted by start; `(repair rN)` suffix on reruns).
- [x] H2: `app.py` sidebar contains no env-var name strings (`grep` regression check passes).
- [x] H2: `agents/runtime_status.py` adapter present; sidebar uses feature-level labels only.
- [x] H2: Sidebar caption directs users to `scripts/validate_env.py` for diagnostics; no inline troubleshooting copy that names env vars.
- [x] H2: `ARCHITECTURE.md` §11 includes the "no env-var names in UI" invariant.

### Phase 4 — Tests
- [x] `tests/conftest.py` exposes shared fixtures (`basic_intent`, plan stubs); 238 total tests pass in < 3s.
- [x] `pytest tests/unit` passes in < 30s on a clean checkout (no API keys required).
- [x] `pytest tests/integration` passes with mocked LLM (no real Gemini / Claude calls during the run).
- [x] `tests/integration/test_multi_provider.py` stubs `AnthropicVertex` and verifies the Claude code path produces a valid plan.
- [x] `tests/unit/test_llm_cache.py` proves provider+model+schema separation: same prompt on two models = two misses + two hits on rerun.
- [ ] `.github/workflows/test.yml` runs unit tests on every push (workspace not a git repo — N/A for now).

### Phase 5 — Single-agent UX
- [ ] `baseline/single_agent.py` accepts `tool_context` and works with both live and sandbox data.
- [ ] Streamlit sidebar has a `Mode` toggle that switches between multi-agent and single-agent.
- [ ] Single-agent live mode produces a renderable plan for a real query (manual smoke).
- [ ] Eval sweep covers single-agent rows at the same model variants as multi-agent (no missing cells in the matrix).

### Phase 6 — Prompt audit
- [ ] `Project_Code/PROMPT_AUDIT.md` exists with a section per prompt: rule-coverage matrix, observed-failure analysis, planned edits.
- [ ] After Phase 6.2 edits, the per-rule pass-rate CSV shows ≥ 5pp improvement on at least one targeted rule per audited prompt at the default model.
- [ ] No CS or HC rule regresses by > 2pp; aggregate `final_pass` does not regress.
- [ ] Phase 6.3 cross-model regression check shows no Claude-specific degradation.

### Phase 7 — Live-API observability + strict-runtime fix
- [ ] `LIVE_API_AUDIT.md` exists with a row per `try: live.<method>(...)` call site documenting current and planned behavior.
- [ ] `agents/coordinator.py` no longer has bare `try / except Exception: pass` blocks around live API calls; all sites use the `_call_live` helper.
- [ ] `baseline/no_specialization.py` mirrors the coordinator's swallow-fix at lines 51–62 and 63–70.
- [ ] `state["live_api_status"]` and `Timeline.summary()["live_api_status"]` are populated for every live run with the three-state model (`ok` / `skipped (no key)` / `error: <msg>`).
- [ ] Streamlit Performance expander shows the per-API status table; sidebar shows colored status dots; top-of-plan banner appears on key-set errors.
- [ ] `scripts/validate_env.py` exists and exits non-zero on any misconfigured key with an actionable message.
- [ ] Live-API tests pass: `pytest tests/unit/test_live_apis_serpapi.py tests/unit/test_live_apis_maps.py tests/unit/test_env_validation.py tests/unit/test_no_secret_logging.py tests/unit/test_validate_env.py`.
- [ ] `tests/integration/test_research_node_status.py` covers all three live-API states (ok / skipped / error) with mocked transports.
- [ ] No API key value (`"SECRET-MARKER-12345"` test injection) appears in any formatted exception or log line.
- [ ] `ARCHITECTURE.md` §9.1 explicitly documents that **the eval path does not call SerpAPI** (sandbox-only) — distinct from "SerpAPI broken in eval".
- [ ] `README.md` setup section instructs the user to run `python scripts/validate_env.py` after editing `.env`.

---

## 9. Risks (engineering only — full register is in `../CODE_PLAN_MULTI_MODEL.md`)

| Risk | Mitigation |
|---|---|
| Cache key collision across models / providers | Provider + model in cache key; smoke test in Phase 1 validation |
| Claude not enabled in GCP region | Phase 0 gate; ship Gemini-only fallback if blocked |
| Claude tool-use forced-output occasionally fails to call the tool | Retry with stronger forcing prompt; tally per-model parse failures separately |
| Schema-output behavior differs across Gemini families | Existing Pydantic `model_validator(mode="before")` normalizers cover most drift; log per-model parse failures |
| Vertex rate limits during 12-cell sweep | Disk cache + sequential model rotation + `--cooldown-seconds` + existing retry/backoff |
| ADC not configured on eval machine | Document `GOOGLE_APPLICATION_CREDENTIALS=sa.json` workflow in README |
| Streamlit `st.session_state` drift on model change | Reset relevant state on selectbox change |
| `importlib.reload(agents.llm)` hack masks regressions after refactor | Delete it as part of Phase 1; verify cache-dir env still picked up |
| Test fixtures going stale when schemas change (Phase 4) | `scripts/regen_fixtures.py` rebuilds canned LLM responses from a small canonical query set |
| Mocked LLM tests don't exercise Claude code path (Phase 4) | Dedicated `test_multi_provider.py` stubs `AnthropicVertex` to assert provider router and tool-use parsing |
| Live single-agent hits `max_tokens` for 7-day trips (Phase 5) | Bump `max_tokens` from 3500 → 4500 with a guard log when truncated; document threshold in code |
| Single-agent UX confuses users about which mode produced a plan (Phase 5) | Plan title and Performance expander explicitly label the mode + model |
| Prompt edit improves one rule, regresses another (Phase 6) | One-prompt-per-commit; gate on per-rule diff; revert on net regression |
| Prompt tuned for Gemini underperforms on Claude (Phase 6) | Track per-model pass rates separately; document divergence rather than over-fit one provider |
| Few-shot bloat in prompts increases latency (Phase 6) | Re-run Phase 2 perf set after Phase 6; revert if P50 regresses |
| Audit reveals a constraint specialists can't satisfy with the current sandbox | Document as a sandbox limitation in the report rather than papering over with prompt tricks |
| Bare `except Exception` in coordinator silently masks SerpAPI / Maps failures (root cause of "SerpAPI not executing") | Phase 7: `_call_live` helper distinguishes key-unset / error / ok; status surfaces in UI |
| Surfacing transient errors confuses users on quota hits (Phase 7) | Map HTTP status to actionable message ("Retry in 30s — quota exceeded"; "Check billing on key"); never include raw key value |
| `Timeline` schema change breaks Phase-4 fixtures (Phase 7) | Land Timeline change before fixture commit; rerun `scripts/regen_fixtures.py` once |
| API keys leak into error messages or logs (Phase 7) | `tests/unit/test_no_secret_logging.py` injects a marker key into env vars and asserts it never appears in exceptions |
| User reports "SerpAPI not executing" because eval mode bypasses it by design | Document explicitly in `ARCHITECTURE.md` §9.1 and in this plan (Phase 7 §"Why now"); not a bug |
| Live-API integration tests need real keys (Phase 4 / Phase 7) | Mark `@pytest.mark.requires_api_key`; CI default skips them; document how to run locally |
| `LIVE_API_STRICT=1` debug knob accidentally enabled in production | Document as debug-only; preflight script sets it itself; `app.py` ignores it; not exposed in `.env.example` |
| Sandbox fallback in `search_places` interacts with new status reporting (Phase 7) | `_sandbox_fallback` populated → status `"ok (sandbox fallback)"`; existing behavior preserved, just labeled |
| Parallel coordinator nodes write the same `timeline` key, raising LangGraph's `InvalidUpdateError` (Hotfix H1) | `Annotated[Timeline, merge_timelines]` reducer + each node returns a fresh single-phase Timeline; sorted in `summary()` |
| Concurrent mutation of shared `Timeline.phases` list across parallel threads (Hotfix H1) | Fresh-Timeline-per-node fix avoids shared mutation; documented as a code-review checklist item |
| Repair-loop reruns produce duplicate phase names in merged timeline (Hotfix H1) | `(repair rN)` suffix via `phase_with_round` helper; UI groups by base name |
| `app.py` sidebar leaks internal env-var names to end users (Hotfix H2) | Remove env-var listing entirely; replace with feature-level "Live data sources" panel from `agents/runtime_status.py` adapter |
| Future contributors reintroduce env-var names in UI surfaces (Hotfix H2) | `tests/unit/test_no_env_leak.py` regression grep + `ARCHITECTURE.md` §11 invariant |
| Users can't diagnose "disabled" features without seeing env-var names | Sidebar caption points to `scripts/validate_env.py` (Phase 7) which gives actionable per-feature diagnostics |

---

## 10. References

- `../PROJECT_REPORT_PLAN.md` — paper-aligned analysis and report structure.
- `../CODE_PLAN_MULTI_MODEL.md` — full how/why with provider-auth deep dive.
- `../CLAUDE.md` — workspace-wide conventions for working in this repo.
- `ARCHITECTURE.md` — current architecture + planned multi-provider LLM layer (§4.2).
- `README.md` — setup, run instructions, GPT-exclusion note.
- `PROMPT_AUDIT.md` *(Phase 6 deliverable)* — per-prompt rule-coverage matrix, observed-failure analysis, planned edits.
- `LIVE_API_AUDIT.md` *(Phase 7 deliverable)* — per-call-site decision matrix for every `live.*` invocation; documents the swallow-vs-raise-vs-skip behavior change.
- `scripts/validate_env.py` *(Phase 7 deliverable)* — preflight that pings each configured API once and reports actionable failures before the user runs `app.py`.
- `tests/` *(Phase 4 deliverable)* — pytest suite (unit + integration + fixtures).
- Paper 1: Xie et al. (2024), *TravelPlanner* (arXiv:2402.01622) — `../Project_Literature_Review/Paper 1.pdf`.
