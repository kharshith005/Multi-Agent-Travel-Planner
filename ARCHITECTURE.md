# Architecture

System architecture reference for the Multi-Agent Travel Itinerary Planner.

---

## 1. System overview

A coordinator-led multi-agent system that converts a natural-language travel query into a day-by-day itinerary satisfying user-stated hard constraints and TravelPlanner-style commonsense constraints.

Two execution paths share the same coordinator and specialist code:

1. **Live API path** (`app.py`) — research node calls Google Maps + SerpAPI; specialists plan over real-world data. Both multi-agent and single-agent modes use live APIs in this path via `build_live_tool_context()`.
2. **Sandbox eval path** (`eval/run_eval.py`) — research node is replaced by a `ToolContext` built from TravelPlanner reference data; specialists are unaware of the substitution because the dict shape is identical.

This dual-path design is load-bearing: the same multi-agent code ships in the app and produces the benchmark numbers.

---

## 2. Component map

```
                      ┌─────────────────────────────┐
          user query →│  Streamlit UI (app.py)       │
                      │  - clarification flow        │
                      │  - model selector            │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Coordinator                 │
                      │  (agents/coordinator.py)     │
                      │  parse_intent → constraint   │
                      │  gate → past-date guard →    │
                      │  research → specialists →    │
                      │  assemble → budget →         │
                      │  verify → repair (≤3 rounds) │
                      └──────────────┬───────────────┘
                                     │ ToolContext + Intent
                      ┌──────────────┴───────────────┐
                      │  Research node               │
                      │  Live: tools/live_apis.py    │
                      │  Eval: tools/sandbox.py      │
                      │  Populates trip_windows here │
                      └──────────────┬───────────────┘
                                     │ shared ToolContext
        ┌──────────┬──────────┬──────┴──────┬────────────┐
        ▼          ▼          ▼             ▼            │
  ┌──────────┐┌──────────┐┌──────────┐┌────────────┐    │
  │Transport ││ Lodging  ││ Dining   ││Sightseeing │    │
  └────┬─────┘└────┬─────┘└────┬─────┘└─────┬──────┘    │
       └───────────┴─────┬─────┴────────────┘            │
                         ▼                               │
               ┌──────────────────┐                      │
               │ Assembler        │                      │
               │ → FullPlan       │                      │
               └────────┬─────────┘                      │
                        ▼                               ▼
               ┌──────────────────┐◄────────────────────┘
               │ Budget Agent     │  post-hoc deterministic
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │ Verifier         │  8 CS + 5 HC rules;
               │ (verifier.py)    │  tags violations with
               │                  │  responsible specialist
               └────────┬─────────┘
                        │ passed → return FullPlan
                        ▼ failed
               ┌──────────────────┐
               │ Repair router    │  Stage 1: deterministic fix
               │ (≤3 rounds)      │    (diversity, sandbox)
               │                  │  Stage 2: LLM re-run on
               │                  │    responsible specialists
               └──────────────────┘

  LLM facade:       agents/llm.py
  Provider backends: agents/providers/{gemini,vertex_requests}.py
  Model registry:   agents/models.py
```

---

## 3. Runtime flow

```
parse_intent (LLM)
  → constraint gate (required fields present)
  → past-date guard (bypass in eval mode)
  → research node
      Live: Google Maps + SerpAPI + airport resolution (IATA + lat/lon)
            + geo-radius lodging search near destination
            + trip_windows derived from outbound/return flights
      Eval: _sandbox_tool_context() + derive_trip_windows()
            Multi-city: city sequence from per-row reference_information
  → specialists fan-out (transport, lodging, dining, sightseeing)
      All receive trip_windows in ToolContext
  → assemble FullPlan
  → Budget Agent (deterministic, no LLM)
  → Verifier (8 CS + 5 HC rules)
      passed → return
      failed and round < REPAIR_MAX_ROUNDS (3):
          repair_prepare_node:
            Stage 1 — Deterministic pre-repair (no LLM):
              diversity + sandbox violations fixed from ToolContext directly
            Stage 2 — LLM repair:
              remaining violations routed to responsible specialists only
          → specialists rerun (only those flagged)
          → assemble → budget → verify → loop
      failed at cap → return plan + VerifierReport
```

---

## 4. LLM provider layer

### 4.1 Facade — `agents/llm.py`

Public API: `call_json`, `call_text`, `reset_call_stats`, `get_call_stats`.

- Active model set via `agents/runtime.py` `ContextVar`; `use_model("llama-3.3")` resolves aliases and sets canonical ID for the request's lifetime.
- **Two-tier cache:** in-memory LRU (`LLM_CACHE_SIZE`, default 256) + optional disk cache (`LLM_CACHE_DIR`, content-addressed `.json` files). Cache key: `sha256(provider | model_id | schema_name | max_tokens | think_first | system | user)`.
- **Retry loop:** exponential backoff on 429/5xx. 429 escalates to a longer recovery policy (Vertex TPM buckets refill over ~60 s).
- **Thread-safe stats:** `{llm_calls, input_tokens, output_tokens, cache_hits, cache_misses}`. Reset with `reset_call_stats()` before a query; read with `get_call_stats()` after.

### 4.2 Model registry — `agents/models.py`

- `ModelEntry` dataclass: `id`, `display_name`, `provider`, `tier`, `family`, `auth_mode`, `est_relative_cost`, `est_latency_ms_per_call`, `available_regions`.
- **Short-alias resolution:** `resolve_model_id("llama-3.3")` → `"llama-3.3-70b-instruct-maas"`. Aliases are defined in `_ALIASES` and resolved transparently by `get_model()`, `default_model_id()`, and `current_model()`.
- `available_models()` filters by configured auth so unconfigured providers never appear in the UI or CLI.

| Auth mode | Env vars required |
|---|---|
| `vertex_api_key` | `VERTEX_AI_API_KEY` |
| `vertex_oauth` | `GOOGLE_CLOUD_PROJECT` + google-auth installed |

### 4.3 Provider backends — `agents/providers/`

**`gemini.py`** — Gemini via `google-genai` Vertex API-key mode.
- `@functools.lru_cache(maxsize=1)` singleton client, 120 s timeout.
- `json_mode=True` → `response_mime_type="application/json"`.

**`vertex_requests.py`** — Llama 3.3 and Mistral Small 3.1 via Vertex AI (two different endpoint patterns).
- **Llama** → OpenAI-compatible MaaS endpoint: `.../endpoints/openapi/chat/completions`; body `"model": "meta/llama-3.3-70b-instruct-maas"`.
- **Mistral** → rawPredict endpoint: `.../publishers/mistralai/models/mistral-small-2503:rawPredict`; body `"model": "mistral-small-2503"` (publisher encoded in URL).
- Uses `requests` (already in `requirements.txt`) + `google-auth` (transitive dep). No new package.
- Auth: ADC bearer token refreshed before each request via `google.auth.transport.requests.Request()`.
- `response_format=json_object` sent only for Llama; Mistral doesn't support it — plain-text JSON is parsed by `_extract_json` in `llm.py`.
- On 4xx/5xx, re-raises `RuntimeError(status_code=N)` so `agents/llm.py` retry logic handles 429 backoff.

**`__init__.py`** — `call_provider()` routes by `model.provider`:
```
"gemini"            → gemini.generate()
"meta" | "mistral"  → vertex_requests.generate()
```

### 4.4 Provider and model reference

| Model | Provider | Short alias | Auth mode | Default region |
|---|---|---|---|---|
| `gemini-3.1-flash-lite-preview` | gemini | `gemini-flash-lite` | `vertex_api_key` | — |
| `gemini-2.5-flash-lite` | gemini | — | `vertex_api_key` | — |
| `llama-3.3-70b-instruct-maas` | meta | `llama-3.3` | `vertex_oauth` | `us-central1` |
| `mistral-small-2503` | mistral | `mistral-small-3.1` | `vertex_oauth` | `us-central1` |

GPT/OpenAI is excluded — Vertex AI does not host OpenAI models.

---

## 5. Data contracts — `agents/schemas.py`

All inter-agent communication uses Pydantic models with `model_validator(mode="before")` normalizers that absorb common LLM shape drift. Preserve these normalizers when editing schemas.

**`Intent`** has a `model_validator(mode="after")` (`_derive_days_from_dates`) that computes `days` from the start and end dates when the LLM returns a date range but no explicit duration — e.g. `dates: ["2026-06-01", "2026-06-05"]` → `days = 5`. The validator only fires when `days == 1` (the default), leaving explicitly stated durations untouched.

| Schema | Producer | Consumer |
|---|---|---|
| `Intent` | `parse_intent` | All specialists, verifier, evaluator |
| `ToolContext` | Research node (live or sandbox) | All specialists |
| `TransportPlan`, `LodgingPlan`, `DiningPlan`, `SightseeingPlan` | Each specialist | Assembler |
| `FullPlan` (list of `PlanDay`) | Assembler | Verifier, app, eval |
| `BudgetReport` | Budget Agent | Verifier, repair router |
| `VerifierReport` (list of `Violation`) | Verifier | App, eval, repair router |

**`ToolContext`** is a `TypedDict(total=False)`. Key fields populated by research:

| Field | Live source | Sandbox source |
|---|---|---|
| `flights_outbound`, `flights_return` | SerpAPI | `sandbox.flight_search()` |
| `trip_windows` | Derived from flights in research node | Same |
| `hotels` | `search_lodging_near()` geo-radius | `sandbox.accommodation_search()` |
| `restaurants`, `attractions` | Google Places | `sandbox.restaurant_search()`, `.attraction_search()` |
| `dest_cities` | — | Expanded from state name via `state_city_index.json` + ref_info |
| `intercity_routes` | — | `sandbox.distance_matrix()` pairs between consecutive cities |
| `category_caps`, `meal_cost_targets`, `transport_one_way_target` | Budget agent inputs | Same |

`PlanDay` field strings follow `Dataset/example_submission.jsonl` style. Both verifier and evaluator parse `Cost: $N` and `Cost: $N for D nights` fragments — use `rules.format_cost()` to generate them.

---

## 6. Constraint model

### 6.1 Commonsense constraints (8)

| Key | Rule |
|---|---|
| `complete_info` | Every day has at least one meal and one attraction; `current_city` set |
| `diverse_attractions` | No attraction repeated across days |
| `diverse_restaurants` | No restaurant repeated across any meal slot |
| `within_sandbox` | All venues appear in sandbox reference data |
| `within_current_city` | Each day's venues belong to that day's `current_city` (state → city expansion applied) |
| `reasonable_city_route` | Every inter-city transition is feasible (flight or driving data exists) |
| `min_nights` | Consecutive hotel stay ≥ hotel's `minimum nights` |
| `non_conflicting_transport` | No flight + self-driving mixed on the same leg or trip |

### 6.2 Hard constraints (5)

Budget · Room Rule · Room Type · Cuisine · Transportation mode

### 6.3 Verifier vs. evaluator

- **`agents/verifier.py`** — in-loop. Tags each `Violation` with the responsible specialist for targeted repair routing.
- **`eval/constraints.py`** — scoring only. Produces `ConstraintReport` with per-rule pass booleans and aggregates micro/macro pass rates.

Both share rule-name constants from `agents/rules.py` but are not merged.

---

## 7. Verification and repair loop

1. Verifier runs deterministic checks on the assembled `FullPlan`.
2. Each `Violation` is tagged with the responsible specialist.
3. **Repair router** (`repair_prepare_node`), two stages:
   - **Stage 1 — Deterministic pre-repair (no LLM):** diversity and sandbox violations are fixed by substituting unused venues from `ToolContext`. Runs only when all violations for a specialist are mechanical. Saves ~30–40% LLM calls on failing queries.
   - **Stage 2 — LLM repair:** remaining violations are routed to their responsible specialists with targeted repair notes. Only flagged specialists rerun.
4. After re-assembly, verifier runs again. Loop cap: `REPAIR_MAX_ROUNDS = 3`.
5. **Escalation guard:** same `(rule, responsible)` pair in two consecutive rounds → repair note prefixed with `[URGENT]`.
6. At the cap, the plan is returned with the final `VerifierReport`.

---

## 8. Sandbox eval path — multi-city query handling

5-day and 7-day TravelPlanner validation queries use US state names as `dest` (e.g. `"North Carolina"`). `_sandbox_tool_context()` in `eval/run_eval.py` resolves these to the actual visited cities via:

1. **Per-row reference city sequence** parsed from `reference_information` column (flight leg descriptions like `"Flight from Org to City1 on date"`).
2. **`eval/state_city_index.json`** fallback — static state → city list built from the training corpus.
3. **`sandbox.cities_in(state)`** — last-resort expansion.

Once cities are resolved, attractions/restaurants/hotels are aggregated across all destination cities (each record tagged with its source city). Inter-city driving/taxi routes are populated into `ctx["intercity_routes"]` for the transport specialist.

---

## 9. Performance instrumentation — `agents/perf.py`

`Timeline` object in `CoordinatorState`:
- One entry per pipeline phase: `parse_intent`, `research`, `transport`, `lodging`, `dining`, `sightseeing`, `assemble`, `budget`, `verify`, `repair_round_N`.
- Aggregated per query in `eval/results.csv`: `latency_seconds`, `llm_calls`, `input_tokens`, `output_tokens`, `cache_hits`, `cache_misses`.

---

## 10. Baselines — `baseline/`

Paper §4.7 architectural ablations:

| Baseline | What it removes | Purpose |
|---|---|---|
| `single_agent.py` | Coordinator + specialists + verifier-repair | Paper Baseline 1 |
| `no_verify.py` | Verifier + repair loop (`skip_verify=True`) | Paper Baseline 2 |
| `no_specialization.py` | Role specialists (one generalist Worker) | Paper Baseline 3 |

In the **app path**, `single_agent.py` calls `build_live_tool_context()` (from `coordinator.py`) to populate a `ToolContext` from live APIs before invoking `plan_trip_single()` — the same live data as the multi-agent path. In the **eval path** it falls back to the TravelPlanner sandbox.

---

## 11. Design invariants

1. **Strict live runtime.** No silent fallbacks. Missing keys or API failures are errors.
2. **`ToolContext` schema is the contract** between live and sandbox paths. Validated by `ToolContextSchema.model_validate(ctx)`.
3. **Pydantic `model_validator(mode="before")` normalizers** absorb LLM shape drift — preserve them when editing schemas.
4. **Day-string format** (`Cost: $N`, `Cost: $N for D nights`, `"-"` for missing) is parsed by verifier and evaluator. Use `rules.format_cost()` to generate it.
5. **Verifier tags every violation with a responsible specialist.** Required by the repair router.
6. **Repair is bounded** — `REPAIR_MAX_ROUNDS = 3`.
7. **Cache key includes provider + model_id + schema name.** Required for cross-provider cache isolation.
8. **One model per pipeline run** — no per-agent model routing.
9. **GPT/OpenAI intentionally excluded** — Vertex AI does not host OpenAI models.
10. **Eval path never calls live APIs.** SerpAPI and Google Maps are bypassed; `tool_context` is pre-built from sandbox.
11. **API key strings never appear in error messages or logs.**
12. **All API calls have a 120 s timeout.** Gemini: `http_options={"timeout": 120_000}` at client construction. Llama/Mistral: `timeout=120` on `requests.post`.
13. **Parallel-write state keys use `Annotated[T, reducer]`** (e.g., `timeline`).
14. **Short model aliases are resolved to canonical IDs before use.** `resolve_model_id()` is called in `current_model()`, `get_model()`, and `default_model_id()` so cache keys and provider dispatch always see the full ID.

---

## 12. Layout

```
app.py                      Streamlit chat UI

agents/
  coordinator.py            LangGraph state machine; parallel specialist fan-out; mechanical pre-repair + LLM repair
  transport.py              Transport specialist (flights, routes, inter-city legs)
  lodging.py                Lodging specialist
  dining.py                 Dining specialist
  sightseeing.py            Sightseeing specialist
  budget.py                 Budget Agent (post-hoc deterministic)
  verifier.py               In-loop verifier (8 CS + 5 HC rules + responsibility tagging)
  rules.py                  Rule constants + format_cost() + format_trip_windows()
  schemas.py                Pydantic contracts (Intent, FullPlan, ToolContext, …)
  llm.py                    LLM facade (two-tier cache, retry loop, provider dispatch)
  models.py                 Model registry (_ALIASES, resolve_model_id, available_models)
  runtime.py                ContextVar for active model; alias resolution in current_model()
  providers/
    __init__.py             call_provider() router + clear_caches()
    gemini.py               Gemini via google-genai (Vertex API-key, lru_cache, 120 s)
    vertex_requests.py      Llama + Mistral via Vertex MaaS + requests + google-auth

baseline/
  single_agent.py           Single-LLM-call baseline (paper §4.7 Baseline 1)
  no_specialization.py      Coordinator + one generalist Worker (Baseline 3)
  no_verify.py              Full pipeline minus verify-and-repair (Baseline 2)

eval/
  run_eval.py               Evaluation CLI (alias resolution, multi-model sweep,
                            per-row city sequence parsing, sandbox ToolContext builder)
  constraints.py            TravelPlanner-style scoring (8 CS + 5 HC, micro/macro)
  format_report.py          Markdown tables + optional Pareto chart from results.csv
  state_city_index.json     Static state → cities fallback for multi-city queries
  results.csv               Sweep output (gitignored)

tools/
  live_apis.py              Google Maps + SerpAPI; airport → (IATA, city, leg, lat, lon); geo-radius hotel search; round-trip flight protocol
  sandbox.py                TravelPlanner reference-data sandbox + cities_in()

scripts/
  refresh_sandbox.py        2022 → 2026 sandbox regeneration (optional)
  build_state_index.py      Rebuild eval/state_city_index.json from training corpus

```
