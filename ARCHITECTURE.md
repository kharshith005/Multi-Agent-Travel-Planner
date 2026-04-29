# Architecture

Architectural reference for the Multi-Agent Travel Itinerary Planner (current state post-Phases 1–13).

Companion documents:
- `../PROJECT_REPORT_PLAN.md` — paper-aligned analysis (Paper 1: TravelPlanner) and report structure.

---

## 1. System overview

A coordinator-led multi-agent system that converts a natural-language travel query into a day-by-day itinerary that satisfies user-stated hard constraints and TravelPlanner-style commonsense constraints. Built on **LangGraph** for the coordinator's state machine, **Pydantic** for inter-agent contracts, and **Streamlit** for the user-facing chat interface.

Two execution paths share the same coordinator and specialist code:

1. **Live API path** (`app.py`) — research node hits Google Maps + SerpAPI; specialists plan over real-world data.
2. **Sandbox eval path** (`eval/run_eval.py`) — research node is replaced by a `ToolContext` built from the TravelPlanner reference data; specialists are unaware of the substitution because the dict shape is identical.

This dual-path design is **load-bearing**: the same multi-agent code is what ships in the app *and* what produces the report's benchmark numbers.

---

## 2. Component map

```
                          ┌────────────────────────────┐
              user query →│   Streamlit UI (app.py)     │
                          │   - clarification flow      │
                          │   - model selector          │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────┐
                          │    Coordinator              │
                          │    (agents/coordinator.py)  │
                          │  parse_intent → constraint  │
                          │  gate → past-date guard →   │
                          │  research → 4-way parallel  │
                          │  specialists → assemble →   │
                          │  budget → verify →          │
                          │  [pre-repair + LLM repair]  │
                          └──────────────┬──────────────┘
                                         │ ToolContext (+ trip_windows), Intent
                          ┌──────────────┴──────────────┐
                          │      Research node          │
                          │  Live: tools/live_apis.py   │
                          │  Eval: tools/sandbox.py     │
                          │  Derives trip_windows here  │
                          └──────────────┬──────────────┘
                                         │ shared ToolContext (trip_windows pre-populated)
            ┌────────────┬───────────┬───┴───────┬────────────┐
            ▼            ▼           ▼           ▼            │
     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐   │
     │Transport │ │ Lodging  │ │ Dining   │ │Sightseeing │   │
     │transport.│ │lodging.py│ │dining.py │ │sightseeing │   │
     │py        │ │          │ │          │ │.py         │   │
     └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬──────┘   │
           │            │            │            │           │
           └────────────┴───────┬────┴────────────┘           │
                                ▼                             │
                      ┌────────────────────┐                  │
                      │ Assembler           │                  │
                      │ (coordinator.py)    │                  │
                      │ → FullPlan          │                  │
                      └─────────┬──────────┘                  │
                                ▼                             ▼
                      ┌────────────────────┐ ◄────────────────┘
                      │ Budget Agent        │   post-hoc deterministic total
                      └─────────┬──────────┘
                                ▼
                      ┌────────────────────┐
                      │ Verifier            │   8 CS + 5 HC rules; tags each
                      │ (verifier.py +      │   violation with responsible
                      │  rules.py)          │   specialist
                      └─────────┬──────────┘
                                │
                   passed?  no  │   yes → return (FullPlan, VerifierReport)
                                ▼
                      ┌────────────────────┐
                      │ Repair router       │   1. Mechanical pre-repair (no LLM):
                      │ (repair_prepare_    │      diversity, sandbox violations
                      │  node)              │   2. LLM re-run for remaining
                      │                     │   ≤ REPAIR_MAX_ROUNDS (=3)
                      └────────────────────┘

       LLM facade (multi-provider):   agents/llm.py
       Provider backends:             agents/providers/{gemini,claude}.py
       Model registry:                agents/models.py
```

---

## 3. Runtime flow

```
parse_intent (LLM)
  → constraint gate (required fields)
  → past-date guard
  → research node
      Live: Maps + SerpAPI + airport resolution (5-tuple with lat/lon)
            + geo-radius lodging search near destination
            + trip_windows pre-populated from flights_outbound[0]/flights_return[0]
      Eval: _sandbox_tool_context() + derive_trip_windows()
  → 4-way parallel fan-out (transport, lodging, dining, sightseeing all start simultaneously)
      All four receive trip_windows in ToolContext — no timing context dependency
  → assemble FullPlan
  → Budget Agent (deterministic rule-based, no LLM)
  → Verifier (8 CS + 5 HC rules)
      passed → return
      failed and round < REPAIR_MAX_ROUNDS:
          repair_prepare_node:
              1. Deterministic pre-repair: diversity and sandbox violations are fixed
                 directly from ToolContext data (no LLM call)
              2. LLM repair: route remaining violations to responsible specialists
                 (targeted — only the failing specialist reruns)
          → specialists rerun (only those in rerun_specialists list)
          → assemble → budget → verify → [loop]
      failed at cap → return plan with violation report
```

Key implementation note: `rerun_specialists = None` means "run all" (first pass); `rerun_specialists = []` means "skip all LLM reruns" (all violations resolved by mechanical pre-repair); `rerun_specialists = ["dining"]` means run only that specialist.

---

## 4. LLM provider layer

### 4.1 Current (multi-provider)

**`agents/llm.py`** — thin facade. Public API: `call_json`, `call_text`, `reset_call_stats`, `get_call_stats`.
- Active model is set via `agents/runtime.py`'s `ContextVar` (`use_model()` context manager).
- Two-tier cache: in-memory LRU (`LLM_CACHE_SIZE`, default 256) + optional disk cache (`LLM_CACHE_DIR`, content-addressed `.json` files).
- Retry loop: `_MAX_ATTEMPTS = 8`, exponential backoff, retryable on 429/5xx/known hints.
- Cache key: `sha256(provider | model_id | schema_name | max_tokens | think_first | system | user)`.
- Thread-safe stats: `{llm_calls, input_tokens, output_tokens, cache_hits, cache_misses}`.

**`agents/models.py`** — model registry.
- `ModelEntry` dataclass: `id` (versioned, e.g. `claude-haiku-4-5@20251001`), `display_name`, `provider`, `tier`, `auth_mode`, `est_relative_cost`, `est_latency_ms_per_call`.
- `_REGISTRY_BY_ID` keys by bare name (`split("@", 1)[0]`) so callers use `get_model("claude-haiku-4-5")`.
- `available_models()` filters by configured auth (Gemini: `VERTEX_AI_API_KEY`; Claude: `GOOGLE_CLOUD_PROJECT` + `CLAUDE_VERTEX_REGION`).
- `default_model_id()` reads `LLM_MODEL` env or falls back to `gemini-2.5-flash-lite`.

**`agents/providers/gemini.py`** — Gemini via `google-genai` Vertex API-key mode.
- `@functools.lru_cache(maxsize=1)` singleton client (`clear_client_cache()` for test teardown).
- Client constructed with `http_options={"timeout": 120}` — 120s uniform timeout.
- `json_mode=True` → `response_mime_type="application/json"`.

**`agents/providers/claude.py`** — Claude via `anthropic[vertex]` (AnthropicVertex).
- `@functools.lru_cache(maxsize=1)` singleton client (`clear_client_cache()` for test teardown).
- Client: `region=CLAUDE_VERTEX_REGION (default: "global")`, `project_id=GOOGLE_CLOUD_PROJECT`, `timeout=120`, **`max_retries=0`** (llm.py owns the retry loop).
- Structured output: forced tool-use. Pydantic schema declared as a single tool; model is forced to call it; response is always valid JSON.
- `CLAUDE_DEBUG=1` env prints outgoing model/region/project to stderr.
- `clear_caches()` in `agents/providers/__init__.py` clears both client singletons.

**`agents/runtime.py`** — `ContextVar` holding the current model ID. `use_model("gemini-2.5-flash")` context manager.

### 4.2 Provider scope on Vertex AI

| Provider | Hosted on Vertex? | SDK | Auth |
|---|---|---|---|
| Gemini | yes (native) | `google-genai` | `VERTEX_AI_API_KEY` |
| Claude | yes (Model Garden) | `anthropic[vertex]` (`AnthropicVertex`) | ADC + `GOOGLE_CLOUD_PROJECT` + `CLAUDE_VERTEX_REGION=global` |
| GPT/OpenAI | **no** | n/a | n/a |

GPT is excluded because Vertex AI does not host OpenAI models.

### 4.3 Claude diagnostics

`scripts/diag_claude.py` — standalone diagnostic matrix script. Runs 6 cells (`region × system`) plus two control cells that mirror Anthropic's working sample. Run `python scripts/diag_claude.py` to distinguish quota issues from wrapper bugs.

---

## 5. Data contracts (`agents/schemas.py`)

All inter-agent communication goes through Pydantic models with `model_validator(mode="before")` normalizers that absorb common LLM shape drift. **Preserve these normalizers** when editing schemas.

| Schema | Producer | Consumer |
|---|---|---|
| `Intent` | `parse_intent` | All specialists, verifier, evaluator |
| `ToolContext` | Research node (live or sandbox) | All specialists |
| `TransportPlan`, `LodgingPlan`, `DiningPlan`, `SightseeingPlan` | Each specialist | Assembler |
| `FullPlan` (list of `PlanDay`) | Assembler | Verifier, app/eval |
| `BudgetReport` | Budget Agent | Verifier, repair router |
| `VerifierReport` (list of `Violation`) | Verifier | App/eval, repair router |
| `AttributionLog` (list of `AgentDecision`) | Each specialist | Repair router |

**`ToolContext`** is a `TypedDict(total=False)`. Key fields populated by research:
- `flights_outbound`, `flights_return` — formatted flight strings with `HH:MM->HH:MM`.
- `trip_windows` — derived in the research node from `flights_outbound[0]`/`flights_return[0]`; available to all four specialists before they run.
- `hotels` — from geo-radius `places_nearby` (live) or sandbox; filtered by budget tier.
- `category_caps`, `meal_cost_targets`, `lodging_cost_targets`, `transport_one_way_target`.

**`ToolContextSchema(BaseModel)`** mirrors `ToolContext` for contract validation:
```python
ToolContextSchema.model_validate(ctx)  # raises ValidationError if contract is violated
```

`PlanDay` field strings follow the `Dataset/example_submission.jsonl` style. The verifier and evaluator both parse `Cost: $N` and `Cost: $N for D nights` fragments — use `rules.format_cost(n, nights=n)` to generate these to prevent regex-breaking typos.

---

## 6. Constraint model

Mirrors Paper 1's three constraint families exactly.

### 6.1 Commonsense constraints (8) — `agents/rules.py` + `agents/verifier.py` + `eval/constraints.py`

| Key | Rule |
|---|---|
| `complete_info` | Every non-travel day has at least one meal and one attraction |
| `diverse_attractions` | No attraction repeated across days |
| `diverse_restaurants` | No restaurant repeated across any meal slot |
| `within_sandbox` | All venues must appear in the sandbox data |
| `within_current_city` | Each day's venues belong to that day's `current_city` |
| `reasonable_city_route` | Consecutive city transitions must exist in distance_matrix |
| `min_nights` | Consecutive hotel stay ≥ hotel's minimum_nights requirement |
| `non_conflicting_transport` | No flight + self-driving on the same inter-city leg |

### 6.2 Hard constraints (5) — extracted from `Intent`

Budget · Room Rule · Room Type · Cuisine · Transportation

### 6.3 Verifier vs evaluator (different jobs, shared rule names)

- **`agents/verifier.py`** — *in-loop* verifier. Tags each `Violation` with the responsible specialist for repair routing.
- **`eval/constraints.py`** — *scoring* evaluator. Produces `ConstraintReport` with per-rule pass booleans and micro/macro pass rates.

They share rule name constants from `agents/rules.py` but are **not merged**.

---

## 7. Verification + repair loop

Failure-mode driven design:

1. Verifier runs deterministic checks against the assembled `FullPlan`.
2. Each `Violation` is tagged with the responsible specialist.
3. **Repair router** (`repair_prepare_node`) runs in two stages:
   - **Stage 1 — Deterministic pre-repair (no LLM):** diversity violations (`diverse_restaurants`, `diverse_attractions`) are fixed by substituting unused venues from `ToolContext` directly into the specialist plans. Only runs when ALL violations for that specialist are mechanical (no other rule violations). Saves ~30–40% LLM calls on failing queries.
   - **Stage 2 — LLM repair:** remaining violations are routed to their responsible specialists with targeted repair notes. Only failing specialists rerun.
4. After re-assembly, verifier runs again. Loop continues for at most `REPAIR_MAX_ROUNDS = 3` rounds.
5. **Escalation guard:** if the same `(rule, responsible)` pair appears in two consecutive rounds, the repair note is prefixed with `[URGENT]` and the loop escalates.
6. After the cap, the plan is returned with the final `VerifierReport`.

---

## 8. Performance instrumentation (`agents/perf.py`)

`Timeline` object in `CoordinatorState`:
- One entry per phase: parse_intent, research, transport, lodging, dining, sightseeing, assemble, budget, verify, repair_round_N.
- Aggregated into `eval/results.csv` per query: `latency_seconds`, `cache_hits`, `cache_misses`.

---

## 9. Two execution paths

### 9.1 Live API path (`app.py`)

- `tools/live_apis.py` builds `ToolContext` from Google Maps + SerpAPI.
- Airport resolution: `_resolve_with_fallback` returns `(iata, fallback_city, drive_leg, lat, lon)` (5-tuple). `lat/lon` are used for geo-radius hotel search.
- Hotel search: `search_lodging_near(latlng, radius_m=48000, budget_levels=...)` using `places_nearby`; falls back to text search if geo search returns nothing.
- SerpAPI round-trip uses the two-call `departure_token` protocol.
- Strict runtime: no silent fallbacks. Missing keys or API failures surface as errors.

### 9.2 Sandbox eval path (`eval/run_eval.py`)

- `_sandbox_tool_context()` builds a `ToolContext` with the **same dict shape** as the live path.
- `derive_trip_windows()` is called in `_sandbox_tool_context` to populate `trip_windows` from sandbox flight strings — mirrors the live research node.
- `plan_trip(..., tool_context=ctx, bypass_date_check=True)` handles historical (2022) queries.
- SerpAPI and Google Maps are NOT called in eval mode.

---

## 10. Baselines (`baseline/`) — paper §4.7 ablations

| Baseline | What it removes |
|---|---|
| `single_agent.py` | Coordinator + specialists + verifier-repair |
| `no_specialization.py` | Role specialists (one generalist Worker) |
| `no_verify.py` | Verifier + repair loop (`skip_verify=True`) |

---

## 11. Design invariants

1. **Strict live runtime.** No deterministic fallback paths in the live runtime. Failures are errors, not stubs.
2. **`ToolContext` schema is the contract** between live and sandbox paths. Validated by `ToolContextSchema.model_validate(ctx)`.
3. **Pydantic `model_validator(mode="before")` normalizers** absorb LLM shape drift; preserve them.
4. **Day-string format** (`Cost: $N`, `Cost: $N for D nights`, `"-"` for missing) is parsed by verifier and evaluator. Use `rules.format_cost(n, nights=n)` to generate it.
5. **Verifier tags every violation with a responsible specialist.** Required for the repair router.
6. **Repair is bounded** — `REPAIR_MAX_ROUNDS = 3`.
7. **Cache key includes provider + model + schema name.** Required for cross-provider correctness.
8. **One model per pipeline run** — no per-agent model routing.
9. **GPT/OpenAI is intentionally excluded** because Vertex AI does not host it.
10. **Eval path never calls live APIs.** SerpAPI and Google Maps are bypassed; `tool_context` is pre-built from sandbox.
11. **API key strings never appear in error messages or logs.**
12. **User-facing UI never displays env-var names.**
13. **Parallel-write state keys must use `Annotated[T, reducer]`** (e.g., `timeline`).
14. **All API calls have a 120s timeout.** Gemini: `http_options={"timeout": 120}` at client construction. Claude: `timeout=120` on `AnthropicVertex` + per-call `messages.create(..., timeout=120)`.
15. **Live runtime never sources sandbox data.** `_sandbox_fallback` was removed (Phase 6). An empty live result raises immediately so the user knows data is unavailable.

---

## 12. Layout

```
app.py                      Streamlit chat UI

agents/
  coordinator.py            LangGraph state machine; 4-way parallel fan-out;
                            mechanical pre-repair + LLM repair
  transport.py              Transport specialist
  lodging.py                Lodging specialist
  dining.py                 Dining specialist
  sightseeing.py            Sightseeing specialist
  budget.py                 Budget Agent (post-hoc deterministic)
                            price_levels_for_nightly_target() for lodging geo filter
  verifier.py               In-loop verifier (8 CS + 5 HC rules + responsibility tagging)
  rules.py                  Rule constants + format_cost() + format_trip_windows()
  schemas.py                Pydantic contracts (Intent, FullPlan, ToolContext,
                            ToolContextSchema for contract validation)
  llm.py                    LLM facade (provider-aware, two-tier cache, retry loop)
  models.py                 Model registry (versioned IDs, bare-name lookup)
  providers/
    __init__.py             call_provider() router + clear_caches()
    gemini.py               google-genai backend (lru_cache client, 120s timeout)
    claude.py               anthropic[vertex] backend (lru_cache, max_retries=0, 120s)
  runtime.py                ContextVar for current model

baseline/
  single_agent.py           Single-LLM-call baseline
  no_specialization.py      Coordinator + one generalist Worker
  no_verify.py              Full pipeline minus verify-and-repair

eval/
  run_eval.py               Evaluation CLI (model sweep, --probe-models)
  constraints.py            TravelPlanner-style scoring evaluator (8 CS + 5 HC)
  results.csv               Sweep output (gitignored)

tools/
  live_apis.py              Google Maps + SerpAPI clients;
                            _resolve_with_fallback → 5-tuple (iata, city, leg, lat, lon);
                            search_lodging_near() geo-radius hotel search
  sandbox.py                TravelPlanner reference-data sandbox

scripts/
  diag_claude.py            Claude Vertex diagnostic matrix (6 cells + 2 controls)
  refresh_sandbox.py        2022 → 2026 schema-preserving regeneration
  smoke_models.py           Smoke test all registered models
  validate_env.py           Preflight env checker

tests/unit/                 227 unit tests (no LLM calls)
```
