# Architecture

Architectural reference for the Multi-Agent Travel Itinerary Planner. Describes the **current implementation** (what is in `agents/`, `tools/`, `eval/`, and `baseline/` today) and the **planned multi-provider LLM layer** that's the subject of `PLAN_MULTI_MODEL.md`.

Companion documents:
- `PLAN_MULTI_MODEL.md` — engineering execution plan for multi-model + latency + per-model TravelPlanner comparison.
- `../PROJECT_REPORT_PLAN.md` — paper-aligned analysis (Paper 1: TravelPlanner) and report structure.
- `../CODE_PLAN_MULTI_MODEL.md` — full how/why with provider-auth deep dive.

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
                              │   - model selection (planned)│
                              │   - perf panel (planned)    │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌────────────────────────────┐
                              │    Coordinator              │
                              │    (agents/coordinator.py)  │
                              │  parse_intent → constraint  │
                              │  gate → past-date guard →   │
                              │  research → specialists →   │
                              │  assemble → verify → repair │
                              └──────────────┬──────────────┘
                                             │ ToolContext, Intent
                              ┌──────────────┴──────────────┐
                              │      Research node          │
                              │  Live: tools/live_apis.py   │
                              │  Eval: tools/sandbox.py     │
                              └──────────────┬──────────────┘
                                             │ shared ToolContext
                ┌────────────┬───────────┬───┴───────┬────────────────┐
                ▼            ▼           ▼           ▼                ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐
         │Transport │ │ Lodging  │ │ Dining   │ │Sightseeing │ │  Budget  │
         │transport.│ │lodging.py│ │dining.py │ │sightseeing │ │budget.py │
         │py        │ │          │ │          │ │.py         │ │(post-hoc)│
         └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬──────┘ └────┬─────┘
               │            │            │            │             │
               └────────────┴───────┬────┴────────────┘             │
                                    ▼                               │
                          ┌────────────────────┐                    │
                          │ Assembler           │                   │
                          │ (in coordinator.py) │                   │
                          │ → FullPlan          │                   │
                          └─────────┬──────────┘                    │
                                    ▼                               │
                          ┌────────────────────┐ ◄──────────────────┘
                          │ Verifier            │   8 commonsense + 5 hard
                          │ (verifier.py +      │   rules; tags each violation
                          │  rules.py)          │   with responsible specialist
                          └─────────┬──────────┘
                                    │
                       passed?  no  │   yes → return (FullPlan, VerifierReport)
                                    ▼
                          ┌────────────────────┐
                          │ Repair router       │   ≤ REPAIR_MAX_ROUNDS (=3);
                          │                     │   escalate on repeated
                          │                     │   (rule, responsible) pairs
                          └────────────────────┘

           LLM facade (current) — single Gemini wrapper:    agents/llm.py
           LLM facade (planned) — provider-aware:           agents/llm.py + agents/providers/{gemini,claude}.py
```

---

## 3. Runtime flow

```mermaid
flowchart TD
    U[User query] --> APP[Streamlit app.py]
    APP --> PARSE[parse_intent (LLM)]
    PARSE --> GATE{Required fields present?}
    GATE -->|no| ASK[Ask user for missing fields]
    ASK --> APP
    GATE -->|yes| DATE{Dates in the past?}
    DATE -->|yes| FAIL[Surface clear error]
    DATE -->|no| RES[Research node: build ToolContext]
    RES --> DISPATCH[Dispatch specialists]

    DISPATCH --> T[Transport]
    DISPATCH --> L[Lodging]
    DISPATCH --> D[Dining]
    DISPATCH --> S[Sightseeing]

    T --> ASM[Assemble FullPlan]
    L --> ASM
    D --> ASM
    S --> ASM

    ASM --> BUD[Budget Agent: post-hoc total + per-category]
    BUD --> VER[Verifier: 8 CS + 5 HC rules]
    VER -->|passed| OUT[Render plan]
    VER -->|failed and round < cap| REP[Targeted specialist rerun]
    REP --> ASM
    VER -->|failed at cap or repeat| OUT_FLAG[Render with violation report]
```

The coordinator is implemented as a LangGraph `StateGraph` (`CoordinatorState` is a `TypedDict`). Specialists run **sequential** or **parallel** depending on a UI/CLI toggle; both modes converge at the assemble node.

---

## 4. LLM provider layer

### 4.1 Current (single-provider, single model)
- One file: `agents/llm.py`.
- One client: `google.genai` configured via `VERTEX_AI_API_KEY` (Vertex API-key auth).
- One model at a time, selected by the `LLM_MODEL` env var (default `gemini-2.5-flash-lite`).
- Two-tier cache: in-memory LRU (size `LLM_CACHE_SIZE`, default 256) + optional disk cache (`LLM_CACHE_DIR`).
- Thread-safe stats counters (`reset_call_stats` / `get_call_stats`) used by `eval/run_eval.py` for paper §5 efficiency reporting.
- Pydantic schema-bound generation (no free-text-then-parse).

### 4.2 Planned (multi-provider via facade + backends — see `PLAN_MULTI_MODEL.md`)
- `agents/llm.py` becomes a thin facade with a per-call `model=` argument.
- `agents/models.py` — registry: `id`, `display_name`, `provider`, `tier`, `family`, `default_temperature`, `est_relative_cost`, `auth_mode`. Filtered to `available_models()` based on configured auth.
- `agents/providers/gemini.py` — existing logic, lifted out.
- `agents/providers/claude.py` — `anthropic[vertex]` `AnthropicVertex` client; structured output via **forced tool-use** (declare a single tool with the Pydantic JSON schema, force the model to call it, parse the tool input).
- `agents/runtime.py` — `ContextVar` holding the current model + a `use_model()` context manager so internal helpers don't have to forward `model=` through every call.
- Cache key changes from `sha256(system + user)` to `sha256(provider + model + system + user + schema_name)` — load-bearing for cross-provider correctness.
- Stats become `{model_id: {llm_calls, input_tokens, output_tokens, latency_ms_total, cache_hits, cache_misses}}`.

### 4.3 Provider scope on Vertex AI
| Provider | Hosted on Vertex? | SDK | Auth |
|---|---|---|---|
| Gemini | yes (native) | `google-genai` | `VERTEX_AI_API_KEY` |
| Claude | yes (Anthropic partnership / Model Garden) | `anthropic[vertex]` (`AnthropicVertex`) | ADC + `GOOGLE_CLOUD_PROJECT` + region |
| GPT/OpenAI | **no** | n/a | n/a |

GPT is excluded because Vertex AI does not host OpenAI models. Accessing GPT would require OpenAI direct or Azure OpenAI Service — out of scope for this project.

---

## 5. Data contracts (`agents/schemas.py`)

All inter-agent communication goes through Pydantic models with `model_validator(mode="before")` normalizers that absorb common LLM shape drift (e.g. `{day, description}` instead of `{days: [...]}`). The normalizers are invariants of the design — **must be preserved** when editing schemas.

| Schema | Producer | Consumer |
|---|---|---|
| `Intent` | `parse_intent` | All specialists, verifier, evaluator |
| `ToolContext` | Research node (live or sandbox) | All specialists |
| `TransportPlan`, `LodgingPlan`, `DiningPlan`, `SightseeingPlan` | Each specialist | Assembler |
| `FullPlan` (list of `PlanDay`) | Assembler | Verifier, app/eval |
| `BudgetReport` | Budget Agent | Verifier, repair router |
| `VerifierReport` (list of `Violation`) | Verifier | App/eval, repair router |
| `AttributionLog` (list of `AgentDecision`) | Each specialist | Repair router (routes by `responsible` field) |

`PlanDay` field strings follow the `Dataset/example_submission.jsonl` style. The verifier and the evaluator both parse `Cost: $N` and `Cost: $N for D nights` fragments out of these strings — keep this format when generating or editing them.

---

## 6. Constraint model

Mirrors Paper 1's three constraint families exactly so the verifier and evaluator are paper-comparable.

### 6.1 Commonsense constraints (8) — `agents/rules.py` + `agents/verifier.py`
Within Sandbox · Complete Information · Within Current City · Reasonable City Route · Diverse Restaurants · Diverse Attractions · Non-conflicting Transportation · Minimum Nights Stay

### 6.2 Hard constraints (5) — extracted from `Intent`
Budget · Room Rule · Room Type · Cuisine · Transportation

### 6.3 Environment constraints (2) — emitted by sandbox/live tools
Unavailable Transportation · Unavailable Attractions

### 6.4 Verifier vs evaluator (different jobs, shared rule names)
- **`agents/verifier.py`** — *in-loop* verifier. Tags each `Violation` with the responsible specialist so the repair router can target the right agent. Used by the live runtime.
- **`eval/constraints.py`** — *scoring* evaluator. Produces `ConstraintReport` with per-rule pass booleans and computes micro/macro pass rates. Used only by `run_eval.py`.

They share rule names from `agents/rules.py` but are **not merged**. The verifier needs attribution; the evaluator needs scorable booleans.

---

## 7. Verification + repair loop

Failure-mode driven design (see `../PROJECT_REPORT_PLAN.md` §A.4 for paper findings):

1. Verifier runs deterministic checks against the assembled `FullPlan`.
2. Each `Violation` is tagged with the responsible specialist (`transport` | `lodging` | `dining` | `sightseeing` | `budget` | `coordinator`).
3. Repair router groups violations by responsible specialist, builds a per-specialist repair note, and reruns only those specialists.
4. After re-assembly, verifier runs again. Loop continues for at most `REPAIR_MAX_ROUNDS = 3` rounds.
5. **Escalation guard**: if the same `(rule, responsible)` pair appears in two consecutive rounds, the loop exits early (prevents dead loops on unfixable violations).
6. After the cap, the plan is returned with the final `VerifierReport` so the user sees the unmet constraints.

---

## 8. Performance instrumentation (planned — `agents/perf.py`)

`Timeline` object passed through `CoordinatorState`:
- One entry per phase: parse_intent, research, transport, lodging, dining, sightseeing, assemble, budget, verify, repair_round_N.
- Each entry: `name`, `start_perf_counter`, `end_perf_counter`, optional `cache_hits` / `cache_misses` for LLM-bound phases.
- Surfaced in the Streamlit "Performance" expander after a plan renders.
- Aggregated into `eval/results.csv` per query: `latency_seconds`, `cache_hits`, `cache_misses`, plus per-model token counts already collected.

Instrumentation lands in Phase 2 of `PLAN_MULTI_MODEL.md`. No optimization is committed without a perf-benchmark run that beats baseline.

---

## 9. Two execution paths

### 9.1 Live API path (`app.py`)
- `tools/live_apis.py` builds `ToolContext` from Google Maps + SerpAPI responses.
- Strict runtime — no silent fallbacks. Missing keys or API failures surface as errors *at the source*. (`live_apis.py` raises `RuntimeError`; **callers must propagate, not swallow**.)
- Streamlit chat with a clarification flow (`coordinator.intent_constraint_violations`) that asks the user for missing fields before any specialists run.
- **SerpAPI is invoked only here.** The eval path (§9.2) builds flights from the sandbox and never calls SerpAPI — this is intentional because the dataset uses 2022 historical dates that no live API will return. If SerpAPI appears "not executing" in eval mode, that is by design, not a bug.
- **Per-API status (Phase 7 — planned):** the Performance expander surfaces a three-state status per API call (`ok` / `skipped (no key)` / `error: <msg>`). Today, bare `try / except Exception: pass` blocks in `coordinator.research_node` (lines 468–481) and `baseline/no_specialization.py` (lines 51–62) silently turn every failure into an empty list. Phase 7 (`PLAN_MULTI_MODEL.md`) replaces those with a `_call_live` helper that distinguishes the three states and propagates them to UI and tests. `LIVE_API_STRICT=1` env flag flips the helper into raise-on-error mode for the preflight script and unit tests.

### 9.2 Sandbox eval path (`eval/run_eval.py`)
- `_sandbox_tool_context()` builds a `ToolContext` with the **same dict shape** that `live_apis.py` produces, populated from `tools/sandbox.py` records (TravelPlanner reference data).
- `plan_trip(..., tool_context=ctx, bypass_date_check=True)` lets historical (2022) queries run.
- This is why the same coordinator and specialists handle both paths — the `ToolContext` schema in `agents/schemas.py` is the contract.
- **SerpAPI / Google Maps are NOT called in eval mode.** Flights, hotels, restaurants, attractions, and routes all come from `tools/sandbox.py`. The `state["live_api_status"]` (Phase 7) field is therefore empty for eval rows; the CSV adds no live-API columns.

---

## 10. Baselines (`baseline/`) — paper §4.7 ablations

| Baseline | What it removes | Purpose | Live-app exposed? |
|---|---|---|---|
| `single_agent.py` | Coordinator + specialists + verifier-repair | Paper-style single-prompt floor: one LLM call sees all tools and emits the full plan | Planned in Phase 5 (sidebar `Mode` toggle) |
| `no_specialization.py` | Role specialists (one generalist Worker instead) | Isolates the specialization contribution | Eval only |
| `no_verify.py` | Verifier + repair loop | Isolates the verify-and-repair contribution | Eval only |

The full multi-agent system minus each of these gives the architectural ablation table. See `PLAN_MULTI_MODEL.md` §6 for the per-model sweep matrix.

**Single-agent vs no-specialization (clarification):** Both run a single LLM call that produces the entire plan. The differences are:
- `single_agent.py` is sandbox-only today (Phase 5 lifts this restriction); has no Budget Agent and no verify-and-repair (verifier runs once for scoring only).
- `no_specialization.py` runs against either live or sandbox tool_context; includes the Budget Agent; runs the verifier once at the end (still no repair, to keep it a fair "no orchestration" comparison).

Both prompts are subjects of the Phase 6 audit (`PROMPT_AUDIT.md`), since they cover the union of all specialist rules in one shot and are the most likely to underperform on long-tail constraints.

---

## 11. Design invariants

These are non-negotiable; violating them breaks something subtle.

1. **Strict live runtime.** No deterministic / provider-switch fallback paths in the live runtime. Failures are errors, not silent stubs. *Caller-side enforcement (Phase 7):* coordinator/baseline call sites use a `_call_live` helper that distinguishes `ok` / `skipped (no key)` / `error` and surfaces each state — bare `try / except Exception: pass` around live API calls is forbidden.
2. **`ToolContext` schema is the contract** between live and sandbox paths. Specialists never branch on which path produced their context.
3. **Pydantic `model_validator(mode="before")` normalizers** absorb LLM shape drift; preserve them.
4. **Day-string format** (`Cost: $N`, `Cost: $N for D nights`, `"-"` for missing) is parsed by both verifier and evaluator. Don't change it without updating both.
5. **Verifier tags every violation with a responsible specialist.** Required for the repair router to work.
6. **Repair is bounded** — `REPAIR_MAX_ROUNDS = 3` and same-violation-twice escalation. No infinite loops.
7. **Cache key includes provider + model + schema name** (post Phase 1 of `PLAN_MULTI_MODEL.md`). Required for cross-provider correctness.
8. **One model per pipeline run** in v1 — no per-agent model routing. Keeps the eval matrix readable.
9. **GPT/OpenAI is intentionally excluded** because Vertex AI does not host it. This is documented and enforced by the registry.
10. **Eval path never calls live APIs.** SerpAPI and Google Maps are bypassed entirely in eval mode (`tool_context` is pre-built from sandbox). This is intentional: the dataset uses 2022 historical dates and no live API will return useful results for them. Anyone instrumenting "where does SerpAPI fire?" should know this up front.
11. **API key strings never appear in error messages or logs.** All `RuntimeError`s in `tools/live_apis.py` and `agents/providers/*` strip the key value. Enforced by `tests/unit/test_no_secret_logging.py` (Phase 7).
12. **User-facing UI never displays env-var names.** Sidebar, performance expander, error banners, and progress messages use feature-level labels (`Flights`, `Places & routes`, `LLM`) — never `SERPAPI_API_KEY`, `GOOGLE_MAPS_API_KEY`, etc. Configuration diagnostics live in `scripts/validate_env.py` only. Enforced by `tests/unit/test_no_env_leak.py` (Hotfix H2).
13. **Parallel-write state keys must use `Annotated[T, reducer]`.** Any `CoordinatorState` field that more than one node writes in the same superstep needs an explicit reducer. The current parallel-fanout (`lodging`, `dining`, `sightseeing`) writes `timeline`, so `timeline: Annotated[Timeline, merge_timelines]`. Enforced by code review and `tests/integration/test_coordinator_e2e.py` running in parallel mode (Hotfix H1).

---

## 12. Layout

```
app.py                      Streamlit chat UI

agents/
  coordinator.py            LangGraph state machine (sequential/parallel) → verify → repair
  transport.py              Transport specialist
  lodging.py                Lodging specialist
  dining.py                 Dining specialist
  sightseeing.py            Sightseeing specialist
  budget.py                 Budget Agent (post-hoc deterministic)
  verifier.py               In-loop verifier (8 CS + 5 HC rules + responsibility tagging)
  rules.py                  Rule constants and helpers shared by verifier and evaluator
  schemas.py                Pydantic contracts (Intent, FullPlan, ToolContext, …)
  llm.py                    LLM facade (provider-aware after Phase 1 of PLAN_MULTI_MODEL.md)
  models.py                 (Phase 1) Model registry
  providers/                (Phase 1) gemini.py, claude.py
  runtime.py                (Phase 1) ContextVar for current model
  perf.py                   (Phase 2) Timeline instrumentation

baseline/
  single_agent.py           Single-LLM-call baseline
  no_specialization.py      Coordinator + one generalist Worker
  no_verify.py              Full pipeline minus verify-and-repair

eval/
  run_eval.py               Evaluation CLI (per-model after Phase 3)
  constraints.py            TravelPlanner-style scoring evaluator
  format_report.py          (Phase 3) Markdown table generator for the report
  perf_queries.txt          (Phase 2) Frozen 10-query latency benchmark set
  perf_baseline.csv         (Phase 2) Recorded baseline timings
  results.csv               Sweep output (gitignored)

tools/
  live_apis.py              Google Maps + SerpAPI clients
  sandbox.py                TravelPlanner reference-data sandbox

scripts/
  refresh_sandbox.py        2022 → 2026 schema-preserving regeneration
  smoke_models.py           (Phase 1) Smoke test all registered models
  validate_env.py           (Phase 7 — planned) Preflight that pings each
                            configured live API once and reports actionable
                            failures before the user runs app.py.

tests/                      (Phase 4 — planned)
  conftest.py               Shared fixtures (sandbox, intent, tool_context, fake_backend)
  unit/                     Pure-function tests (no LLM): budget, constraints,
                            verifier, assemble, timing, models, runtime, llm_cache, perf,
                            live_apis_serpapi, live_apis_maps, env_validation,
                            no_secret_logging, validate_env
  integration/              Coordinator e2e with mocked LLM, repair routing,
                            multi-provider parity (stubbed AnthropicVertex),
                            research_node_status (Phase 7)
  fixtures/                 Canned LLM responses + optional real-response cache

PROMPT_AUDIT.md             (Phase 6 — planned) Per-prompt rule-coverage matrix
                            and observed-failure analysis for the 4 specialists
                            and 2 single-prompt baselines.

LIVE_API_AUDIT.md           (Phase 7 — planned) Per-call-site decision matrix
                            for every live.* invocation in coordinator/baseline.
                            Documents the swallow → raise/skip behavior change.
```
