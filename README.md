# Multi-Agent Travel Itinerary Planner

A coordinator-led multi-agent system for travel-itinerary planning. A Streamlit chat UI drives a LangGraph coordinator + four domain specialists (transport, lodging, dining, sightseeing) + a Budget Agent + a deterministic constraint verifier with a bounded repair loop.

CSE 572 Spring 2026 final project — inspired by **Xie et al. (2024), TravelPlanner: A Benchmark for Real-World Planning with Language Agents** (arXiv:2402.01622). The paper shows GPT-4-Turbo + ReAct hits **0.6%** final pass rate on the TravelPlanner benchmark; the goal here is a multi-agent counter-design that closes that gap.

**Documents in this directory:**
- [ARCHITECTURE.md](ARCHITECTURE.md) — system architecture reference.

**Workspace-level documents** (one directory up):
- `../PROJECT_REPORT_PLAN.md` — paper-aligned analysis and report structure.
- `../CODE_PLAN_MULTI_MODEL.md` — multi-model and provider-auth background.
- `../CLAUDE.md` — repo conventions for Claude Code sessions.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env and fill in your keys
```

### Required environment

- `VERTEX_AI_API_KEY` — Gemini on Vertex AI (default model: `gemini-2.5-flash-lite`; override via `LLM_MODEL`).
- `GOOGLE_MAPS_API_KEY` — live sightseeing/lodging and route summaries.
- `SERPAPI_API_KEY` — live flight options (Google Flights via SerpAPI), with city names resolved to IATA codes.

`.env` is gitignored. Runtime is **strict** — missing keys or API failures surface as errors, not silent fallbacks.

### Optional: Claude on Vertex AI

Additional vars required for Claude models:

- `GOOGLE_CLOUD_PROJECT` — GCP project with Vertex AI billing enabled and Anthropic Model Garden terms accepted.
- `CLAUDE_VERTEX_REGION` — e.g. `global` (region varies by Claude SKU).
- `GOOGLE_APPLICATION_CREDENTIALS` — path to a service-account JSON for headless eval, OR run `gcloud auth application-default login` for interactive use.

When these are not set, Claude options are filtered out of the model picker and the system runs Gemini-only with no behavior change.

### Provider scope

| Provider | On Vertex AI | Status |
|---|---|---|
| **Gemini** | yes | Wired up (current default) |
| **Claude** | yes (Anthropic-Vertex partnership) | Supported via `agents/providers/claude.py` |
| **GPT/OpenAI** | **no** | Excluded — not hosted on Vertex AI |

GPT models are intentionally not supported. They are not available on Vertex AI; using them would require OpenAI direct or Azure OpenAI Service, which is out of scope for this project.

---

## Run the chat app

```bash
streamlit run app.py
```

Sidebar:
- **Sub-agent execution** — sequential (specialists run one after another) or parallel (specialists fan out).
- **Model selection** *(planned)* — pick from the curated low-cost / fast-response Vertex models registered in `agents/models.py`.

---

## Run the evaluator

The evaluator runs against the frozen TravelPlanner sandbox (reproducible, no live-API dependency):

```bash
# Sanity check — annotated train plans should pass the evaluator near 100%
python -m eval.run_eval --split train --system annotated --limit 10

# Full multi-agent on the first 20 validation queries
python -m eval.run_eval --split validation --system multi --limit 20

# Full 180-query evaluation, multi-agent + single-agent baseline
python -m eval.run_eval --split validation --system both

# All four systems for the architectural ablation table
python -m eval.run_eval --split validation --system all

# Persistent disk cache — makes reruns near-zero-cost
python -m eval.run_eval --split validation --system multi --llm-cache-dir .cache/llm
```

`--system` accepts: `multi` (full multi-agent), `single` (single-agent baseline), `no_verify`, `no_specialization`, `annotated` (sanity check), `both` (multi + single), `all` (multi + single + no_verify + no_specialization).

Additional flags:
- `--model <id>` — run on a specific registered model.
- `--models all` or `--models id1,id2,...` — sweep multiple models in one invocation.
- `--cooldown-seconds N` — sleep between models to manage Vertex quotas.

Results are written to `eval/results.csv` and a Markdown summary is printed.

---

## Refresh the sandbox

The shipped TravelPlanner data is from 2022. To regenerate against current-year data while preserving the schema:

```bash
# Dry-run 5 cities to eyeball output
python scripts/refresh_sandbox.py --subset 5 --provider llm

# Full refresh (writes Dataset/validation_fresh.csv and validation_fresh_ref_info.jsonl)
python scripts/refresh_sandbox.py --provider llm --year 2026 --inflation 1.20
```

---

## Layout

```
app.py                      Streamlit chat UI

agents/
  coordinator.py            LangGraph state machine: parse → research → specialists → verify → repair
  transport.py              Transport specialist (flights, routes)
  lodging.py                Lodging specialist
  dining.py                 Dining specialist
  sightseeing.py            Sightseeing specialist
  budget.py                 Budget Agent (post-hoc, deterministic)
  verifier.py               In-loop verifier (8 commonsense + 5 hard rules + responsibility tagging)
  rules.py                  Rule constants and helpers shared by verifier and evaluator
  schemas.py                Pydantic contracts (Intent, FullPlan, ToolContext, …)
  llm.py                    LLM wrapper (Gemini today; provider-aware facade after Phase 1)

baseline/
  single_agent.py           Single-LLM-call baseline (paper §4.7 baseline 1)
  no_specialization.py      Coordinator + one generalist Worker (paper §4.7 baseline 3)
  no_verify.py              Full pipeline minus verify-and-repair (paper §4.7 baseline 2)

eval/
  run_eval.py               Evaluation CLI
  constraints.py            TravelPlanner-style scoring evaluator
  results.csv               Sweep output (gitignored)

tools/
  live_apis.py              Google Maps + SerpAPI clients
  sandbox.py                TravelPlanner reference-data sandbox

scripts/
  refresh_sandbox.py        2022 → 2026 schema-preserving regeneration
```

For detail on each component and the planned multi-provider LLM layer, see [ARCHITECTURE.md](ARCHITECTURE.md).
