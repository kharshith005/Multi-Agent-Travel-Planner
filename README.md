# Multi-Agent Travel Itinerary Planner

A coordinator-led multi-agent system that converts a natural-language travel query into a day-by-day itinerary satisfying both user-stated hard constraints and TravelPlanner-style commonsense constraints.

Built on **LangGraph** (coordinator state machine), **Pydantic** (inter-agent contracts), and **Streamlit** (chat UI). Evaluated against the **TravelPlanner benchmark** — Xie et al. (2024), arXiv:2402.01622.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

### Required keys (Gemini + live APIs)

| Variable | Purpose |
|---|---|
| `VERTEX_AI_API_KEY` | Gemini on Vertex AI |
| `GOOGLE_MAPS_API_KEY` | Live sightseeing, lodging, and route lookups |
| `SERPAPI_API_KEY` | Live flight options via Google Flights |

`.env` is gitignored. The runtime is **strict** — missing keys surface as errors, never silent fallbacks.

### Optional: Llama 3.3 / Mistral Small 3.1 on Vertex AI

Both models use Application Default Credentials (ADC):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | — | GCP project with Vertex AI enabled |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to service-account JSON (or use `gcloud auth application-default login`) |
| `META_VERTEX_REGION` | `us-central1` | Region for Llama 3.3 70B |
| `MISTRAL_VERTEX_REGION` | `us-central1` | Region for Mistral Small 3.1 |

Enable each model in Vertex AI Model Garden before use (one-time per-project click-through). When `GOOGLE_CLOUD_PROJECT` is unset or `google-auth` is not installed, these models are silently hidden from the model picker.

### Supported models

| Model | Short alias | Provider | Auth |
|---|---|---|---|
| `gemini-3.1-flash-lite-preview` | `gemini-flash-lite` | Gemini | `VERTEX_AI_API_KEY` |
| `gemini-2.5-flash-lite` | — | Gemini | `VERTEX_AI_API_KEY` |
| `llama-3.3-70b-instruct-maas` | `llama-3.3` | Meta (Vertex MaaS) | ADC + `GOOGLE_CLOUD_PROJECT` |
| `mistral-small-2503` | `mistral-small-3.1` | Mistral (Vertex MaaS) | ADC + `GOOGLE_CLOUD_PROJECT` |

Set `LLM_MODEL=<id or alias>` in `.env` or pass `--model <id or alias>` to the eval CLI.

---

## Run the chat app

```bash
streamlit run app.py
```

The sidebar exposes sequential vs. parallel specialist execution and a model selector that shows only models whose auth is configured.

---

## Run the evaluator

Runs against the frozen TravelPlanner sandbox — no live API calls, reproducible.

```bash
# Sanity check: annotated reference plans should pass near 100%
python -m eval.run_eval --split train --system annotated --limit 10

# Full multi-agent on first 20 validation queries
python -m eval.run_eval --split validation --system multi --limit 20

# Multi-agent + single-agent baseline (180 queries)
python -m eval.run_eval --split validation --system both

# All four systems for the architectural ablation table
python -m eval.run_eval --split validation --system all

# Persistent disk cache (makes reruns near-zero-cost)
python -m eval.run_eval --split validation --system multi --llm-cache-dir .cache/llm

# Run on a specific model (full ID or short alias)
python -m eval.run_eval --split validation --system multi --model llama-3.3

# Sweep multiple models
python -m eval.run_eval --split validation --system multi --models gemini-flash-lite,llama-3.3
```

`--system` options: `multi`, `single`, `no_verify`, `no_specialization`, `annotated`, `both`, `all`.

Results are written to `eval/results.csv` and a Markdown summary is printed.

---

## Layout

```
app.py                      Streamlit chat UI

agents/
  coordinator.py            LangGraph state machine: parse → research → specialists → assemble → budget → verify → repair
  transport.py              Transport specialist (flights, routes, inter-city legs)
  lodging.py                Lodging specialist
  dining.py                 Dining specialist
  sightseeing.py            Sightseeing specialist
  budget.py                 Budget Agent (post-hoc, deterministic)
  verifier.py               In-loop verifier (8 commonsense + 5 hard rules)
  rules.py                  Rule constants and helpers
  schemas.py                Pydantic contracts (Intent, FullPlan, ToolContext, …)
  llm.py                    LLM facade (two-tier cache, retry, provider dispatch)
  models.py                 Model registry with short-alias resolution
  runtime.py                ContextVar for the active model per request
  providers/
    __init__.py             call_provider() router + clear_caches()
    gemini.py               Gemini via google-genai (Vertex API-key mode)
    vertex_requests.py      Llama + Mistral via Vertex MaaS + requests

baseline/
  single_agent.py           Single-LLM-call baseline (paper §4.7 Baseline 1)
  no_specialization.py      Coordinator + one generalist Worker (Baseline 3)
  no_verify.py              Full pipeline minus verify-and-repair (Baseline 2)

eval/
  run_eval.py               Evaluation CLI (multi-model sweep, alias resolution)
  constraints.py            TravelPlanner-style scoring (8 CS + 5 HC rules)
  format_report.py          Generate Markdown tables + Pareto chart from results.csv
  state_city_index.json     State → cities map for multi-city query resolution
  results.csv               Sweep output (gitignored)

tools/
  live_apis.py              Google Maps + SerpAPI clients
  sandbox.py                TravelPlanner reference-data sandbox

scripts/
  refresh_sandbox.py        2022 → 2026 sandbox regeneration (optional)
  build_state_index.py      Rebuild eval/state_city_index.json from training corpus

```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design reference.
