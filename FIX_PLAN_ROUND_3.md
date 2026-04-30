# Fix Plan — Round 3 (Llama 3.3 + Mistral Small 3.1 providers; Claude Haiku Vertex validation)

Drafted 2026-04-29.
- Updated 2026-04-29: switched from `openai` SDK to `requests` (already in deps) for the Llama/Mistral backend — no new package required.
- Updated 2026-04-29: **Part A fully implemented.** Short aliases `llama-3.3` and `mistral-small-3.1` supported in `LLM_MODEL` env var and `--model` CLI flag.

Two independent workstreams:
- **A.** Add Llama 3.3 70B Instruct and Mistral Small 3.1 to the model registry, including a new `meta`/`mistral` provider backend that mirrors the existing Gemini/Claude pattern.
- **B.** Test/validate the existing Claude Haiku 4.5 path on Vertex AI end-to-end against the official Vertex Model Garden docs, before committing it as the canonical Claude entry for the report.

---

## Part A — Llama 3.3 + Mistral Small 3.1 as supported LLMs

### A.0 Background — what Vertex actually exposes

Per Vertex AI partner-model docs (verified 2026-04-29):

| Property | Llama 3.3 70B Instruct | Mistral Small 3.1 |
|---|---|---|
| Model ID | `llama-3.3-70b-instruct-maas` | `mistral-small-2503` |
| Publisher | `meta` | `mistralai` |
| Regions | `us-central1` (and others; verify in Model Garden console) | `us-central1`, `europe-west4` |
| Auth modes | API key, ADC, OAuth | gcloud / service-account ADC |
| Endpoint shape | OpenAI-compatible chat completions (`/endpoints/openapi/chat/completions`) | `:rawPredict` / `:streamRawPredict` (also OpenAI-compat available via the same MaaS endpoint) |
| Tool/function calling | Supported | Not explicitly documented |
| JSON mode | Supported (`response_format={"type":"json_object"}`) | Not explicitly documented |
| Quota (default) | per-region MaaS quota | 60 QPM / 200k TPM (us-central1) |
| Context window | 128k | 128k |

**Design choice: call both models via `requests` (already in deps) posting directly to Vertex `:rawPredict`.**
- **No new package.** `requests` is already in `requirements.txt`. `google-auth` arrives transitively via `google-genai`.
- **No OpenAI SDK.** The `openai` package was considered (Vertex exposes an OpenAI-compat endpoint) but rejected: we already own retries in `agents/llm.py`, there is no benefit to the SDK's client-class abstraction, and adding a fourth LLM-related package is unnecessary weight.
- **One backend module for both models.** Llama uses the MaaS OpenAI-compat path; Mistral uses `:rawPredict`. Both accept the same chat-completions-style JSON body and return the same response shape, so a single `generate()` function handles both by passing the correct URL for the active model.

Endpoint URLs (non-streaming `:rawPredict`):

- **Llama 3.3:**
  ```
  POST https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/publishers/meta/models/llama-3.3-70b-instruct-maas:rawPredict
  ```
- **Mistral Small 3.1:**
  ```
  POST https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/publishers/mistralai/models/mistral-small-2503:rawPredict
  ```

Request body (same for both — OpenAI chat-completions schema):
```json
{
  "model": "<model_id>",
  "messages": [
    {"role": "system", "content": "<system>"},
    {"role": "user",   "content": "<user>"}
  ],
  "max_tokens": <N>,
  "response_format": {"type": "json_object"}   // only when json_mode=True
}
```

Auth = short-lived OAuth bearer token from ADC (`google.auth.default()` → refresh → `credentials.token`). Token expires after ~60 min; refresh before each request if expired.

> **Why not the `vertexai` Python SDK?** Adds a third Google LLM library on top of `google-genai` and `anthropic[vertex]`, with no tangible benefit for a plain POST call.

### A.1 Files touched (no new packages — `requests` + `google-auth` already present)

| File | Change |
|---|---|
| `Project_Code/agents/models.py` | Add two `ModelEntry` rows (provider="meta", "mistral"); update cost calibration comments. |
| `Project_Code/agents/providers/__init__.py` | Extend `call_provider` dispatcher to route `meta` and `mistral` to a new `vertex_requests` backend. |
| `Project_Code/agents/providers/vertex_requests.py` | **NEW.** Single backend using `requests` for both Llama and Mistral via Vertex `:rawPredict`. Same shape as `gemini.py` / `claude.py`: `generate(model_id, system, user, *, max_tokens, json_mode) -> (text, in_tok, out_tok)`. |
| `Project_Code/agents/llm.py` | No changes needed — `call_json` already dispatches cleanly; `meta`/`mistral` go through the same JSON-mode path as Gemini. |
| `Project_Code/.env.example` | Document `META_VERTEX_REGION` / `MISTRAL_VERTEX_REGION` (default `us-central1`). Reuse `GOOGLE_CLOUD_PROJECT`. |
| `Project_Code/requirements.txt` | No changes needed. `google-auth` already present transitively. |
| `Project_Code/scripts/probe_provider.py` | **NEW.** Tiny smoke-test script (Part B validation). |

### A.2 New `agents/providers/vertex_requests.py` — design sketch

Module shape mirrors `claude.py`, using `requests` instead of an SDK:

1. **Auth helper.** `_access_token()`: calls `google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])`, then `google.auth.transport.requests.Request()` to refresh if expired. Returns `credentials.token` (a short-lived bearer string). No caching of the token itself — it's checked and refreshed on each call (cheap; avoids stale-token 401s mid-batch).
2. **Region + project resolvers.** `_region(provider)` reads `META_VERTEX_REGION` (for `"meta"`) or `MISTRAL_VERTEX_REGION` (for `"mistral"`), defaulting to `"us-central1"`. `_project()` reads `GOOGLE_CLOUD_PROJECT` and raises a clear error if unset — same pattern as `claude.py`.
3. **URL builder.** `_endpoint_url(model_id, provider)`:
   - meta → `https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/meta/models/{model_id}:rawPredict`
   - mistral → same pattern but `publishers/mistralai`
4. **`generate()` function** with the same signature as `gemini.py`:
   ```
   def generate(model_id, system, user, *, max_tokens, json_mode) -> (text, in_tok, out_tok)
   ```
   - Builds the chat-completions JSON body: `{"model": model_id, "messages": [...], "max_tokens": N}`.
   - When `json_mode=True`, adds `"response_format": {"type": "json_object"}`. If the response returns HTTP 400 with "response_format not supported", retries once without it (Mistral fallback).
   - Calls `requests.post(url, json=body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, timeout=120)`.
   - On HTTP 4xx/5xx, extracts the status code and re-raises as `RuntimeError(status_code=N)` so `agents/llm.py`'s `_with_retry` handles the 429 backoff correctly.
   - Parses `resp.json()["choices"][0]["message"]["content"]` for the text; reads `resp.json()["usage"]["prompt_tokens"]` / `"completion_tokens"` for stats.
5. **`clear_client_cache()`** stub for parity with `gemini.py` / `claude.py` — no-op since there is no persistent client object, but keeps `providers/__init__.py:clear_caches()` uniform.

### A.3 `agents/providers/__init__.py` — dispatcher widening

Today `call_provider` has a 2-way branch (`gemini` / `claude`). Widen to 4-way:
```
provider in {"gemini"} → gemini.generate(...)
provider in {"meta", "mistral"} → vertex_openai.generate(...)
provider in {"claude"} → claude.generate(...)
```
Keep the `schema_cls` argument plumbed through but **only Claude uses it** (Anthropic's tool-forcing is more reliable than OpenAI-style JSON mode). Llama/Mistral get `schema_cls=None` and rely on `json_mode=True` plus the post-call `model_validate_json()` in `call_json`. This matches what Gemini does today and is sufficient because:
- The retry loop in `agents/llm.py:_call_json_inner` already catches `ValidationError` and retries with a clarified prompt.
- The cache key in `_cache_key` already includes the provider, so cached Gemini JSON won't be served on a Llama call.

### A.4 `agents/models.py` — registry rows

Two new entries (cost is a rough placeholder; calibrate with first run output):

| ID | Display name | Provider | Tier | Family | Auth mode | Est. cost (× baseline) | Latency hint |
|---|---|---|---|---|---|---|---|
| `llama-3.3-70b-instruct-maas` | "Llama 3.3 70B Instruct" | `meta` | `standard` | `llama-3` | `vertex_oauth` | ~6× | 1200 ms |
| `mistral-small-2503` | "Mistral Small 3.1" | `mistral` | `lite` | `mistral-3` | `vertex_oauth` | ~3× | 800 ms |

`auth_mode="vertex_oauth"` is a new mode constant. Update `_meta_auth_ok()` / `_mistral_auth_ok()` (or one combined `_vertex_oauth_ok()` since both use the same `GOOGLE_CLOUD_PROJECT` + ADC requirement). `available_models()` filters them out when ADC is missing — same pattern as Claude today.

### A.5 Pre-flight & cache-key invariants

- The pre-flight LLM check in `app.py` (added in Round 2) already reads the active model and issues a tiny `call_json`. It will work unchanged for Llama and Mistral — we just need to make sure `vertex_openai.generate` raises with `status_code` set on auth/quota errors, so the banner renders the right reason.
- `_cache_key` already namespaces by provider + model_id. No change needed.
- The eval CLI's `--llm-cache-dir` continues to give per-model reproducibility because `provider` is part of the key.

### A.6 Testing checklist (one-shot, before committing)

A small `scripts/probe_provider.py` (10–20 lines, **gitignored**) that:
1. For each provider in `{gemini, claude, meta, mistral}`, calls `call_json` with a 1-field schema (`{"ok": bool}`) and `max_tokens=16`.
2. Prints the (model, latency, tokens, response) tuple.
3. Exits non-zero if any provider raises.

Run order:
- Probe **Mistral first** — smallest quota, fastest to hit issues.
- Probe **Llama** with `response_format={"type":"json_object"}` and confirm we get strict JSON.
- Probe **Llama with tool_choice="auto"** as a stretch goal — if it works we can later mirror Claude's tool-forcing pattern for harder JSON cases. Defer if not free.
- Re-run the existing 5-row eval (`--limit 5 --system multi --model llama-3.3-70b-instruct-maas`) to confirm the multi-agent pipeline survives an end-to-end run on each new model.

### A.7 Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Mistral does not enforce `response_format`** → invalid JSON, stuck in retry loop. | The "swallow once and retry without" branch in §A.2(4) plus the existing `ValidationError` retry in `_call_json_inner` cap losses at 2 retries. Worst case: drop Mistral from `--system multi` and only test it on the single-agent baseline. |
| **OAuth token expiry mid-batch.** | `_access_token()` refreshes via `google.auth.transport.requests.Request()` on every call — no cached stale token. 401 from the API is still re-raised as retryable so `_with_retry` catches it. |
| **Region mismatch** (Mistral not in `us-east5`). | `models.py` carries `available_regions` per entry; the runtime asserts the configured region is in the list and raises a clear error otherwise. Same pattern as Claude. |
| **`requests` response shape differs between Llama and Mistral.** | Both use the same OpenAI chat-completions JSON shape on Vertex `:rawPredict`; tested in the tier-1 probe script before committing. |

### A.8 Rough effort

- Provider module + dispatcher: ~1 hr
- Registry + env wiring + probe script: ~30 min
- 5-row eval per model + cost calibration: ~30 min
- Total: ~2 hr; one PR's worth.

---

## Part B — Validate Claude Haiku 4.5 on Vertex AI

The codebase already has a Claude provider (`agents/providers/claude.py`) using `AnthropicVertex`. Round 2 dropped Sonnet entries and kept `claude-haiku-4-5@20251001`. Goal of this part is **prove it works end-to-end** before the report run, against the official Vertex Model Garden doc:
https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/haiku-4-5

### B.0 Doc-derived facts (from WebFetch on the URL above)

- Model ID: `claude-haiku-4-5` (no `@revision` required for Vertex; the SDK accepts both — current code strips the suffix in `_vertex_model_id` at `claude.py:73-74`).
- Supported regions: `us-east5`, `europe-west1`, `asia-east1`, **and `global`**. Quota at `global` is the highest (2,500 QPM / 2.5M input TPM / 250k output TPM).
- Auth: ADC + project; `anthropic[vertex]` `AnthropicVertex` client is the supported Python entry point.
- Function calling **is supported** (used by `claude.py:103-127` for forced tool-use structured output).
- Limits: 200k input, 64k output. Pricing: see Vertex pricing page (not exposed in the docs page).

The current `claude.py` is consistent with all of these. The validation work is operational, not code-change.

### B.1 Pre-conditions to verify

| Check | How |
|---|---|
| `gcloud auth application-default login` has been run, or `GOOGLE_APPLICATION_CREDENTIALS` points to a valid SA key. | `gcloud auth application-default print-access-token` returns a token. |
| `GOOGLE_CLOUD_PROJECT` is set and the project has Vertex AI API enabled. | `gcloud services list --enabled --project=$GOOGLE_CLOUD_PROJECT \| grep aiplatform`. |
| Claude Haiku 4.5 has been **enabled in Model Garden** (one-time per-project click-through). | Vertex console → Model Garden → search "Claude Haiku 4.5" → "Enable" → accept Anthropic terms. Without this, ADC requests get a 403 "Model not enabled". |
| The SA / user has `roles/aiplatform.user`. | `gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT` shows the binding. |
| `CLAUDE_VERTEX_REGION=global` (already the default in `.env.example`). | `echo $CLAUDE_VERTEX_REGION`. |

### B.2 Three-tier validation ladder

**Tier 1 — raw curl (5 min).** Confirm the project is provisioned and the `global` endpoint actually serves Haiku 4.5 for our project. Use the Vertex AI rawPredict shape from the doc:

```
ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
curl -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  "https://aiplatform.googleapis.com/v1/projects/$GOOGLE_CLOUD_PROJECT/locations/global/publishers/anthropic/models/claude-haiku-4-5:rawPredict" \
  -d '{
    "anthropic_version": "vertex-2023-10-16",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Reply with the word OK."}]
  }'
```

Expected: 200 with a content block whose text starts with "OK". 4xx here means auth/region/Model-Garden enablement — fix before tier 2.

**Tier 2 — `anthropic[vertex]` SDK probe (5 min).** Tiny script that exercises the same client our provider uses:
- Calls `AnthropicVertex(region="global", project_id=…).messages.create(model="claude-haiku-4-5", max_tokens=32, messages=[…])`.
- Prints `resp.content[0].text` and `resp.usage`.
- Assert: text non-empty, `usage.input_tokens > 0`.

If tier 1 passed but this fails, the issue is in the SDK install — check `pip show anthropic` shows the `vertex` extras, and that the SDK version in our `.venv` matches the version that was tested against Vertex.

**Tier 3 — forced tool-use structured output (10 min).** Exercise the structured-output path our specialists actually use. Reuse a tiny Pydantic schema like:
```
class _Probe(BaseModel):
    ok: bool
    note: str
```
Call `agents/providers/claude.generate()` directly with `schema_cls=_Probe`, `max_tokens=64`. Assert:
- The returned text parses to valid `_Probe(...)`.
- `usage.input_tokens` and `output_tokens` are non-zero.

If this fails, the issue is the tool-forcing block in `claude.py:103-127` — most likely the schema's `additionalProperties` causing Anthropic to refuse the tool call. Fix by adding `"additionalProperties": False` to the schema (already standard in Pydantic v2 output) OR setting `tool_choice={"type": "tool", "name": "structured_output", "disable_parallel_tool_use": true}`.

### B.3 End-to-end smoke against the eval harness

After the three-tier probe passes:

1. **Single row, single specialist.**
   ```
   LLM_MODEL=claude-haiku-4-5 python -m eval.run_eval --split validation --system multi --limit 1
   ```
   Expectation: completes in <60s, `final=True` or a deterministic violation pattern. 429 here means our quota footprint is too high — drop concurrent worker count or reduce tool-context size.

2. **Five-row sanity.**
   ```
   LLM_MODEL=claude-haiku-4-5 python -m eval.run_eval --split validation --system multi --limit 5 --llm-cache-dir .cache/llm-claude
   ```
   Expectation: ≥3/5 `final=True`. Tokens/sec roughly comparable to Gemini Flash-Lite (Haiku 4.5 is the fastest Claude tier).

3. **Cost / latency capture.** Log avg input + output tokens per query and avg wall-clock per query. These two numbers go in the report's efficiency table next to Gemini.

### B.4 Quota strategy for the report run

Doc-confirmed quotas at `global`: **2,500 QPM / 2.5M input TPM / 250k output TPM.** At ~10k input + ~1.5k output tokens per query and 5–7 LLM calls per row, one row consumes ~70k input / ~10k output tokens. The 180-row validation set ⇒ ~12.6M input tokens — **5× the per-minute TPM**. Therefore:
- Run with `--limit 30` partitions sequentially, sleeping 60s between batches; or
- Run overnight at a steady cadence — Anthropic's `global` endpoint enforces the TPM, not a daily ceiling.
- **Do NOT run multiple specialists in parallel** while validating; the Round 2 dispatcher already serialises specialists by default (`coordinator.py` has a `parallel_specialists` toggle — keep it off for Claude).

### B.5 Failure modes & explicit decisions

| Symptom | Probable cause | Decision |
|---|---|---|
| Tier 1 curl returns 403 "Model not enabled". | Model Garden click-through missing. | Enable in console; not a code problem. |
| Tier 1 returns 401. | ADC token stale or wrong project. | `gcloud auth application-default login` again. |
| Tier 2 hangs >2 min. | Anthropic SDK swallowing 429 internally despite `max_retries=0`. | Confirm `claude.py:55-60` still passes `max_retries=0`. |
| Tier 3 returns "Claude did not call the structured_output tool". | Pydantic schema has unsupported keyword (e.g. `$ref` w/o `definitions`). | Inline the schema definitions before passing to `tools=[…]`. |
| Per-row wall-clock >2× Gemini. | `global` endpoint routing through a far region. | Try `CLAUDE_VERTEX_REGION=us-east5` and re-probe; if quota is enough, switch the report run there. |
| Eval row 161 still TruncationError. | Round 2 raised dining `max_tokens` to 4096; Haiku may emit longer JSON. | Bump dining ceiling to 6144 just for Claude runs. |

### B.6 Deliverables for the report

- Exit-1 cell of the per-model results CSV (`eval/results.csv`) for `claude-haiku-4-5`.
- Average wall-clock per query (Haiku vs Gemini Flash-Lite vs Gemini 2.5 Flash).
- Per-rule pass rates as the diagnostic that tells us which rule (cuisine? room_type?) Haiku does or doesn't help with.

### B.7 Rough effort

- B.1 + B.2 (tiers 1–3): 30 min.
- B.3 (5-row eval + cost capture): 30 min.
- B.4 (full 180-row run, mostly waiting): 2–3 hours wall-clock.
- Total active effort: ~1 hr.

---

## Suggested execution order

| # | Workstream | Step | Status |
|---|---|---|---|
| 1 | A | `vertex_requests.py` backend + dispatcher + registry + aliases | ✅ Done |
| 2 | A | Short aliases `llama-3.3` / `mistral-small-3.1` in env var + CLI | ✅ Done |
| 3 | B | `scripts/probe_claude_haiku.py` three-tier validation script | ✅ Done |
| 4 | B | Run Haiku probe — confirm tiers 1–3 pass | ⏳ Pending (operational) |
| 5 | A | 5-row eval per new model + cost calibration | ⏳ Pending (needs Model Garden enablement) |
| 6 | B | 180-row Haiku eval run for the report | ⏳ Pending (run after tier probe passes) |

No schema changes that affect verifier or evaluator output formats. No retries logic in the cross-provider layer (each provider raises with `status_code` and `agents/llm.py` handles it).
