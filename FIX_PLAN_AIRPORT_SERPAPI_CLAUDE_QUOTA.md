# Fix Plan — NYC Airport Resolution, SerpAPI Empty Results, Claude Vertex 429

Drafted 2026-04-29. Three related issues surfaced in a live app run for a `tempe → new york city` query. Plan only — no code changes yet.

---

## Issue 1 — NYC resolves to "NYS" (seaplane base) instead of JFK/LGA/EWR

### Symptom
```
No direct airport for 'new york city'; nearest is NYS (2 mi away) — drive transfer included.
```

### Root cause
**Files:** `Project_Code/tools/live_apis.py:766-802` (`_resolve_airport_code`) and `:674-764` (`_resolve_with_fallback`).

1. `_resolve_airport_code("new york city")` matches by `city == target` OR `target in name`. In the `airportsdata` package:
   - JFK / LGA / EWR have `city == "New York"` (no "city" suffix), so `city == "new york city"` is False.
   - Their names ("John F Kennedy International Airport", etc.) do not contain the substring `"new york city"`.
   - Direct lookup raises `RuntimeError`.
2. Control falls through to nearest-airport mode (`:709-749`): geocode "new york city" → Manhattan center, then haversine over **every** entry in `airportsdata`. That dataset includes seaplane bases, heliports, and private fields with no scheduled commercial service.
3. **NYS = "New York Skyports SPB"** (East River seaplane base, ~2 mi from Manhattan) beats LGA (~8 mi), EWR (~13 mi), JFK (~15 mi) on raw distance.
4. SerpAPI's `google_flights` engine has no inventory for `NYS` → cascades into Issue 2.

### Fix plan

**A. City-alias map** in `live_apis.py` (top of module) for high-traffic ambiguous metros that resolve to airport *systems*:
```python
_CITY_AIRPORT_ALIASES = {
    "new york": ["JFK", "LGA", "EWR"],
    "new york city": ["JFK", "LGA", "EWR"],
    "nyc": ["JFK", "LGA", "EWR"],
    "washington": ["IAD", "DCA", "BWI"],
    "washington dc": ["IAD", "DCA", "BWI"],
    "chicago": ["ORD", "MDW"],
    "houston": ["IAH", "HOU"],
    "dallas": ["DFW", "DAL"],
    "los angeles": ["LAX", "BUR", "LGB", "SNA"],
    "san francisco": ["SFO", "OAK", "SJC"],
    "london": ["LHR", "LGW", "STN", "LCY"],
    "paris": ["CDG", "ORY"],
    "tokyo": ["HND", "NRT"],
    # ...extend as needed
}
```
Check this map *before* the substring search inside `_resolve_airport_code`. Return the first code; later improvement could pick by SerpAPI hit rate.

**B. Filter the nearest-airport haversine sweep** (`_resolve_with_fallback:729-743`) to **major commercial airports only**. Two options:
- **Cheap heuristic (preferred for now):** skip entries whose `name` contains `Seaplane Base`, `Heliport`, `SPB`, `Heli`, `Airpark`, `Field`, `Strip`. One-line `if any(tag in name for tag in BLACKLIST): continue`.
- **Robust fallback:** ship a bundled allowlist of ~500 IATA codes for `large_airport`+`medium_airport` from the OurAirports CSV data dump. One-time data file, not a runtime dependency. Use only if heuristic proves leaky.

**C. SerpAPI verification with retry-on-empty** (optional, in `coordinator.py:670-746`): if the first flight search returns zero options *with no time filter*, retry once with the next candidate from the alias list (or the second-nearest commercial airport). Cap at one retry to bound cost.

**D. Clarify progress message** at `live_apis.py:758-762`. When the alias map was consulted, change to:
```
"'new york city' has no single primary airport; using JFK (16 mi), LGA (8 mi), EWR (13 mi)"
```

---

## Issue 2 — SerpAPI round-trip + one-way both fail

### Symptom
```
No round-trip itineraries matched outbound_times=4,13,12,21 — broadening.
No flights matched outbound_times=4,13 — broadening.
No flights matched outbound_times=12,21 — broadening.
SerpAPI round-trip search returned no outbound options with departure_token;
outbound flight search failed: SerpAPI flights returned no options;
return flight search failed: SerpAPI flights returned no options
— transport specialist will use driving/taxi fallback.
```

### Root cause
Same as Issue 1. The dep/arr codes passed to `flight_search_round_trip` (`coordinator.py:693`) and `flight_search` (`:713,727`) were `NYS`. SerpAPI's `google_flights` engine has zero listings for that base, so `payload.get("best_flights")` and `other_flights` are empty in every call (with and without `outbound_times`). The "broadening" branches at `live_apis.py:266-275` and `:537-544` cannot recover — they remove the time filter, but the airport itself has zero scheduled commercial inventory.

### Fix plan

This is fully resolved by Issue 1 fixes A+B+C. Additionally:

**E. Explicit "no inventory" warning path**: when SerpAPI returns zero flights even after dropping `outbound_times`, log:
```
"SerpAPI has no Google Flights inventory for {dep_id}↔{arr_id} — falling back to next candidate / driving"
```
so it's clear in eval logs that the broader search did not just hit a time-filter edge case.

**F. Past-date guard for SerpAPI**: `_serpapi_get` already retries past-date errors (`live_apis.py:217-219`). Re-verify that `bypass_date_check=True` (eval path) cannot reach `flight_search` with a past `outbound_date`. `_normalize_outbound_date:817-828` forces past dates to today, which is fine for the app path; the eval path uses sandbox, so it should never call SerpAPI. Confirm with a grep before closing the issue.

---

## Issue 3 — Claude 429 RESOURCE_EXHAUSTED on Vertex

### Symptom
```
Planning failed: RuntimeError: Claude request failed (429):
Error code: 429 - {'error': {'code': 429,
'message': 'Resource has been exhausted (e.g. check quota).',
'status': 'RESOURCE_EXHAUSTED'}}.
Check GOOGLE_CLOUD_PROJECT, CLAUDE_VERTEX_REGION (e.g. global), and ADC configuration.
```

### Root cause
**Files:** `Project_Code/agents/providers/claude.py:144-153` + `Project_Code/agents/llm.py:135-185`.

- Anthropic-on-Vertex SDK raises `APIStatusError` with `status_code=429`. `claude.py:147-153` wraps it in `RuntimeError`, preserving `status_code` via `setattr`. `llm.py:_is_retryable` correctly identifies 429 as retryable.
- Retry policy: `_MAX_ATTEMPTS=3`, `_backoff = min(2**attempt, 12) + jitter`. Total ceiling ≈ 1+2+4+jitter ≈ **~7 seconds across 3 attempts**.
- Vertex AI Anthropic quota is a **project-level TPM/RPM bucket** in the chosen region. Once exhausted it does not refill in 7 seconds — typical refill window is 60s.
- Vertex 429 responses do not always include `Retry-After`, so `_parse_retry_after:156-167` returns `None` and we fall to the short backoff.
- A single TravelPlanner query traverses `coordinator → research → 4 specialists (parallel fan-out) → budget → verifier → up to 3 repair rounds` ≈ 15–25 Claude calls. The parallel specialist fan-out is the spike that trips per-minute quota; back-to-back eval queries keep the bucket empty.
- Error message currently misattributes the failure to ADC/auth config.

### Fix plan

**G. Per-status retry policy** in `agents/llm.py:170-185`. Split:
```python
_MAX_ATTEMPTS_429 = 6
_BACKOFF_429 = lambda a: min(15 * (2 ** a), 90) + random.uniform(0, 5)
# Sequence: 15s, 30s, 60s, 90s, 90s, 90s — total ≤ ~6 min worst case.
```
Other 5xx keep the current short policy. Acceptable for batch eval where the alternative is a hard failure.

**H. Serialize specialist fan-out when on Claude.** Currently `coordinator.py` runs the 4 specialists in parallel (find via `ThreadPoolExecutor`). Add a provider-aware semaphore: cap concurrent in-flight Claude requests to 1–2; keep parallel for Gemini. This prevents TPM spikes that trigger the 429 in the first place.

**I. Better error message.** After exhausting retries, raise with quota-specific guidance instead of auth guidance:
```
"Claude on Vertex quota exhausted in region {region} for project {project}.
Increase quota at https://console.cloud.google.com/iam-admin/quotas
(filter: aiplatform.googleapis.com, online_prediction_requests_per_base_model)
or switch region with CLAUDE_VERTEX_REGION."
```

**J. Eval-time mitigation.** Always enable the disk LLM cache (`--llm-cache-dir .cache/llm`) for paper runs so repeat eval runs don't re-spend quota on identical prompts. Already supported per `CLAUDE.md`; document as required for the final report run.

**K. Optional: provider auto-fallback.** If Claude exhausts after all retries, degrade *that single call* to Gemini and continue, marking the result as `provider=gemini-fallback`. Gate behind `CLAUDE_FALLBACK_TO_GEMINI=1` and **disable for eval** — mixing providers within one ablation row defeats the purpose of the table. App-path resilience only.

---

## Suggested order of execution

| # | Task | Effort | Why first |
|---|------|--------|-----------|
| 1 | G + I (retry policy + clearer error) | 15 min | Unblocks current Claude usage immediately. |
| 2 | A (NYC alias map) | 30 min | Fixes the immediate user-visible NYC bug. |
| 3 | B (blacklist seaplane/heliport in haversine) | 30 min | Prevents the same class of bug for any other city. |
| 4 | H (serial fan-out on Claude) | ~1 hr | Prevents future quota spikes during eval. |
| 5 | C, E, F, K | nice-to-have | Layer on after 1–4 land. |

## Files touched

- `Project_Code/tools/live_apis.py` — alias map, blacklist filter, clearer progress message.
- `Project_Code/agents/llm.py` — split retry policy by status code.
- `Project_Code/agents/providers/claude.py` — quota-specific error message.
- `Project_Code/agents/coordinator.py` — provider-aware semaphore for specialist fan-out; optional retry-on-empty SerpAPI verification.

No schema changes, no file moves, no eval contract changes.
