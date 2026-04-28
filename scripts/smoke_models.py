"""Smoke test all registered models (Phase 1 validation gate).

For each model in available_models(), calls parse_intent on a fixed query
and reports model, latency, token counts, and cache status.

Run twice: first run → all cache misses; second run → all cache hits.
Different models should never share cache entries (verifies provider+model key separation).

Usage:
    cd Project_Code && source .venv/bin/activate
    python scripts/smoke_models.py
    python scripts/smoke_models.py  # second run — expect cache hits
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Enable disk cache for key-separation verification
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "smoke"
os.environ["LLM_CACHE_DIR"] = str(CACHE_DIR)

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from agents.models import available_models
from agents.runtime import use_model
from agents.coordinator import parse_intent
from agents.llm import get_call_stats, reset_call_stats

FIXED_QUERY = (
    "Plan a 3-day trip from Washington DC to Myrtle Beach for 2 people "
    "from 2026-06-01 to 2026-06-03 with a budget of $1400. "
    "I prefer coastal hotels and seafood restaurants."
)

print(f"Smoke test — fixed query:\n  {FIXED_QUERY}\n")
print(f"Cache dir: {CACHE_DIR}\n")

models = available_models()
if not models:
    print("ERROR: No models available. Check VERTEX_AI_API_KEY (Gemini) or "
          "GOOGLE_CLOUD_PROJECT + CLAUDE_VERTEX_REGION (Claude).", file=sys.stderr)
    sys.exit(1)

for entry in models:
    reset_call_stats()
    t0 = time.perf_counter()
    try:
        with use_model(entry.id):
            intent = parse_intent(FIXED_QUERY)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = get_call_stats()
        status = "CACHE HIT" if stats["cache_hits"] > 0 else "CACHE MISS"
        print(
            f"[{entry.provider:6s}] {entry.display_name:<40s} "
            f"{elapsed_ms:6.0f}ms  "
            f"calls={stats['llm_calls']}  "
            f"in={stats['input_tokens']}  out={stats['output_tokens']}  "
            f"{status}"
        )
        print(f"          intent: {intent.org!r} -> {intent.dest!r}, {intent.days}d, budget={intent.budget}")
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[{entry.provider:6s}] {entry.display_name:<40s} FAILED ({elapsed_ms:.0f}ms): {e}")

print("\nDone. Run again to verify cache hits on same model, no cross-model collisions.")
