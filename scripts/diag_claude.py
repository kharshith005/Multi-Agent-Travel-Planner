"""Claude Vertex AI diagnostic script.

Runs a 6-cell matrix (region × system) plus two control cells that mirror
the working Anthropic sample exactly. Prints a per-cell verdict table.

Usage (from Project_Code/):
    python scripts/diag_claude.py
    CLAUDE_DEBUG=1 python scripts/diag_claude.py   # verbose outgoing-call dump

Plausible outcomes:
  All fail 429       → pure quota issue; consider us-east5 or requesting quota increase.
  C/D succeed only   → quota is region-scoped; switch CLAUDE_VERTEX_REGION=us-east5.
  F succeeds, E fails→ bug in our wrapper; bisect claude.py against the raw call.
  All fail auth      → project not enabled in Vertex AI Model Garden.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load .env so GOOGLE_CLOUD_PROJECT and CLAUDE_VERTEX_REGION are available.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

try:
    from anthropic import AnthropicVertex
except ImportError:
    print("ERROR: anthropic[vertex] not installed. Run: pip install 'anthropic[vertex]'")
    sys.exit(1)

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
if not PROJECT:
    print("ERROR: GOOGLE_CLOUD_PROJECT not set. Check .env or export it.")
    sys.exit(1)

MODEL = "claude-haiku-4-5"   # cheapest / fastest for the probe
PING_USER = "Reply with the single word: pong"
TIMEOUT = 30.0

# ── Cell definitions ──────────────────────────────────────────────────────────
cells: list[dict] = [
    {"name": "A", "region": "global",   "system": "",               "via": "wrapper"},
    {"name": "B", "region": "global",   "system": "Reply pong",     "via": "wrapper"},
    {"name": "C", "region": "us-east5", "system": "",               "via": "wrapper"},
    {"name": "D", "region": "us-east5", "system": "Reply pong",     "via": "wrapper"},
    # Control cells — mirror Anthropic's working sample exactly
    {"name": "E", "region": "global",   "system": "",               "via": "wrapper",  "model": "claude-sonnet-4-6"},
    {"name": "F", "region": "global",   "system": "",               "via": "raw",      "model": "claude-sonnet-4-6"},
]

# ── Print header ─────────────────────────────────────────────────────────────

print(f"\nProject: {PROJECT}")
print(f"Model (default): {MODEL}")
print(f"Timeout per call: {TIMEOUT}s\n")
print(f"{'Cell':<5}{'Region':<12}{'System':<20}{'Via':<10}{'ms':>7}  Status / reply (truncated)")
print("-" * 80)


def _run_cell(cell: dict) -> None:
    region = cell["region"]
    system = cell["system"]
    via = cell["via"]
    model_id = cell.get("model", MODEL)

    t0 = time.monotonic()
    try:
        if via == "wrapper":
            # Use our agents/providers/claude.py wrapper.
            # Temporarily override env so the wrapper picks the right region.
            old_region = os.environ.get("CLAUDE_VERTEX_REGION", "")
            os.environ["CLAUDE_VERTEX_REGION"] = region
            try:
                # Import fresh to avoid cached client from a different region.
                from agents.providers.claude import clear_client_cache, generate
                clear_client_cache()
                text, _in, _out = generate(
                    model_id, system, PING_USER,
                    max_tokens=32,
                    schema_cls=None,
                )
            finally:
                os.environ["CLAUDE_VERTEX_REGION"] = old_region
        else:
            # Raw AnthropicVertex call — mirrors Anthropic's getting-started sample.
            client = AnthropicVertex(
                region=region,
                project_id=PROJECT,
                timeout=TIMEOUT,
                max_retries=0,
            )
            msgs = [{"role": "user", "content": PING_USER}]
            kwargs: dict = {"model": model_id, "max_tokens": 32, "messages": msgs}
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        reply = (text or "").replace("\n", " ").strip()[:60]
        print(f"{cell['name']:<5}{region:<12}{system!r:<20}{via:<10}{elapsed_ms:>7}  OK: {reply!r}")

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        status = getattr(exc, "status_code", None)
        err = str(exc).replace("\n", " ").strip()[:120]
        print(f"{cell['name']:<5}{region:<12}{system!r:<20}{via:<10}{elapsed_ms:>7}  FAIL ({status}): {err}")


for cell in cells:
    _run_cell(cell)

print()
print("Interpretation:")
print("  All cells FAIL 429      → quota exhausted; file Vertex quota increase")
print("  C/D pass, A/B fail      → quota is region-scoped; set CLAUDE_VERTEX_REGION=us-east5")
print("  F passes, E fails       → bug in our wrapper; compare generate() vs. raw call")
print("  All FAIL auth/forbidden → Claude not enabled in Vertex AI Model Garden for this project")
