"""Validate runtime environment before launching the app or running eval.

Checks all required API keys and reports which features are active.
Exits non-zero if any required key is missing.

Usage:
    cd Project_Code && source .venv/bin/activate
    python scripts/validate_env.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def _check(label: str, key: str, required: bool) -> bool:
    val = os.environ.get(key, "").strip()
    status = "OK     " if val else ("MISSING" if required else "not set")
    tag = " [required]" if required and not val else ""
    print(f"  {status}  {label}{tag}")
    return bool(val)


print("=== Environment validation ===\n")

ok_llm     = _check("LLM planning (Gemini on Vertex)",        "VERTEX_AI_API_KEY",        required=True)
ok_maps    = _check("Places & maps (Google Maps API)",         "GOOGLE_MAPS_API_KEY",      required=True)
ok_flights = _check("Flight search (SerpAPI)",                 "SERPAPI_API_KEY",          required=True)
_check("Claude on Vertex (project ID)",                        "GOOGLE_CLOUD_PROJECT",     required=False)
_check("Claude on Vertex (region)",                            "CLAUDE_VERTEX_REGION",     required=False)
_check("GCP service-account credentials file",                 "GOOGLE_APPLICATION_CREDENTIALS", required=False)

print()
if ok_llm and ok_maps and ok_flights:
    print("All required keys are set. Live planning is fully operational.")
    sys.exit(0)
else:
    missing = [
        name for name, ok in [
            ("VERTEX_AI_API_KEY", ok_llm),
            ("GOOGLE_MAPS_API_KEY", ok_maps),
            ("SERPAPI_API_KEY", ok_flights),
        ] if not ok
    ]
    print(f"Missing required keys: {', '.join(missing)}")
    print("Copy .env.example to .env and fill in the missing values.")
    sys.exit(1)
