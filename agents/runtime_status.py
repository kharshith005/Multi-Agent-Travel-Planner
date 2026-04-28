"""Feature-level auth status for the Streamlit UI.

Reports which live-data features are available based on env auth,
without exposing env-var names directly in the user-facing interface.
"""
from __future__ import annotations

import os


def llm_available() -> bool:
    return bool(os.environ.get("VERTEX_AI_API_KEY", "").strip())


def flights_available() -> bool:
    return bool(os.environ.get("SERPAPI_API_KEY", "").strip())


def places_available() -> bool:
    return bool(os.environ.get("GOOGLE_MAPS_API_KEY", "").strip())
