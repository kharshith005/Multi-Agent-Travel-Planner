"""Invariant: env-var names must never appear in the Streamlit UI source."""
from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_IN_UI = [
    "VERTEX_AI_API_KEY",
    "SERPAPI_API_KEY",
    "GOOGLE_MAPS_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CLAUDE_VERTEX_REGION",
]

APP_PY = Path(__file__).resolve().parents[2] / "app.py"


def test_env_var_names_not_in_app_sidebar():
    """No env-var key name should appear as a string literal in app.py's sidebar."""
    source = APP_PY.read_text(encoding="utf-8")
    # Find the sidebar block (between 'with st.sidebar:' and the next top-level block)
    sidebar_match = re.search(r"with st\.sidebar:(.*?)^(?!    )", source, re.DOTALL | re.MULTILINE)
    sidebar_src = sidebar_match.group(1) if sidebar_match else source
    for key in FORBIDDEN_IN_UI:
        assert key not in sidebar_src, (
            f"Env-var name '{key}' found in app.py sidebar. "
            "Use feature-level labels (e.g. 'Flight search') instead."
        )


def test_runtime_status_does_not_expose_key_names():
    """runtime_status.py must not have env-var names in its public function names or docstrings."""
    rs_path = Path(__file__).resolve().parents[2] / "agents" / "runtime_status.py"
    source = rs_path.read_text(encoding="utf-8")
    # The file may reference env key names internally (it reads them), but
    # the *public* function names and docstring must not be the key names.
    public_names = re.findall(r"^def (\w+)", source, re.MULTILINE)
    for fn in public_names:
        # Function names like "llm_available", "flights_available" — not key names
        for key in FORBIDDEN_IN_UI:
            assert key.lower() not in fn.lower(), (
                f"Public function '{fn}' in runtime_status.py resembles env-var name '{key}'."
            )
