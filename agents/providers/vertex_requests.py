"""Llama and Mistral backend via Vertex AI.

Two different endpoint patterns (per Vertex AI Model Garden docs):

  Llama 3.3 — OpenAI-compatible MaaS endpoint:
    https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}
      /locations/{REGION}/endpoints/openapi/chat/completions
    Body "model" field: "meta/llama-3.3-70b-instruct-maas"

  Mistral Small 3.1 — rawPredict endpoint:
    https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}
      /locations/{REGION}/publishers/mistralai/models/mistral-small-2503:rawPredict
    Body "model" field: "mistral-small-2503" (no publisher prefix; it's in the URL)

Auth: Application Default Credentials (ADC) bearer token via google-auth.
No new Python packages — uses `requests` (already in requirements.txt) and
`google-auth` (transitive via google-genai).
"""
from __future__ import annotations

import os

import requests as _requests

try:
    import google.auth
    import google.auth.transport.requests
    _google_auth_available = True
except ImportError:
    _google_auth_available = False

API_TIMEOUT_SECONDS = 120.0

# Env var that controls the Vertex region for each provider.
_REGION_ENV = {
    "meta":    "META_VERTEX_REGION",
    "mistral": "MISTRAL_VERTEX_REGION",
}

# Default region per provider when the env var is unset.
_DEFAULT_REGION = {
    "meta":    "us-central1",
    "mistral": "us-central1",
}


def _project() -> str:
    p = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not p:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT for Llama/Mistral on Vertex AI. "
            "See .env.example for required variables."
        )
    return p


def _region(provider: str) -> str:
    env_key = _REGION_ENV.get(provider, "META_VERTEX_REGION")
    default = _DEFAULT_REGION.get(provider, "us-central1")
    return os.environ.get(env_key, default).strip() or default


def _access_token() -> str:
    """Return a fresh ADC bearer token, refreshing if expired."""
    if not _google_auth_available:
        raise RuntimeError(
            "google-auth is not installed. "
            "Run: pip install google-auth google-auth-httplib2"
        )
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


def _endpoint_url(model_id: str, provider: str) -> str:
    """Return the correct Vertex AI endpoint URL for the given provider.

    Llama uses the shared OpenAI-compatible MaaS endpoint.
    Mistral uses its own rawPredict endpoint under publishers/mistralai.
    """
    region = _region(provider)
    project = _project()
    base = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}"
    if provider == "meta":
        return f"{base}/endpoints/openapi/chat/completions"
    else:  # mistral
        return f"{base}/publishers/mistralai/models/{model_id}:rawPredict"


def _body_model_id(model_id: str, provider: str) -> str:
    """Return the model ID string for the request body.

    Llama's OpenAI-compat endpoint requires the publisher prefix.
    Mistral's rawPredict endpoint encodes the model in the URL, so the body
    uses the bare model ID.
    """
    if provider == "meta":
        return f"meta/{model_id}"
    return model_id  # mistral: publisher already in URL


def clear_client_cache() -> None:
    """No-op — no persistent client. Exists for providers/__init__.py parity."""
    pass


def generate(
    model_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    json_mode: bool,
    provider: str,
) -> tuple[str, int, int]:
    """Call Llama or Mistral on Vertex AI.

    Returns (text, input_tokens, output_tokens).
    Llama uses the OpenAI-compatible MaaS endpoint; Mistral uses rawPredict.
    response_format=json_object is sent only for Llama; Mistral doesn't support
    it, so plain-text JSON is relied on and parsed by _extract_json in llm.py.
    """
    url = _endpoint_url(model_id, provider)
    token = _access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body: dict = {
        "model": _body_model_id(model_id, provider),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
    }
    # Mistral on Vertex AI does not support response_format.
    if json_mode and provider == "meta":
        body["response_format"] = {"type": "json_object"}

    resp = _post(url, headers, body)
    _raise_for_status(resp, provider)

    data = resp.json()
    text = data["choices"][0]["message"]["content"]

    if not text or not text.strip():
        raise RuntimeError(f"{provider.upper()} model returned empty response")

    usage  = data.get("usage") or {}
    in_tok  = int(usage.get("prompt_tokens")    or 0)
    out_tok = int(usage.get("completion_tokens") or 0)

    return text, in_tok, out_tok


def _post(url: str, headers: dict, body: dict) -> _requests.Response:
    try:
        return _requests.post(
            url,
            json=body,
            headers=headers,
            timeout=API_TIMEOUT_SECONDS,
        )
    except _requests.Timeout as e:
        err = RuntimeError(
            f"Vertex MaaS request timed out after {API_TIMEOUT_SECONDS}s: {e}"
        )
        setattr(err, "status_code", 504)
        raise err from e
    except _requests.ConnectionError as e:
        err = RuntimeError(f"Vertex MaaS connection error: {e}")
        setattr(err, "status_code", 503)
        raise err from e


def _raise_for_status(resp: _requests.Response, provider: str = "") -> None:
    if resp.status_code < 400:
        return
    status = resp.status_code
    try:
        payload = resp.json()
        # Vertex rawPredict wraps errors in a list: [{error: {...}}]
        if isinstance(payload, list) and payload:
            payload = payload[0]
        detail = payload.get("error", {}).get("message", resp.text[:400])
    except Exception:
        detail = resp.text[:400]
    region_env = _REGION_ENV.get(provider, "META_VERTEX_REGION")
    if status == 429:
        err = RuntimeError(
            f"Vertex MaaS quota exhausted (429): {detail}. "
            f"Switch region via {region_env} or request a quota increase."
        )
    elif status == 401:
        err = RuntimeError(
            f"Vertex MaaS auth failed (401): {detail}. "
            "Run `gcloud auth application-default login` or check GOOGLE_APPLICATION_CREDENTIALS."
        )
    elif status == 403:
        err = RuntimeError(
            f"Vertex MaaS permission denied (403): {detail}. "
            "Enable the model in Vertex AI Model Garden and ensure roles/aiplatform.user is granted."
        )
    else:
        err = RuntimeError(f"Vertex MaaS request failed ({status}): {detail}")
    setattr(err, "status_code", status)
    raise err
