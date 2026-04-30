"""Llama and Mistral backend via Vertex AI OpenAI-compatible MaaS endpoint.

Correct endpoint pattern (Vertex AI docs):
  https://{HOST}/v1/projects/{PROJECT}/locations/{REGION}/endpoints/openapi/chat/completions

  HOST = {REGION}-aiplatform.googleapis.com  (e.g. us-central1-aiplatform.googleapis.com)

Model ID in request body must carry the publisher prefix:
  Llama 3.3   → "meta/llama-3.3-70b-instruct-maas"
  Mistral 3.1 → "mistralai/mistral-small-2503"

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

# Publisher prefix used in the request body "model" field.
_BODY_MODEL_PREFIX = {
    "meta":    "meta",
    "mistral": "mistralai",
}

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
            "Set GOOGLE_CLOUD_PROJECT for Llama/Mistral/GLM on Vertex AI. "
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


def _endpoint_url(provider: str) -> str:
    """OpenAI-compatible MaaS chat completions URL for the given provider."""
    region = _region(provider)
    project = _project()
    return (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{region}/endpoints/openapi/chat/completions"
    )


def _body_model_id(model_id: str, provider: str) -> str:
    """Return the model ID with publisher prefix for the request body.

    Vertex AI MaaS requires the publisher prefix, e.g.:
      "meta/llama-3.3-70b-instruct-maas"
      "mistralai/mistral-small-2503"
      "zai-org/glm-5-maas"
    """
    prefix = _BODY_MODEL_PREFIX.get(provider)
    if prefix is None:
        raise ValueError(f"Unknown provider {provider!r} for vertex_requests backend")
    return f"{prefix}/{model_id}"


def clear_client_cache() -> None:
    """No-op — no persistent client. Exists for providers/__init__.py parity."""
    pass


def _build_messages(system: str, user: str, provider: str) -> list[dict]:
    """Build the messages list for the chat request.

    GLM-5 (ZhipuAI) doesn't reliably handle the system role on Vertex AI —
    it silently returns empty content when a system message is present.
    For glm, merge the system prompt into the user turn instead.
    """
    if provider == "glm":
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def generate(
    model_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    json_mode: bool,
    provider: str,
) -> tuple[str, int, int]:
    """Call Llama, Mistral, or GLM-5 on Vertex AI.

    Returns (text, input_tokens, output_tokens).
    Posts to /endpoints/openapi/chat/completions with the publisher-prefixed
    model ID. When json_mode=True, requests response_format=json_object for
    Llama/Mistral; GLM-5 skips response_format entirely (unsupported).
    Mistral falls back silently if the model rejects response_format (HTTP 400).
    """
    url = _endpoint_url(provider)
    token = _access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body: dict = {
        "model": _body_model_id(model_id, provider),
        "messages": _build_messages(system, user, provider),
        "max_tokens": max_tokens,
    }
    # GLM-5 doesn't support response_format; skip it to avoid empty responses.
    if json_mode and provider != "glm":
        body["response_format"] = {"type": "json_object"}

    resp = _post(url, headers, body)

    # Mistral may reject response_format with HTTP 400; retry once without it.
    if resp.status_code == 400 and json_mode:
        err_text = resp.text.lower()
        if "response_format" in err_text or "not supported" in err_text:
            body_no_fmt = {k: v for k, v in body.items() if k != "response_format"}
            resp = _post(url, headers, body_no_fmt)

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
        detail = resp.json().get("error", {}).get("message", resp.text[:400])
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
