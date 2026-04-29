"""LLM facade — routes to the appropriate provider backend.

Provider is determined by agents/models.py based on the active model.
Active model is set via agents/runtime.use_model() or the LLM_MODEL env var.

Public API (backward-compatible):
    call_json(system, user, schema_cls, *, max_tokens, think_first) -> T
    call_text(system, user, *, max_tokens) -> str
    reset_call_stats() -> None
    get_call_stats() -> dict[str, int]   # aggregate; adds cache_hits / cache_misses
"""
from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


T = TypeVar("T", bound=BaseModel)


class TruncationError(RuntimeError):
    """Raised when the model response is truncated before the JSON object closes."""


# ── efficiency tracking ──────────────────────────────────────────────────────

_stats_lock = threading.Lock()
_stats: dict[str, int] = {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}


def reset_call_stats() -> None:
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0


def get_call_stats() -> dict[str, int]:
    with _stats_lock:
        return dict(_stats)


# ── in-memory LRU cache ──────────────────────────────────────────────────────

_cache_lock = threading.Lock()
_json_cache: dict[str, str] = {}
_cache_order: deque[str] = deque()
_CACHE_SIZE = int(os.environ.get("LLM_CACHE_SIZE", "256"))


def _get_disk_cache_dir() -> str:
    """Read at call time so LLM_CACHE_DIR set after import is picked up."""
    return os.environ.get("LLM_CACHE_DIR", "").strip()


def _disk_cache_path(key: str) -> Path | None:
    d = _get_disk_cache_dir()
    if not d:
        return None
    return Path(d) / f"{key}.json"


def _disk_cache_get(key: str) -> str | None:
    path = _disk_cache_path(key)
    if path is None or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _disk_cache_put(key: str, value: str) -> None:
    path = _disk_cache_path(key)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    except OSError:
        pass


def _cache_key(model_id: str, system: str, user: str, schema_name: str, max_tokens: int, think_first: bool) -> str:
    provider = _provider_for(model_id)
    raw = f"{provider}|{model_id}|{schema_name}|{max_tokens}|{think_first}|{system}|{user}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    with _cache_lock:
        hit = _json_cache.get(key)
    if hit is not None:
        return hit
    disk_hit = _disk_cache_get(key)
    if disk_hit is not None:
        _cache_put(key, disk_hit, _write_disk=False)
    return disk_hit


def _cache_put(key: str, value: str, *, _write_disk: bool = True) -> None:
    if _CACHE_SIZE <= 0 and not _get_disk_cache_dir():
        return
    with _cache_lock:
        if _CACHE_SIZE > 0:
            if key not in _json_cache:
                _cache_order.append(key)
                while len(_cache_order) > _CACHE_SIZE:
                    old = _cache_order.popleft()
                    _json_cache.pop(old, None)
            _json_cache[key] = value
    if _write_disk:
        _disk_cache_put(key, value)


# ── retry ─────────────────────────────────────────────────────────────────────

_MAX_ATTEMPTS = 3
# 429 quota exhaustion needs much longer waits; Vertex TPM buckets refill over ~60s.
_MAX_ATTEMPTS_429 = 6
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRY_HINTS = (
    "rate_limit",
    "too many requests",
    "high demand",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUSES:
        return True
    return any(h in str(exc).lower() for h in _RETRY_HINTS)


def _parse_retry_after(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if not val:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int) -> float:
    return min(2 ** attempt, 12) + random.uniform(0, 0.75)


def _backoff_429(attempt: int) -> float:
    # 15s, 30s, 60s, 90s, 90s, 90s — ~6 min total; gives Vertex TPM quota time to refill.
    return min(15 * (2 ** attempt), 90) + random.uniform(0, 5)


def _with_retry(fn, *, max_attempts: int = _MAX_ATTEMPTS):
    last_exc: Exception | None = None
    effective_max = max_attempts
    attempt = 0
    while attempt < effective_max:
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable(e):
                raise
            status = getattr(e, "status_code", None)
            # On first 429 hit, escalate to the longer quota-recovery policy.
            if status == 429 and effective_max == max_attempts:
                effective_max = _MAX_ATTEMPTS_429
            delay = _parse_retry_after(e) or (
                _backoff_429(attempt) if status == 429 else _backoff(attempt)
            )
            time.sleep(delay)
            attempt += 1
    raise last_exc  # type: ignore[misc]


# ── provider helpers ──────────────────────────────────────────────────────────

def _active_model() -> str:
    from agents.runtime import current_model
    return current_model()


def _provider_for(model_id: str) -> str:
    from agents.models import get_model
    return get_model(model_id).provider


# ── public API ────────────────────────────────────────────────────────────────

def call_json(
    system: str,
    user: str,
    schema_cls: type[T],
    *,
    max_tokens: int = 1024,
    think_first: bool = False,
) -> T:
    """Structured-output call. Returns a validated Pydantic instance.

    When think_first=True the model is asked to reason through constraints before
    producing the JSON answer. The response is still parsed as JSON.
    """
    effective_max = max(max_tokens, 6000) if think_first else max_tokens
    model_id = _active_model()
    key = _cache_key(model_id, system, user, schema_cls.__name__, effective_max, think_first)

    cached = _cache_get(key)
    if cached is not None:
        try:
            result = schema_cls.model_validate_json(cached)
            with _stats_lock:
                _stats["cache_hits"] += 1
            return result
        except ValidationError:
            with _cache_lock:
                _json_cache.pop(key, None)
                try:
                    _cache_order.remove(key)
                except ValueError:
                    pass
            disk_path = _disk_cache_path(key)
            if disk_path is not None and disk_path.exists():
                try:
                    disk_path.unlink()
                except OSError:
                    pass

    if think_first:
        prompt_system = (
            system
            + "\n\nThink step by step: reason through the constraints and "
            "available options before deciding. Then output your final "
            "answer as a single JSON object."
        )
        use_json_mode = False
    else:
        prompt_system = system + "\n\nRespond with a single JSON object, no prose."
        use_json_mode = True

    with _stats_lock:
        _stats["llm_calls"] += 1
        _stats["cache_misses"] += 1

    provider = _provider_for(model_id)
    _MAX_TOKENS_CAP = 32768
    cur_max = [effective_max]  # mutable so the retry loop can escalate on truncation

    def do_call() -> str:
        from agents.providers import call_provider
        from agents.models import get_model
        model_entry = get_model(model_id)

        if provider == "claude":
            text, in_tok, out_tok = call_provider(
                model_entry, prompt_system, user,
                max_tokens=cur_max[0],
                json_mode=True,
                schema_cls=schema_cls,
            )
        else:
            text, in_tok, out_tok = call_provider(
                model_entry, prompt_system, user,
                max_tokens=cur_max[0],
                json_mode=use_json_mode,
                schema_cls=None,
            )
            text = _extract_json(text)

        with _stats_lock:
            _stats["input_tokens"] += in_tok
            _stats["output_tokens"] += out_tok

        try:
            schema_cls.model_validate_json(text)
        except ValidationError as ve:
            err = RuntimeError(
                f"LLM response failed {schema_cls.__name__} schema validation: {ve}. Retrying."
            )
            setattr(err, "status_code", 503)
            raise err
        return text

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            raw = do_call()
            break
        except TruncationError as e:
            last_exc = e
            # Double the token budget for the next attempt; don't sleep.
            cur_max[0] = min(cur_max[0] * 2, _MAX_TOKENS_CAP)
        except Exception as e:
            last_exc = e
            if not _is_retryable(e):
                raise
            delay = _parse_retry_after(e) or _backoff(attempt)
            time.sleep(delay)
    else:
        raise last_exc  # type: ignore[misc]

    _cache_put(key, raw)
    return schema_cls.model_validate_json(raw)


def call_text(system: str, user: str, *, max_tokens: int = 2048) -> str:
    """Unstructured call for the single-agent baseline's ReAct loop."""
    model_id = _active_model()

    from agents.providers import call_provider
    from agents.models import get_model
    model_entry = get_model(model_id)

    def do_call() -> str:
        text, in_tok, out_tok = call_provider(
            model_entry, system, user,
            max_tokens=max_tokens,
            json_mode=False,
            schema_cls=None,
        )
        with _stats_lock:
            _stats["input_tokens"] += in_tok
            _stats["output_tokens"] += out_tok
        return text

    with _stats_lock:
        _stats["llm_calls"] += 1
        _stats["cache_misses"] += 1

    return _with_retry(do_call)


def _extract_json(text: str) -> str:
    """Grab the first balanced {...} from a model response.

    Raises RuntimeError (retryable) when no JSON is found or response is truncated.
    """
    start = text.find("{")
    if start < 0:
        err = RuntimeError(
            "LLM response contained no JSON object — model produced only prose "
            f"(got {len(text)} chars). Retrying."
        )
        setattr(err, "status_code", 503)
        raise err
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body = text[start: i + 1]
                if body.replace(" ", "").replace("\n", "") == "{}":
                    err = RuntimeError(
                        "LLM returned an empty JSON object ({}); required fields missing. Retrying."
                    )
                    setattr(err, "status_code", 503)
                    raise err
                return body
    err = TruncationError(
        "LLM response was truncated before the JSON object was closed "
        f"(got {len(text)} chars). Increase max_tokens or reduce input size."
    )
    setattr(err, "status_code", 503)
    raise err
