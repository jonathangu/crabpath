"""OpenAI chat callbacks for optional LLM-assisted workflows."""

from __future__ import annotations

import os
import sys
import threading


_OPENAI_TIMEOUT_SECONDS = 60
_OPENAI_MAX_RETRIES = 2
_thread_local = threading.local()


def _extract_unsupported_kwarg(exc: TypeError) -> str | None:
    """Return a clearly unsupported keyword argument from an OpenAI TypeError."""
    message = str(exc).lower()
    if "unexpected keyword argument" not in message:
        return None
    for candidate in ("timeout", "max_retries"):
        if f"'{candidate}'" in message:
            return candidate
    return None


def _is_timeout_kwarg_error(exc: TypeError) -> bool:
    """Return True when the TypeError is specifically about timeout."""
    return _extract_unsupported_kwarg(exc) == "timeout"


def _openai_client_kwargs() -> dict[str, object]:
    return {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "timeout": _OPENAI_TIMEOUT_SECONDS,
        "max_retries": _OPENAI_MAX_RETRIES,
    }


def _build_openai_client_with_fallback(openai_module: object, kwargs: dict[str, object]):
    """Build an OpenAI client while preserving compatibility with older signatures."""
    try:
        return openai_module.OpenAI(**kwargs)
    except TypeError as exc:
        unsupported = _extract_unsupported_kwarg(exc)
        if unsupported is None or unsupported not in kwargs:
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop(unsupported)
        return _build_openai_client_with_fallback(openai_module, fallback_kwargs)


def _get_client():
    """Create a lazy OpenAI client on first use."""
    client = getattr(_thread_local, "client", None)
    if client is not None:
        return client

    import openai

    client = _build_openai_client_with_fallback(openai, _openai_client_kwargs())
    _thread_local.client = client
    return client


def openai_llm_fn(system: str, user: str) -> str:
    """Run a single OpenAI chat request."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=_OPENAI_TIMEOUT_SECONDS,
        )
    except TypeError as exc:
        # Backwards compatibility: some OpenAI clients may not support the timeout kwarg.
        if not _is_timeout_kwarg_error(exc):
            raise
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    if not response.choices:
        return ""
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is None or not isinstance(message.content, str):
        return ""
    return message.content


def chat_completion(system: str, user: str) -> str:
    """Backward-compatible wrapper for single completion calls."""
    return openai_llm_fn(system, user)


def openai_llm_batch_fn(requests: list[dict], *, max_workers: int = 8) -> list[dict]:
    """Run OpenAI chat requests concurrently.

    This is used by `batch_or_single()` when an explicit batch fn is provided.
    Historically this implementation was sequential, which made `init --llm openai`
    take hours for medium workspaces.

    Args:
        requests: list of {id, system, user}
        max_workers: thread pool size
    """
    if not requests:
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(requests)
    results: list[dict] = []

    def _call(req: dict) -> dict:
        system = str(req.get("system", ""))
        user = str(req.get("user", ""))
        request_id = req.get("id")
        try:
            response = openai_llm_fn(system, user)
        except Exception as exc:  # noqa: BLE001
            return {"id": request_id, "response": "", "error": str(exc)}
        return {"id": request_id, "response": response}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call, req) for req in requests]
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"LLM batch progress: processing {completed}/{total}", file=sys.stderr)
            results.append(fut.result())

    return results
