"""OpenAI chat callbacks for optional LLM-assisted workflows."""

from __future__ import annotations

import os
import sys


_OPENAI_TIMEOUT_SECONDS = 60

_client = None


def _get_client():
    """Create a lazy OpenAI client on first use."""
    global _client
    if _client is not None:
        return _client

    import openai

    _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


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
    except TypeError:
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

    def _call(idx: int, req: dict) -> dict:
        system = str(req.get("system", ""))
        user = str(req.get("user", ""))
        request_id = req.get("id")
        try:
            response = openai_llm_fn(system, user)
        except Exception as exc:  # noqa: BLE001
            return {"id": request_id, "response": "", "error": str(exc)}
        return {"id": request_id, "response": response}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call, i, req): i for i, req in enumerate(requests, start=1)}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"LLM batch progress: processing {completed}/{total}", file=sys.stderr)
            results.append(fut.result())

    return results
