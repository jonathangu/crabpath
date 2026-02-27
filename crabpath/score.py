"""Scoring helpers for optional LLM-powered retrieval quality estimation."""

from __future__ import annotations

from collections.abc import Callable

from ._batch import batch_or_single
from ._util import _extract_json

def _coerce_score(value: object) -> float:
    """ coerce score."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 1.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def score_retrieval(
    query: str,
    fired_nodes: list[tuple[str, str]],
    llm_fn: Callable[[str, str], str] | None = None,
    llm_batch_fn: Callable[[list[dict]], list[dict]] | None = None,
) -> dict[str, float]:
    """Score how useful each fired node was for the query.

    Returns ``{node_id: score}`` where score is 0.0-1.0.
    """
    if not fired_nodes:
        return {}
    defaults = {node_id: 1.0 for node_id, _ in fired_nodes}
    if llm_fn is None and llm_batch_fn is None:
        return defaults

    lines = [f"- {node_id}: {(content or '').replace(chr(10), ' ')[:240]}" for node_id, content in fired_nodes]
    user = f"Query: {query}\n\nRetrieved chunks:\n{chr(10).join(lines)}"
    system = (
        'Score how useful each document chunk was for answering this query. For each chunk, return a '
        'score from 0.0 (useless) to 1.0 (essential). Return JSON: {"scores": {"node_id": float}}'
    )
    try:
        responses = batch_or_single(
            [{"id": "request", "system": system, "user": user}],
            llm_fn=llm_fn,
            llm_batch_fn=llm_batch_fn,
        )
        if not responses:
            return defaults
        payload = _extract_json(responses[0].get("response", ""))  # type: ignore[index]
        if not isinstance(payload, dict):
            return defaults
        raw_scores = payload.get("scores", {})
        if not isinstance(raw_scores, dict):
            return defaults

        scores = defaults.copy()
        for node_id, value in raw_scores.items():
            if node_id in scores:
                scores[node_id] = _coerce_score(value)
        return scores
    except (Exception, SystemExit):
        return defaults
