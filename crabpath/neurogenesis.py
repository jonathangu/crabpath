"""Neurogenesis heuristics for automatic concept node creation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1

from .graph import Edge, Graph


BLOCKED_QUERIES = {
    "hello",
    "thanks",
    "yes",
    "no",
    "ok",
    "hi",
    "bye",
    "got it",
    "sure",
    "yeah",
}


@dataclass(frozen=True)
class NoveltyResult:
    should_create: bool
    top_score: float
    band: str
    blocked: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class NeurogenesisConfig:
    known_threshold: float = 0.60
    novel_min: float = 0.28
    novel_max: float = 0.58
    noise_threshold: float = 0.28
    min_tokens: int = 3


def _normalize(text: str) -> str:
    return " ".join(token.lower() for token in re.findall(r"[a-zA-Z0-9']+", text))


def deterministic_auto_id(query_text: str) -> str:
    normalized = _normalize(query_text)
    digest = sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"auto:{digest}"


def assess_novelty(
    query_text: str,
    raw_scores: list[tuple[str, float]],
    config: NeurogenesisConfig = NeurogenesisConfig(),
) -> NoveltyResult:
    """Assess novelty from raw cosine hits and simple quality gates."""
    normalized = _normalize(query_text)
    token_count = len(normalized.split()) if normalized else 0

    if token_count < config.min_tokens:
        return NoveltyResult(
            should_create=False,
            top_score=0.0,
            band="blocked",
            blocked=True,
            reason="too_few_tokens",
        )

    if normalized in BLOCKED_QUERIES:
        return NoveltyResult(
            should_create=False,
            top_score=0.0,
            band="blocked",
            blocked=True,
            reason="blocked_phrase",
        )

    top_score = raw_scores[0][1] if raw_scores else 0.0

    if top_score >= config.known_threshold:
        return NoveltyResult(
            should_create=False,
            top_score=top_score,
            band="known",
        )

    if top_score < config.noise_threshold:
        return NoveltyResult(
            should_create=False,
            top_score=top_score,
            band="noise",
        )

    if config.novel_min <= top_score <= config.novel_max:
        return NoveltyResult(
            should_create=True,
            top_score=top_score,
            band="novel",
        )

    return NoveltyResult(
        should_create=False,
        top_score=top_score,
        band="novelty_miss",
        reason="outside_novel_band",
    )


def _weight_for_seed(seed_id: str, weights: float | dict[str, float]) -> float:
    if isinstance(weights, dict):
        return float(weights.get(seed_id, 0.15))
    return float(weights)


def connect_new_node(
    graph: Graph,
    new_node_id: str,
    current_seed_ids: list[str],
    weights: float | dict[str, float],
) -> None:
    """Create incoming connections from seeds to the new node."""
    if not current_seed_ids:
        return
    for seed_id in current_seed_ids:
        if seed_id == new_node_id:
            continue
        graph.add_edge(
            Edge(
                source=seed_id,
                target=new_node_id,
                weight=_weight_for_seed(seed_id, weights),
            )
        )
