"""Deterministic runtime routing policy for daemon traversal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .index import VectorIndex
from .protocol import parse_float, parse_int, parse_route_mode
from .route_model import RouteModel


@dataclass(frozen=True)
class RoutingPolicy:
    """Runtime edge routing options for habitual traversal."""

    route_mode: str = "off"
    top_k: int = 5
    alpha_sim: float = 0.5
    use_relevance: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        route_mode: object,
        top_k: object,
        alpha_sim: object,
        use_relevance: object,
    ) -> "RoutingPolicy":
        """Validate and normalize policy fields."""
        return cls(
            route_mode=parse_route_mode(route_mode),
            top_k=parse_int(top_k, "route_top_k", default=5),
            alpha_sim=parse_float(alpha_sim, "route_alpha_sim", default=0.5),
            use_relevance=True if use_relevance is None else _parse_use_relevance(use_relevance),
        )


@dataclass(frozen=True)
class DecisionMetrics:
    """Per-source learned-routing decision telemetry."""

    router_entropy: float
    router_conf: float
    router_margin: float
    relevance_entropy: float
    relevance_conf: float
    chosen_edge: tuple[str, str]
    graph_prior_score: float
    router_score_raw: float
    final_score: float
    candidate_count: int
    policy_disagreement: float


def _parse_use_relevance(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("route_use_relevance must be a boolean")
    return value


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    denom = sum(exp_values)
    if denom <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / denom for value in exp_values]


def _normalized_entropy(probs: list[float]) -> float:
    n = len(probs)
    if n <= 1:
        return 0.0
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
    normalized = entropy / math.log(float(n))
    return max(0.0, min(1.0, normalized))


def _margin(probs: list[float]) -> float:
    if not probs:
        return 0.0
    if len(probs) == 1:
        return 1.0
    ordered = sorted(probs, reverse=True)
    return max(0.0, min(1.0, ordered[0] - ordered[1]))


def _confidence(values: list[float]) -> tuple[float, float, float]:
    probs = _softmax(values)
    entropy = _normalized_entropy(probs)
    margin = _margin(probs)
    conf = margin if len(values) <= 3 else (1.0 - entropy)
    return entropy, max(0.0, min(1.0, conf)), margin


def _constant_feature_vector(df: int) -> list[float]:
    if df <= 0:
        raise ValueError("route model feature dimension must be positive")
    values = [0.0] * df
    values[-1] = 1.0
    return values


def make_runtime_route_fn(
    *,
    policy: RoutingPolicy,
    query_vector: list[float],
    index: VectorIndex,
    learned_model: RouteModel | None = None,
    target_projections: dict[str, object] | None = None,
    decision_log: list[DecisionMetrics] | None = None,
) -> Callable[[str | None, list[object], str], list[str]] | None:
    """Build deterministic local route policy for habitual candidates."""
    if policy.route_mode == "off":
        return None
    if policy.route_mode == "learned":
        if learned_model is None:
            return None
        return make_learned_route_fn(
            model=learned_model,
            target_projections=target_projections or {},
            index=index,
            config=policy,
            query_vector=query_vector,
            decision_log=decision_log,
        )

    use_similarity = policy.route_mode == "edge+sim"

    def _score(edge: object) -> float:
        # `traverse` passes graph.Edge values; keep this function generic for tests.
        weight = float(getattr(edge, "weight", 0.0))
        relevance = 0.0
        if policy.use_relevance:
            metadata = getattr(edge, "metadata", None)
            if isinstance(metadata, dict):
                raw_relevance = metadata.get("relevance", 0.0)
                if isinstance(raw_relevance, (int, float)):
                    relevance = float(raw_relevance)

        similarity = 0.0
        if use_similarity:
            target_id = str(getattr(edge, "target", ""))
            target_vector = index._vectors.get(target_id)
            if target_vector is not None:
                similarity = VectorIndex.cosine(query_vector, target_vector)

        return weight + relevance + (policy.alpha_sim * similarity)

    def _route_fn(_source_id: str | None, candidates: list[object], _query_text: str) -> list[str]:
        ranked = sorted(
            ((str(getattr(edge, "target", "")), _score(edge)) for edge in candidates),
            key=lambda item: (-item[1], item[0]),
        )
        return [target_id for target_id, _score_value in ranked[: policy.top_k] if target_id]

    return _route_fn


def make_learned_route_fn(
    model: RouteModel,
    target_projections: dict[str, object],
    index: VectorIndex,
    config: RoutingPolicy,
    query_vector: list[float],
    decision_log: list[DecisionMetrics] | None = None,
) -> Callable[[str | None, list[object], str], list[str]]:
    """Build deterministic learned route function using low-rank scoring."""
    q_proj = model.project_query(query_vector)
    feat_vec = _constant_feature_vector(model.df)

    def _target_proj(target_id: str):
        cached = target_projections.get(target_id)
        if cached is not None:
            return cached
        raw = index._vectors.get(target_id)
        if raw is None:
            return None
        projected = model.project_target(raw)
        target_projections[target_id] = projected
        return projected

    def _route_fn(_source_id: str | None, candidates: list[object], _query_text: str) -> list[str]:
        source_id = _source_id or ""
        scored: list[tuple[str, float, float, float]] = []
        for edge in candidates:
            target_id = str(getattr(edge, "target", ""))
            if not target_id:
                continue
            t_proj = _target_proj(target_id)
            if t_proj is None:
                continue
            weight = float(getattr(edge, "weight", 0.0))
            metadata = getattr(edge, "metadata", None)
            relevance = 0.0
            if isinstance(metadata, dict):
                raw_relevance = metadata.get("relevance", 0.0)
                if isinstance(raw_relevance, (int, float)):
                    relevance = float(raw_relevance)
            router_score = model.score_projected(q_proj, t_proj, feat_vec)
            scored.append((target_id, weight, relevance, router_score))

        if not scored:
            return []

        relevances = [relevance for _target_id, _weight, relevance, _router_score in scored]
        router_scores = [router_score for _target_id, _weight, _relevance, router_score in scored]
        relevance_entropy, relevance_conf, _relevance_margin = _confidence(relevances)
        router_entropy, router_conf, router_margin = _confidence(router_scores)

        ranked: list[tuple[str, float, float, float]] = []
        for target_id, weight, relevance, router_score in scored:
            graph_prior = (relevance_conf * relevance) + ((1.0 - relevance_conf) * weight)
            final_score = (router_conf * router_score) + ((1.0 - router_conf) * graph_prior)
            ranked.append((target_id, graph_prior, router_score, final_score))

        ranked.sort(key=lambda item: (-item[3], item[0]))

        if decision_log is not None and ranked:
            chosen_target, chosen_graph_prior, chosen_router, chosen_final = ranked[0]
            decision_log.append(
                DecisionMetrics(
                    router_entropy=router_entropy,
                    router_conf=router_conf,
                    router_margin=router_margin,
                    relevance_entropy=relevance_entropy,
                    relevance_conf=relevance_conf,
                    chosen_edge=(source_id, chosen_target),
                    graph_prior_score=chosen_graph_prior,
                    router_score_raw=chosen_router,
                    final_score=chosen_final,
                    candidate_count=len(ranked),
                    policy_disagreement=abs(chosen_router - chosen_graph_prior),
                )
            )

        return [target_id for target_id, _graph_prior, _router, _final in ranked[: config.top_k]]

    return _route_fn
