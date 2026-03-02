"""Deterministic runtime routing policy for daemon traversal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .index import VectorIndex
from .protocol import parse_float, parse_int, parse_route_mode


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



def _parse_use_relevance(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("route_use_relevance must be a boolean")
    return value



def make_runtime_route_fn(
    *,
    policy: RoutingPolicy,
    query_vector: list[float],
    index: VectorIndex,
) -> Callable[[str | None, list[object], str], list[str]] | None:
    """Build deterministic local route policy for habitual candidates."""
    if policy.route_mode == "off":
        return None

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
