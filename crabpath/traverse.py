"""Traversal functions for dynamic retrieval over a learned memory graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .graph import Graph


RouteFn = Callable[[str | None, list[str]], list[str]]


@dataclass
class TraversalConfig:
    """Configuration controlling query-time traversal."""

    max_hops: int = 15
    beam_width: int = 3
    edge_damping: float = 0.3
    visit_penalty: float = 0.0
    reflex_threshold: float = 0.8
    habitual_range: tuple[float, float] = (0.3, 0.8)


@dataclass
class TraversalStep:
    """One transition chosen by traversal."""

    from_node: str
    to_node: str
    edge_weight: float
    effective_weight: float
    tier: str


@dataclass
class TraversalResult:
    """Traversal output used by retrieval and learning."""

    fired: list[str]
    steps: list[TraversalStep]
    context: str


def _tier(weight: float, config: TraversalConfig) -> str:
    if weight >= config.reflex_threshold:
        return "reflex"
    if config.habitual_range[0] <= weight < config.habitual_range[1]:
        return "habitual"
    return "dormant"


def traverse(
    graph: Graph,
    seeds: list[tuple[str, float]],
    config: TraversalConfig | None = None,
    route_fn: RouteFn | None = None,
) -> TraversalResult:
    """Traverse graph from seed nodes using edge tiers and fatigue damping.

    Edge tiers:
    - reflex (w >= threshold): auto follow.
    - habitual (within range): follow by weight, or by ``route_fn`` when supplied.
    - dormant: skipped.

    Every directed edge is discounted by ``damping^k`` where ``k`` is how many times
    that edge was used in this traversal episode.
    """
    cfg = config or TraversalConfig()
    if cfg.max_hops <= 0 or cfg.beam_width <= 0:
        return TraversalResult([], [], "")

    valid_seeds = [(node_id, score) for node_id, score in seeds if graph.get_node(node_id)]
    if not valid_seeds:
        return TraversalResult([], [], "")

    frontier: list[tuple[str, float]] = sorted(valid_seeds, key=lambda item: item[1], reverse=True)[: cfg.beam_width]
    seen_counts: dict[str, int] = {node_id: 1 for node_id, _ in frontier}
    fired: list[str] = [node_id for node_id, _ in frontier]
    steps: list[TraversalStep] = []
    used_edges: dict[tuple[str, str], int] = {}

    for _ in range(cfg.max_hops):
        if not frontier:
            break

        raw_candidates: list[tuple[str, str, float, float, str]] = []
        for source_id, source_score in frontier:
            for target_node, edge in graph.outgoing(source_id):
                tier = _tier(edge.weight, cfg)
                if tier == "dormant":
                    continue

                use_count = used_edges.get((source_id, target_node.id), 0)
                effective = edge.weight * (cfg.edge_damping**use_count)
                if effective <= 0.0:
                    continue
                score = source_score * effective
                if target_node.id in seen_counts:
                    score -= cfg.visit_penalty
                raw_candidates.append(
                    (source_id, target_node.id, score, effective, tier)
                )

        if not raw_candidates:
            break

        reflexive = [item for item in raw_candidates if item[4] == "reflex"]
        habitual = [item for item in raw_candidates if item[4] == "habitual"]

        selected: list[tuple[str, str, float, float, str]] = []

        if route_fn is not None and habitual:
            wanted = set(route_fn(None, [target_id for _, target_id, _, _, _ in habitual]))
            if wanted:
                selected.extend(item for item in habitual if item[1] in wanted)
        else:
            selected.extend(sorted(habitual, key=lambda item: item[2], reverse=True))

        selected.extend(sorted(reflexive, key=lambda item: item[2], reverse=True))
        selected = sorted(selected, key=lambda item: item[2], reverse=True)

        next_frontier: list[tuple[str, float]] = []
        target_seen: set[str] = set()

        for source_id, target_id, score, effective, tier in selected[: cfg.beam_width]:
            if target_id in target_seen:
                continue

            edge = graph._edges.get(source_id, {}).get(target_id)
            if edge is None:
                continue

            target_seen.add(target_id)
            if target_id not in seen_counts:
                seen_counts[target_id] = 1
                fired.append(target_id)
            else:
                seen_counts[target_id] += 1

            used_edges[(source_id, target_id)] = used_edges.get((source_id, target_id), 0) + 1
            steps.append(
                TraversalStep(
                    from_node=source_id,
                    to_node=target_id,
                    edge_weight=edge.weight,
                    effective_weight=effective,
                    tier=tier,
                )
            )
            next_frontier.append((target_id, score))

        frontier = next_frontier

    context = "\n\n".join(node.content for node in [graph.get_node(node_id) for node_id in fired] if node)
    return TraversalResult(fired=fired, steps=steps, context=context)
