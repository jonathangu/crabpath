"""Policy updates for graph edges and Hebbian co-firing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .graph import Edge, Graph


@dataclass
class LearningConfig:
    """Hyperparameters for policy-like updates."""

    learning_rate: float = 0.1
    discount: float = 0.95
    hebbian_increment: float = 0.06
    weight_bounds: tuple[float, float] = (-1.0, 1.0)


def _clip_weight(weight: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], weight))


def hebbian_update(
    graph: Graph,
    fired_nodes: list[str],
    config: LearningConfig | None = None,
) -> None:
    """Apply co-firing updates between all observed node pairs.

    Every fired node pair in the trajectory strengthens their edge by
    ``hebbian_increment``. Edges are created if missing.
    """
    cfg = config or LearningConfig()
    if len(fired_nodes) < 2:
        return

    for i, source_id in enumerate(fired_nodes):
        for target_id in fired_nodes[i + 1 :]:
            edge = graph._edges.get(source_id, {}).get(target_id)
            if edge is None:
                graph.add_edge(
                    Edge(
                        source=source_id,
                        target=target_id,
                        weight=cfg.hebbian_increment,
                        kind="sibling",
                    )
                )
            else:
                edge.weight = _clip_weight(
                    edge.weight + cfg.hebbian_increment,
                    cfg.weight_bounds,
                )
                graph._edges[source_id][target_id] = edge


def apply_outcome(
    graph: Graph,
    fired_nodes: list[str],
    outcome: float,
    config: LearningConfig | None = None,
) -> dict:
    """Apply outcome-based policy updates over the full fired trajectory.

    Positive outcome strengthens traversed edges; negative outcome weakens them.
    Negative outcomes may create inhibitory edges when missing.
    Returns a mapping of ``"source->target"`` to weight delta.
    """
    cfg = config or LearningConfig()
    updates: dict[str, float] = defaultdict(float)

    if len(fired_nodes) < 2:
        hebbian_update(graph, fired_nodes, cfg)
        return dict(updates)

    sign = 1.0 if outcome >= 0 else -1.0

    for idx in range(len(fired_nodes) - 1):
        source_id = fired_nodes[idx]
        target_id = fired_nodes[idx + 1]
        edge = graph._edges.get(source_id, {}).get(target_id)
        delta = cfg.learning_rate * (cfg.discount ** (idx + 1)) * sign

        if edge is None:
            graph.add_edge(
                Edge(
                    source=source_id,
                    target=target_id,
                    weight=_clip_weight(delta, cfg.weight_bounds),
                    kind="inhibitory" if sign < 0 else "sibling",
                )
            )
            updates[f"{source_id}->{target_id}"] = delta
            continue

        edge.weight = _clip_weight(edge.weight + delta, cfg.weight_bounds)
        if sign < 0 and edge.weight <= 0:
            edge.kind = "inhibitory"
        graph._edges[source_id][target_id] = edge
        updates[f"{source_id}->{target_id}"] = delta

    hebbian_update(graph, fired_nodes, cfg)
    return dict(updates)
