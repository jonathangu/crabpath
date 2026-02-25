"""Offline consolidation operations for graph maintenance."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .graph import Graph, Edge, Node


@dataclass
class ConsolidationConfig:
    min_edge_weight: float = 0.03
    max_node_chars: int = 1500
    min_fires_to_split: int = 10
    merge_cosine_threshold: float = 0.90
    merge_cofire_threshold: float = 0.80
    probation_max_turns: int = 50


@dataclass
class ConsolidationResult:
    edges_pruned: int
    nodes_pruned: int
    nodes_split: int
    nodes_merged: int


def prune_weak_edges(graph: Graph, min_weight: float) -> int:
    pruned = 0
    for edge in list(graph.edges()):
        if abs(edge.weight) >= min_weight:
            continue
        if graph._remove_edge(edge.source, edge.target):
            pruned += 1
    return pruned


def prune_orphan_nodes(graph: Graph, protected_ids: set[str] | None = None) -> int:
    protected = set(protected_ids or [])
    pruned = 0
    orphan_ids = [node.id for node in graph.nodes() if not graph._incoming.get(node.id)]
    for node_id in orphan_ids:
        if graph.is_node_protected(node_id) or node_id in protected:
            continue
        graph.remove_node(node_id)
        pruned += 1
    return pruned


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_probationary(metadata: dict) -> bool:
    return bool(metadata.get("probationary")) or bool(metadata.get("auto_probationary"))


def prune_probationary(graph: Graph, max_turns: int) -> int:
    now = time.time()
    pruned = 0
    for node in list(graph.nodes()):
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        if not _is_probationary(metadata):
            continue
        if _as_int(metadata.get("fired_count")) >= 3:
            continue

        created_ts = _as_float(metadata.get("created_ts"), default=0.0)
        if now - created_ts <= max_turns:
            continue
        graph.remove_node(node.id)
        pruned += 1
    return pruned


def should_split(node: Node, config: ConsolidationConfig) -> bool:
    if not isinstance(node, Node):
        return False
    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    fires = _as_int(metadata.get("fired_count"), default=0)
    return len(node.content) > config.max_node_chars and fires > config.min_fires_to_split


def split_node(graph: Graph, node_id: str, parts: list[dict]) -> list[str]:
    parent = graph.get_node(node_id)
    if parent is None:
        return []

    now = time.time()
    child_ids: list[str] = []

    parent_outgoing = list(graph._outgoing.get(node_id, []))
    for part in parts:
        child_id = str(part.get("id", f"{node_id}:split"))
        idx = 1
        while graph.get_node(child_id) is not None:
            child_id = f"{child_id}:{idx}"
            idx += 1

        child = Node(
            id=child_id,
            content=str(part.get("content", "")),
            summary=str(part.get("summary", "")),
            type=parent.type,
            metadata={
                "fired_count": 0,
                "last_fired_ts": 0.0,
                "created_ts": now,
                "protected": bool(parent.metadata.get("protected"))
                if isinstance(parent.metadata, dict)
                else False,
            },
        )
        graph.add_node(child)
        child_ids.append(child_id)

        graph.add_edge(Edge(source=node_id, target=child_id, weight=0.7))

    for target in parent_outgoing:
        parent_target_edge = graph.get_edge(node_id, target)
        if parent_target_edge is None:
            continue
        graph._remove_edge(node_id, target)
        for child_id in child_ids:
            existing = graph.get_edge(child_id, target)
            if existing is None:
                graph.add_edge(
                    Edge(
                        source=child_id,
                        target=target,
                        weight=parent_target_edge.weight,
                        decay_rate=parent_target_edge.decay_rate,
                        last_followed_ts=parent_target_edge.last_followed_ts,
                        created_by=parent_target_edge.created_by,
                        follow_count=parent_target_edge.follow_count,
                        skip_count=parent_target_edge.skip_count,
                    )
                )
            elif abs(parent_target_edge.weight) > abs(existing.weight):
                existing.weight = parent_target_edge.weight
                existing.decay_rate = parent_target_edge.decay_rate
                existing.last_followed_ts = parent_target_edge.last_followed_ts
                existing.created_by = parent_target_edge.created_by
                existing.follow_count += parent_target_edge.follow_count
                existing.skip_count += parent_target_edge.skip_count

    parent.summary = f"See: {', '.join(child_ids)}"
    parent.content = parent.summary

    return child_ids


def should_merge(
    graph: Graph,
    node_a_id: str,
    node_b_id: str,
    cofire_count: int,
    total_fires: int,
    cosine_sim: float,
    config: ConsolidationConfig,
) -> bool:
    if total_fires <= 0:
        return False
    if node_a_id == node_b_id:
        return False
    if graph.get_node(node_a_id) is None or graph.get_node(node_b_id) is None:
        return False
    return (
        _as_float(cosine_sim, default=0.0) > config.merge_cosine_threshold
        and _as_float(cofire_count, default=0.0) / total_fires > config.merge_cofire_threshold
    )


def consolidate(
    graph: Graph,
    config: ConsolidationConfig | None = None,
    protected_ids: set[str] | None = None,
) -> ConsolidationResult:
    if config is None:
        config = ConsolidationConfig()

    edges_pruned = prune_weak_edges(graph, config.min_edge_weight)
    nodes_pruned = prune_orphan_nodes(graph, protected_ids=protected_ids)
    nodes_pruned += prune_probationary(graph, config.probation_max_turns)

    return ConsolidationResult(
        edges_pruned=edges_pruned,
        nodes_pruned=nodes_pruned,
        nodes_split=0,
        nodes_merged=0,
    )
