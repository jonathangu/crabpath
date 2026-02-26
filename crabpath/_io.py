"""Shared IO helpers for CLI and MCP entrypoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from ._structural_utils import count_cross_file_edges
from .autotune import HEALTH_TARGETS
from .embeddings import EmbeddingIndex
from .graph import Graph
from .legacy.activation import Firing, activate
from .mitosis import MitosisState
from .synaptogenesis import edge_tier_stats


def load_graph(path: str) -> Graph:
    """Load a graph with stable CLI/MCP error messages."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"graph file not found: {path}")

    try:
        return Graph.load(path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"failed to load graph: {path}: {exc}") from exc


def load_index(path: str) -> EmbeddingIndex:
    """Load embedding index from disk, or return empty if missing."""
    file_path = Path(path)
    if not file_path.exists():
        return EmbeddingIndex()

    try:
        return EmbeddingIndex.load(path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"failed to load index: {path}: {exc}") from exc


def load_query_stats(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"query-stats file not found: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
        stats = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load query-stats: {path}: {exc}") from exc

    if not isinstance(stats, dict):
        raise ValueError(f"query-stats must be a JSON object: {path}")

    return stats


def load_mitosis_state(path: str | None) -> MitosisState:
    if path is None:
        return MitosisState()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"mitosis-state file not found: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
        state_data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load mitosis-state: {path}: {exc}") from exc

    if not isinstance(state_data, dict):
        raise ValueError(f"mitosis-state must be a JSON object: {path}")

    return MitosisState(
        families=state_data.get("families", {}),
        generations=state_data.get("generations", {}),
        chunk_to_parent=state_data.get("chunk_to_parent", {}),
    )


def load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON line in snapshots file: {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"invalid snapshot row in snapshots file: {path}")
            rows.append(row)

    return rows


def split_csv(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("fired-ids must contain at least one id")
    return ids


def graph_stats(graph: Graph) -> dict[str, Any]:
    """Compute canonical graph summary statistics."""
    edges = graph.edges()
    avg_weight = sum(edge.weight for edge in edges) / len(edges) if edges else 0.0

    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
    top = sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:5]

    return {
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "avg_weight": avg_weight,
        "top_hubs": [node_id for node_id, _ in top],
    }


def keyword_seed(graph: Graph, query_text: str) -> dict[str, float]:
    if not query_text:
        return {}

    needles = {token.strip().lower() for token in query_text.split() if token.strip()}
    seeds: dict[str, float] = {}
    for node in graph.nodes():
        haystack = f"{node.id} {node.content}".lower()
        score = 0.0
        for needle in needles:
            if needle in haystack:
                score += 1.0
        if score:
            seeds[node.id] = score

    return seeds


def run_query(
    graph: Graph,
    index: EmbeddingIndex,
    query_text: str,
    *,
    top_k: int,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> Firing:
    seeds: dict[str, float] = {}
    if embed_fn is not None and index.vectors:
        try:
            seeds = index.seed(query_text, embed_fn=embed_fn, top_k=top_k)
        except (TypeError, ValueError):
            seeds = {}

    if not seeds:
        seeds = keyword_seed(graph, query_text)

    return activate(
        graph,
        seeds,
        max_steps=3,
        decay=0.1,
        top_k=top_k,
        reset=False,
    )


def build_firing(graph: Graph, fired_ids: list[str]) -> Firing:
    if not fired_ids:
        raise ValueError("fired-ids must contain at least one id")

    nodes: list[tuple[Any, float]] = []
    fired_at: dict[str, int] = {}
    for idx, node_id in enumerate(fired_ids):
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"unknown node id: {node_id}")
        nodes.append((node, 1.0))
        fired_at[node_id] = idx

    return Firing(fired=nodes, inhibited=[], fired_at=fired_at)


def build_snapshot(graph: Graph) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "tier_counts": edge_tier_stats(graph),
        "cross_file_edges": count_cross_file_edges(graph),
    }


def health_metric_available(metric: str, has_query_stats: bool) -> bool:
    if metric in {
        "avg_nodes_fired_per_query",
        "context_compression",
        "proto_promotion_rate",
        "reconvergence_rate",
    }:
        return has_query_stats
    return True


def health_metric_status(
    value: float | None,
    target: tuple[float | None, float | None],
    available: bool,
) -> str:
    if not available or value is None:
        return "warn"

    min_v, max_v = target
    if min_v is not None and value < min_v:
        return "low"
    if max_v is not None and value > max_v:
        return "high"
    return "ok"


def build_health_rows(
    health: Any,
    has_query_stats: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, target in HEALTH_TARGETS.items():
        available = health_metric_available(metric, has_query_stats)
        raw_value = getattr(health, metric, None)
        value = raw_value if available else None
        value_num = raw_value if isinstance(raw_value, (int, float)) else None
        status = health_metric_status(
            float(value_num) if value_num is not None else None,
            target,
            available,
        )
        rows.append(
            {
                "metric": metric,
                "value": value if available else None,
                "target_range": target,
                "status": status,
            }
        )

    return rows
