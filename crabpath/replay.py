from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from .graph import Graph
from .learn import LearningConfig, apply_outcome
from .traverse import TraversalConfig, traverse
from ._util import _tokenize


def _extract_user_query_content(content: object) -> str | None:
    if isinstance(content, str):
        value = content.strip()
        return value if value else None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue

            if isinstance(item, dict):
                item_text = _extract_user_query_content(item.get("text"))
                if item_text:
                    parts.append(item_text)
                    continue
                item_content = _extract_user_query_content(item.get("content"))
                if item_content:
                    parts.append(item_content)
        return " ".join(parts) if parts else None

    return None


def _extract_openclaw_query(payload: dict) -> str | None:
    if payload.get("type") != "message":
        return None

    message = payload.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return None

    return _extract_user_query_content(message.get("content"))


def _extract_flat_query(payload: dict) -> str | None:
    if payload.get("role") != "user":
        return None
    return _extract_user_query_content(payload.get("content"))


def _extract_query_timestamp(payload: dict) -> float | None:
    timestamp_keys = ("ts", "timestamp", "created_at", "time")
    for key in timestamp_keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                iso_value = value[:-1] + "+00:00" if value.endswith("Z") else value
                try:
                    return datetime.fromisoformat(iso_value).timestamp()
                except ValueError:
                    continue
    return None


def _extract_query_record(raw: str) -> tuple[str | None, float | None]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() or None, None

    if not isinstance(payload, dict):
        return None, None

    query = _extract_openclaw_query(payload)
    if query is None:
        query = _extract_flat_query(payload)
    return query, _extract_query_timestamp(payload)


def extract_queries(session_path: str | Path, since_ts: float | None = None) -> list[str]:
    """Extract user queries from an OpenClaw session log.

    OpenClaw format: JSONL with records like:
    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "..."}]}}

    Also handles flat format: {"role": "user", "content": "..."}
    Also handles plain text lines.

    Returns list of query strings.
    """
    path = Path(session_path).expanduser()
    if not path.exists():
        raise SystemExit(f"missing sessions file: {path}")

    queries: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            query, query_ts = _extract_query_record(raw.strip())
            if query is None:
                continue
            if since_ts is not None and query_ts is not None and query_ts <= since_ts:
                continue
            queries.append(query)

    return queries


def extract_query_records(
    session_path: str | Path,
    since_ts: float | None = None,
) -> list[tuple[str, float | None]]:
    """Extract (query, timestamp) pairs from session log."""
    path = Path(session_path).expanduser()
    if not path.exists():
        raise SystemExit(f"missing sessions file: {path}")

    records: list[tuple[str, float | None]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            query, query_ts = _extract_query_record(raw)
            if query is None:
                continue
            if since_ts is not None and query_ts is not None and query_ts <= since_ts:
                continue
            records.append((query, query_ts))
    return records


def extract_queries_from_dir(sessions_dir: str | Path, since_ts: float | None = None) -> list[str]:
    """Extract queries from all .jsonl files in a directory."""
    path = Path(sessions_dir).expanduser()
    if not path.exists():
        raise SystemExit(f"missing sessions directory: {path}")
    if not path.is_dir():
        raise SystemExit(f"not a directory: {path}")

    queries: list[str] = []
    for session_file in sorted(path.glob("*.jsonl")):
        queries.extend(extract_queries(session_file, since_ts=since_ts))
    return queries


def extract_query_records_from_dir(
    sessions_dir: str | Path,
    since_ts: float | None = None,
) -> list[tuple[str, float | None]]:
    """Extract (query, timestamp) pairs from all .jsonl files in a directory."""
    path = Path(sessions_dir).expanduser()
    if not path.exists():
        raise SystemExit(f"missing sessions directory: {path}")
    if not path.is_dir():
        raise SystemExit(f"not a directory: {path}")

    records: list[tuple[str, float | None]] = []
    for session_file in sorted(path.glob("*.jsonl")):
        records.extend(extract_query_records(session_file, since_ts=since_ts))
    return records


def default_keyword_seed_fn(graph: Graph, query_text: str) -> list[tuple[str, float]]:
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    scores: dict[str, float] = {}
    for node in graph.nodes():
        node_tokens = _tokenize(node.content)
        overlap = len(query_tokens & node_tokens)
        if overlap > 0:
            scores[node.id] = overlap / len(query_tokens)

    if not scores:
        return []

    for target_id in list(scores):
        for source_node, _edge in graph.incoming(target_id):
            if source_node.id not in scores:
                scores[source_node.id] = 0.0

    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return ranked[:10]


def _snapshot_edges(graph: Graph) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for source_id, edges in graph._edges.items():
        for target_id, edge in edges.items():
            weights[(source_id, target_id)] = edge.weight
    return weights


def _cross_file_edges(graph: Graph) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for source_id, source_edges in graph._edges.items():
        source_node = graph.get_node(source_id)
        source_file = source_node.metadata.get("file") if source_node else None
        for target_id in source_edges:
            target_node = graph.get_node(target_id)
            target_file = target_node.metadata.get("file") if target_node else None
            if source_file is not None and target_file is not None and source_file != target_file:
                edges.add((source_id, target_id))
    return edges


def replay_queries(
    graph: Graph,
    queries: list[str | tuple[str, float | None]],
    config: TraversalConfig | None = None,
    keyword_seed_fn: Callable[[Graph, str], list[tuple[str, float]]] | None = None,
    outcome: float = 1.0,
    outcome_fn: Callable[[str], float] | None = None,
    verbose: bool = False,
    since_ts: float | None = None,
) -> dict:
    """Replay historical queries to warm up graph edges.

    For each query:
    1. Seed from keyword matching (or provided seed_fn)
    2. Traverse the graph
    3. Apply outcome weighting (positive, negative, or custom)
    4. Apply Hebbian co-firing for co-selected nodes
    """
    cfg = config or TraversalConfig()
    seed_fn = keyword_seed_fn or default_keyword_seed_fn

    normalized_queries: list[tuple[str, float | None]] = []
    for entry in queries:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
            query, query_ts = entry
        else:
            query, query_ts = str(entry), None
        if not query.strip():
            continue
        normalized_queries.append((query, query_ts))

    if since_ts is not None:
        normalized_queries = [
            (query, query_ts) for query, query_ts in normalized_queries if query_ts is None or query_ts > since_ts
        ]

    if not normalized_queries:
        return {
            "queries_replayed": 0,
            "edges_reinforced": 0,
            "cross_file_edges_created": 0,
            "last_replayed_ts": None,
        }

    stats = {
        "queries_replayed": 0,
        "edges_reinforced": 0,
        "cross_file_edges_created": 0,
        "last_replayed_ts": None,
    }
    total_queries = len(normalized_queries)
    latest_ts = None

    for query, query_ts in normalized_queries:
        stats["queries_replayed"] += 1

        seeds = seed_fn(graph, query)
        result = traverse(graph=graph, seeds=seeds, config=cfg)
        if not result.fired:
            if verbose:
                print(
                    f"Replayed {stats['queries_replayed']}/{total_queries} queries, "
                    f"{stats['cross_file_edges_created']} cross-file edges created"
                )
            if query_ts is not None:
                latest_ts = query_ts if latest_ts is None else max(latest_ts, query_ts)
            continue

        before_weights = _snapshot_edges(graph)
        before_cross_edges = _cross_file_edges(graph)

        query_outcome = outcome_fn(query) if outcome_fn is not None else outcome
        fired_nodes = [result.steps[0].from_node, *[step.to_node for step in result.steps]] if result.steps else result.fired
        apply_outcome(graph=graph, fired_nodes=fired_nodes, outcome=query_outcome, config=LearningConfig())

        after_weights = _snapshot_edges(graph)
        after_cross_edges = _cross_file_edges(graph)

        for key, weight in after_weights.items():
            if before_weights.get(key) != weight:
                stats["edges_reinforced"] += 1

        new_cross_edges = after_cross_edges - before_cross_edges
        stats["cross_file_edges_created"] += len(new_cross_edges)

        if verbose:
            print(
                f"Replayed {stats['queries_replayed']}/{total_queries} queries, "
                f"{stats['cross_file_edges_created']} cross-file edges created"
            )
        if query_ts is not None:
            latest_ts = query_ts if latest_ts is None else max(latest_ts, query_ts)

    stats["last_replayed_ts"] = latest_ts
    return stats
