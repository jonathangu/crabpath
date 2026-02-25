"""Command-line interface for CrabPath.

All output is machine-readable JSON to keep agents simple:
- stdout carries success payloads
- stderr carries structured errors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .activation import Firing
from .activation import learn as _learn
from .embeddings import EmbeddingIndex, openai_embed
from .feedback import auto_outcome, map_correction_to_snapshot, snapshot_path
from .graph import Graph
from .lifecycle_sim import run_simulation, workspace_scenario, SimConfig
from .migrate import MigrateConfig, migrate
from .mitosis import MitosisConfig, MitosisState, split_node
from .migrate import fallback_llm_split


DEFAULT_GRAPH_PATH = "crabpath_graph.json"
DEFAULT_INDEX_PATH = "crabpath_embeddings.json"
DEFAULT_TOP_K = 12


class CLIError(Exception):
    """Raised for user-facing CLI errors."""


class JSONArgumentParser(argparse.ArgumentParser):
    """Argparse parser that prints JSON errors and exits with code 1."""

    def error(self, message: str) -> None:  # pragma: no cover - exercised via CLI tests
        print(json.dumps({"error": message}), file=sys.stderr)
        raise SystemExit(1)


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload))


def _emit_error(message: str) -> int:
    print(json.dumps({"error": message}), file=sys.stderr)
    return 1


def _load_graph(path: str) -> Graph:
    file_path = Path(path)
    if not file_path.exists():
        raise CLIError(f"graph file not found: {path}")
    try:
        return Graph.load(path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CLIError(f"failed to load graph: {path}: {exc}") from exc


def _load_index(path: str) -> EmbeddingIndex:
    file_path = Path(path)
    if not file_path.exists():
        return EmbeddingIndex()
    try:
        return EmbeddingIndex.load(path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CLIError(f"failed to load index: {path}: {exc}") from exc


def _split_csv(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise CLIError("fired-ids must contain at least one id")
    return ids


def _keyword_seed(graph: Graph, query_text: str) -> dict[str, float]:
    if not query_text:
        return {}

    needles = {token.strip() for token in query_text.lower().split() if token.strip()}
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


def _safe_openai_embed_fn() -> Optional[Callable[[list[str]], list[list[float]]]]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return openai_embed()
    except Exception:
        return None


def cmd_query(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    index = _load_index(args.index)

    seeds: dict[str, float] = {}
    if os.getenv("OPENAI_API_KEY"):
        embed_fn = _safe_openai_embed_fn()
        if embed_fn is not None and index.vectors:
            seeds = index.seed(
                query_text=args.query,
                embed_fn=embed_fn,
                top_k=args.top,
            )

    if not seeds:
        seeds = _keyword_seed(graph, args.query)

    from .activation import activate

    firing = activate(
        graph,
        seeds,
        max_steps=3,
        decay=0.1,
        top_k=args.top,
        reset=False,
    )

    return {
        "fired": [
            {"id": node.id, "content": node.content, "energy": score}
            for node, score in firing.fired
        ],
        "inhibited": list(firing.inhibited),
        "guardrails": list(firing.inhibited),
    }


def _build_firing(graph: Graph, fired_ids: list[str]) -> Firing:
    if not fired_ids:
        raise CLIError("fired-ids must contain at least one id")

    nodes: list[tuple[Any, float]] = []
    fired_at: dict[str, int] = {}
    for idx, node_id in enumerate(fired_ids):
        node = graph.get_node(node_id)
        if node is None:
            raise CLIError(f"unknown node id: {node_id}")
        nodes.append((node, 1.0))
        fired_at[node_id] = idx

    return Firing(fired=nodes, inhibited=[], fired_at=fired_at)


def cmd_learn(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)

    fired_ids = _split_csv(args.fired_ids)
    try:
        outcome = float(args.outcome)
    except ValueError as exc:
        raise CLIError(f"invalid outcome: {args.outcome}") from exc

    before = {(edge.source, edge.target): edge.weight for edge in graph.edges()}
    firing = _build_firing(graph, fired_ids)
    _learn(graph, firing, outcome=outcome)

    after = {(edge.source, edge.target): edge.weight for edge in graph.edges()}
    edges_updated = 0
    for key, weight in after.items():
        if key not in before or before[key] != weight:
            edges_updated += 1

    graph.save(args.graph)
    return {"ok": True, "edges_updated": edges_updated}


def cmd_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    fired_ids = _split_csv(args.fired_ids)

    record = {
        "session_id": args.session,
        "turn_id": args.turn,
        "timestamp": time.time(),
        "fired_ids": fired_ids,
        "fired_scores": [1.0 for _ in fired_ids],
        "fired_at": {node_id: idx for idx, node_id in enumerate(fired_ids)},
        "inhibited": [],
        "attributed": False,
    }

    path = snapshot_path(args.graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"ok": True, "snapshot_path": str(path)}


def cmd_feedback(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = map_correction_to_snapshot(
        session_id=args.session,
        turn_window=args.turn_window,
    )
    if snapshot is None:
        raise CLIError(f"no attributable snapshot found for session: {args.session}")

    turns_since_fire = snapshot.get("turns_since_fire", 0)
    return {
        "turn_id": snapshot.get("turn_id"),
        "fired_ids": snapshot.get("fired_ids", []),
        "turns_since_fire": turns_since_fire,
        "suggested_outcome": auto_outcome(corrections_count=1, turns_since_fire=int(turns_since_fire)),
    }


def cmd_stats(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    edges = graph.edges()

    if edges:
        avg_weight = sum(edge.weight for edge in edges) / len(edges)
    else:
        avg_weight = 0.0

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


def cmd_migrate(args: argparse.Namespace) -> dict[str, Any]:
    config = MigrateConfig(
        include_memory=args.include_memory,
        include_docs=args.include_docs,
    )
    embed_fn = _safe_openai_embed_fn()
    embeddings_index = EmbeddingIndex()
    embed_callback = None
    if args.output_embeddings is not None and embed_fn is not None:
        embeddings_index = _load_index(args.output_embeddings)

        def embed_callback(node_id: str, content: str) -> None:
            embeddings_index.upsert(node_id, content, embed_fn=embed_fn)

    graph, info = migrate(
        workspace_dir=args.workspace,
        session_logs=args.session_logs or None,
        config=config,
        embed_callback=embed_callback,
        verbose=False,
    )
    if "states" in info:
        info = dict(info)
        info.pop("states", None)

    graph_path = args.output_graph
    graph.save(graph_path)

    embeddings_path = args.output_embeddings
    if embeddings_path:
        if embed_callback is not None:
            embeddings_index.save(embeddings_path)
        else:
            EmbeddingIndex().save(embeddings_path)

    return {
        "ok": True,
        "graph_path": str(graph_path),
        "embeddings_path": str(embeddings_path) if embeddings_path else None,
        "info": info,
    }


def cmd_add(args: argparse.Namespace) -> dict[str, Any]:
    graph_path = Path(args.graph)
    if graph_path.exists():
        graph = Graph.load(args.graph)
    else:
        graph = Graph()

    from .graph import Node, Edge

    node_id = args.id
    if graph.get_node(node_id) is not None:
        # Update existing node
        node = graph.get_node(node_id)
        node.content = args.content
        if args.threshold is not None:
            node.threshold = args.threshold
        graph.save(args.graph)
        return {"ok": True, "action": "updated", "id": node_id}

    threshold = args.threshold if args.threshold is not None else 0.5
    graph.add_node(Node(id=node_id, content=args.content, threshold=threshold))

    # Connect to existing nodes if --connect provided
    edges_added = 0
    if args.connect:
        connect_ids = [c.strip() for c in args.connect.split(",") if c.strip()]
        for target_id in connect_ids:
            if graph.get_node(target_id) is not None and target_id != node_id:
                graph.add_edge(Edge(source=node_id, target=target_id, weight=0.5))
                graph.add_edge(Edge(source=target_id, target=node_id, weight=0.5))
                edges_added += 2

    graph.save(args.graph)
    return {"ok": True, "action": "created", "id": node_id, "edges_added": edges_added}


def cmd_remove(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    node = graph.get_node(args.id)
    if node is None:
        raise CLIError(f"node not found: {args.id}")
    graph.remove_node(args.id)
    graph.save(args.graph)
    return {"ok": True, "action": "removed", "id": args.id}


def cmd_consolidate(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    result = graph.consolidate(min_weight=args.min_weight)
    graph.save(args.graph)
    return {"ok": True, **result}


def cmd_split_node(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    state = MitosisState()
    index = _load_index(args.index)
    embed_fn = _safe_openai_embed_fn()
    embed_callback = None
    if embed_fn is not None:
        def embed_callback(node_id: str, content: str) -> None:
            index.upsert(node_id, content, embed_fn=embed_fn)

    result = split_node(
        graph,
        node_id=args.node_id,
        llm_call=fallback_llm_split,
        state=state,
        config=MitosisConfig(),
        embed_callback=embed_callback,
    )

    if result is None:
        raise CLIError(f"could not split node: {args.node_id}")

    if args.save:
        graph.save(args.graph)
        if embed_callback is not None:
            index.save(args.index)

    return {
        "ok": True,
        "action": "split",
        "node_id": args.node_id,
        "chunk_ids": result.chunk_ids,
        "chunk_count": len(result.chunk_ids),
        "edges_created": result.edges_created,
    }


def cmd_sim(args: argparse.Namespace) -> dict[str, Any]:
    files, queries = workspace_scenario()
    selected_queries = queries[: args.queries]

    if not selected_queries:
        raise CLIError("queries must be a positive integer")

    config = SimConfig(
        decay_interval=args.decay_interval,
        decay_half_life=args.decay_half_life,
    )
    result = run_simulation(files, selected_queries, config=config)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))

    payload = {
        "ok": True,
        "queries": args.queries,
        "result": result,
    }
    if args.output:
        payload["output"] = args.output
    return payload


def _build_parser() -> JSONArgumentParser:
    parser = JSONArgumentParser(
        prog="crabpath",
        description="CrabPath CLI: JSON-in / JSON-out for agent use",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    q = subparsers.add_parser("query", help="Run query + activation against a graph")
    q.add_argument("query")
    q.add_argument("--top", type=int, default=DEFAULT_TOP_K)
    q.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    q.add_argument("--index", default=DEFAULT_INDEX_PATH)
    q.set_defaults(func=cmd_query)

    learn = subparsers.add_parser("learn", help="Apply STDP on specified fired node ids")
    learn.add_argument("--outcome", required=True)
    learn.add_argument("--fired-ids", required=True)
    learn.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    learn.set_defaults(func=cmd_learn)

    snap = subparsers.add_parser("snapshot", help="Persist a turn snapshot")
    snap.add_argument("--session", required=True)
    snap.add_argument("--turn", type=int, required=True)
    snap.add_argument("--fired-ids", required=True)
    snap.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    snap.set_defaults(func=cmd_snapshot)

    fb = subparsers.add_parser("feedback", help="Find most attributable snapshot")
    fb.add_argument("--session", required=True)
    fb.add_argument("--turn-window", type=int, default=5)
    fb.set_defaults(func=cmd_feedback)

    st = subparsers.add_parser("stats", help="Show simple graph stats")
    st.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    st.set_defaults(func=cmd_stats)

    mig = subparsers.add_parser("migrate", help="Bootstrap CrabPath from workspace files")
    mig.add_argument("--workspace", default="~/.openclaw/workspace")
    mig.add_argument("--session-logs", action="append", default=[])
    mig.add_argument("--include-memory", dest="include_memory", action="store_true")
    mig.add_argument("--no-include-memory", dest="include_memory", action="store_false")
    mig.set_defaults(include_memory=True)
    mig.add_argument("--include-docs", action="store_true", default=False)
    mig.add_argument("--output-graph", default=DEFAULT_GRAPH_PATH)
    mig.add_argument("--output-embeddings", default=None)
    mig.add_argument("--verbose", action="store_true", default=False)
    mig.set_defaults(func=cmd_migrate)

    split = subparsers.add_parser("split", help="Split a node into coherent chunks")
    split.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    split.add_argument("--index", default=DEFAULT_INDEX_PATH)
    split.add_argument("--node-id", required=True, dest="node_id")
    split.add_argument("--save", action="store_true")
    split.set_defaults(func=cmd_split_node)

    sim = subparsers.add_parser("sim", help="Run the lifecycle simulation")
    sim.add_argument("--queries", type=int, default=100)
    sim.add_argument("--decay-interval", type=int, default=5)
    sim.add_argument("--decay-half-life", type=int, default=80)
    sim.add_argument("--output", default=None)
    sim.set_defaults(func=cmd_sim)

    add = subparsers.add_parser("add", help="Add or update a node in the graph")
    add.add_argument("--id", required=True, help="Node ID")
    add.add_argument("--content", required=True, help="Node content text")
    add.add_argument("--threshold", type=float, default=None, help="Firing threshold (default: 0.5)")
    add.add_argument("--connect", default=None, help="Comma-separated node IDs to connect to")
    add.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    add.set_defaults(func=cmd_add)

    rm = subparsers.add_parser("remove", help="Remove a node and all its edges")
    rm.add_argument("--id", required=True, help="Node ID to remove")
    rm.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    rm.set_defaults(func=cmd_remove)

    cons = subparsers.add_parser("consolidate", help="Consolidate and prune weak connections")
    cons.add_argument("--min-weight", type=float, default=0.05)
    cons.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    cons.set_defaults(func=cmd_consolidate)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        result = args.func(args)
        _emit_json(result)
        return 0
    except CLIError as exc:
        return _emit_error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
