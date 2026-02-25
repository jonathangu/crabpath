"""Command-line interface for CrabPath.

All output is machine-readable JSON to keep agents simple:
- stdout carries success payloads
- stderr carries structured errors
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .activation import Firing
from .activation import learn as _learn
from .embeddings import EmbeddingIndex, openai_embed
from .feedback import auto_outcome, map_correction_to_snapshot, snapshot_path
from .graph import Graph
from .lifecycle_sim import run_simulation, workspace_scenario, SimConfig
from .autotune import HEALTH_TARGETS, measure_health
from .migrate import MigrateConfig, migrate
from .mitosis import MitosisConfig, MitosisState, split_node
from .migrate import fallback_llm_split
from .synaptogenesis import edge_tier_stats


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


def _load_query_stats(path: str | None) -> tuple[dict[str, Any], bool]:
    if path is None:
        return {}, False

    file_path = Path(path)
    if not file_path.exists():
        raise CLIError(f"query-stats file not found: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
        stats = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CLIError(f"failed to load query-stats: {path}: {exc}") from exc

    if not isinstance(stats, dict):
        raise CLIError(f"query-stats must be a JSON object: {path}")
    return stats, True


def _load_mitosis_state(path: str | None) -> MitosisState:
    if path is None:
        return MitosisState()

    file_path = Path(path)
    if not file_path.exists():
        raise CLIError(f"mitosis-state file not found: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
        state_data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CLIError(f"failed to load mitosis-state: {path}: {exc}") from exc

    if not isinstance(state_data, dict):
        raise CLIError(f"mitosis-state must be a JSON object: {path}")

    return MitosisState(
        families=state_data.get("families", {}),
        generations=state_data.get("generations", {}),
        chunk_to_parent=state_data.get("chunk_to_parent", {}),
    )


def _node_file_id(node_id: str) -> str:
    return str(node_id).split("::", 1)[0]


def _cross_file_edges(graph: Graph) -> int:
    if graph.node_count <= 1:
        return 0

    cross = 0
    for edge in graph.edges():
        if _node_file_id(edge.source) != _node_file_id(edge.target):
            cross += 1
    return cross


def _format_health_target(target: tuple[float | None, float | None]) -> str:
    min_v, max_v = target
    if min_v is None and max_v is None:
        return "*"
    if min_v is None:
        return f"<= {max_v}"
    if max_v is None:
        return f">= {min_v}"
    return f"{min_v} - {max_v}"


def _status_for_health_metric(
    value: float | None,
    target: tuple[float | None, float | None],
    available: bool,
) -> str:
    if not available:
        return "⚠️"

    if value is None:
        return "⚠️"

    min_v, max_v = target
    if min_v is not None and value < min_v:
        return "❌"
    if max_v is not None and value > max_v:
        return "❌"
    return "✅"


def _format_metric_value(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"

    if metric.endswith("_pct") or metric == "context_compression":
        return f"{value:.2f}%"

    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _health_metric_available(metric: str, has_query_stats: bool) -> bool:
    if metric in {
        "avg_nodes_fired_per_query",
        "context_compression",
        "proto_promotion_rate",
        "reconvergence_rate",
    }:
        return has_query_stats
    return True


def _build_health_report_lines(
    graph: Graph,
    health: dict[str, Any],
    has_query_stats: bool,
    *,
    with_status: bool = False,
) -> list[str]:
    lines: list[str] = [
        f"Graph Health: {graph.node_count} nodes, {graph.edge_count} edges",
        "-" * 46,
    ]
    for metric, target in HEALTH_TARGETS.items():
        available = _health_metric_available(metric, has_query_stats)
        raw_value = health.get(metric)
        value = raw_value if available else None
        status = _status_for_health_metric(value if available else None, target, available)
        value_text = (
            _format_metric_value(metric, float(value))
            if value is not None
            else "n/a (collect query stats)"
        )
        if with_status:
            lines.append(
                f"{metric:24} | {value_text:>20} | target { _format_health_target(target):15} | {status}"
            )
        else:
            lines.append(f"{metric}: {value_text} (target {_format_health_target(target)}) {status}")
    return lines


def _build_health_payload(
    args: argparse.Namespace,
    graph: Graph,
    health: Any,
    has_query_stats: bool,
) -> dict[str, Any]:
    rows = []
    for metric, target in HEALTH_TARGETS.items():
        value = getattr(health, metric)
        available = _health_metric_available(metric, has_query_stats)
        status = _status_for_health_metric(value if available else None, target, available)
        rows.append(
            {
                "metric": metric,
                "value": value if available else None,
                "target_range": target,
                "status": status,
            }
        )

    return {
        "ok": True,
        "graph": args.graph,
        "query_stats_provided": has_query_stats,
        "mitosis_state": args.mitosis_state,
        "metrics": rows,
    }


def cmd_health(args: argparse.Namespace) -> dict[str, Any] | str:
    graph = _load_graph(args.graph)
    state = _load_mitosis_state(args.mitosis_state)
    query_stats, has_query_stats = _load_query_stats(args.query_stats)
    health = measure_health(graph, state, query_stats)
    if args.json:
        return _build_health_payload(args, graph, health, has_query_stats)

    return "\n".join(
        _build_health_report_lines(
            graph,
            dataclasses.asdict(health),
            has_query_stats,
            with_status=True,
        )
    )


def _snapshot_path(path_value: str | None) -> Path:
    if path_value is None:
        raise CLIError("--snapshots is required for evolve")
    return Path(path_value)


def _load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CLIError(f"invalid JSON line in snapshots file: {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise CLIError(f"invalid snapshot row in snapshots file: {path}")
            rows.append(row)
    return rows


def _build_snapshot(graph: Graph) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "tier_counts": edge_tier_stats(graph),
        "cross_file_edges": _cross_file_edges(graph),
    }


def _format_timeline(snapshots: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Evolution timeline"]
    if not snapshots:
        return "No snapshots yet."

    previous: dict[str, Any] | None = None
    for idx, snapshot in enumerate(snapshots, start=1):
        timestamp = snapshot.get("timestamp")
        try:
            ts = float(timestamp) if timestamp is not None else 0.0
            label = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError, OverflowError):
            label = "invalid-timestamp"

        nodes = int(snapshot.get("nodes", 0))
        edges = int(snapshot.get("edges", 0))
        cross_file = int(snapshot.get("cross_file_edges", 0))
        tiers = snapshot.get("tier_counts", {})
        dormant = int((tiers or {}).get("dormant", 0))
        habitual = int((tiers or {}).get("habitual", 0))
        reflex = int((tiers or {}).get("reflex", 0))

        if previous is None:
            lines.append(
                f"#{idx:>2} {label} | nodes {nodes} | edges {edges} | cross-file {cross_file} | tiers d={dormant} h={habitual} r={reflex}"
            )
        else:
            delta_nodes = nodes - int(previous.get("nodes", 0))
            delta_edges = edges - int(previous.get("edges", 0))
            delta_cross = cross_file - int(previous.get("cross_file_edges", 0))
            prev_tiers = previous.get("tier_counts", {})
            prev_dormant = int((prev_tiers or {}).get("dormant", 0))
            prev_habitual = int((prev_tiers or {}).get("habitual", 0))
            prev_reflex = int((prev_tiers or {}).get("reflex", 0))

            lines.append(
                f"#{idx:>2} {label} | nodes {nodes} ({delta_nodes:+d}) | edges {edges} ({delta_edges:+d}) | cross-file {cross_file} ({delta_cross:+d}) | tiers d={dormant} ({delta_dormant:+d}) h={habitual} ({delta_habitual:+d}) r={reflex} ({delta_reflex:+d})"
            )

        previous = snapshot

        if previous is None or previous.get("timestamp") is None:
            continue
    return "\n".join(lines)


def cmd_evolve(args: argparse.Namespace) -> dict[str, Any] | str:
    graph = _load_graph(args.graph)
    path = _snapshot_path(args.snapshots)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = _build_snapshot(graph)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")

    if not args.report:
        return {"ok": True, "snapshot": snapshot, "snapshots": str(path)}

    snapshots = _load_snapshot_rows(path)
    return _format_timeline(snapshots)


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
                args.query,
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

    health = subparsers.add_parser("health", help="Measure graph health from graph state + optional stats")
    health.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    health.add_argument("--mitosis-state", default=None)
    health.add_argument("--query-stats", default=None)
    health.add_argument("--json", action="store_true", default=False)
    health.set_defaults(func=cmd_health)

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

    evolve = subparsers.add_parser("evolve", help="Append graph snapshot stats to a JSONL timeline")
    evolve.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    evolve.add_argument("--snapshots", required=True)
    evolve.add_argument("--report", action="store_true", default=False)
    evolve.set_defaults(func=cmd_evolve)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        result = args.func(args)
        if args.command == "health" and not args.json:
            if not isinstance(result, str):
                _emit_json(result)
            else:
                print(result)
            return 0
        if args.command == "evolve" and getattr(args, "report", False):
            if not isinstance(result, str):
                _emit_json(result)
            else:
                print(result)
            return 0
        _emit_json(result)
        return 0
    except CLIError as exc:
        return _emit_error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
