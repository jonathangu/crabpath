"""Minimal MCP server for CrabPath (stdio JSON-RPC transport)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ._io import (
    build_firing,
    build_health_rows,
    build_snapshot,
    graph_stats,
    load_graph,
    load_index,
    load_mitosis_state,
    load_query_stats,
    load_snapshot_rows,
    run_query,
    split_csv,
)
from .autotune import measure_health
from .embeddings import EmbeddingIndex, openai_embed
from .feedback import auto_outcome, map_correction_to_snapshot, snapshot_path
from .graph import Edge, Graph
from .legacy.activation import learn as _learn
from .lifecycle_sim import SimConfig, run_simulation, workspace_scenario
from .migrate import MigrateConfig, fallback_llm_split
from .migrate import migrate as run_migration
from .mitosis import MitosisConfig, MitosisState, split_node


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _error(req_id: Any, code: int, message: str) -> None:
    _emit(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
    )


def _result(req_id: Any, result: dict[str, Any]) -> None:
    _emit({"jsonrpc": "2.0", "id": req_id, "result": result})


def _load_graph(path: str) -> Graph:
    try:
        return load_graph(path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _load_index(path: str) -> EmbeddingIndex:
    try:
        return load_index(path)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _load_query_stats(path: str | None) -> tuple[dict[str, Any], bool]:
    try:
        return load_query_stats(path), path is not None
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _load_mitosis_state(path: str | None) -> MitosisState:
    try:
        return load_mitosis_state(path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _safe_openai_embed_fn() -> Callable[[list[str]], list[list[float]]] | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return openai_embed()
    except Exception:
        return None


def _load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    return load_snapshot_rows(path)


def mcp_query(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments.get("graph", "crabpath_graph.json"))
    index = _load_index(arguments.get("index", "crabpath_embeddings.json"))
    embed_fn = _safe_openai_embed_fn() if os.getenv("OPENAI_API_KEY") else None
    firing = run_query(
        graph,
        index,
        arguments["query"],
        top_k=arguments.get("top", 12),
        embed_fn=embed_fn,
    )
    return {
        "fired": [
            {"id": node.id, "content": node.content, "energy": score}
            for node, score in firing.fired
        ],
        "inhibited": list(firing.inhibited),
        "guardrails": list(firing.inhibited),
    }


def mcp_migrate(arguments: dict[str, Any]) -> dict[str, Any]:
    config = MigrateConfig(
        include_memory=arguments.get("include_memory", True),
        include_docs=arguments.get("include_docs", False),
    )
    graph, info = run_migration(
        workspace_dir=arguments["workspace"],
        session_logs=arguments.get("session_logs"),
        config=config,
        verbose=False,
    )

    graph_path = arguments.get("output_graph") or arguments.get("graph") or "crabpath_graph.json"
    graph.save(graph_path)

    embeddings_path = arguments.get("output_embeddings")
    embeddings_output = None
    if embeddings_path:
        embed_fn = _safe_openai_embed_fn()
        if embed_fn is not None:
            index = EmbeddingIndex()
            index.build(graph, embed_fn=embed_fn)
            if index.vectors:
                index.save(embeddings_path)
                embeddings_output = str(embeddings_path)

    return {
        "ok": True,
        "graph_path": str(graph_path),
        "embeddings_path": embeddings_output,
        "info": info,
    }


def mcp_learn(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments["graph"])
    fired_ids = split_csv(arguments["fired_ids"])
    outcome = float(arguments["outcome"])

    before = {(edge.source, edge.target): edge.weight for edge in graph.edges()}
    firing = build_firing(graph, fired_ids)
    _learn(graph, firing, outcome=outcome)

    after = {(edge.source, edge.target): edge.weight for edge in graph.edges()}
    edges_updated = sum(
        1 for key, weight in after.items() if key not in before or before[key] != weight
    )
    graph.save(arguments["graph"])

    return {"ok": True, "edges_updated": edges_updated}


def mcp_stats(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments["graph"])
    return graph_stats(graph)


def mcp_split(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments["graph"])
    state = MitosisState()
    result = split_node(
        graph=graph,
        node_id=arguments["node_id"],
        llm_call=fallback_llm_split,
        state=state,
        config=MitosisConfig(),
    )
    if result is None:
        raise ValueError(f"could not split node: {arguments['node_id']}")

    if arguments.get("save", False):
        graph.save(arguments["graph"])

    return {
        "ok": True,
        "action": "split",
        "node_id": arguments["node_id"],
        "chunk_ids": result.chunk_ids,
        "chunk_count": len(result.chunk_ids),
        "edges_created": result.edges_created,
    }


def mcp_add(arguments: dict[str, Any]) -> dict[str, Any]:
    graph_path = Path(arguments["graph"])
    if graph_path.exists():
        graph = Graph.load(arguments["graph"])
    else:
        graph = Graph()

    node_id = arguments["id"]
    if graph.get_node(node_id) is not None:
        node = graph.get_node(node_id)
        node.content = arguments["content"]
        if arguments.get("threshold") is not None:
            node.threshold = arguments["threshold"]
        graph.save(arguments["graph"])
        return {"ok": True, "action": "updated", "id": node_id}

    threshold = arguments.get("threshold") if arguments.get("threshold") is not None else 0.5
    from .graph import Node

    graph.add_node(Node(id=node_id, content=arguments["content"], threshold=threshold))

    edges_added = 0
    for target_id in arguments.get("connect", "").split(",") if arguments.get("connect") else []:
        target_id = target_id.strip()
        if target_id and graph.get_node(target_id) is not None and target_id != node_id:
            graph.add_edge(Edge(source=node_id, target=target_id, weight=0.5))
            graph.add_edge(Edge(source=target_id, target=node_id, weight=0.5))
            edges_added += 2

    graph.save(arguments["graph"])
    return {"ok": True, "action": "created", "id": node_id, "edges_added": edges_added}


def mcp_remove(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments["graph"])
    if graph.get_node(arguments["id"]) is None:
        raise ValueError(f"node not found: {arguments['id']}")
    graph.remove_node(arguments["id"])
    graph.save(arguments["graph"])
    return {"ok": True, "action": "removed", "id": arguments["id"]}


def mcp_health(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments.get("graph", "crabpath_graph.json"))
    state = _load_mitosis_state(arguments.get("mitosis_state"))
    query_stats, has_query_stats = _load_query_stats(arguments.get("query_stats"))
    health = measure_health(graph, state, query_stats)
    metrics = build_health_rows(health, has_query_stats)

    return {
        "ok": True,
        "graph": arguments.get("graph", "crabpath_graph.json"),
        "query_stats_provided": has_query_stats,
        "mitosis_state": arguments.get("mitosis_state"),
        "metrics": metrics,
    }


def mcp_evolve(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments.get("graph", "crabpath_graph.json"))
    snapshots_path = arguments.get("snapshots")
    if not snapshots_path:
        raise ValueError("--snapshots is required for evolve")

    path = Path(snapshots_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(graph)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")

    response: dict[str, Any] = {
        "ok": True,
        "snapshot": snapshot,
        "snapshots": str(path),
    }

    if arguments.get("report", False):
        response["report_rows"] = _load_snapshot_rows(path)
    return response


def mcp_consolidate(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(arguments["graph"])
    result = graph.consolidate(min_weight=arguments.get("min_weight", 0.05))
    graph.save(arguments["graph"])
    return {"ok": True, **result}


def mcp_sim(arguments: dict[str, Any]) -> dict[str, Any]:
    files, queries = workspace_scenario()
    selected_queries = queries[: int(arguments.get("queries", 100))]
    if not selected_queries:
        raise ValueError("queries must be a positive integer")

    config = SimConfig(
        decay_interval=int(arguments.get("decay_interval", 5)),
        decay_half_life=int(arguments.get("decay_half_life", 80)),
    )
    result = run_simulation(files, selected_queries, config=config)

    output = arguments.get("output")
    if output:
        Path(output).write_text(json.dumps(result, indent=2))

    payload = {"ok": True, "result": result}
    if output:
        payload["output"] = str(output)
    return payload


def mcp_snapshot(arguments: dict[str, Any]) -> dict[str, Any]:
    _load_graph(arguments["graph"])
    fired_ids = split_csv(arguments["fired_ids"])

    record = {
        "session_id": arguments["session"],
        "turn_id": arguments["turn"],
        "timestamp": time.time(),
        "fired_ids": fired_ids,
        "fired_scores": [1.0 for _ in fired_ids],
        "fired_at": {node_id: idx for idx, node_id in enumerate(fired_ids)},
        "inhibited": [],
        "attributed": False,
    }

    path = snapshot_path(arguments["graph"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"ok": True, "snapshot_path": str(path)}


def mcp_feedback(arguments: dict[str, Any]) -> dict[str, Any]:
    snapshot = map_correction_to_snapshot(
        session_id=arguments["session"],
        turn_window=int(arguments.get("turn_window", 5)),
    )
    if snapshot is None:
        raise ValueError(f"no attributable snapshot found for session: {arguments['session']}")

    turns_since_fire = snapshot.get("turns_since_fire", 0)
    return {
        "turn_id": snapshot.get("turn_id"),
        "fired_ids": snapshot.get("fired_ids", []),
        "turns_since_fire": turns_since_fire,
        "suggested_outcome": auto_outcome(
            corrections_count=1,
            turns_since_fire=int(turns_since_fire),
        ),
    }


TOOLS = [
    {
        "name": "query",
        "description": "Run activation over the graph for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "graph": {"type": "string"},
                "index": {"type": "string"},
                "top": {"type": "integer", "default": 12},
            },
            "required": ["query", "graph", "index"],
        },
    },
    {
        "name": "migrate",
        "description": "Bootstrap and replay a workspace into a memory graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workspace": {"type": "string"},
                "session_logs": {"type": "array", "items": {"type": "string"}},
                "include_memory": {"type": "boolean", "default": True},
                "include_docs": {"type": "boolean", "default": False},
                "output_graph": {"type": "string"},
                "output_embeddings": {"type": "string"},
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["workspace", "output_graph"],
        },
    },
    {
        "name": "learn",
        "description": "Apply learning updates based on fired node ids and outcome.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "number"},
                "fired_ids": {"type": "string"},
                "graph": {"type": "string"},
            },
            "required": ["outcome", "fired_ids", "graph"],
        },
    },
    {
        "name": "stats",
        "description": "Return simple graph statistics.",
        "inputSchema": {
            "type": "object",
            "properties": {"graph": {"type": "string"}},
            "required": ["graph"],
        },
    },
    {
        "name": "split",
        "description": "Split a node into coherent chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string"},
                "node_id": {"type": "string"},
                "save": {"type": "boolean", "default": False},
            },
            "required": ["graph", "node_id"],
        },
    },
    {
        "name": "add",
        "description": "Add or update a node in the graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "threshold": {"type": "number"},
                "connect": {"type": "string"},
                "graph": {"type": "string"},
            },
            "required": ["id", "content", "graph"],
        },
    },
    {
        "name": "remove",
        "description": "Remove a node and all edges.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "graph": {"type": "string"}},
            "required": ["id", "graph"],
        },
    },
    {
        "name": "consolidate",
        "description": "Consolidate graph by pruning weak edges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string"},
                "min_weight": {"type": "number", "default": 0.05},
            },
            "required": ["graph"],
        },
    },
    {
        "name": "health",
        "description": "Measure graph health from state and optional stats.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string"},
                "mitosis_state": {"type": "string"},
                "query_stats": {"type": "object"},
            },
            "required": ["graph"],
        },
    },
    {
        "name": "evolve",
        "description": "Append graph snapshot stats to a timeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string"},
                "snapshots": {"type": "string"},
                "report": {"type": "boolean", "default": False},
            },
            "required": ["graph", "snapshots"],
        },
    },
    {
        "name": "sim",
        "description": "Run lifecycle simulation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "queries": {"type": "integer", "default": 100},
                "decay_interval": {"type": "integer", "default": 5},
                "decay_half_life": {"type": "integer", "default": 80},
                "output": {"type": "string"},
            },
            "required": ["queries"],
        },
    },
    {
        "name": "snapshot",
        "description": "Persist a turn snapshot.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string"},
                "session": {"type": "string"},
                "turn": {"type": "integer"},
                "fired_ids": {"type": "string"},
            },
            "required": ["graph", "session", "turn", "fired_ids"],
        },
    },
    {
        "name": "feedback",
        "description": "Find attributable snapshot for a session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string"},
                "turn_window": {"type": "integer", "default": 5},
            },
            "required": ["session"],
        },
    },
]


HANDLERS = {
    "query": lambda args: mcp_query(args),
    "migrate": lambda args: mcp_migrate(args),
    "learn": lambda args: mcp_learn(args),
    "stats": lambda args: mcp_stats(args),
    "split": lambda args: mcp_split(args),
    "add": lambda args: mcp_add(args),
    "remove": lambda args: mcp_remove(args),
    "consolidate": lambda args: mcp_consolidate(args),
    "health": lambda args: mcp_health(args),
    "evolve": lambda args: mcp_evolve(args),
    "sim": lambda args: mcp_sim(args),
    "snapshot": lambda args: mcp_snapshot(args),
    "feedback": lambda args: mcp_feedback(args),
}


def _handle_initialize(req_id: Any, _params: dict[str, Any] | None) -> None:
    _result(
        req_id,
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "crabpath-mcp", "version": "1.0.0"},
        },
    )


def _handle_tools_list(req_id: Any) -> None:
    _result(req_id, {"tools": TOOLS})


def _handle_tools_call(req_id: Any, params: dict[str, Any] | None) -> None:
    if not params:
        raise ValueError("Missing params")
    name = params.get("name")
    if not isinstance(name, str) or name not in HANDLERS:
        raise ValueError(f"Unknown tool: {name}")

    args = params.get("arguments") or {}
    if not isinstance(args, dict):
        raise ValueError("Arguments must be an object")

    payload = HANDLERS[name](args)
    _result(
        req_id,
        {"content": [{"type": "text", "text": json.dumps(payload)}]},
    )


def _handle_request(request: dict[str, Any]) -> None:
    req_id = request.get("id")
    if request.get("jsonrpc") != "2.0":
        _error(req_id, -32600, "Invalid JSON-RPC version")
        return

    method = request.get("method")
    params = request.get("params")
    params_dict = params if isinstance(params, dict) else None

    try:
        if method == "initialize":
            _handle_initialize(req_id, params_dict)
        elif method == "tools/list":
            _handle_tools_list(req_id)
        elif method == "tools/call":
            _handle_tools_call(req_id, params_dict)
        else:
            _error(req_id, -32601, f"Unknown method: {method}")
    except ValueError as exc:
        _error(req_id, -32602, str(exc))
    except Exception as exc:  # pragma: no cover - thin transport wrapper
        _error(req_id, -32000, f"Internal error: {exc}")


def main() -> None:
    while True:
        raw = sys.stdin.readline()
        if not raw:
            break

        raw = raw.strip()
        if not raw:
            continue

        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            _error(None, -32700, "Parse error")
            continue

        if not isinstance(request, dict):
            _error(None, -32600, "Invalid request")
            continue

        _handle_request(request)


if __name__ == "__main__":
    main()
