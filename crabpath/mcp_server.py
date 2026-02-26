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
from . import __version__
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


def _coerce_str(value: Any, name: str, *, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _coerce_bool(value: Any, name: str, *, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _coerce_int(
    value: Any,
    name: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")

    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def _coerce_float(
    value: Any,
    name: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")

    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def _coerce_path(value: Any, name: str, *, default: str | None = None) -> str:
    path = _coerce_str(value, name, default=default)
    parts = Path(path).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe path traversal in {name}: {path!r}")
    return path


def _coerce_optional_path(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _coerce_path(value, name)


def _coerce_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} entries must be strings")
        if not item:
            raise ValueError(f"{name} entries must not be empty")
        items.append(item)
    return items


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


def _coerce_query_stats(value: Any) -> tuple[dict[str, Any], bool]:
    if value is None:
        return {}, False
    if isinstance(value, dict):
        return value, True
    path = _coerce_path(value, "query_stats")
    return _load_query_stats(path)


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
    graph = _load_graph(_coerce_path(arguments.get("graph"), "graph", default="crabpath_graph.json"))
    index = _load_index(_coerce_path(arguments.get("index"), "index", default="crabpath_embeddings.json"))
    embed_fn = _safe_openai_embed_fn() if os.getenv("OPENAI_API_KEY") else None
    query = _coerce_str(arguments.get("query"), "query")
    top_k = _coerce_int(arguments.get("top"), "top", default=12, minimum=1)
    firing = run_query(graph, index, query, top_k=top_k, embed_fn=embed_fn)
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
        include_memory=_coerce_bool(arguments.get("include_memory"), "include_memory", default=True),
        include_docs=_coerce_bool(arguments.get("include_docs"), "include_docs", default=False),
    )
    workspace = _coerce_path(arguments.get("workspace"), "workspace")
    session_logs = (
        _coerce_string_list(arguments.get("session_logs"), "session_logs")
        if arguments.get("session_logs") is not None
        else None
    )
    graph, info = run_migration(
        workspace_dir=workspace,
        session_logs=session_logs,
        config=config,
        verbose=False,
    )

    graph_path = _coerce_path(
        arguments.get("output_graph") or arguments.get("graph"),
        "output_graph",
        default="crabpath_graph.json",
    )
    graph.save(graph_path)

    embeddings_path = (
        _coerce_path(arguments.get("output_embeddings"), "output_embeddings")
        if arguments.get("output_embeddings") is not None
        else None
    )
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
    graph = _load_graph(_coerce_path(arguments.get("graph"), "graph"))
    fired_ids = split_csv(_coerce_str(arguments.get("fired_ids"), "fired_ids"))
    outcome = _coerce_float(arguments.get("outcome"), "outcome", minimum=-1.0, maximum=1.0)

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
    graph = _load_graph(_coerce_path(arguments.get("graph"), "graph"))
    return graph_stats(graph)


def mcp_split(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(_coerce_path(arguments.get("graph"), "graph"))
    state = MitosisState()
    result = split_node(
        graph=graph,
        node_id=_coerce_str(arguments.get("node_id"), "node_id"),
        llm_call=fallback_llm_split,
        state=state,
        config=MitosisConfig(),
    )
    if result is None:
        raise ValueError(f"could not split node: {_coerce_str(arguments.get('node_id'), 'node_id')}")

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
    graph_path = Path(_coerce_path(arguments.get("graph"), "graph"))
    if graph_path.exists():
        graph = Graph.load(str(graph_path))
    else:
        graph = Graph()

    node_id = _coerce_str(arguments.get("id"), "id")
    if graph.get_node(node_id) is not None:
        node = graph.get_node(node_id)
        node.content = _coerce_str(arguments.get("content"), "content")
        threshold = arguments.get("threshold")
        if threshold is not None:
            node.threshold = _coerce_float(threshold, "threshold")
        graph.save(str(graph_path))
        return {"ok": True, "action": "updated", "id": node_id}

    threshold = (
        _coerce_float(arguments.get("threshold"), "threshold", default=0.5)
        if arguments.get("threshold") is not None
        else 0.5
    )
    from .graph import Node

    graph.add_node(Node(id=node_id, content=arguments["content"], threshold=threshold))

    edges_added = 0
    for target_id in arguments.get("connect", "").split(",") if arguments.get("connect") else []:
        target_id = target_id.strip()
        if target_id and graph.get_node(target_id) is not None and target_id != node_id:
            graph.add_edge(Edge(source=node_id, target=target_id, weight=0.5))
            graph.add_edge(Edge(source=target_id, target=node_id, weight=0.5))
            edges_added += 2

    graph.save(str(graph_path))
    return {"ok": True, "action": "created", "id": node_id, "edges_added": edges_added}


def mcp_remove(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(_coerce_path(arguments.get("graph"), "graph"))
    node_id = _coerce_str(arguments.get("id"), "id")
    if graph.get_node(node_id) is None:
        raise ValueError(f"node not found: {node_id}")
    graph.remove_node(node_id)
    graph.save(_coerce_path(arguments.get("graph"), "graph"))
    return {"ok": True, "action": "removed", "id": node_id}


def mcp_health(arguments: dict[str, Any]) -> dict[str, Any]:
    graph_path = _coerce_path(arguments.get("graph"), "graph", default="crabpath_graph.json")
    graph = _load_graph(graph_path)
    state_path = _coerce_optional_path(arguments.get("mitosis_state"), "mitosis_state")
    state = _load_mitosis_state(state_path)
    query_stats, has_query_stats = _coerce_query_stats(arguments.get("query_stats"))
    health = measure_health(graph, state, query_stats)
    metrics = build_health_rows(health, has_query_stats)

    return {
        "ok": True,
        "graph": graph_path,
        "query_stats_provided": has_query_stats,
        "mitosis_state": arguments.get("mitosis_state"),
        "metrics": metrics,
    }


def mcp_evolve(arguments: dict[str, Any]) -> dict[str, Any]:
    graph = _load_graph(_coerce_path(arguments.get("graph"), "graph", default="crabpath_graph.json"))
    snapshots_path = _coerce_path(
        arguments.get("snapshots"),
        "snapshots",
    )
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
    graph_path = _coerce_path(arguments.get("graph"), "graph")
    graph = _load_graph(graph_path)
    min_weight = _coerce_float(arguments.get("min_weight"), "min_weight", default=0.05, minimum=0.0)
    result = graph.consolidate(min_weight=min_weight)
    graph.save(graph_path)
    return {"ok": True, **result}


def mcp_sim(arguments: dict[str, Any]) -> dict[str, Any]:
    files, queries = workspace_scenario()
    selected_queries = queries[: _coerce_int(arguments.get("queries"), "queries", default=100, minimum=1)]
    if not selected_queries:
        raise ValueError("queries must be a positive integer")

    config = SimConfig(
        decay_interval=_coerce_int(arguments.get("decay_interval"), "decay_interval", default=5, minimum=1),
        decay_half_life=_coerce_int(arguments.get("decay_half_life"), "decay_half_life", default=80, minimum=1),
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
    graph = _load_graph(_coerce_path(arguments.get("graph"), "graph"))
    session = _coerce_str(arguments.get("session"), "session")
    turn = _coerce_int(arguments.get("turn"), "turn", minimum=0)
    fired_ids = split_csv(_coerce_str(arguments.get("fired_ids"), "fired_ids"))

    record = {
        "session_id": session,
        "turn_id": turn,
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
        session_id=_coerce_str(arguments.get("session"), "session"),
        turn_window=_coerce_int(arguments.get("turn_window"), "turn_window", default=5, minimum=1),
    )
    if snapshot is None:
        raise ValueError(f"no attributable snapshot found for session: {_coerce_str(arguments.get('session'), 'session')}")

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
                "query_stats": {
                    "oneOf": [
                        {"type": "object"},
                        {"type": "string"},
                    ]
                },
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
            "serverInfo": {"name": "crabpath-mcp", "version": __version__},
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

    args = params.get("arguments")
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
    except (ValueError, TypeError, KeyError) as exc:
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
