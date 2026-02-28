"""Persistent long-running NDJSON worker for OpenClawBrain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import deque
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .autotune import measure_health
from .graph import Graph, Node
from .hasher import HashEmbedder
from .index import VectorIndex
from .journal import log_health, log_learn, log_query
from .learn import apply_outcome, apply_outcome_pg
from .maintain import run_maintenance
from .inject import _apply_inhibitory_edges, inject_node
from .store import load_state, save_state
from .traverse import TraversalConfig, traverse


FIRED_LOG_LIMIT_PER_CHAT = 100
FIRED_LOG_LIMIT_TOTAL = 1000

FIRED_LOG_FILE = "fired_log.jsonl"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openclawbrain daemon")
    parser.add_argument("--state", required=True)
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--auto-save-interval", type=int, default=10)
    return parser


def _make_embed_fn(embed_model: str) -> Callable[[str], list[float]] | None:
    """Create embedding callback when OPENAI_API_KEY is present."""
    if not os.getenv("OPENAI_API_KEY"):
        return None

    from openai import OpenAI

    client = OpenAI()
    return lambda text: [float(v) for v in client.embeddings.create(model=embed_model, input=[text]).data[0].embedding]


def _journal_path(state_path: str) -> str:
    """Resolve journal.jsonl path for the loaded state directory."""
    return str(Path(state_path).expanduser().parent / "journal.jsonl")


def _fired_log_path(state_path: str) -> str:
    """Resolve fired_log.jsonl path for the loaded state directory."""
    return str(Path(state_path).expanduser().parent / FIRED_LOG_FILE)


def _ms(start: float, end: float) -> float:
    """Convert elapsed duration in seconds to milliseconds."""
    return round((end - start) * 1000.0, 3)


def _node_authority(node: Node | None) -> str:
    """Extract node authority with safe fallback."""
    if node is None or not isinstance(node.metadata, dict):
        return "overlay"
    authority = node.metadata.get("authority")
    if authority in {"constitutional", "canonical", "overlay"}:
        return authority
    return "overlay"


def _node_ids_with_authority(graph: Graph, authority: str) -> set[str]:
    """Collect node ids for a given authority label."""
    return {node.id for node in graph.nodes() if _node_authority(node) == authority}


def _real_graph_updates(updates: dict[str, float]) -> int:
    """Count updates that target concrete edges and ignore pseudo STOP transitions."""
    return sum(1 for key in updates if not key.endswith("->__STOP__"))


def _health_payload(graph: Graph) -> dict[str, object]:
    """Build health payload with node/edge counts included."""
    health = measure_health(graph)
    return health.__dict__ | {"nodes": graph.node_count(), "edges": graph.edge_count()}


def _emit_response(req_id: object, payload: object, error: dict[str, object] | None = None) -> None:
    """Emit one NDJSON object."""
    if error is None:
        print(json.dumps({"id": req_id, "result": payload}), flush=True)
    else:
        print(json.dumps({"id": req_id, "error": error}), flush=True)


def _parse_int(value: object, label: str, default: int | None = None, min_value: int = 1) -> int:
    """Validate integer input."""
    if value is None:
        if default is None:
            raise ValueError(f"{label} is required")
        return default
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < min_value:
        raise ValueError(f"{label} must be >= {min_value}")
    return value


def _parse_float(value: object, label: str, required: bool = False, default: float | None = None) -> float:
    """Validate float input."""
    if value is None:
        if not required:
            if default is None:
                raise ValueError(f"{label} is required")
            return default
        raise ValueError(f"{label} is required")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _parse_str_list(value: object, label: str, required: bool = True) -> list[str]:
    """Parse a JSON list of non-empty strings."""
    if value is None:
        if required:
            raise ValueError(f"{label} is required")
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    entries: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} must only contain non-empty strings")
        entries.append(item)
    if not entries and required:
        raise ValueError(f"{label} must not be empty")
    return entries


def _parse_chat_id(value: object, label: str, required: bool = True) -> str | None:
    """Parse optional chat_id with non-empty normalization."""
    if value is None:
        if required:
            raise ValueError(f"{label} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{label} is required")
    return value if value else None


def _append_fired_log(state_path: str, entry: dict[str, object]) -> None:
    path = Path(_fired_log_path(state_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _append_fired_history(
    daemon_state: "_DaemonState",
    chat_id: str,
    query: str,
    fired_nodes: list[str],
    timestamp: float,
) -> None:
    entries = daemon_state.fired_log.setdefault(chat_id, deque(maxlen=FIRED_LOG_LIMIT_PER_CHAT))
    entries.append(
        {
            "chat_id": chat_id,
            "query": query,
            "fired_nodes": list(fired_nodes),
            "ts": timestamp,
            "timestamp": timestamp,
        }
    )

    while _fired_history_size(daemon_state) > FIRED_LOG_LIMIT_TOTAL:
        _trim_oldest_fired_history_entry(daemon_state)


def _trim_oldest_fired_history_entry(daemon_state: "_DaemonState") -> None:
    oldest_chat = None
    oldest_timestamp = float("inf")

    for chat_id, entries in daemon_state.fired_log.items():
        if not entries:
            continue
        candidate = entries[0].get("ts")
        if isinstance(candidate, (int, float)) and candidate < oldest_timestamp:
            oldest_timestamp = float(candidate)
            oldest_chat = chat_id

    if oldest_chat is None:
        return
    entries = daemon_state.fired_log[oldest_chat]
    if entries:
        entries.popleft()
        if not entries:
            daemon_state.fired_log.pop(oldest_chat, None)


def _fired_history_size(daemon_state: "_DaemonState") -> int:
    return sum(len(entries) for entries in daemon_state.fired_log.values())


def _recent_fired_nodes(daemon_state: "_DaemonState", chat_id: str, lookback: int) -> list[str]:
    history = daemon_state.fired_log.get(chat_id, deque())
    if not history:
        return []

    seen: set[str] = set()
    fired_nodes: list[str] = []
    for entry in reversed(history):
        for node_id in entry.get("fired_nodes", []):
            if isinstance(node_id, str) and node_id and node_id not in seen:
                seen.add(node_id)
                fired_nodes.append(node_id)
        lookback -= 1
        if lookback <= 0:
            break
    return fired_nodes


def _correction_node_id(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"correction::{digest[:12]}"


def _handle_query(
    graph: Graph,
    index: VectorIndex,
    meta: dict[str, object],
    embed_fn: Callable[[str], list[float]] | None,
    params: dict[str, object],
    state_path: str,
) -> dict[str, object]:
    """Handle query requests with embed + traversal timings."""
    query_text = params.get("query")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query must be a non-empty string")

    top_k = _parse_int(params.get("top_k"), "top_k", default=4)

    total_start = time.perf_counter()
    resolved_embed = embed_fn or HashEmbedder().embed

    embed_start = time.perf_counter()
    query_vector = resolved_embed(query_text)
    embed_stop = time.perf_counter()

    expected_dim = meta.get("embedder_dim")
    if isinstance(expected_dim, int) and len(query_vector) != expected_dim:
        raise ValueError(
            f"query embedding dimension mismatch: expected {expected_dim}, got {len(query_vector)}"
        )

    traverse_start = time.perf_counter()
    seeds = index.search(query_vector, top_k=top_k)
    result = traverse(
        graph=graph,
        seeds=seeds,
        config=TraversalConfig(max_hops=15, max_context_chars=20000),
        query_text=query_text,
    )
    traverse_stop = time.perf_counter()
    total_stop = time.perf_counter()

    log_query(
        query_text=query_text,
        fired_ids=result.fired,
        node_count=graph.node_count(),
        journal_path=_journal_path(state_path),
        metadata={"chat_id": params.get("chat_id")},
    )

    return {
        "fired_nodes": result.fired,
        "context": result.context,
        "seeds": [[node_id, score] for node_id, score in seeds],
        "embed_query_ms": _ms(embed_start, embed_stop),
        "traverse_ms": _ms(traverse_start, traverse_stop),
        "total_ms": _ms(total_start, total_stop),
    }


def _handle_inject(
    graph: Graph,
    index: VectorIndex,
    embed_fn: Callable[[str], list[float]] | None,
    params: dict[str, object],
    meta: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Handle injection requests from daemon clients."""
    node_id = params.get("id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("id is required")

    content = params.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content is required")

    node_type = params.get("type")
    if node_type not in {"TEACHING", "CORRECTION", "DIRECTIVE"}:
        raise ValueError("type must be one of: TEACHING, CORRECTION, DIRECTIVE")

    raw_metadata = params.get("metadata")
    if raw_metadata is not None and not isinstance(raw_metadata, dict):
        raise ValueError("metadata must be an object")

    metadata = dict(raw_metadata or {})
    metadata.setdefault("type", node_type)
    metadata.setdefault("source", "daemon")
    resolved_embed = embed_fn or HashEmbedder().embed
    resolved_meta = str(meta.get("embedder_name", "hash-v1"))
    connect_min_sim = 0.0 if resolved_meta == "hash-v1" else 0.3

    existing = graph.get_node(node_id) is not None
    result = inject_node(
        graph=graph,
        index=index,
        node_id=node_id,
        content=content,
        metadata=metadata,
        embed_fn=resolved_embed,
        connect_min_sim=connect_min_sim,
    )

    return {"injected": not existing, "node_id": result["node_id"]}, not existing


def _do_learn(
    graph: Graph,
    index: VectorIndex,
    embed_fn: Callable[[str], list[float]] | None,
    state_path: str,
    *,
    fired_ids: list[str],
    outcome: float,
    content: str = "",
    node_type: str = "CORRECTION",
    source: str = "daemon",
    log_metadata: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    """Unified learning: penalize/reinforce fired path + optionally inject node.

    All three public handlers (learn, correction, self_learn) delegate here.
    """
    should_write = False
    edges_updated = 0

    # 1. Apply outcome to fired path
    if fired_ids and outcome != 0:
        updates = apply_outcome_pg(
            graph=graph, fired_nodes=fired_ids, outcome=outcome,
            baseline=0.0, temperature=1.0,
        )
        edges_updated = _real_graph_updates(updates)
        if edges_updated:
            should_write = True
        log_learn(
            fired_ids=fired_ids,
            outcome=outcome,
            journal_path=_journal_path(state_path),
            metadata=log_metadata,
        )

    # 2. Inject node if content provided
    node_id = None
    node_injected = False
    if content:
        resolved_embed = embed_fn or HashEmbedder().embed
        node_id = _correction_node_id(content)
        if graph.get_node(node_id) is None:
            inject_node(
                graph=graph,
                index=index,
                node_id=node_id,
                content=content,
                metadata={"type": node_type, "source": source},
                embed_fn=resolved_embed,
            )
            node_injected = True
            should_write = True

        # 3. Inhibitory edges for CORRECTION type
        if node_type == "CORRECTION" and fired_ids:
            edges_added = _apply_inhibitory_edges(
                graph=graph,
                source_id=node_id,
                targets=fired_ids,
                inhibition_strength=-0.5,
                inhibition_lr=0.08,
            )
            edges_updated += edges_added
            if edges_added:
                should_write = True

    payload: dict[str, object] = {
        "edges_updated": edges_updated,
        "fired_ids_penalized": fired_ids,
        "node_injected": node_injected,
    }
    if node_id is not None:
        payload["node_id"] = node_id

    return payload, should_write


# ── Thin wrappers: resolve parameters, then delegate to _do_learn ──


def _handle_correction(
    daemon_state: "_DaemonState",
    graph: Graph,
    index: VectorIndex,
    embed_fn: Callable[[str], list[float]] | None,
    state_path: str,
    params: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Human-initiated correction via chat_id lookback."""
    chat_id = _parse_chat_id(params.get("chat_id"), "chat_id", required=True)
    outcome = _parse_float(params.get("outcome"), "outcome", required=True)
    lookback = _parse_int(params.get("lookback"), "lookback", default=1)
    raw_content = params.get("content")
    content = raw_content.strip() if isinstance(raw_content, str) else ""
    fired_ids = _recent_fired_nodes(daemon_state, chat_id, lookback)

    payload, should_write = _do_learn(
        graph, index, embed_fn, state_path,
        fired_ids=fired_ids,
        outcome=outcome,
        content=content,
        node_type="CORRECTION",
        source="daemon",
        log_metadata={"chat_id": chat_id},
    )
    # Backward-compat key name
    payload["correction_injected"] = payload.pop("node_injected")
    return payload, should_write


def _handle_self_learn(
    daemon_state: "_DaemonState",
    graph: Graph,
    index: VectorIndex,
    embed_fn: Callable[[str], list[float]] | None,
    state_path: str,
    params: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Agent-initiated learning — corrections and positive reinforcement."""
    raw_content = params.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ValueError("content is required")

    fired_ids = _parse_str_list(params.get("fired_ids"), "fired_ids", required=False)
    outcome = _parse_float(params.get("outcome"), "outcome", required=False, default=-1.0)

    raw_node_type = params.get("node_type")
    if raw_node_type is None:
        node_type = "CORRECTION"
    elif raw_node_type in {"CORRECTION", "TEACHING"}:
        node_type = raw_node_type
    else:
        raise ValueError("node_type must be one of: CORRECTION, TEACHING")

    return _do_learn(
        graph, index, embed_fn, state_path,
        fired_ids=fired_ids,
        outcome=outcome,
        content=raw_content.strip(),
        node_type=node_type,
        source="self",
        log_metadata={"source": "self"},
    )


def _handle_learn(graph: Graph, index: VectorIndex, embed_fn: Callable[[str], list[float]] | None,
                   state_path: str, params: dict[str, object]) -> tuple[dict[str, object], bool]:
    """Bare learning — just outcome on fired nodes, no injection."""
    fired_nodes = _parse_str_list(params.get("fired_nodes"), "fired_nodes")
    outcome = _parse_float(params.get("outcome"), "outcome", required=True)

    return _do_learn(
        graph, index, embed_fn, state_path,
        fired_ids=fired_nodes,
        outcome=outcome,
    )


def _handle_maintain(
    daemon_state: "_DaemonState",
    params: dict[str, object],
    embed_fn: Callable[[str], list[float]] | None,
    state_path: str,
) -> tuple[dict[str, object], bool]:
    """Handle maintenance request by delegating to shared maintenance workflow."""
    raw_tasks = params.get("tasks")

    if raw_tasks is None:
        requested = None
    elif isinstance(raw_tasks, str):
        requested = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    elif isinstance(raw_tasks, list):
        requested = [task for task in raw_tasks if isinstance(task, str)]
    else:
        raise ValueError("tasks must be a list or comma-separated string")

    dry_run = params.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")

    max_merges = _parse_int(params.get("max_merges"), "max_merges", default=5)
    prune_below = _parse_float(params.get("prune_below"), "prune_below", required=False, default=0.01)
    report = run_maintenance(
        state_path=state_path,
        tasks=requested,
        embed_fn=embed_fn,
        llm_fn=None,
        journal_path=_journal_path(state_path),
        dry_run=dry_run,
        max_merges=max_merges,
        prune_below=prune_below,
    )
    should_write = False
    if not dry_run and any(task in report.tasks_run for task in ("decay", "scale", "split", "prune", "merge", "connect")):
        daemon_state.graph, daemon_state.index, daemon_state.meta = _load_new_state(state_path)
        should_write = True

    return asdict(report), should_write


def _handle_health(graph: Graph) -> dict[str, object]:
    """Handle health request."""
    return _health_payload(graph)


def _handle_info(graph: Graph, meta: dict[str, object]) -> dict[str, object]:
    """Handle info request."""
    return {
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "embedder": str(meta.get("embedder_name", "hash-v1")),
    }


def _load_new_state(state_path: str) -> tuple[Graph, VectorIndex, dict[str, object]]:
    """Reload state.json from disk."""
    graph, index, meta = load_state(state_path)
    return graph, index, dict(meta)


@dataclass
class _DaemonState:
    graph: Graph
    index: VectorIndex
    meta: dict[str, object]
    fired_log: dict[str, deque[dict[str, object]]]
    write_count: int = 0


def main(argv: list[str] | None = None) -> int:
    """Run NDJSON worker loop."""
    args = _build_parser().parse_args(argv)
    state_path = str(Path(args.state).expanduser())

    graph, index, meta = load_state(state_path)
    daemon_state = _DaemonState(graph=graph, index=index, meta=meta, fired_log={})
    embed_fn = _make_embed_fn(args.embed_model)
    auto_save_interval = max(1, args.auto_save_interval) if args.auto_save_interval > 0 else 0

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    stop_requested = False

    def _on_signal(_sig: int, _frame: object | None) -> None:
        nonlocal stop_requested
        stop_requested = True

    prev_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        for raw_line in sys.stdin:
            if stop_requested:
                break

            line = raw_line.strip()
            if not line:
                continue

            req_id: object = None
            should_write = False
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                req_id = request.get("id")
                method = request.get("method")
                params = request.get("params", {})
                if not isinstance(params, dict):
                    raise ValueError("params must be a JSON object")
                if not isinstance(method, str):
                    raise ValueError("method must be a string")

                if method == "query":
                    payload = _handle_query(
                        graph=daemon_state.graph,
                        index=daemon_state.index,
                        meta=daemon_state.meta,
                        embed_fn=embed_fn,
                        params=params,
                        state_path=state_path,
                    )
                    query_chat_id = _parse_chat_id(params.get("chat_id"), "chat_id", required=False)
                    if query_chat_id is not None:
                        query_nodes = payload.get("fired_nodes", [])
                        if not isinstance(query_nodes, list):
                            query_nodes = []
                        query_text = params.get("query")
                        if not isinstance(query_text, str):
                            query_text = ""

                        _append_fired_history(
                            daemon_state=daemon_state,
                            chat_id=query_chat_id,
                            query=query_text,
                            fired_nodes=[node_id for node_id in query_nodes if isinstance(node_id, str)],
                            timestamp=time.time(),
                        )
                        _append_fired_log(
                            state_path=state_path,
                            entry={
                                "chat_id": query_chat_id,
                                "query": query_text,
                                "fired_nodes": [node_id for node_id in query_nodes if isinstance(node_id, str)],
                                "ts": time.time(),
                            },
                        )
                elif method == "learn":
                    payload, should_write = _handle_learn(
                        daemon_state.graph, daemon_state.index, embed_fn,
                        state_path, params,
                    )
                elif method == "inject":
                    payload, should_write = _handle_inject(
                        graph=daemon_state.graph,
                        index=daemon_state.index,
                        embed_fn=embed_fn,
                        params=params,
                        meta=daemon_state.meta,
                    )
                elif method == "maintain":
                    payload, should_write = _handle_maintain(
                        daemon_state,
                        params,
                        embed_fn,
                        state_path,
                    )
                elif method == "health":
                    payload = _handle_health(daemon_state.graph)
                elif method == "info":
                    payload = _handle_info(daemon_state.graph, daemon_state.meta)
                elif method == "save":
                    save_state(
                        graph=daemon_state.graph,
                        index=daemon_state.index,
                        path=state_path,
                        meta=daemon_state.meta,
                    )
                    daemon_state.write_count = 0
                    payload = {"saved": True}
                elif method == "reload":
                    daemon_state.graph, daemon_state.index, daemon_state.meta = _load_new_state(state_path)
                    daemon_state.write_count = 0
                    payload = {"reloaded": True}
                elif method == "correction":
                    payload, should_write = _handle_correction(
                        daemon_state=daemon_state,
                        graph=daemon_state.graph,
                        index=daemon_state.index,
                        embed_fn=embed_fn,
                        state_path=state_path,
                        params=params,
                    )
                elif method in ("self_learn", "self_correct"):
                    payload, should_write = _handle_self_learn(
                        daemon_state=daemon_state,
                        graph=daemon_state.graph,
                        index=daemon_state.index,
                        embed_fn=embed_fn,
                        state_path=state_path,
                        params=params,
                    )
                elif method == "shutdown":
                    if daemon_state.write_count > 0:
                        save_state(
                            graph=daemon_state.graph,
                            index=daemon_state.index,
                            path=state_path,
                            meta=daemon_state.meta,
                        )
                    _emit_response(req_id, {"shutdown": True})
                    break
                else:
                    _emit_response(req_id, None, {"code": -32601, "message": f"unknown method: {method}"})
                    continue

                if should_write:
                    daemon_state.write_count += 1
                    if auto_save_interval and daemon_state.write_count % auto_save_interval == 0:
                        save_state(
                            graph=daemon_state.graph,
                            index=daemon_state.index,
                            path=state_path,
                            meta=daemon_state.meta,
                        )

                _emit_response(req_id, payload)
            except Exception as exc:  # noqa: BLE001
                _emit_response(req_id, None, {"code": -1, "message": str(exc)})

    except KeyboardInterrupt:
        pass
    finally:
        if stop_requested and daemon_state.write_count > 0:
            save_state(
                graph=daemon_state.graph,
                index=daemon_state.index,
                path=state_path,
                meta=daemon_state.meta,
            )
        signal.signal(signal.SIGINT, prev_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, prev_handlers[signal.SIGTERM])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
