"""Persistent long-running NDJSON worker for OpenClawBrain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import deque
import signal
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable
from contextlib import nullcontext

from .autotune import measure_health
from .graph import Graph, Node
from .hasher import HashEmbedder
from .index import VectorIndex
from .learn import apply_outcome, apply_outcome_pg
from .maintain import run_maintenance
from .inject import _apply_inhibitory_edges, inject_correction, inject_node
from .policy import RoutingPolicy, make_runtime_route_fn
from .protocol import QueryParams, QueryRequest, QueryResponse
from .protocol import (
    parse_bool,
    parse_chat_id,
    parse_float,
    parse_int,
    parse_route_mode,
    parse_str_list,
)
from .route_model import RouteModel
from .storage import EventStore, JsonStateStore, JsonlEventStore, StateStore
from .state_lock import state_write_lock
from .traverse import TraversalConfig, traverse
from .prompt_context import build_prompt_context_ranked_with_stats


FIRED_LOG_LIMIT_PER_CHAT = 100
FIRED_LOG_LIMIT_TOTAL = 1000

FIRED_LOG_FILE = "fired_log.jsonl"
INJECTED_FEEDBACK_LOG_FILE = "injected_feedback.jsonl"
INJECTED_FEEDBACK_TAIL_SIZE = 10_000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openclawbrain daemon")
    parser.add_argument("--state", required=True)
    parser.add_argument("--embed-model", default="auto")
    parser.add_argument("--max-prompt-context-chars", type=int, default=30000)
    parser.add_argument("--max-fired-nodes", type=int, default=30)
    parser.add_argument("--route-mode", choices=["off", "edge", "edge+sim", "learned"], default="off")
    parser.add_argument("--route-top-k", type=int, default=5)
    parser.add_argument("--route-alpha-sim", type=float, default=0.5)
    parser.add_argument("--route-use-relevance", choices=["true", "false"], default="true")
    parser.add_argument("--route-model", default=None)
    parser.add_argument("--auto-save-interval", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Bypass state lock (expert use)")
    return parser


def _make_embed_fn(embed_model: str) -> Callable[[str], list[float]] | None:
    """Create OpenAI embedding callback for a model name."""
    from openai import OpenAI

    client = OpenAI()
    return lambda text: [float(v) for v in client.embeddings.create(model=embed_model, input=[text]).data[0].embedding]


def _resolve_embed_fn(embed_model: str, meta: dict[str, object]) -> Callable[[str], list[float]] | None:
    """Resolve daemon query embedder from CLI flag and state metadata."""
    model = str(embed_model or "").strip()
    model_lower = model.lower()
    embedder_name = meta.get("embedder_name")
    embedder_name_str = embedder_name.strip().lower() if isinstance(embedder_name, str) else ""

    if model_lower in {"auto", ""}:
        if embedder_name_str == "hash-v1":
            return None
        resolved_model = "text-embedding-3-small"
        if "text-embedding-3-small" in embedder_name_str:
            resolved_model = "text-embedding-3-small"
        return _make_embed_fn(resolved_model)
    if model_lower == "hash":
        return None
    return _make_embed_fn(model)


def _journal_path(state_path: str) -> str:
    """Resolve journal.jsonl path for the loaded state directory."""
    return str(Path(state_path).expanduser().parent / "journal.jsonl")


def _fired_log_path(state_path: str) -> str:
    """Resolve fired_log.jsonl path for the loaded state directory."""
    return str(Path(state_path).expanduser().parent / FIRED_LOG_FILE)


def _injected_feedback_log_path(state_path: str) -> str:
    """Resolve injected_feedback.jsonl path for deduped feedback events."""
    return str(Path(state_path).expanduser().parent / INJECTED_FEEDBACK_LOG_FILE)


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


def _append_query_event(
    event_store: EventStore,
    *,
    query_text: str,
    fired_ids: list[str],
    node_count: int | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    resolved_node_count = node_count if node_count is not None else len(fired_ids)
    event_store.append(
        {
            "type": "query",
            "query": query_text,
            "fired": list(fired_ids),
            "fired_count": len(fired_ids),
            "node_count": resolved_node_count,
            "metadata": metadata,
        }
    )


def _append_learn_event(
    event_store: EventStore,
    *,
    fired_ids: list[str],
    outcome: float,
    metadata: dict[str, object] | None = None,
) -> None:
    event_store.append(
        {
            "type": "learn",
            "fired": list(fired_ids),
            "outcome": float(outcome),
            "metadata": metadata,
        }
    )


def _emit_response(req_id: object, payload: object, error: dict[str, object] | None = None) -> None:
    """Emit one NDJSON object."""
    if error is None:
        print(json.dumps(QueryResponse(id=req_id, result=payload).to_dict()), flush=True)
    else:
        print(json.dumps(QueryResponse(id=req_id, error=error).to_dict()), flush=True)


def _build_query_route_fn(
    *,
    route_mode: str,
    route_top_k: int,
    route_alpha_sim: float,
    route_use_relevance: bool,
    query_vector: list[float],
    index: VectorIndex,
) -> Callable[[str | None, list[object], str], list[str]] | None:
    """Backwards-compatible wrapper around policy runtime route fn builder."""
    policy = RoutingPolicy(
        route_mode=parse_route_mode(route_mode),
        top_k=parse_int(route_top_k, "route_top_k", default=5),
        alpha_sim=parse_float(route_alpha_sim, "route_alpha_sim", default=0.5),
        use_relevance=parse_bool(route_use_relevance, "route_use_relevance", default=True),
    )
    return make_runtime_route_fn(policy=policy, query_vector=query_vector, index=index)


def _append_fired_log(state_path: str, entry: dict[str, object]) -> None:
    path = Path(_fired_log_path(state_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _append_injected_feedback_log(state_path: str, entry: dict[str, object]) -> None:
    path = Path(_injected_feedback_log_path(state_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _tail_jsonl(path: Path, *, max_entries: int) -> list[dict[str, object]]:
    """Read last N JSONL objects without loading entire file into memory."""
    if max_entries <= 0 or not path.exists():
        return []

    rows: deque[dict[str, object]] = deque(maxlen=max_entries)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return list(rows)


def _load_injected_feedback_dedup_keys(state_path: str, *, max_entries: int) -> set[str]:
    keys: set[str] = set()
    for row in _tail_jsonl(Path(_injected_feedback_log_path(state_path)), max_entries=max_entries):
        dedup_key = row.get("dedup_key")
        if isinstance(dedup_key, str) and dedup_key:
            keys.add(dedup_key)
    return keys


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


def _feedback_node_id(kind: str, content: str) -> str:
    if kind == "CORRECTION":
        return _correction_node_id(content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{kind.lower()}::{digest[:12]}"


def _handle_query(
    graph: Graph,
    index: VectorIndex,
    meta: dict[str, object],
    embed_fn: Callable[[str], list[float]] | None,
    params: dict[str, object],
    event_store: EventStore | None = None,
    state_path: str | None = None,
    learned_model: RouteModel | None = None,
    target_projections: dict[str, object] | None = None,
    *,
    query_defaults: "QueryDefaults | None" = None,
) -> dict[str, object]:
    """Handle query requests with embed + traversal timings.

    Includes a deterministic `prompt_context` block suitable for prompt caching.
    """
    effective_defaults = query_defaults or QueryDefaults()
    resolved_event_store = event_store
    if resolved_event_store is None:
        if state_path is None:
            raise ValueError("state_path is required when event_store is not provided")
        resolved_event_store = JsonlEventStore(_journal_path(state_path))
    query = QueryParams.from_dict(_with_query_defaults(params, effective_defaults))
    query_text = query.query
    top_k = query.top_k
    max_prompt_chars = query.max_prompt_context_chars
    max_context_chars = query.max_context_chars
    max_fired_nodes = query.max_fired_nodes
    prompt_context_include_node_ids = query.prompt_context_include_node_ids
    exclude_files = set(query.exclude_files)
    exclude_file_prefixes = list(query.exclude_file_prefixes)

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
    policy = RoutingPolicy(
        route_mode=query.route_mode,
        top_k=query.route_top_k,
        alpha_sim=query.route_alpha_sim,
        use_relevance=query.route_use_relevance,
    )
    route_fn = make_runtime_route_fn(
        policy=policy,
        query_vector=query_vector,
        index=index,
        learned_model=learned_model,
        target_projections=target_projections,
    )
    result = traverse(
        graph=graph,
        seeds=seeds,
        config=TraversalConfig(max_hops=15, max_context_chars=max_context_chars, max_fired_nodes=max_fired_nodes),
        query_text=query_text,
        route_fn=route_fn,
    )
    traverse_stop = time.perf_counter()
    total_stop = time.perf_counter()

    fired_node_ids = [str(node_id) for node_id in result.fired]
    fired_node_scores = {str(node_id): float(score) for node_id, score in result.fired_scores.items()}

    excluded_node_ids: set[str] = set()
    excluded_files: set[str] = set()
    if exclude_files or exclude_file_prefixes:
        for node_id in fired_node_ids:
            node = graph.get_node(node_id)
            metadata = node.metadata if node is not None and isinstance(node.metadata, dict) else {}
            file_path = metadata.get("file")
            if not isinstance(file_path, str) or not file_path:
                continue
            if file_path in exclude_files or any(file_path.startswith(prefix) for prefix in exclude_file_prefixes):
                excluded_node_ids.add(node_id)
                excluded_files.add(file_path)

    prompt_node_ids = [node_id for node_id in fired_node_ids if node_id not in excluded_node_ids]
    prompt_node_scores = {node_id: score for node_id, score in fired_node_scores.items() if node_id in prompt_node_ids}

    prompt_context, prompt_context_stats = build_prompt_context_ranked_with_stats(
        graph=graph,
        node_ids=prompt_node_ids,
        node_scores=prompt_node_scores,
        max_chars=max_prompt_chars,
        include_node_ids=prompt_context_include_node_ids,
    )
    prompt_context_stats["prompt_context_excluded_files_count"] = len(excluded_files)
    prompt_context_stats["prompt_context_excluded_node_ids_count"] = len(excluded_node_ids)
    _append_query_event(
        resolved_event_store,
        query_text=query_text,
        fired_ids=result.fired,
        node_count=graph.node_count(),
        metadata={"chat_id": query.chat_id, "max_fired_nodes": max_fired_nodes, **prompt_context_stats},
    )

    return {
        "fired_nodes": result.fired,
        "max_fired_nodes": max_fired_nodes,
        "context": result.context,
        "prompt_context": prompt_context,
        **prompt_context_stats,
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
    event_store: EventStore,
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
        _append_learn_event(
            event_store,
            fired_ids=fired_ids,
            outcome=outcome,
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
    event_store: EventStore | None = None,
    state_path: str | None = None,
    params: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    """Human-initiated correction via chat_id lookback."""
    resolved_params = params or {}
    chat_id = parse_chat_id(resolved_params.get("chat_id"), "chat_id", required=True)
    outcome = parse_float(resolved_params.get("outcome"), "outcome", required=True)
    lookback = parse_int(resolved_params.get("lookback"), "lookback", default=1)
    raw_content = resolved_params.get("content")
    content = raw_content.strip() if isinstance(raw_content, str) else ""
    fired_ids = _recent_fired_nodes(daemon_state, chat_id, lookback)

    resolved_event_store = event_store
    if resolved_event_store is None:
        if state_path is None:
            raise ValueError("state_path is required when event_store is not provided")
        resolved_event_store = JsonlEventStore(_journal_path(state_path))

    payload, should_write = _do_learn(
        graph, index, embed_fn, resolved_event_store,
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


def _handle_last_fired(
    daemon_state: "_DaemonState",
    params: dict[str, object],
) -> dict[str, object]:
    """Return recent fired node ids for a chat_id."""
    chat_id = parse_chat_id(params.get("chat_id"), "chat_id", required=True)
    lookback = parse_int(params.get("lookback"), "lookback", default=1)
    return {
        "chat_id": chat_id,
        "lookback": lookback,
        "fired_nodes": _recent_fired_nodes(daemon_state, chat_id, lookback),
    }


def _handle_learn_by_chat_id(
    daemon_state: "_DaemonState",
    graph: Graph,
    index: VectorIndex,
    embed_fn: Callable[[str], list[float]] | None,
    event_store: EventStore | None = None,
    state_path: str | None = None,
    params: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    """Learn by chat_id lookback using in-memory fired history only."""
    resolved_params = params or {}
    chat_id = parse_chat_id(resolved_params.get("chat_id"), "chat_id", required=True)
    outcome = parse_float(resolved_params.get("outcome"), "outcome", required=True)
    lookback = parse_int(resolved_params.get("lookback"), "lookback", default=1)
    fired_ids = _recent_fired_nodes(daemon_state, chat_id, lookback)
    resolved_event_store = event_store
    if resolved_event_store is None:
        if state_path is None:
            raise ValueError("state_path is required when event_store is not provided")
        resolved_event_store = JsonlEventStore(_journal_path(state_path))
    return _do_learn(
        graph,
        index,
        embed_fn,
        resolved_event_store,
        fired_ids=fired_ids,
        outcome=outcome,
        log_metadata={"chat_id": chat_id},
    )


def _handle_self_learn(
    daemon_state: "_DaemonState",
    graph: Graph,
    index: VectorIndex,
    embed_fn: Callable[[str], list[float]] | None,
    event_store: EventStore | None = None,
    state_path: str | None = None,
    params: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    """Agent-initiated learning — corrections and positive reinforcement."""
    resolved_params = params or {}
    raw_content = resolved_params.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ValueError("content is required")

    fired_ids = parse_str_list(resolved_params.get("fired_ids"), "fired_ids", required=False)
    outcome = parse_float(resolved_params.get("outcome"), "outcome", required=False, default=-1.0)

    raw_node_type = resolved_params.get("node_type")
    if raw_node_type is None:
        node_type = "CORRECTION"
    elif raw_node_type in {"CORRECTION", "TEACHING"}:
        node_type = raw_node_type
    else:
        raise ValueError("node_type must be one of: CORRECTION, TEACHING")

    resolved_event_store = event_store
    if resolved_event_store is None:
        if state_path is None:
            raise ValueError("state_path is required when event_store is not provided")
        resolved_event_store = JsonlEventStore(_journal_path(state_path))

    return _do_learn(
        graph, index, embed_fn, resolved_event_store,
        fired_ids=fired_ids,
        outcome=outcome,
        content=raw_content.strip(),
        node_type=node_type,
        source="self",
        log_metadata={"source": "self"},
    )


def _handle_capture_feedback(
    daemon_state: "_DaemonState",
    graph: Graph,
    index: VectorIndex,
    meta: dict[str, object],
    embed_fn: Callable[[str], list[float]] | None,
    event_store: EventStore | None = None,
    state_path: str | None = None,
    params: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    """Capture real-time correction/teaching/directive with optional learning + dedup."""
    resolved_params = params or {}
    resolved_event_store = event_store
    if resolved_event_store is None:
        if state_path is None:
            raise ValueError("state_path is required when event_store is not provided")
        resolved_event_store = JsonlEventStore(_journal_path(state_path))
    chat_id = parse_chat_id(resolved_params.get("chat_id"), "chat_id", required=True)
    lookback = parse_int(resolved_params.get("lookback"), "lookback", default=1)

    kind = resolved_params.get("kind")
    if kind not in {"CORRECTION", "TEACHING", "DIRECTIVE"}:
        raise ValueError("kind must be one of: CORRECTION, TEACHING, DIRECTIVE")

    raw_content = resolved_params.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ValueError("content is required")
    content = raw_content.strip()

    dedup_key_used: str | None = None
    dedup_key = resolved_params.get("dedup_key")
    message_id = resolved_params.get("message_id")
    if dedup_key is not None:
        if not isinstance(dedup_key, str):
            raise ValueError("dedup_key must be a string")
        dedup_key_used = dedup_key.strip()
        if not dedup_key_used:
            raise ValueError("dedup_key must be a non-empty string")
    elif message_id is not None:
        if not isinstance(message_id, str):
            raise ValueError("message_id must be a string")
        dedup_key_used = message_id.strip()
        if not dedup_key_used:
            raise ValueError("message_id must be a non-empty string")

    fired_ids = _recent_fired_nodes(daemon_state, chat_id, lookback)

    if dedup_key_used is not None and dedup_key_used in daemon_state.feedback_dedup_keys:
        deduped_payload: dict[str, object] = {
            "deduped": True,
            "edges_updated": 0,
            "fired_ids_used": fired_ids,
        }
        deduped_payload["dedup_key_used"] = dedup_key_used
        return deduped_payload, False

    outcome_used: float | None = None
    if resolved_params.get("outcome") is not None:
        outcome_used = parse_float(resolved_params.get("outcome"), "outcome", required=True)
    elif kind == "CORRECTION":
        outcome_used = -1.0

    node_id = _feedback_node_id(kind, content)
    injected_node_id: str | None = None
    should_write = False

    if graph.get_node(node_id) is None:
        resolved_embed = embed_fn or HashEmbedder().embed
        node_metadata = {
            "type": kind,
            "source": "capture_feedback",
            "chat_id": chat_id,
        }
        if kind == "CORRECTION":
            inject_correction(
                graph=graph,
                index=index,
                node_id=node_id,
                content=content,
                metadata=node_metadata,
                embed_fn=resolved_embed,
            )
        else:
            resolved_meta = str(meta.get("embedder_name", "hash-v1"))
            connect_min_sim = 0.0 if resolved_meta == "hash-v1" else 0.3
            inject_node(
                graph=graph,
                index=index,
                node_id=node_id,
                content=content,
                metadata=node_metadata,
                embed_fn=resolved_embed,
                connect_min_sim=connect_min_sim,
            )
        injected_node_id = node_id
        should_write = True

    edges_updated = 0
    if outcome_used is not None and fired_ids:
        updates = apply_outcome_pg(
            graph=graph,
            fired_nodes=fired_ids,
            outcome=outcome_used,
            baseline=0.0,
            temperature=1.0,
        )
        edges_updated = _real_graph_updates(updates)
        if edges_updated:
            should_write = True
        _append_learn_event(
            resolved_event_store,
            fired_ids=fired_ids,
            outcome=outcome_used,
            metadata={"chat_id": chat_id, "source": "capture_feedback", "kind": kind},
        )

    if dedup_key_used is not None:
        _append_injected_feedback_log(
            state_path=state_path,
            entry={
                "dedup_key": dedup_key_used,
                "chat_id": chat_id,
                "kind": kind,
                "node_id": node_id,
                "ts": time.time(),
            },
        )
        daemon_state.feedback_dedup_keys.add(dedup_key_used)

    payload: dict[str, object] = {
        "deduped": False,
        "edges_updated": edges_updated,
        "fired_ids_used": fired_ids,
    }
    if dedup_key_used is not None:
        payload["dedup_key_used"] = dedup_key_used
    if injected_node_id is not None:
        payload["injected_node_id"] = injected_node_id
    if outcome_used is not None:
        payload["outcome_used"] = outcome_used
    return payload, should_write


def _handle_learn(graph: Graph, index: VectorIndex, embed_fn: Callable[[str], list[float]] | None,
                   event_store: EventStore | None = None, state_path: str | None = None, params: dict[str, object] | None = None) -> tuple[dict[str, object], bool]:
    """Bare learning — just outcome on fired nodes, no injection."""
    resolved_params = params or {}
    fired_nodes = parse_str_list(resolved_params.get("fired_nodes"), "fired_nodes")
    outcome = parse_float(resolved_params.get("outcome"), "outcome", required=True)
    resolved_event_store = event_store
    if resolved_event_store is None:
        if state_path is None:
            raise ValueError("state_path is required when event_store is not provided")
        resolved_event_store = JsonlEventStore(_journal_path(state_path))

    return _do_learn(
        graph, index, embed_fn, resolved_event_store,
        fired_ids=fired_nodes,
        outcome=outcome,
    )


def _handle_maintain(
    daemon_state: "_DaemonState",
    params: dict[str, object],
    embed_fn: Callable[[str], list[float]] | None,
    state_path: str,
    state_store: StateStore,
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

    max_merges = parse_int(params.get("max_merges"), "max_merges", default=5)
    prune_below = parse_float(params.get("prune_below"), "prune_below", required=False, default=0.01)
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
        daemon_state.graph, daemon_state.index, daemon_state.meta = _load_new_state(state_path, state_store)
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


def _load_new_state(state_path: str, state_store: StateStore) -> tuple[Graph, VectorIndex, dict[str, object]]:
    """Reload state.json from disk."""
    graph, index, meta = state_store.load(state_path)
    return graph, index, dict(meta)


@dataclass
class _DaemonState:
    graph: Graph
    index: VectorIndex
    meta: dict[str, object]
    fired_log: dict[str, deque[dict[str, object]]]
    feedback_dedup_keys: set[str] = field(default_factory=set)
    write_count: int = 0


@dataclass(frozen=True)
class QueryDefaults:
    max_prompt_context_chars: int = 30000
    max_fired_nodes: int = 30
    route_mode: str = "off"
    route_top_k: int = 5
    route_alpha_sim: float = 0.5
    route_use_relevance: bool = True


def _with_query_defaults(params: dict[str, object], defaults: QueryDefaults) -> dict[str, object]:
    merged = dict(params)
    merged.setdefault("max_prompt_context_chars", defaults.max_prompt_context_chars)
    merged.setdefault("max_context_chars", defaults.max_prompt_context_chars)
    merged.setdefault("max_fired_nodes", defaults.max_fired_nodes)
    merged.setdefault("route_mode", defaults.route_mode)
    merged.setdefault("route_top_k", defaults.route_top_k)
    merged.setdefault("route_alpha_sim", defaults.route_alpha_sim)
    merged.setdefault("route_use_relevance", defaults.route_use_relevance)
    return merged


def main(argv: list[str] | None = None) -> int:
    """Run NDJSON worker loop."""
    args = _build_parser().parse_args(argv)
    state_path = str(Path(args.state).expanduser())
    journal_path = _journal_path(state_path)
    state_store = JsonStateStore()
    event_store = JsonlEventStore(journal_path)
    route_model_path = (
        Path(args.route_model).expanduser()
        if args.route_model
        else Path(state_path).expanduser().parent / "route_model.npz"
    )
    lock_cm = state_write_lock(state_path, force=args.force, command_hint="openclawbrain daemon")
    with lock_cm if state_path else nullcontext():
        graph, index, meta = state_store.load(state_path)
        route_model: RouteModel | None = None
        target_projections: dict[str, object] = {}
        if route_model_path.exists():
            route_model = RouteModel.load_npz(route_model_path)
            target_projections = route_model.precompute_target_projections(index)
        daemon_state = _DaemonState(
            graph=graph,
            index=index,
            meta=meta,
            fired_log={},
            feedback_dedup_keys=_load_injected_feedback_dedup_keys(
                state_path=state_path,
                max_entries=INJECTED_FEEDBACK_TAIL_SIZE,
            ),
        )
        embed_fn = _resolve_embed_fn(args.embed_model, meta)
        query_defaults = QueryDefaults(
            max_prompt_context_chars=parse_int(
                args.max_prompt_context_chars,
                "max_prompt_context_chars",
                default=30000,
            ),
            max_fired_nodes=parse_int(
                args.max_fired_nodes,
                "max_fired_nodes",
                default=30,
            ),
            route_mode=parse_route_mode(args.route_mode),
            route_top_k=parse_int(
                args.route_top_k,
                "route_top_k",
                default=5,
            ),
            route_alpha_sim=parse_float(
                args.route_alpha_sim,
                "route_alpha_sim",
                default=0.5,
            ),
            route_use_relevance=str(args.route_use_relevance).strip().lower() == "true",
        )
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
                    request_envelope = QueryRequest.from_dict(json.loads(line))
                    req_id = request_envelope.id
                    method = request_envelope.method
                    params = request_envelope.params

                    if method == "query":
                        payload = _handle_query(
                            graph=daemon_state.graph,
                            index=daemon_state.index,
                            meta=daemon_state.meta,
                            embed_fn=embed_fn,
                            params=params,
                            event_store=event_store,
                            learned_model=route_model,
                            target_projections=target_projections,
                            query_defaults=query_defaults,
                        )
                        query_chat_id = parse_chat_id(params.get("chat_id"), "chat_id", required=False)
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
                            daemon_state.graph,
                            daemon_state.index,
                            embed_fn,
                            event_store=event_store,
                            params=params,
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
                            state_store,
                        )
                    elif method == "health":
                        payload = _handle_health(daemon_state.graph)
                    elif method == "info":
                        payload = _handle_info(daemon_state.graph, daemon_state.meta)
                    elif method == "save":
                        state_store.save(state_path, graph=daemon_state.graph, index=daemon_state.index, meta=daemon_state.meta)
                        daemon_state.write_count = 0
                        payload = {"saved": True}
                    elif method == "reload":
                        daemon_state.graph, daemon_state.index, daemon_state.meta = _load_new_state(state_path, state_store)
                        embed_fn = _resolve_embed_fn(args.embed_model, daemon_state.meta)
                        target_projections = route_model.precompute_target_projections(daemon_state.index) if route_model else {}
                        daemon_state.write_count = 0
                        payload = {"reloaded": True}
                    elif method == "correction":
                        payload, should_write = _handle_correction(
                            daemon_state=daemon_state,
                            graph=daemon_state.graph,
                            index=daemon_state.index,
                            embed_fn=embed_fn,
                            event_store=event_store,
                            params=params,
                        )
                    elif method == "last_fired":
                        payload = _handle_last_fired(
                            daemon_state=daemon_state,
                            params=params,
                        )
                    elif method == "learn_by_chat_id":
                        payload, should_write = _handle_learn_by_chat_id(
                            daemon_state=daemon_state,
                            graph=daemon_state.graph,
                            index=daemon_state.index,
                            embed_fn=embed_fn,
                            event_store=event_store,
                            params=params,
                        )
                    elif method == "capture_feedback":
                        payload, should_write = _handle_capture_feedback(
                            daemon_state=daemon_state,
                            graph=daemon_state.graph,
                            index=daemon_state.index,
                            meta=daemon_state.meta,
                            embed_fn=embed_fn,
                            event_store=event_store,
                            state_path=state_path,
                            params=params,
                        )
                    elif method in ("self_learn", "self_correct"):
                        payload, should_write = _handle_self_learn(
                            daemon_state=daemon_state,
                            graph=daemon_state.graph,
                            index=daemon_state.index,
                            embed_fn=embed_fn,
                            event_store=event_store,
                            params=params,
                        )
                    elif method == "shutdown":
                        if daemon_state.write_count > 0:
                            state_store.save(state_path, graph=daemon_state.graph, index=daemon_state.index, meta=daemon_state.meta)
                        _emit_response(req_id, {"shutdown": True})
                        break
                    else:
                        _emit_response(req_id, None, {"code": -32601, "message": f"unknown method: {method}"})
                        continue

                    if should_write:
                        daemon_state.write_count += 1
                        if auto_save_interval and daemon_state.write_count % auto_save_interval == 0:
                            state_store.save(state_path, graph=daemon_state.graph, index=daemon_state.index, meta=daemon_state.meta)

                    _emit_response(req_id, payload)
                except Exception as exc:  # noqa: BLE001
                    _emit_response(req_id, None, {"code": -1, "message": str(exc)})

        except KeyboardInterrupt:
            pass
        finally:
            if stop_requested and daemon_state.write_count > 0:
                state_store.save(state_path, graph=daemon_state.graph, index=daemon_state.index, meta=daemon_state.meta)
            signal.signal(signal.SIGINT, prev_handlers[signal.SIGINT])
            signal.signal(signal.SIGTERM, prev_handlers[signal.SIGTERM])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    resolved_event_store = event_store
    if resolved_event_store is None:
        if state_path is None:
            raise ValueError("state_path is required when event_store is not provided")
        resolved_event_store = JsonlEventStore(_journal_path(state_path))
