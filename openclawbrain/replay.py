from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .decay import apply_decay
from .graph import Graph
from .learn import LearningConfig, apply_outcome
from .traverse import TraversalConfig, traverse
from ._util import _tokenize

DEFAULT_TOOL_RESULT_ALLOWLIST = frozenset(
    {
        "image",
        "openai-whisper",
        "openai-whisper-api",
        "openai-whisper-local",
        "summarize",
    }
)
DEFAULT_TOOL_RESULT_MAX_CHARS = 20_000


def _extract_user_query_content(content: object) -> str | None:
    """ extract user query content."""
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


def _extract_message_payload(payload: dict) -> dict | None:
    """ extract message payload."""
    if not isinstance(payload, dict):
        return None

    if payload.get("type") == "message":
        message = payload.get("message")
        return message if isinstance(message, dict) else None

    role = payload.get("role")
    if isinstance(role, str) and role.strip().lower() in {"user", "assistant", "toolresult", "tool_result"}:
        return payload
    return None


def _normalize_tool_result_allowlist(tool_names: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    """Normalize a tool-result allowlist to lowercase tool names."""
    if tool_names is None:
        tool_names = set(DEFAULT_TOOL_RESULT_ALLOWLIST)
    normalized: set[str] = set()
    for name in tool_names:
        if not isinstance(name, str):
            continue
        value = name.strip().lower()
        if value:
            normalized.add(value)
    return normalized


def _is_media_stub_query(query: str) -> bool:
    """Detect OpenClaw media attachment stubs in user query text."""
    text = query.strip().lower()
    if not text:
        return False
    if "[media attached:" in text:
        return True
    if "media attached" in text and any(marker in text for marker in ("audio/", "image/", "video/", "application/pdf")):
        return True
    return False


def _extract_tool_result(message: dict) -> tuple[str | None, str | None]:
    """Extract tool name and text body from a toolResult message payload."""
    tool_name = message.get("toolName")
    if isinstance(tool_name, str):
        normalized_name = tool_name.strip().lower()
        if not normalized_name:
            normalized_name = None
    else:
        normalized_name = None

    text = _extract_user_query_content(message.get("content"))
    return normalized_name, text


def _extract_openclaw_query(payload: dict) -> str | None:
    """ extract openclaw query."""
    message = _extract_message_payload(payload)
    if message is None or message.get("role") != "user":
        return None

    return _extract_user_query_content(message.get("content"))


def _extract_openclaw_response(payload: dict) -> str | None:
    """ extract openclaw response."""
    message = _extract_message_payload(payload)
    if message is None or message.get("role") != "assistant":
        return None

    return _extract_user_query_content(message.get("content"))


def _extract_flat_query(payload: dict) -> str | None:
    """ extract flat query."""
    if payload.get("role") != "user":
        return None
    return _extract_user_query_content(payload.get("content"))


def _extract_flat_response(payload: dict) -> str | None:
    """ extract flat response."""
    if payload.get("role") != "assistant":
        return None
    return _extract_user_query_content(payload.get("content"))


def _extract_tool_call(payload: dict) -> dict[str, object] | None:
    """ extract tool call."""
    function = payload.get("function")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if name is None and isinstance(function, dict):
        fn_name = function.get("name")
        if isinstance(fn_name, str):
            name = fn_name
        if arguments is None:
            arguments = function.get("arguments")

    call: dict[str, object] = {}
    if isinstance(payload.get("id"), str):
        call["id"] = payload["id"]
    if isinstance(name, str) and name.strip():
        call["name"] = name
    if arguments is not None:
        call["arguments"] = arguments

    return call or None


def _extract_tool_calls(payload: dict, *, is_assistant: bool) -> list[dict[str, object]]:
    """ extract tool calls."""
    if not is_assistant:
        return []

    calls: list[dict[str, object]] = []
    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            call = _extract_tool_call(item)
            if call is not None:
                calls.append(call)

    content = payload.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_call":
                continue
            call = _extract_tool_call(item)
            if call is not None:
                calls.append(call)

    return calls


def _extract_query_timestamp(payload: dict) -> float | None:
    """ extract query timestamp."""
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

    message = _extract_message_payload(payload)
    if message is not None:
        for key in timestamp_keys:
            value = message.get(key)
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
    """ extract query record."""
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


def extract_interactions(
    session_path: str | Path,
    since_ts: float | None = None,
    *,
    include_tool_results: bool = True,
    tool_result_allowlist: set[str] | list[str] | tuple[str, ...] | None = None,
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> list[dict[str, object]]:
    """Extract user/assistant interactions from a session log.

    Output entries use:
    {"query": <user message>, "response": <assistant message>, "tool_calls": [...], "ts": <float|None>}
    """
    path = Path(session_path).expanduser()
    if not path.exists():
        print(f"warning: skipping missing session file: {path}", file=sys.stderr)
        return []

    interactions: list[dict[str, object]] = []
    last_user_index: int | None = None
    tool_result_allow = _normalize_tool_result_allowlist(tool_result_allowlist)
    max_tool_chars = max(0, int(tool_result_max_chars))
    appended_tool_chars: dict[int, int] = {}

    try:
        fh = path.open("r", encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        print(f"warning: skipping unreadable session file: {path} ({exc})", file=sys.stderr)
        return []

    for raw in fh:
        raw = raw.strip()
        if not raw:
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            interactions.append({"query": raw, "response": None, "tool_calls": [], "ts": None})
            continue

        if not isinstance(payload, dict):
            continue

        message = _extract_message_payload(payload)
        if message is None:
            continue
        role = message.get("role")
        if not isinstance(role, str):
            continue

        record_ts = _extract_query_timestamp(payload)
        if role == "user":
            query = _extract_user_query_content(message.get("content"))
            if query is None:
                last_user_index = None
                continue
            if since_ts is not None and record_ts is not None and record_ts <= since_ts:
                last_user_index = None
                continue
            interactions.append({"query": query, "response": None, "tool_calls": [], "ts": record_ts})
            last_user_index = len(interactions) - 1
            appended_tool_chars[last_user_index] = 0
            continue

        normalized_role = role.strip().lower()
        if normalized_role in {"toolresult", "tool_result"}:
            if (
                include_tool_results
                and last_user_index is not None
                and max_tool_chars > 0
                and isinstance(interactions[last_user_index].get("query"), str)
            ):
                base_query = interactions[last_user_index]["query"]
                if isinstance(base_query, str) and _is_media_stub_query(base_query):
                    tool_name, tool_text = _extract_tool_result(message)
                    if tool_name in tool_result_allow and isinstance(tool_text, str):
                        cleaned = tool_text.strip()
                        if cleaned:
                            used = appended_tool_chars.get(last_user_index, 0)
                            remaining = max_tool_chars - used
                            if remaining > 0:
                                snippet = cleaned[:remaining]
                                interactions[last_user_index]["query"] = (
                                    f"{base_query}\n\n[toolResult:{tool_name}] {snippet}"
                                )
                                appended_tool_chars[last_user_index] = used + len(snippet)
            continue

        if normalized_role != "assistant":
            continue

        response = _extract_user_query_content(message.get("content"))
        tool_calls = _extract_tool_calls(
            message,
            is_assistant=True,
        )
        assistant_record = {
            "query": None,
            "response": response,
            "tool_calls": tool_calls,
            "ts": record_ts,
        }

        if last_user_index is not None:
            if interactions[last_user_index]["query"] is not None and (
                since_ts is None or record_ts is None or record_ts > since_ts
            ):
                interactions[last_user_index]["response"] = response
                interactions[last_user_index]["tool_calls"] = tool_calls
                if record_ts is not None:
                    interactions[last_user_index]["ts"] = record_ts
            else:
                interactions[last_user_index]["response"] = None
                interactions[last_user_index]["tool_calls"] = []
            continue

        if since_ts is not None and record_ts is not None and record_ts <= since_ts:
            continue
        interactions.append(assistant_record)

    return interactions


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
        print(f"warning: skipping missing session file: {path}", file=sys.stderr)
        return []

    queries: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        print(f"warning: skipping unreadable session file: {path} ({exc})", file=sys.stderr)
        return []
    with handle:
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
        print(f"warning: skipping missing session file: {path}", file=sys.stderr)
        return []

    records: list[tuple[str, float | None]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        print(f"warning: skipping unreadable session file: {path} ({exc})", file=sys.stderr)
        return []
    with handle:
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
    """extract queries from dir."""
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


def _auto_score_query_outcome(
    outcome: float,
    response: str | None,
    fired_nodes: list[str],
    graph: Graph,
) -> float:
    """Heuristically score a query outcome based on fired node overlap."""
    if response is None:
        return outcome

    response_text = response.lower()
    if not response_text:
        return outcome

    for node_id in set(fired_nodes):
        node = graph.get_node(node_id)
        if node is None or not node.content:
            continue
        if node.content.lower() in response_text:
            return min(2.0, outcome + 0.5)

    return outcome


def _snapshot_edges(graph: Graph) -> dict[tuple[str, str], float]:
    """ snapshot edges."""
    weights: dict[tuple[str, str], float] = {}
    for source_id, edges in graph._edges.items():
        for target_id, edge in edges.items():
            weights[(source_id, target_id)] = edge.weight
    return weights


def _cross_file_edges(graph: Graph) -> set[tuple[str, str]]:
    """ cross file edges."""
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


def _interaction_query_outcome(entry: dict[str, object]) -> tuple[str, float | None, str | None, list[dict[str, object]]]:
    """ interaction query outcome."""
    raw_query = entry.get("query")
    query: str | None = None
    if isinstance(raw_query, str):
        query = raw_query
    elif raw_query is not None:
        query = str(raw_query)

    response = entry.get("response")
    response_text = response if isinstance(response, str) else None

    query_ts = _extract_query_timestamp(entry)
    if query_ts is None and "ts" in entry and isinstance(entry.get("ts"), (float, int)):
        query_ts = float(entry["ts"])  # type: ignore[assignment]

    raw_tool_calls = entry.get("tool_calls")
    tool_calls = [tool for tool in raw_tool_calls if isinstance(tool, dict)] if isinstance(raw_tool_calls, list) else []

    return query, query_ts, response_text, tool_calls


def _normalize_replay_queries(
    queries: list[str | tuple[str, float | None] | dict[str, object]],
    *,
    since_ts: float | None = None,
) -> list[tuple[str, float | None, str | None, list[dict[str, object]]]]:
    """Normalize replay inputs into (query, ts, response, tool_calls)."""
    normalized_queries: list[tuple[str, float | None, str | None, list[dict[str, object]]]] = []
    for entry in queries:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
            query, query_ts = entry
            if not query.strip():
                continue
            normalized_queries.append((query, query_ts, None, []))
            continue
        if isinstance(entry, dict):
            query, query_ts, response_text, tool_calls = _interaction_query_outcome(entry)
            if query is None or not query.strip():
                continue
            normalized_queries.append((query, query_ts, response_text, tool_calls))
            continue

        query = str(entry)
        if query.strip():
            normalized_queries.append((query, None, None, []))

    if since_ts is not None:
        normalized_queries = [
            (query, query_ts, response, tools)
            for query, query_ts, response, tools in normalized_queries
            if query_ts is None or query_ts > since_ts
        ]
    return normalized_queries


def replay_queries(
    graph: Graph,
    queries: list[str | tuple[str, float | None] | dict[str, object]],
    config: TraversalConfig | None = None,
    keyword_seed_fn: Callable[[Graph, str], list[tuple[str, float]]] | None = None,
    outcome: float = 1.0,
    outcome_fn: Callable[[str], float] | None = None,
    verbose: bool = False,
    since_ts: float | None = None,
    auto_decay: bool = False,
    decay_interval: int = 10,
) -> dict:
    """Replay historical queries to warm up graph edges.

    For each query:
    1. Seed from keyword matching (or provided seed_fn)
    2. Traverse the graph
    3. Apply outcome weighting (positive, negative, or custom)
    4. Apply Hebbian co-firing for co-selected nodes

    When ``auto_decay`` is True, decay is applied every ``decay_interval``
    learning steps, causing unrelated edges to weaken over time.
    """
    cfg = config or TraversalConfig()
    seed_fn = keyword_seed_fn or default_keyword_seed_fn

    normalized_queries = _normalize_replay_queries(queries=queries, since_ts=since_ts)

    if not normalized_queries:
        return {
            "queries_replayed": 0,
            "edges_reinforced": 0,
            "cross_file_edges_created": 0,
            "last_replayed_ts": None,
            "last_replayed_ts_source": None,
        }

    stats = {
        "queries_replayed": 0,
        "edges_reinforced": 0,
        "cross_file_edges_created": 0,
        "last_replayed_ts": None,
        "last_replayed_ts_source": None,
    }
    total_queries = len(normalized_queries)
    latest_ts = None

    for idx, (query, query_ts, query_response, _) in enumerate(normalized_queries, start=1):
        stats["queries_replayed"] += 1

        if idx > 0 and idx % 100 == 0:
            print(f"Replayed {idx}/{total_queries} queries...", file=sys.stderr)

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
        query_outcome = _auto_score_query_outcome(query_outcome, query_response, result.fired, graph)

        fired_nodes = [result.steps[0].from_node, *[step.to_node for step in result.steps]] if result.steps else result.fired
        apply_outcome(
            graph=graph,
            fired_nodes=fired_nodes,
            outcome=query_outcome,
            config=LearningConfig(),
            auto_decay=auto_decay,
            decay_interval=decay_interval,
        )

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

    if stats["queries_replayed"] > 0 and latest_ts is None:
        stats["last_replayed_ts"] = time.time()
        stats["last_replayed_ts_source"] = "wall_clock"
    else:
        stats["last_replayed_ts"] = latest_ts
        if latest_ts is not None:
            stats["last_replayed_ts_source"] = "query_ts"
    return stats


def replay_queries_parallel(
    graph: Graph,
    queries: list[str | tuple[str, float | None] | dict[str, object]],
    *,
    workers: int = 1,
    merge_every: int = 50,
    config: TraversalConfig | None = None,
    keyword_seed_fn: Callable[[Graph, str], list[tuple[str, float]]] | None = None,
    outcome: float = 1.0,
    outcome_fn: Callable[[str], float] | None = None,
    verbose: bool = False,
    since_ts: float | None = None,
    auto_decay: bool = False,
    decay_interval: int = 10,
    on_merge: Callable[[dict[str, object]], None] | None = None,
) -> dict:
    """Approximate true-parallel replay with deterministic shard merges."""
    worker_count = max(1, workers)
    batch_size = max(1, merge_every)
    if worker_count <= 1:
        return replay_queries(
            graph=graph,
            queries=queries,
            config=config,
            keyword_seed_fn=keyword_seed_fn,
            outcome=outcome,
            outcome_fn=outcome_fn,
            verbose=verbose,
            since_ts=since_ts,
            auto_decay=auto_decay,
            decay_interval=decay_interval,
        )

    cfg = config or TraversalConfig()
    seed_fn = keyword_seed_fn or default_keyword_seed_fn
    normalized = _normalize_replay_queries(queries=queries, since_ts=since_ts)
    if not normalized:
        return {
            "queries_replayed": 0,
            "edges_reinforced": 0,
            "cross_file_edges_created": 0,
            "last_replayed_ts": None,
            "last_replayed_ts_source": None,
            "merge_batches": 0,
            "replay_workers": worker_count,
        }

    indexed = [
        (idx, query, query_ts, query_response)
        for idx, (query, query_ts, query_response, _tool_calls) in enumerate(normalized)
    ]
    shards: list[list[tuple[int, str, float | None, str | None]]] = [[] for _ in range(worker_count)]
    for item in indexed:
        shards[item[0] % worker_count].append(item)

    def _worker_shard(
        shard_id: int,
        shard_items: list[tuple[int, str, float | None, str | None]],
    ) -> tuple[int, list[list[dict[str, object]]]]:
        shard_batches: list[list[dict[str, object]]] = []
        pending: list[dict[str, object]] = []
        for global_idx, query, query_ts, query_response in shard_items:
            seeds = seed_fn(graph, query)
            result = traverse(graph=graph, seeds=seeds, config=cfg)
            event: dict[str, object] = {"idx": global_idx, "ts": query_ts}
            if result.fired:
                query_outcome = outcome_fn(query) if outcome_fn is not None else outcome
                query_outcome = _auto_score_query_outcome(query_outcome, query_response, result.fired, graph)
                fired_nodes = (
                    [result.steps[0].from_node, *[step.to_node for step in result.steps]]
                    if result.steps
                    else result.fired
                )
                event["fired_nodes"] = fired_nodes
                event["outcome"] = query_outcome
            pending.append(event)
            if len(pending) >= batch_size:
                shard_batches.append(pending)
                pending = []
        if pending:
            shard_batches.append(pending)
        return shard_id, shard_batches

    shard_batches_by_id: list[list[list[dict[str, object]]]] = [[] for _ in range(worker_count)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_worker_shard, shard_id, shard_items) for shard_id, shard_items in enumerate(shards)]
        for future in concurrent.futures.as_completed(futures):
            shard_id, shard_batches = future.result()
            shard_batches_by_id[shard_id] = shard_batches

    stats = {
        "queries_replayed": len(normalized),
        "edges_reinforced": 0,
        "cross_file_edges_created": 0,
        "last_replayed_ts": None,
        "last_replayed_ts_source": None,
        "merge_batches": 0,
        "replay_workers": worker_count,
    }
    latest_ts = None
    processed = 0
    decay_counter = 0
    max_rounds = max((len(shard_batches) for shard_batches in shard_batches_by_id), default=0)

    for round_idx in range(max_rounds):
        for shard_id, shard_batches in enumerate(shard_batches_by_id):
            if round_idx >= len(shard_batches):
                continue
            batch = sorted(
                shard_batches[round_idx],
                key=lambda item: int(item.get("idx", -1)),
            )
            for event in batch:
                query_ts = event.get("ts")
                if isinstance(query_ts, (float, int)):
                    query_ts_value = float(query_ts)
                    latest_ts = query_ts_value if latest_ts is None else max(latest_ts, query_ts_value)
                else:
                    query_ts_value = None

                fired_nodes = event.get("fired_nodes")
                query_outcome = event.get("outcome")
                if isinstance(fired_nodes, list) and len(fired_nodes) >= 2 and isinstance(query_outcome, (float, int)):
                    before_weights = _snapshot_edges(graph)
                    before_cross_edges = _cross_file_edges(graph)
                    apply_outcome(
                        graph=graph,
                        fired_nodes=[str(node_id) for node_id in fired_nodes],
                        outcome=float(query_outcome),
                        config=LearningConfig(),
                        auto_decay=False,
                    )
                    decay_counter += 1
                    after_weights = _snapshot_edges(graph)
                    after_cross_edges = _cross_file_edges(graph)
                    for key, weight in after_weights.items():
                        if before_weights.get(key) != weight:
                            stats["edges_reinforced"] += 1
                    stats["cross_file_edges_created"] += len(after_cross_edges - before_cross_edges)
                    if auto_decay and decay_interval > 0 and decay_counter % max(1, decay_interval) == 0:
                        apply_decay(graph)
                if query_ts_value is not None:
                    latest_ts = query_ts_value if latest_ts is None else max(latest_ts, query_ts_value)
                processed += 1

            stats["merge_batches"] += 1
            if on_merge is not None:
                on_merge(
                    {
                        "merged_queries": processed,
                        "total_queries": len(normalized),
                        "round": round_idx + 1,
                        "shard": shard_id + 1,
                        "batch_size": len(batch),
                        "merge_batches": stats["merge_batches"],
                        "last_replayed_ts": latest_ts,
                        "last_replayed_ts_source": "query_ts" if latest_ts is not None else None,
                    }
                )
            if verbose:
                print(
                    f"Merged batch {stats['merge_batches']} ({processed}/{len(normalized)} queries)",
                    file=sys.stderr,
                )

    if stats["queries_replayed"] > 0 and latest_ts is None:
        stats["last_replayed_ts"] = time.time()
        stats["last_replayed_ts_source"] = "wall_clock"
    else:
        stats["last_replayed_ts"] = latest_ts
        if latest_ts is not None:
            stats["last_replayed_ts_source"] = "query_ts"
    return stats


def default_keyword_seed_fn(graph: Graph, query_text: str) -> list[tuple[str, float]]:
    """default keyword seed fn."""
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
