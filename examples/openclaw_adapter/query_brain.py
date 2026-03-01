#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from openclawbrain import HashEmbedder, TraversalConfig, load_state, traverse
from openclawbrain.prompt_context import build_prompt_context_ranked_with_stats
from openclawbrain.socket_client import OCBClient


EMBED_MODEL = "text-embedding-3-small"
FIRED_LOG_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_PROMPT_CONTEXT_CHARS = 12000
BOOTSTRAP_EXCLUDE_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "active-tasks.md"]


def _fired_log_path(state_path: Path) -> Path:
    return state_path.parent / "fired_log.jsonl"


def _read_fired_log(log_path: Path) -> list[dict[str, object]]:
    if not log_path.exists():
        return []

    entries = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        entries.append(record)
    return entries


def _write_fired_log(log_path: Path, entries: list[dict[str, object]]) -> None:
    tmp = log_path.with_name(f"{log_path.name}.tmp")
    payload = "\n".join(json.dumps(entry) for entry in entries)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(log_path)


def _prune_fired_log(entries: list[dict[str, object]], now: float) -> list[dict[str, object]]:
    cutoff = now - FIRED_LOG_TTL_SECONDS
    output = []
    for entry in entries:
        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and ts >= cutoff:
            output.append(entry)
    return output


def _append_fired_log(log_path: Path, entry: dict[str, object], now: float | None = None) -> None:
    entries = _read_fired_log(log_path)
    entries.append(entry)
    entries = _prune_fired_log(entries, now if now is not None else time.time())
    _write_fired_log(log_path, entries)


def _load_query_via_socket(
    socket_path: str | None,
    query_text: str,
    chat_id: str | None,
    top: int,
    max_prompt_context_chars: int,
    exclude_files: list[str],
    exclude_file_prefixes: list[str],
) -> dict[str, object] | None:
    if socket_path is None:
        return None

    with OCBClient(socket_path) as client:
        params: dict[str, Any] = {
            "query": query_text,
            "top_k": top,
            "max_prompt_context_chars": max_prompt_context_chars,
        }
        if chat_id is not None:
            params["chat_id"] = chat_id
        if exclude_files:
            params["exclude_files"] = exclude_files
        if exclude_file_prefixes:
            params["exclude_file_prefixes"] = exclude_file_prefixes
        payload = client.request("query", params)
        if not isinstance(payload, dict):
            raise RuntimeError("invalid socket response payload")
        return payload


def _collect_prompt_context_stats(payload: dict[str, object]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(key, str) and key.startswith("prompt_context_"):
            stats[key] = value
    return stats


def _expand_recent_memory_files(values: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        candidates = [value]
        if "/" not in value:
            candidates.append(f"memory/{value}")
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _build_json_output(
    *,
    state_path: Path,
    query_text: str,
    fired_nodes: list[str],
    prompt_context: str,
    prompt_context_stats: dict[str, object],
    seeds: object | None = None,
    context: object | None = None,
    timings: dict[str, object] | None = None,
    compact: bool,
) -> dict[str, object]:
    if compact:
        output: dict[str, object] = {
            "state": str(state_path),
            "query": query_text,
            "fired_nodes": fired_nodes,
            "prompt_context": prompt_context,
        }
        output.update(prompt_context_stats)
        if timings:
            output.update(timings)
        return output

    output = {
        "state": str(state_path),
        "query": query_text,
        "seeds": seeds if seeds is not None else [],
        "fired_nodes": fired_nodes,
        "context": context,
        "prompt_context": prompt_context,
    }
    output.update(prompt_context_stats)
    if timings:
        output.update(timings)
    return output


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "This script must run inside the agent framework exec environment where OPENAI_API_KEY is injected."
        )
    return api_key


def embed_query(client, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding


def _embed_fn_from_state(meta: dict[str, object]) -> tuple[Callable[[str], list[float]], str]:
    embedder_name = str(meta.get("embedder_name", "hash-v1"))
    hash_dim = meta.get("embedder_dim")

    if embedder_name == "hash-v1" and hash_dim == HashEmbedder().dim:
        return HashEmbedder().embed, embedder_name
    if embedder_name == "hash-v1":
        # Legacy fallback: some historical states used hash-v1 with non-hash dims.
        from openai import OpenAI

        api_key = require_api_key()
        client = OpenAI(api_key=api_key)
        return lambda text: embed_query(client, text), "text-embedding-3-small"
    if embedder_name in {"text-embedding-3-small", "openai-text-embedding-3-small"}:
        from openai import OpenAI
        api_key = require_api_key()
        client = OpenAI(api_key=api_key)
        return lambda text: embed_query(client, text), embedder_name
    return HashEmbedder().embed, "hash-v1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Query a OpenClawBrain state.json with OpenAI embeddings"
    )
    parser.add_argument("state_path", help="Path to state.json")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--chat-id", help="Conversation id to persist fired nodes")
    parser.add_argument("--socket", help="Unix socket path for daemon mode")
    parser.add_argument("--top", type=int, default=4, help="Top-k vector matches")
    parser.add_argument(
        "--max-prompt-context-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CONTEXT_CHARS,
        help="Max chars for deterministic prompt_context",
    )
    parser.add_argument(
        "--exclude-bootstrap",
        dest="exclude_bootstrap",
        action="store_true",
        default=True,
        help="Exclude OpenClaw bootstrap files from prompt_context (default: on)",
    )
    parser.add_argument(
        "--no-exclude-bootstrap",
        dest="exclude_bootstrap",
        action="store_false",
        help="Keep bootstrap files in prompt_context",
    )
    parser.add_argument(
        "--exclude-recent-memory",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Exclude specific recent memory note files from prompt_context",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "prompt"],
        default="text",
        help="Output format (json includes both context and prompt_context)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument(
        "--compact",
        dest="compact",
        action="store_true",
        default=None,
        help="Compact JSON output (omit non-deterministic context)",
    )
    parser.add_argument(
        "--no-compact",
        dest="compact",
        action="store_false",
        help="Full JSON output (includes context/seeds)",
    )
    args = parser.parse_args(argv)
    output_format = "json" if args.json else args.format
    compact = args.compact if args.compact is not None else bool(args.json)

    query_text = " ".join(args.query).strip()
    state_path = Path(args.state_path)
    if args.top <= 0:
        raise SystemExit("--top must be >= 1")
    if args.max_prompt_context_chars <= 0:
        raise SystemExit("--max-prompt-context-chars must be >= 1")

    resolved_socket = args.socket
    if resolved_socket is None:
        resolved_socket = OCBClient.default_socket_path(state_path.expanduser().parent.name)

    exclude_files: list[str] = []
    if args.exclude_bootstrap:
        exclude_files.extend(BOOTSTRAP_EXCLUDE_FILES)
    if args.exclude_recent_memory:
        exclude_files.extend(_expand_recent_memory_files(args.exclude_recent_memory))
    # Preserve insertion order and remove duplicates.
    exclude_files = list(dict.fromkeys(exclude_files))

    if args.socket is not None or Path(resolved_socket).expanduser().exists():
        try:
            result = _load_query_via_socket(
                resolved_socket,
                query_text,
                args.chat_id,
                args.top,
                args.max_prompt_context_chars,
                exclude_files,
                [],
            )
            if result is not None:
                fired_nodes = [str(node_id) for node_id in result.get("fired_nodes", [])]
                prompt_context = str(result.get("prompt_context") or "")
                prompt_context_stats = _collect_prompt_context_stats(result)
                if not prompt_context and output_format in {"json", "prompt"} and state_path.exists():
                    graph, _index, _meta = load_state(str(state_path))
                    prompt_context, prompt_context_stats = build_prompt_context_ranked_with_stats(
                        graph=graph,
                        node_ids=fired_nodes,
                        max_chars=args.max_prompt_context_chars,
                    )
                timings = {
                    key: result[key]
                    for key in ("embed_query_ms", "traverse_ms", "total_ms")
                    if key in result
                }
                if output_format == "json":
                    output = _build_json_output(
                        state_path=state_path,
                        query_text=query_text,
                        fired_nodes=fired_nodes,
                        prompt_context=prompt_context,
                        prompt_context_stats=prompt_context_stats,
                        seeds=result.get("seeds", []),
                        context=result.get("context"),
                        timings=timings,
                        compact=compact,
                    )
                    print(json.dumps(output, indent=2))
                    return
                if output_format == "prompt":
                    print(prompt_context)
                    return
                print("Fired nodes:")
                for node_id in fired_nodes:
                    print(f"- {node_id}")
                print("Context:")
                print(result.get("context") or "(no context)")
                return
        except Exception as exc:
            print(f"socket unavailable, falling back to local state: {exc}", file=sys.stderr)

    if not state_path.exists():
        raise SystemExit(f"state file not found: {state_path}")

    graph, index, meta = load_state(str(state_path))
    query_embed_fn, embedder_name = _embed_fn_from_state(meta)
    query_vector = query_embed_fn(query_text)

    expected_dim = meta.get("embedder_dim")
    if isinstance(expected_dim, int) and len(query_vector) != expected_dim:
        raise SystemExit(
            "Embedding dimension mismatch: "
            f"query={len(query_vector)} index={expected_dim}. "
            f"Rebuild state.json with {embedder_name}."
        )

    seeds = index.search(query_vector, top_k=args.top)
    result = traverse(graph=graph, seeds=seeds, query_text=query_text, config=TraversalConfig(max_context_chars=20000))

    excluded_node_ids: set[str] = set()
    excluded_files: set[str] = set()
    if exclude_files:
        for node_id in result.fired:
            node = graph.get_node(node_id)
            metadata = node.metadata if node is not None and isinstance(node.metadata, dict) else {}
            file_path = metadata.get("file")
            if not isinstance(file_path, str) or not file_path:
                continue
            if file_path in exclude_files:
                excluded_node_ids.add(node_id)
                excluded_files.add(file_path)

    prompt_node_ids = [node_id for node_id in result.fired if node_id not in excluded_node_ids]
    prompt_context, prompt_context_stats = build_prompt_context_ranked_with_stats(
        graph=graph,
        node_ids=prompt_node_ids,
        node_scores={str(node_id): float(score) for node_id, score in result.fired_scores.items()},
        max_chars=args.max_prompt_context_chars,
    )
    prompt_context_stats["prompt_context_excluded_files_count"] = len(excluded_files)
    prompt_context_stats["prompt_context_excluded_node_ids_count"] = len(excluded_node_ids)

    if args.chat_id:
        log_entry = {
            "chat_id": args.chat_id,
            "query": query_text,
            "fired_nodes": result.fired,
            "ts": time.time(),
        }
        _append_fired_log(_fired_log_path(state_path), log_entry)

    if output_format == "json":
        output = _build_json_output(
            state_path=state_path,
            query_text=query_text,
            fired_nodes=result.fired,
            prompt_context=prompt_context,
            prompt_context_stats=prompt_context_stats,
            seeds=seeds,
            context=result.context,
            compact=compact,
        )
        print(json.dumps(output, indent=2))
        return
    if output_format == "prompt":
        print(prompt_context)
        return

    print("Fired nodes:")
    for node_id in result.fired:
        print(f"- {node_id}")
    print("Context:")
    print(result.context or "(no context)")


if __name__ == "__main__":
    main()
