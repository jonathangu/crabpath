#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Callable

from openclawbrain import HashEmbedder, TraversalConfig, load_state, traverse
from openclawbrain.socket_client import OCBClient

EMBED_MODEL = "text-embedding-3-small"
BOOTSTRAP_FILES = {"AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "active-tasks.md"}


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for OpenAI-backed states")
    return api_key


def embed_query(client, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding


def _embed_fn_from_state(meta: dict[str, object]) -> Callable[[str], list[float]]:
    embedder_name = str(meta.get("embedder_name", "hash-v1"))
    hash_dim = meta.get("embedder_dim")

    if embedder_name == "hash-v1" and hash_dim == HashEmbedder().dim:
        return HashEmbedder().embed
    if embedder_name == "hash-v1":
        from openai import OpenAI

        client = OpenAI(api_key=require_api_key())
        return lambda text: embed_query(client, text)
    if embedder_name in {"text-embedding-3-small", "openai-text-embedding-3-small"}:
        from openai import OpenAI

        client = OpenAI(api_key=require_api_key())
        return lambda text: embed_query(client, text)
    return HashEmbedder().embed


def _query_via_socket(
    socket_path: str,
    query: str,
    top_k: int,
    max_prompt_context_chars: int,
) -> dict[str, object]:
    with OCBClient(socket_path) as client:
        return client.request(
            "query",
            {
                "query": query,
                "top_k": top_k,
                "max_prompt_context_chars": max_prompt_context_chars,
            },
        )


def _query_local(state_path: Path, query: str, top_k: int) -> tuple[dict[str, object], object]:
    graph, index, meta = load_state(str(state_path))
    query_embed_fn = _embed_fn_from_state(meta)
    query_vector = query_embed_fn(query)
    seeds = index.search(query_vector, top_k=top_k)
    result = traverse(graph=graph, seeds=seeds, query_text=query, config=TraversalConfig(max_context_chars=20000))
    return {"fired_nodes": result.fired}, graph


def _as_percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100.0, 1)


def _is_bootstrap(file_path: str) -> bool:
    basename = Path(file_path).name
    return file_path in BOOTSTRAP_FILES or basename in BOOTSTRAP_FILES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit prompt-context duplication risk from fired nodes")
    parser.add_argument("state_path", help="Path to state.json")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--socket", help="Daemon socket path")
    parser.add_argument("--top", type=int, default=4, help="Top-k vector matches")
    parser.add_argument("--max-prompt-context-chars", type=int, default=12000)
    args = parser.parse_args(argv)

    if args.top <= 0:
        raise SystemExit("--top must be >= 1")
    if args.max_prompt_context_chars <= 0:
        raise SystemExit("--max-prompt-context-chars must be >= 1")

    state_path = Path(args.state_path).expanduser()
    query_text = " ".join(args.query).strip()

    graph = None
    payload: dict[str, object]

    resolved_socket = args.socket
    if resolved_socket is None:
        resolved_socket = OCBClient.default_socket_path(state_path.parent.name)

    if args.socket is not None or Path(resolved_socket).expanduser().exists():
        payload = _query_via_socket(
            resolved_socket,
            query_text,
            args.top,
            args.max_prompt_context_chars,
        )
        graph, _index, _meta = load_state(str(state_path))
    else:
        payload, graph = _query_local(state_path, query_text, args.top)

    fired_nodes = [str(node_id) for node_id in payload.get("fired_nodes", []) if isinstance(node_id, str)]
    total = len(fired_nodes)

    file_counter = Counter()
    bootstrap_count = 0
    recent_memory_count = 0
    unknown_file_count = 0

    recent_memory_files: set[str] = set()

    for node_id in fired_nodes:
        node = graph.get_node(node_id) if graph is not None else None
        metadata = node.metadata if node is not None and isinstance(node.metadata, dict) else {}
        file_path = metadata.get("file")
        if not isinstance(file_path, str) or not file_path:
            unknown_file_count += 1
            continue

        file_counter[file_path] += 1
        if _is_bootstrap(file_path):
            bootstrap_count += 1
        else:
            if file_path.startswith("memory/"):
                recent_memory_count += 1
                recent_memory_files.add(Path(file_path).name)

    other_count = total - bootstrap_count - recent_memory_count - unknown_file_count

    print(f"state: {state_path}")
    print(f"query: {query_text}")
    print(f"fired_nodes_total: {total}")
    print(f"bootstrap_fired_nodes: {bootstrap_count} ({_as_percent(bootstrap_count, total)}%)")
    print(f"recent_memory_fired_nodes: {recent_memory_count} ({_as_percent(recent_memory_count, total)}%)")
    print(f"other_fired_nodes: {other_count} ({_as_percent(other_count, total)}%)")
    print(f"unknown_file_fired_nodes: {unknown_file_count} ({_as_percent(unknown_file_count, total)}%)")

    top_files = file_counter.most_common(5)
    if top_files:
        print("top_files:")
        for file_path, count in top_files:
            print(f"  - {file_path}: {count}")

    print("suggested_adapter_flags:")
    print("  --json --compact --max-prompt-context-chars 12000 --exclude-bootstrap")
    if recent_memory_files:
        notes = " ".join(sorted(recent_memory_files)[:2])
        print(f"  --exclude-recent-memory {notes}")


if __name__ == "__main__":
    main()
