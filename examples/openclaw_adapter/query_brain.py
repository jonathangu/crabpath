#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

from openclawbrain import HashEmbedder, TraversalConfig, build_prompt_context, load_state, traverse
from openclawbrain.socket_client import OCBClient


EMBED_MODEL = "text-embedding-3-small"
FIRED_LOG_TTL_SECONDS = 7 * 24 * 60 * 60


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
) -> dict[str, object] | None:
    if socket_path is None:
        return None

    with OCBClient(socket_path) as client:
        payload = client.query(query=query_text, chat_id=chat_id, top_k=top)
        if not isinstance(payload, dict):
            raise RuntimeError("invalid socket response payload")
        return payload


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
        "--format",
        choices=["text", "json", "prompt"],
        default="text",
        help="Output format (json includes both context and prompt_context)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)
    output_format = "json" if args.json else args.format

    query_text = " ".join(args.query).strip()
    state_path = Path(args.state_path)
    if args.top <= 0:
        raise SystemExit("--top must be >= 1")

    resolved_socket = args.socket
    if resolved_socket is None:
        resolved_socket = OCBClient.default_socket_path(state_path.expanduser().parent.name)

    if args.socket is not None or Path(resolved_socket).expanduser().exists():
        try:
            result = _load_query_via_socket(resolved_socket, query_text, args.chat_id, args.top)
            if result is not None:
                fired_nodes = [str(node_id) for node_id in result.get("fired_nodes", [])]
                prompt_context = ""
                if output_format in {"json", "prompt"} and state_path.exists():
                    graph, _index, _meta = load_state(str(state_path))
                    prompt_context = build_prompt_context(graph=graph, node_ids=fired_nodes)
                if output_format == "json":
                    output = {
                        "state": str(state_path),
                        "query": query_text,
                        "seeds": result.get("seeds", []),
                        "fired_nodes": fired_nodes,
                        "context": result.get("context"),
                        "prompt_context": prompt_context,
                    }
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
    prompt_context = build_prompt_context(graph=graph, node_ids=result.fired)

    if args.chat_id:
        log_entry = {
            "chat_id": args.chat_id,
            "query": query_text,
            "fired_nodes": result.fired,
            "ts": time.time(),
        }
        _append_fired_log(_fired_log_path(state_path), log_entry)

    output = {
        "state": str(state_path),
        "query": query_text,
        "seeds": seeds,
        "fired_nodes": result.fired,
        "context": result.context,
        "prompt_context": prompt_context,
    }

    if output_format == "json":
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
