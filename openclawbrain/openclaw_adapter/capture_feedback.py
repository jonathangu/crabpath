#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openclawbrain import HashEmbedder, apply_outcome_pg, inject_correction, inject_node, load_state, save_state
from openclawbrain.socket_client import OCBClient

FIRE_LOG = "fired_log.jsonl"
INJECTED_FEEDBACK_LOG = "injected_feedback.jsonl"
EMBED_MODEL = "text-embedding-3-small"
DEDUP_TAIL_SIZE = 10_000


def _state_dir(state_path: Path) -> Path:
    return state_path.parent


def _fire_log_path(state_path: Path) -> Path:
    return _state_dir(state_path) / FIRE_LOG


def _injected_feedback_log_path(state_path: Path) -> Path:
    return _state_dir(state_path) / INJECTED_FEEDBACK_LOG


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _tail_jsonl(path: Path, *, max_entries: int) -> list[dict[str, object]]:
    if max_entries <= 0 or not path.exists():
        return []

    rows: deque[dict[str, object]] = deque(maxlen=max_entries)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return list(rows)


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _real_graph_updates(updates: dict[str, float]) -> int:
    return sum(1 for key in updates if not key.endswith("->__STOP__"))


def _unique_fired_nodes(entries: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    fired: list[str] = []
    for entry in entries:
        raw = entry.get("fired_nodes")
        if not isinstance(raw, list):
            continue
        for node_id in raw:
            if isinstance(node_id, str) and node_id and node_id not in seen:
                seen.add(node_id)
                fired.append(node_id)
    return fired


def _load_recent_fired_nodes(state_path: Path, chat_id: str, lookback: int) -> list[str]:
    rows = _read_jsonl(_fire_log_path(state_path))
    candidates: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        if row.get("chat_id") != chat_id:
            continue
        ts = row.get("ts")
        if isinstance(ts, (int, float)):
            candidates.append((float(ts), row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [row for _ts, row in candidates[:lookback]]
    return _unique_fired_nodes(selected)


def _load_dedup_keys(state_path: Path) -> set[str]:
    dedup_keys: set[str] = set()
    for row in _tail_jsonl(_injected_feedback_log_path(state_path), max_entries=DEDUP_TAIL_SIZE):
        dedup_key = row.get("dedup_key")
        if isinstance(dedup_key, str) and dedup_key:
            dedup_keys.add(dedup_key)
    return dedup_keys


def _require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "This script requires OPENAI_API_KEY for non-hash embedding state when socket fallback is local."
        )
    return api_key


def _resolve_embed_fn(meta: dict[str, object]) -> Callable[[str], list[float]]:
    embedder_name = meta.get("embedder_name")
    if embedder_name == "hash-v1":
        return HashEmbedder().embed

    if embedder_name in {"text-embedding-3-small", "openai-text-embedding-3-small"}:
        from openai import OpenAI

        client = OpenAI(api_key=_require_api_key())

        def _embed(content: str) -> list[float]:
            response = client.embeddings.create(model=EMBED_MODEL, input=[content])
            return list(response.data[0].embedding)

        return _embed

    return HashEmbedder().embed


def _feedback_node_id(kind: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if kind == "CORRECTION":
        return f"correction::{digest[:12]}"
    return f"{kind.lower()}::{digest[:12]}"


def _parse_dedup_key(dedup_key: str | None, message_id: str | None) -> str | None:
    if dedup_key is not None:
        parsed = dedup_key.strip()
        if not parsed:
            raise SystemExit("--dedup-key must be a non-empty string")
        return parsed
    if message_id is not None:
        parsed = message_id.strip()
        if not parsed:
            raise SystemExit("--message-id must be a non-empty string")
        return parsed
    return None


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, separators=(",", ":")))
        return

    print(f"deduped={payload.get('deduped')}")
    print(f"edges_updated={payload.get('edges_updated')}")
    print(f"fired_ids_used={payload.get('fired_ids_used')}")
    if "injected_node_id" in payload:
        print(f"injected_node_id={payload.get('injected_node_id')}")
    if "dedup_key_used" in payload:
        print(f"dedup_key_used={payload.get('dedup_key_used')}")
    if "outcome_used" in payload:
        print(f"outcome_used={payload.get('outcome_used')}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture real-time feedback with optional learning and dedup")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument("--chat-id", required=True, help="Conversation id used during query")
    parser.add_argument("--kind", required=True, choices=["CORRECTION", "TEACHING", "DIRECTIVE"], help="Feedback kind")
    parser.add_argument("--content", required=True, help="Feedback content")
    parser.add_argument("--socket", help="Unix socket path for daemon mode")
    parser.add_argument("--outcome", type=float, help="Optional learn outcome")
    parser.add_argument("--lookback", type=int, default=1, help="Number of recent queries to resolve by chat-id")
    parser.add_argument("--dedup-key", help="Stable external dedup key")
    parser.add_argument("--message-id", help="Alias for dedup key when dedup-key is omitted")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.lookback <= 0:
        raise SystemExit("--lookback must be >= 1")

    state_path = Path(args.state).expanduser()
    if not state_path.exists():
        raise SystemExit(f"state file not found: {state_path}")

    dedup_key_used = _parse_dedup_key(args.dedup_key, args.message_id)

    socket_path = args.socket
    if socket_path is None:
        socket_path = OCBClient.default_socket_path(state_path.parent.name)

    if args.socket is not None or (socket_path is not None and Path(socket_path).exists()):
        try:
            with OCBClient(socket_path) as client:
                response = client.capture_feedback(
                    chat_id=args.chat_id,
                    kind=args.kind,
                    content=args.content,
                    outcome=args.outcome,
                    lookback=args.lookback,
                    dedup_key=args.dedup_key,
                    message_id=args.message_id,
                )
            _emit(response, as_json=bool(args.json))
            return
        except Exception as exc:
            print(f"socket unavailable, falling back to local state: {exc}", file=sys.stderr)

    fired_ids = _load_recent_fired_nodes(state_path, args.chat_id, args.lookback)

    if dedup_key_used is not None and dedup_key_used in _load_dedup_keys(state_path):
        payload: dict[str, object] = {
            "deduped": True,
            "dedup_key_used": dedup_key_used,
            "edges_updated": 0,
            "fired_ids_used": fired_ids,
        }
        _emit(payload, as_json=bool(args.json))
        return

    graph, index, meta = load_state(str(state_path))
    embed_fn = _resolve_embed_fn(meta)
    node_id = _feedback_node_id(args.kind, args.content)

    changed = False
    injected_node_id: str | None = None

    if graph.get_node(node_id) is None:
        metadata = {
            "type": args.kind,
            "source": "capture_feedback",
            "chat_id": args.chat_id,
        }
        if args.kind == "CORRECTION":
            inject_correction(
                graph=graph,
                index=index,
                node_id=node_id,
                content=args.content,
                metadata=metadata,
                embed_fn=embed_fn,
            )
        else:
            embedder_name = str(meta.get("embedder_name", "hash-v1"))
            connect_min_sim = 0.0 if embedder_name == "hash-v1" else 0.3
            inject_node(
                graph=graph,
                index=index,
                node_id=node_id,
                content=args.content,
                metadata=metadata,
                embed_fn=embed_fn,
                connect_min_sim=connect_min_sim,
            )
        injected_node_id = node_id
        changed = True

    outcome_used = args.outcome
    if outcome_used is None and args.kind == "CORRECTION":
        outcome_used = -1.0

    edges_updated = 0
    if outcome_used is not None and fired_ids:
        updates = apply_outcome_pg(graph=graph, fired_nodes=fired_ids, outcome=outcome_used, baseline=0.0, temperature=1.0)
        edges_updated = _real_graph_updates(updates)
        if edges_updated:
            changed = True

    if changed:
        save_state(graph=graph, index=index, path=str(state_path), meta=meta)

    if dedup_key_used is not None:
        _append_jsonl(
            _injected_feedback_log_path(state_path),
            {
                "dedup_key": dedup_key_used,
                "chat_id": args.chat_id,
                "kind": args.kind,
                "node_id": node_id,
                "ts": time.time(),
            },
        )

    payload = {
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

    _emit(payload, as_json=bool(args.json))


if __name__ == "__main__":
    main()
