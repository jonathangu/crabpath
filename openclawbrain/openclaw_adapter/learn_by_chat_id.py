#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openclawbrain import apply_outcome_pg, load_state, save_state
from openclawbrain.socket_client import OCBClient


FIRE_LOG = "fired_log.jsonl"


def _state_dir(state_path: Path) -> Path:
    return state_path.parent


def _fire_log_path(state_path: Path) -> Path:
    return _state_dir(state_path) / FIRE_LOG


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn from recent fired nodes resolved by chat-id")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument("--chat-id", required=True, help="Conversation id used during query")
    parser.add_argument("--socket", help="Unix socket path for daemon mode")
    parser.add_argument("--outcome", type=float, required=True, help="Learn outcome value")
    parser.add_argument("--lookback", type=int, default=1, help="Number of recent queries to learn from")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument("--pretty-json", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _emit(payload: dict[str, object], *, as_json: bool, pretty_json: bool) -> None:
    if as_json:
        if pretty_json:
            print(json.dumps(payload, indent=2))
            return
        print(json.dumps(payload, separators=(",", ":")))
        return

    fired = payload.get("fired_ids_penalized")
    edges = payload.get("edges_updated")
    print(f"edges_updated={edges}")
    print(f"fired_ids_penalized={fired}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.lookback <= 0:
        raise SystemExit("--lookback must be >= 1")

    state_path = Path(args.state).expanduser()
    socket_path = args.socket
    if socket_path is None:
        socket_path = OCBClient.default_socket_path(state_path.parent.name)

    if args.socket is not None or (socket_path is not None and Path(socket_path).exists()):
        try:
            with OCBClient(socket_path) as client:
                response = client.learn_by_chat_id(
                    chat_id=args.chat_id,
                    outcome=args.outcome,
                    lookback=args.lookback,
                )
            _emit(response, as_json=bool(args.json), pretty_json=bool(args.pretty_json))
            return
        except Exception as exc:
            print(f"socket unavailable, falling back to local state: {exc}", file=sys.stderr)

    if not state_path.exists():
        raise SystemExit(f"state file not found: {state_path}")

    fired_ids = _load_recent_fired_nodes(state_path, args.chat_id, args.lookback)
    edges_updated = 0
    if fired_ids:
        graph, index, meta = load_state(str(state_path))
        updates = apply_outcome_pg(graph=graph, fired_nodes=fired_ids, outcome=args.outcome)
        edges_updated = len([k for k in updates if not k.endswith("->__STOP__")])
        if edges_updated:
            save_state(graph=graph, index=index, path=str(state_path), meta=meta)

    payload: dict[str, Any] = {
        "edges_updated": edges_updated,
        "fired_ids_penalized": fired_ids,
    }
    _emit(payload, as_json=bool(args.json), pretty_json=bool(args.pretty_json))


if __name__ == "__main__":
    main()
