"""Synchronous Unix socket client for the OpenClawBrain socket server."""

from __future__ import annotations

import argparse
import json
import socket
import uuid
from pathlib import Path
from typing import Any


class OCBClient:
    """Synchronous client for NDJSON socket daemon."""

    def __init__(self, socket_path: str, timeout: float = 30.0):
        self.socket_path = Path(socket_path).expanduser()
        self.timeout = timeout
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(self.timeout)
        self._socket.connect(str(self.socket_path))
        self._reader = self._socket.makefile("r", encoding="utf-8")
        self._writer = self._socket.makefile("w", encoding="utf-8")

    @staticmethod
    def default_socket_path(agent: str) -> str:
        return str(Path.home() / ".openclawbrain" / agent / "daemon.sock")

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = str(uuid.uuid4())
        request = {"id": req_id, "method": method, "params": params or {}}
        line = json.dumps(request) + "\n"
        self._writer.write(line)
        self._writer.flush()

        response_line = self._reader.readline()
        if not response_line:
            raise RuntimeError("socket closed before response")
        response = json.loads(response_line)
        if not isinstance(response, dict):
            raise RuntimeError("invalid daemon response")
        if response.get("id") != req_id:
            raise RuntimeError(f"response id mismatch: expected {req_id}, got {response.get('id')}")

        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                message = str(error.get("message", "daemon error"))
                code = error.get("code")
                if code is None:
                    raise RuntimeError(message)
                raise RuntimeError(f"daemon error {code}: {message}")
            raise RuntimeError(str(error))

        if "result" in response and isinstance(response["result"], dict):
            return response["result"]
        if "result" in response:
            return {"result": response["result"]}
        raise RuntimeError("daemon response missing result")

    def query(self, query: str, chat_id: str | None = None, top_k: int = 4) -> dict[str, Any]:
        params = {"query": query, "top_k": top_k}
        if chat_id is not None:
            params["chat_id"] = chat_id
        return self.request("query", params)

    def learn(self, fired_nodes: list[str], outcome: float) -> dict[str, Any]:
        return self.request("learn", {"fired_nodes": fired_nodes, "outcome": outcome})

    def correction(
        self,
        chat_id: str,
        outcome: float = -1.0,
        content: str | None = None,
        lookback: int = 1,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"chat_id": chat_id, "outcome": outcome, "lookback": lookback}
        if content is not None:
            params["content"] = content
        return self.request("correction", params)

    def self_learn(
        self,
        content: str,
        fired_ids: list[str] | None = None,
        outcome: float = -1.0,
        node_type: str = "CORRECTION",
    ) -> dict[str, Any]:
        """Autonomous agent learning — corrections and positive reinforcement.

        outcome < 0 + CORRECTION: penalize path, inject with inhibitory edges
        outcome = 0 + TEACHING: inject knowledge only, no weight changes
        outcome > 0 + TEACHING: reinforce path, inject positive knowledge
        """
        params: dict[str, Any] = {"content": content, "outcome": outcome, "node_type": node_type}
        if fired_ids is not None:
            params["fired_ids"] = fired_ids
        return self.request("self_learn", params)

    def self_correct(
        self,
        content: str,
        fired_ids: list[str] | None = None,
        outcome: float = -1.0,
        node_type: str = "CORRECTION",
    ) -> dict[str, Any]:
        """Alias for self_learn (backward compatibility)."""
        return self.self_learn(content=content, fired_ids=fired_ids, outcome=outcome, node_type=node_type)

    def inject(
        self,
        node_id: str,
        content: str,
        node_type: str = "TEACHING",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"id": node_id, "content": content, "type": node_type}
        if metadata is not None:
            params["metadata"] = metadata
        return self.request("inject", params)

    def health(self) -> dict[str, Any]:
        return self.request("health")

    def info(self) -> dict[str, Any]:
        return self.request("info")

    def maintain(self, tasks: list[str] | None = None, dry_run: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {"dry_run": dry_run}
        if tasks is not None:
            params["tasks"] = tasks
        return self.request("maintain", params)

    def save(self) -> dict[str, Any]:
        return self.request("save")

    def reload(self) -> dict[str, Any]:
        return self.request("reload")

    def close(self) -> None:
        try:
            self._reader.close()
            self._writer.close()
        finally:
            self._socket.close()

    def __enter__(self) -> "OCBClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenClawBrain socket client")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --params JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise SystemExit("--params must be a JSON object")

    with OCBClient(socket_path=args.socket, timeout=args.timeout) as client:
        result = client.request(args.method, params)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
