from __future__ import annotations

import json
import sys
from pathlib import Path

from openclawbrain import Edge, Graph, HashEmbedder, Node, VectorIndex, save_state
from openclawbrain.openclaw_adapter import capture_feedback as feedback_module
from openclawbrain.store import load_state


def _write_state(path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha"))
    graph.add_node(Node("b", "beta"))
    graph.add_edge(Edge("a", "b", weight=0.5, kind="sibling"))

    embedder = HashEmbedder()
    index = VectorIndex()
    index.upsert("a", embedder.embed("alpha"))
    index.upsert("b", embedder.embed("beta"))
    save_state(graph=graph, index=index, path=path, meta={"embedder_name": "hash-v1", "embedder_dim": embedder.dim})


def test_capture_feedback_adapter_uses_socket(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    class FakeClient:
        def __init__(self, socket_path: str) -> None:
            assert socket_path == str(tmp_path / "daemon.sock")

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def capture_feedback(
            self,
            *,
            chat_id: str,
            kind: str,
            content: str,
            outcome: float | None,
            lookback: int,
            dedup_key: str | None,
            message_id: str | None,
        ) -> dict[str, object]:
            return {
                "deduped": False,
                "chat_id": chat_id,
                "kind": kind,
                "content": content,
                "outcome": outcome,
                "lookback": lookback,
                "dedup_key_used": dedup_key,
                "message_id": message_id,
                "edges_updated": 4,
                "fired_ids_used": ["a", "b"],
                "injected_node_id": "teaching::abc",
            }

    monkeypatch.setattr(feedback_module, "OCBClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_feedback.py",
            "--state",
            str(state_path),
            "--chat-id",
            "chat-1",
            "--kind",
            "TEACHING",
            "--content",
            "Always validate inputs.",
            "--outcome",
            "1.0",
            "--lookback",
            "2",
            "--dedup-key",
            "evt-1",
            "--socket",
            str(tmp_path / "daemon.sock"),
            "--json",
        ],
    )

    feedback_module.main()
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["deduped"] is False
    assert payload["edges_updated"] == 4
    assert payload["fired_ids_used"] == ["a", "b"]
    assert payload["dedup_key_used"] == "evt-1"


def test_capture_feedback_adapter_local_fallback_dedup(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    fire_log = tmp_path / "fired_log.jsonl"
    fire_log.write_text(
        json.dumps({"chat_id": "chat-1", "fired_nodes": ["a", "b"], "ts": 1_000.0}) + "\n",
        encoding="utf-8",
    )

    graph_before, _index_before, _meta_before = load_state(str(state_path))
    before_weight = graph_before._edges["a"]["b"].weight

    class FailingClient:
        def __init__(self, _socket_path: str) -> None:
            raise RuntimeError("socket down")

    monkeypatch.setattr(feedback_module, "OCBClient", FailingClient)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_feedback.py",
            "--state",
            str(state_path),
            "--chat-id",
            "chat-1",
            "--kind",
            "CORRECTION",
            "--content",
            "Do not use alpha here.",
            "--lookback",
            "1",
            "--dedup-key",
            "evt-42",
            "--socket",
            str(tmp_path / "daemon.sock"),
            "--json",
        ],
    )
    feedback_module.main()
    first = json.loads(capsys.readouterr().out.strip())
    assert first["deduped"] is False
    assert first["dedup_key_used"] == "evt-42"
    assert first["outcome_used"] == -1.0
    assert first["edges_updated"] >= 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_feedback.py",
            "--state",
            str(state_path),
            "--chat-id",
            "chat-1",
            "--kind",
            "CORRECTION",
            "--content",
            "Do not use alpha here.",
            "--lookback",
            "1",
            "--dedup-key",
            "evt-42",
            "--socket",
            str(tmp_path / "daemon.sock"),
            "--json",
        ],
    )
    feedback_module.main()
    second = json.loads(capsys.readouterr().out.strip())
    assert second["deduped"] is True
    assert second["dedup_key_used"] == "evt-42"
    assert second["edges_updated"] == 0

    graph_after, _index_after, _meta_after = load_state(str(state_path))
    assert graph_after._edges["a"]["b"].weight != before_weight

    dedup_log = tmp_path / "injected_feedback.jsonl"
    rows = [line for line in dedup_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
