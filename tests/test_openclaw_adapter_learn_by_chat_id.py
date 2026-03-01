from __future__ import annotations

import json
import sys
from pathlib import Path

from openclawbrain import Edge, Graph, HashEmbedder, Node, VectorIndex, save_state
from openclawbrain.openclaw_adapter import learn_by_chat_id as learn_module
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


def test_learn_by_chat_id_adapter_uses_socket(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    class FakeClient:
        def __init__(self, socket_path: str) -> None:
            assert socket_path == str(tmp_path / "daemon.sock")

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def learn_by_chat_id(self, chat_id: str, outcome: float, lookback: int) -> dict[str, object]:
            return {
                "edges_updated": 7,
                "fired_ids_penalized": ["x", "y"],
                "chat_id": chat_id,
                "outcome": outcome,
                "lookback": lookback,
            }

    monkeypatch.setattr(learn_module, "OCBClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "learn_by_chat_id.py",
            "--state",
            str(state_path),
            "--chat-id",
            "chat-1",
            "--outcome",
            "1.0",
            "--socket",
            str(tmp_path / "daemon.sock"),
            "--json",
        ],
    )
    learn_module.main()
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["edges_updated"] == 7
    assert payload["fired_ids_penalized"] == ["x", "y"]


def test_learn_by_chat_id_adapter_falls_back_to_local_fired_log(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    graph_before, _index_before, _meta_before = load_state(str(state_path))
    before_weight = graph_before._edges["a"]["b"].weight
    fire_log = tmp_path / "fired_log.jsonl"
    fire_log.write_text(
        json.dumps({"chat_id": "chat-1", "fired_nodes": ["a", "b"], "ts": 1_000.0}) + "\n",
        encoding="utf-8",
    )

    class FailingClient:
        def __init__(self, _socket_path: str) -> None:
            raise RuntimeError("socket down")

    monkeypatch.setattr(learn_module, "OCBClient", FailingClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "learn_by_chat_id.py",
            "--state",
            str(state_path),
            "--chat-id",
            "chat-1",
            "--outcome",
            "-1.0",
            "--lookback",
            "1",
            "--socket",
            str(tmp_path / "daemon.sock"),
            "--json",
        ],
    )
    learn_module.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["fired_ids_penalized"] == ["a", "b"]
    assert payload["edges_updated"] == 1

    graph_after, _index_after, _meta_after = load_state(str(state_path))
    assert graph_after._edges["a"]["b"].weight != before_weight
