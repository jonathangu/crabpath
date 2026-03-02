from __future__ import annotations

from pathlib import Path

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.index import VectorIndex
from openclawbrain.storage import JsonStateStore, JsonlEventStore


def test_json_state_store_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = JsonStateStore(str(state_path))

    graph = Graph()
    graph.add_node(Node("a", "alpha"))
    graph.add_node(Node("b", "beta"))
    graph.add_edge(Edge("a", "b", weight=0.5))

    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    index.upsert("b", [0.0, 1.0])

    meta = {"embedder_name": "hash-v1", "embedder_dim": 1024, "custom": "ok"}
    store.save(graph, index, meta)

    loaded_graph, loaded_index, loaded_meta = store.load()
    assert loaded_graph.get_node("a") is not None
    assert loaded_graph.get_node("b") is not None
    assert loaded_graph._edges["a"]["b"].weight == 0.5
    assert loaded_index._vectors["a"] == [1.0, 0.0]
    assert loaded_meta["embedder_name"] == "hash-v1"
    assert loaded_meta["custom"] == "ok"


def test_jsonl_event_store_append_read_last_iter_since(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    store = JsonlEventStore(str(journal_path))

    store.append({"type": "query", "query": "alpha"})
    first = store.read_last(1)[0]
    first_ts = float(first["ts"])

    store.append({"type": "learn", "fired": ["a"], "outcome": 1.0})
    store.append({"type": "query", "query": "beta"})

    tail = store.read_last(2)
    assert [entry["type"] for entry in tail] == ["learn", "query"]
    assert all("ts" in entry and "iso" in entry for entry in tail)

    since_first = list(store.iter_since(first_ts))
    assert len(since_first) >= 3
    assert since_first[0]["type"] == "query"

    future_entries = list(store.iter_since(first_ts + 10_000))
    assert future_entries == []
