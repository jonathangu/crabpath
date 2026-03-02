from __future__ import annotations

from pathlib import Path

from openclawbrain import Edge, Graph, Node, VectorIndex
from openclawbrain.storage import JsonStateStore, JsonlEventStore


def test_json_state_store_roundtrip(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha"))
    graph.add_node(Node("b", "beta"))
    graph.add_edge(Edge("a", "b", weight=0.7))

    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    index.upsert("b", [0.0, 1.0])

    store = JsonStateStore()
    path = tmp_path / "state.json"
    store.save(str(path), graph=graph, index=index, meta={"embedder_name": "hash-v1", "embedder_dim": 2})

    loaded_graph, loaded_index, meta = store.load(str(path))
    assert loaded_graph.get_node("a") is not None
    assert loaded_graph.get_node("b") is not None
    assert loaded_graph.edge_count() == 1
    assert loaded_index._vectors["a"] == [1.0, 0.0]
    assert meta["embedder_name"] == "hash-v1"


def test_jsonl_event_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    store = JsonlEventStore(str(path))

    store.append({"type": "query", "query": "alpha", "fired": ["a"]})
    store.append({"type": "learn", "fired": ["a"], "outcome": -1.0})

    all_events = store.iter_since(None)
    assert len(all_events) == 2
    assert all_events[0]["type"] == "query"
    assert all_events[1]["type"] == "learn"

    only_recent = store.iter_since(float(all_events[0]["ts"]))
    assert len(only_recent) >= 1

    last = store.read_last(1)
    assert len(last) == 1
    assert last[0]["type"] == "learn"
