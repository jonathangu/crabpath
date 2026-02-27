from __future__ import annotations

from pathlib import Path

from crabpath import Edge, Graph, ManagedState, Node, VectorIndex, load_state, save_state


def test_save_load_state_with_meta(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta", metadata={"file": "b.md"}))
    graph.add_edge(Edge("a", "b", 0.5, kind="sibling", metadata={"source": "test"}))

    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    index.upsert("b", [0.0, 1.0])

    state_path = tmp_path / "state.json"
    save_state(
        graph=graph,
        index=index,
        path=state_path,
        embedder_name="hash-v1",
        embedder_dim=1024,
    )

    loaded_graph, loaded_index, meta = load_state(str(state_path))
    assert loaded_graph.get_node("a").content == "alpha"
    assert loaded_index._vectors["a"] == [1.0, 0.0]
    assert meta["embedder_name"] == "hash-v1"
    assert meta["embedder_dim"] == 1024
    assert meta["schema_version"] == 1
    assert meta["node_count"] == 2
    assert "created_at" in meta


def test_save_state_preserves_custom_metadata(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])

    state_path = tmp_path / "state.json"
    save_state(
        graph=graph,
        index=index,
        path=state_path,
        meta={"last_replayed_ts": 123.45, "custom": "value"},
    )

    _, _, meta = load_state(str(state_path))
    assert meta["last_replayed_ts"] == 123.45
    assert meta["custom"] == "value"
    assert meta["embedder_name"] == "hash-v1"


def test_managed_state_context_manager_autosaves_and_context_enter_exit(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    state_path = tmp_path / "state.json"
    save_state(graph=graph, index=index, path=state_path)

    with ManagedState(state_path, auto_save_every=1) as state:
        state.graph.add_node(Node("b", "beta", metadata={"file": "b.md"}))
        state.index.upsert("b", [0.0, 1.0])
        state.tick()

    reloaded_graph, _, _ = load_state(str(state_path))
    assert reloaded_graph.get_node("b") is not None
