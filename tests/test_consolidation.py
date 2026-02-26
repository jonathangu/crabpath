from __future__ import annotations

from crabpath import Edge, Graph, Node
from crabpath.graph import (
    ConsolidationConfig,
    consolidate,
    prune_orphan_nodes,
    prune_probationary,
    prune_weak_edges,
    should_merge,
    should_split,
    split_node,
)


def test_prune_weak_edges() -> None:
    graph = Graph()
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="b", content="B"))
    graph.add_node(Node(id="c", content="C"))
    graph.add_edge(Edge(source="a", target="b", weight=0.02))
    graph.add_edge(Edge(source="a", target="c", weight=0.01))
    graph.add_edge(Edge(source="b", target="c", weight=0.10))

    removed = prune_weak_edges(graph, min_weight=0.03)

    assert removed == 2
    assert graph.edge_count == 1
    assert graph.get_edge("b", "c") is not None


def test_prune_orphan_nodes() -> None:
    graph = Graph()
    graph.add_node(Node(id="source", content="source"))
    graph.add_node(Node(id="seed", content="seed"))
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="b", content="B"))
    graph.add_node(Node(id="loner", content="Loner"))
    graph.add_edge(Edge(source="source", target="seed", weight=0.9))
    graph.add_edge(Edge(source="seed", target="source", weight=0.2))
    graph.add_edge(Edge(source="seed", target="a", weight=0.8))
    graph.add_edge(Edge(source="a", target="b", weight=0.8))

    removed = prune_orphan_nodes(graph)

    assert removed == 1
    assert graph.get_node("loner") is None


def test_prune_orphan_preserves_protected() -> None:
    graph = Graph()
    graph.add_node(Node(id="source", content="source"))
    graph.add_node(Node(id="seed", content="seed"))
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="protected", content="important", metadata={"protected": True}))
    graph.add_node(Node(id="loner", content="Loner"))
    graph.add_edge(Edge(source="source", target="seed", weight=0.9))
    graph.add_edge(Edge(source="seed", target="source", weight=0.2))
    graph.add_edge(Edge(source="seed", target="a", weight=0.8))
    graph.add_edge(Edge(source="a", target="protected", weight=0.8))

    removed = prune_orphan_nodes(graph, protected_ids={"protected"})

    assert removed == 1
    assert graph.get_node("protected") is not None
    assert graph.get_node("loner") is None


def test_should_split_fat_node() -> None:
    config = ConsolidationConfig(max_node_chars=16, min_fires_to_split=10)
    graph = Graph()
    graph.add_node(Node(id="fat", content="x" * 30, metadata={"fired_count": 11}))
    node = graph.get_node("fat")

    assert node is not None
    assert should_split(node, config) is True


def test_split_node_creates_children() -> None:
    graph = Graph()
    graph.add_node(Node(id="source", content="source"))
    graph.add_node(Node(id="parent", content="legacy content", metadata={"fired_count": 99}))
    graph.add_node(Node(id="downstream", content="downstream"))
    graph.add_edge(Edge(source="source", target="parent", weight=0.8))
    graph.add_edge(Edge(source="parent", target="downstream", weight=0.9))

    child_ids = split_node(
        graph,
        "parent",
        [
            {"id": "split-a", "content": "part a", "summary": "first"},
            {"id": "split-b", "content": "part b", "summary": "second"},
        ],
    )

    parent = graph.get_node("parent")
    assert parent is not None
    assert parent.content == "See: split-a, split-b"
    assert graph.get_node("source") is not None
    assert graph.get_node("source").id == "source"
    assert all(graph.get_node(child_id) is not None for child_id in child_ids)

    for child_id in child_ids:
        assert graph.get_edge("parent", child_id) is not None
        assert graph.get_edge(child_id, "downstream") is not None

    assert graph.get_edge("source", "parent") is not None


def test_should_merge() -> None:
    config = ConsolidationConfig(merge_cosine_threshold=0.90, merge_cofire_threshold=0.80)
    graph = Graph()
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="b", content="B"))

    assert should_merge(
        graph, "a", "b", cofire_count=9, total_fires=10, cosine_sim=0.91, config=config
    )
    assert (
        should_merge(
            graph, "a", "b", cofire_count=8, total_fires=10, cosine_sim=0.91, config=config
        )
        is False
    )


def test_consolidate_full_sweep(monkeypatch) -> None:
    graph = Graph()
    graph.add_node(Node(id="seed", content="seed"))
    graph.add_node(Node(id="good", content="good"))
    graph.add_node(Node(id="bridge", content="bridge"))
    graph.add_node(Node(id="weak", content="weak"))
    graph.add_node(Node(id="protected", content="protected", metadata={"protected": True}))
    graph.add_node(
        Node(
            id="probationary",
            content="probationary",
            metadata={"probationary": True, "created_ts": 0.0, "fired_count": 1},
        ),
    )

    graph.add_edge(Edge(source="seed", target="bridge", weight=1.0))
    graph.add_edge(Edge(source="bridge", target="seed", weight=1.0))
    graph.add_edge(Edge(source="seed", target="good", weight=1.0))
    graph.add_edge(Edge(source="weak", target="seed", weight=0.01))
    graph.add_edge(Edge(source="good", target="protected", weight=1.0))
    graph.add_edge(Edge(source="good", target="probationary", weight=1.0))

    monkeypatch.setattr("crabpath.graph.time.time", lambda: 100.0)
    result = consolidate(
        graph, config=ConsolidationConfig(min_edge_weight=0.03, probation_max_turns=10)
    )

    assert result.edges_pruned == 1
    assert result.nodes_pruned == 2
    assert graph.get_node("weak") is None
    assert graph.get_node("probationary") is None
    assert graph.get_node("protected") is not None


def test_probationary_pruning(monkeypatch) -> None:
    graph = Graph()
    graph.add_node(Node(id="seed", content="seed"))
    graph.add_node(
        Node(
            id="probationary",
            content="probationary",
            metadata={"probationary": True, "created_ts": 0.0, "fired_count": 2},
        ),
    )
    graph.add_edge(Edge(source="seed", target="probationary", weight=1.0))

    monkeypatch.setattr("crabpath.graph.time.time", lambda: 100.0)
    removed = prune_probationary(graph, max_turns=10)

    assert removed == 1
    assert graph.get_node("probationary") is None
