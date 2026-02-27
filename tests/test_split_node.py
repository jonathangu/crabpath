from __future__ import annotations

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.index import VectorIndex
from openclawbrain.merge import suggest_merges
from openclawbrain.split_node import split_node, suggest_splits


def _topic_embed(text: str) -> list[float]:
    """Minimal deterministic embedder for routing tests."""
    lowered = text.lower()
    if "apple" in lowered:
        return [1.0, 0.0]
    if "banana" in lowered:
        return [0.0, 1.0]
    return [0.5, 0.5]


def test_suggest_splits_detects_long_node() -> None:
    """test suggest splits detects long node."""
    graph = Graph()
    graph.add_node(Node("long", "x" * 2005))

    candidates = suggest_splits(graph, VectorIndex(), min_content_chars=800)
    assert len(candidates) == 1
    assert candidates[0].node_id == "long"
    assert isinstance(candidates[0].score, float)


def test_suggest_splits_skips_constitutional() -> None:
    """test suggest splits skips constitutional nodes."""
    graph = Graph()
    graph.add_node(Node("const", "x" * 2005, metadata={"authority": "constitutional"}))
    graph.add_node(Node("overlay", "x" * 2005, metadata={"authority": "overlay"}))

    candidates = suggest_splits(graph, VectorIndex(), min_content_chars=800)
    node_ids = {candidate.node_id for candidate in candidates}
    assert "const" not in node_ids
    assert "overlay" in node_ids


def test_split_node_creates_children() -> None:
    """test split node creates children."""
    graph = Graph()
    graph.add_node(Node("parent", "alpha beta gamma", metadata={"file": "x.md"}))
    index = VectorIndex()

    result = split_node(
        graph,
        index,
        "parent",
        ["alpha", "beta", "gamma"],
        embed_fn=_topic_embed,
    )

    assert result.parent_id == "parent"
    assert result.child_ids == ["split:parent:0", "split:parent:1", "split:parent:2"]
    assert graph.get_node("split:parent:0") is not None
    assert graph.get_node("split:parent:1") is not None
    assert graph.get_node("split:parent:2") is not None


def test_split_node_rewires_edges() -> None:
    """test split node rewires edges."""
    graph = Graph()
    index = VectorIndex()
    graph.add_node(Node("parent", "apple banana content"))
    graph.add_node(Node("source", "apple source"))
    graph.add_node(Node("apple_target", "apple target"))
    graph.add_node(Node("banana_target", "banana target"))
    graph.add_edge(Edge("source", "parent", 0.9))
    graph.add_edge(Edge("parent", "apple_target", 0.8))
    graph.add_edge(Edge("parent", "banana_target", 0.7))

    result = split_node(
        graph,
        index,
        "parent",
        ["apple piece", "banana piece"],
        embed_fn=_topic_embed,
    )
    _ = result

    outgoing_parent0 = {target.id: edge.weight for target, edge in graph.outgoing("split:parent:0")}
    outgoing_parent1 = {target.id: edge.weight for target, edge in graph.outgoing("split:parent:1")}

    assert any(target == "apple_target" and weight == 0.8 for target, weight in outgoing_parent0.items())
    assert any(target == "banana_target" and weight == 0.7 for target, weight in outgoing_parent1.items())
    assert any(target.id == "split:parent:0" for target, _ in graph.incoming("apple_target"))

    # 1 rewired edge from "source" + 1 sibling edge from "split:parent:1"
    assert len(graph.incoming("split:parent:0")) == 2
    assert any(source.id == "source" for source, _ in graph.incoming("split:parent:0"))


def test_split_node_removes_parent() -> None:
    """test split node removes parent."""
    graph = Graph()
    index = VectorIndex()
    graph.add_node(Node("parent", "x"))

    split_node(graph, index, "parent", ["x", "y"], embed_fn=_topic_embed)
    assert graph.get_node("parent") is None


def test_split_node_siblings() -> None:
    """test split node siblings."""
    graph = Graph()
    index = VectorIndex()
    graph.add_node(Node("parent", "x x x"))

    result = split_node(
        graph,
        index,
        "parent",
        ["A", "B", "C"],
        embed_fn=_topic_embed,
        sibling_weight=0.3,
    )

    assert result.siblings_added == 6
    for src in result.child_ids:
        for dst in result.child_ids:
            if src == dst:
                continue
            edge = graph._edges[src][dst]
            assert edge.weight == 0.3
            assert edge.kind == "sibling"


def test_split_preserves_inhibitory() -> None:
    """test split preserves inhibitory edges across all children."""
    graph = Graph()
    index = VectorIndex()
    graph.add_node(Node("parent", "apple banana"))
    graph.add_node(Node("source", "apple source"))
    graph.add_node(Node("target", "banana target"))
    graph.add_edge(Edge("source", "parent", -0.4, kind="inhibitory"))
    graph.add_edge(Edge("parent", "target", -0.6, kind="inhibitory"))

    result = split_node(graph, index, "parent", ["apple", "banana"], embed_fn=_topic_embed)

    for child_id in result.child_ids:
        assert graph._edges["source"][child_id].weight == -0.4
        assert graph._edges["source"][child_id].kind == "inhibitory"
        assert graph._edges[child_id]["target"].weight == -0.6


def test_split_merge_roundtrip() -> None:
    """test split then split children are mergeable."""
    graph = Graph()
    index = VectorIndex()
    graph.add_node(Node("parent", "alpha beta gamma"))

    result = split_node(
        graph,
        index,
        "parent",
        ["apple part", "banana part"],
        embed_fn=_topic_embed,
        sibling_weight=1.0,
    )

    merge_candidates = suggest_merges(graph)
    assert ("split:parent:0", "split:parent:1") in merge_candidates
