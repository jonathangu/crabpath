"""Tests for save/load."""

import tempfile
from pathlib import Path

from crabpath import Graph, Node, Edge, activate, learn


def test_save_load_roundtrip():
    g = Graph()
    g.add_node(Node(id="a", content="hello", metadata={"type": "fact", "tags": ["test"]}))
    g.add_node(Node(id="b", content="world", threshold=0.5))
    g.add_edge(Edge(source="a", target="b", weight=0.7))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    g.save(path)
    g2 = Graph.load(path)

    assert g2.node_count == 2
    assert g2.edge_count == 1
    assert g2.get_node("a").content == "hello"
    assert g2.get_node("a").metadata["type"] == "fact"
    assert g2.get_node("b").threshold == 0.5
    assert g2.get_edge("a", "b").weight == 0.7

    Path(path).unlink()


def test_save_load_preserves_learned_weights():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=1.5))

    result = activate(g, seeds={"a": 1.0})
    learn(g, result, outcome=1.0, rate=0.2)

    new_weight = g.get_edge("a", "b").weight

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    g.save(path)
    g2 = Graph.load(path)

    assert g2.get_edge("a", "b").weight == new_weight

    Path(path).unlink()


def test_save_omits_defaults():
    """Default threshold/potential/trace/metadata should not bloat the JSON."""
    import json

    g = Graph()
    g.add_node(Node(id="a", content="simple"))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    g.save(path)

    with open(path) as f:
        data = json.load(f)

    node_data = data["nodes"][0]
    assert "threshold" not in node_data  # default 1.0 omitted
    assert "potential" not in node_data  # default 0.0 omitted
    assert "trace" not in node_data       # default 0.0 omitted
    assert "metadata" not in node_data   # empty dict omitted

    Path(path).unlink()


def test_save_load_with_potential():
    """Non-zero potential should persist."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.get_node("a").potential = 3.0

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    g.save(path)
    g2 = Graph.load(path)

    assert g2.get_node("a").potential == 3.0

    Path(path).unlink()


def test_save_load_with_trace():
    """Non-zero trace should persist."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    activate(g, seeds={"a": 2.0})  # sets trace to 2.0

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    g.save(path)
    g2 = Graph.load(path)

    assert g2.get_node("a").trace == 2.0

    Path(path).unlink()
