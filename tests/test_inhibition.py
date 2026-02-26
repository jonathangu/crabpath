"""Tests for the dedicated inhibition module."""

import pytest

from crabpath.graph import Edge, Graph, Node
from crabpath.inhibition import (
    InhibitionConfig,
    apply_correction,
    get_inhibitory_edges,
    inhibition_stats,
    is_inhibited,
    score_with_inhibition,
)


def _graph_with_nodes(*ids):
    g = Graph()
    for nid in ids:
        g.add_node(Node(id=nid, content=f"Content for {nid}"))
    return g


def test_correction_decays_positive_edges():
    g = _graph_with_nodes("A", "B")
    g.add_edge(Edge(source="A", target="B", weight=0.8))

    modified = apply_correction(g, ["A", "B"], reward=-1.0, config=InhibitionConfig())

    assert g.get_edge("A", "B").weight == pytest.approx(0.4)
    assert modified == [
        {"source": "A", "target": "B", "before": 0.8, "after": 0.4}
    ]


def test_correction_creates_negative_edges():
    g = _graph_with_nodes("A", "B")

    modified = apply_correction(g, ["A", "B"], reward=-1.0, config=InhibitionConfig())

    edge = g.get_edge("A", "B")
    assert edge is not None
    assert edge.weight == pytest.approx(-0.15)
    assert modified == [
        {"source": "A", "target": "B", "before": None, "after": -0.15}
    ]


def test_score_with_inhibition_suppresses():
    g = _graph_with_nodes("A", "B", "C")
    g.add_edge(Edge(source="A", target="B", weight=-0.5))

    scored = score_with_inhibition(
        candidates=[("B", 1.0), ("C", 0.9)],
        graph=g,
        source_node="A",
        config=InhibitionConfig(suppression_lambda=1.0),
    )

    assert scored[0] == ("C", 0.9)
    assert scored[1] == ("B", 0.5)


def test_is_inhibited():
    g = _graph_with_nodes("A", "B", "C")
    g.add_edge(Edge(source="A", target="B", weight=-0.2))

    assert is_inhibited(g, "A", "B")
    assert not is_inhibited(g, "A", "C")


def test_inhibition_stats():
    g = _graph_with_nodes("A", "B", "C", "D")
    g.add_edge(Edge(source="A", target="B", weight=0.5))
    g.add_edge(Edge(source="A", target="C", weight=-0.4))
    g.add_edge(Edge(source="A", target="D", weight=-0.6))

    stats = inhibition_stats(g)
    assert stats == {
        "total_inhibitory_edges": 2,
        "strongest_inhibition": -0.6,
        "average_inhibitory_weight": -0.5,
    }
    assert get_inhibitory_edges(g, "A") == [("C", -0.4), ("D", -0.6)]


def test_competition_respects_protected_targets():
    g = _graph_with_nodes("A", "B", "C")
    g.get_node("B").metadata["protected"] = True
    g.add_edge(Edge(source="A", target="B", weight=-0.2))

    config = InhibitionConfig()
    config.max_outgoing = 1

    apply_correction(g, ["A", "C"], reward=-1.0, config=config)
    assert g.get_edge("A", "C") is None
