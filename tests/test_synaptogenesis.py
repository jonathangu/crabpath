"""Tests for CrabPath synaptogenesis — edge formation mechanics."""

import pytest
from crabpath.graph import Graph, Node, Edge
from crabpath.synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    ProtoEdge,
    record_cofiring,
    record_skips,
    decay_proto_edges,
    classify_tier,
    edge_tier_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _graph_with_nodes(*ids):
    g = Graph()
    for nid in ids:
        g.add_node(Node(id=nid, content=f"Content for {nid}"))
    return g


# ---------------------------------------------------------------------------
# Proto-edge creation
# ---------------------------------------------------------------------------


def test_first_cofire_creates_proto():
    g = _graph_with_nodes("A", "B")
    state = SynaptogenesisState()

    result = record_cofiring(g, ["A", "B"], state)

    assert result["proto_created"] >= 1
    assert ("A", "B") in state.proto_edges or ("B", "A") in state.proto_edges
    # No real edge yet
    assert g.get_edge("A", "B") is None


def test_repeated_cofire_promotes():
    g = _graph_with_nodes("A", "B")
    state = SynaptogenesisState()
    config = SynaptogenesisConfig(promotion_threshold=3)

    # Fire together 3 times
    for _ in range(3):
        record_cofiring(g, ["A", "B"], state, config)

    # Should have promoted to real edge
    edge_ab = g.get_edge("A", "B")
    edge_ba = g.get_edge("B", "A")
    assert edge_ab is not None or edge_ba is not None


def test_promotion_threshold_respected():
    g = _graph_with_nodes("A", "B")
    state = SynaptogenesisState()
    config = SynaptogenesisConfig(promotion_threshold=5)

    # Fire 4 times — not enough
    for _ in range(4):
        record_cofiring(g, ["A", "B"], state, config)

    assert g.get_edge("A", "B") is None  # Still proto


def test_promotion_weight_cofire():
    g = _graph_with_nodes("A", "B")
    state = SynaptogenesisState()
    config = SynaptogenesisConfig(promotion_threshold=3, cofire_initial_weight=0.15)

    for _ in range(3):
        record_cofiring(g, ["A", "B"], state, config)

    # Check weight is around initial
    edge = g.get_edge("A", "B") or g.get_edge("B", "A")
    assert edge is not None
    assert 0.1 <= edge.weight <= 0.3


def test_causal_direction_stronger():
    """If A always fires before B, A→B should get causal weight."""
    g = _graph_with_nodes("A", "B")
    state = SynaptogenesisState()
    config = SynaptogenesisConfig(
        promotion_threshold=3,
        cofire_initial_weight=0.15,
        causal_initial_weight=0.25,
    )

    # A always before B (causal)
    for _ in range(3):
        record_cofiring(g, ["A", "B"], state, config)

    edge_ab = g.get_edge("A", "B")
    assert edge_ab is not None
    assert edge_ab.weight == config.causal_initial_weight


# ---------------------------------------------------------------------------
# Reinforcement of existing edges
# ---------------------------------------------------------------------------


def test_existing_edge_reinforced():
    g = _graph_with_nodes("A", "B")
    g.add_edge(Edge(source="A", target="B", weight=0.5))
    state = SynaptogenesisState()
    config = SynaptogenesisConfig(hebbian_increment=0.05)

    result = record_cofiring(g, ["A", "B"], state, config)

    assert result["reinforced"] >= 1
    assert g.get_edge("A", "B").weight == pytest.approx(0.55)


def test_multiple_cofires_accumulate():
    g = _graph_with_nodes("A", "B")
    g.add_edge(Edge(source="A", target="B", weight=0.3))
    state = SynaptogenesisState()
    config = SynaptogenesisConfig(hebbian_increment=0.05)

    for _ in range(5):
        record_cofiring(g, ["A", "B"], state, config)

    assert g.get_edge("A", "B").weight == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Skip penalty
# ---------------------------------------------------------------------------


def test_skip_penalty():
    g = _graph_with_nodes("A", "B", "C")
    g.add_edge(Edge(source="A", target="B", weight=0.5))
    g.add_edge(Edge(source="A", target="C", weight=0.5))
    config = SynaptogenesisConfig(skip_factor=0.9)

    # Select B, skip C
    skipped = record_skips(g, "A", ["B", "C"], ["B"], config)

    assert skipped == 1
    assert g.get_edge("A", "B").weight == 0.5  # Not penalized
    assert g.get_edge("A", "C").weight == pytest.approx(0.45)  # Penalized


def test_skip_no_edge_no_crash():
    g = _graph_with_nodes("A", "B")
    # No edge from A to B
    skipped = record_skips(g, "A", ["B"], [], SynaptogenesisConfig())
    assert skipped == 0


# ---------------------------------------------------------------------------
# Proto-edge decay
# ---------------------------------------------------------------------------


def test_proto_decay():
    state = SynaptogenesisState()
    state.proto_edges[("A", "B")] = ProtoEdge(source="A", target="B", credit=1.5)
    config = SynaptogenesisConfig(proto_decay_rate=0.8, proto_min_credit=0.5)

    removed = decay_proto_edges(state, config)
    assert removed == 0
    assert state.proto_edges[("A", "B")].credit == pytest.approx(1.2)

    # Decay again
    decay_proto_edges(state, config)  # 0.96
    decay_proto_edges(state, config)  # 0.768
    decay_proto_edges(state, config)  # 0.614
    decay_proto_edges(state, config)  # 0.491 < 0.5 → removed

    assert ("A", "B") not in state.proto_edges


def test_proto_decay_removes_weak():
    state = SynaptogenesisState()
    state.proto_edges[("X", "Y")] = ProtoEdge(source="X", target="Y", credit=0.4)
    config = SynaptogenesisConfig(proto_min_credit=0.5)

    removed = decay_proto_edges(state, config)
    assert removed == 1
    assert ("X", "Y") not in state.proto_edges


# ---------------------------------------------------------------------------
# Edge competition (max outgoing)
# ---------------------------------------------------------------------------


def test_max_outgoing_enforced():
    g = _graph_with_nodes("hub", "t1", "t2", "t3", "new")
    config = SynaptogenesisConfig(max_outgoing=3, promotion_threshold=1)

    # Fill hub with 3 outgoing edges
    g.add_edge(Edge(source="hub", target="t1", weight=0.8))
    g.add_edge(Edge(source="hub", target="t2", weight=0.5))
    g.add_edge(Edge(source="hub", target="t3", weight=0.2))

    state = SynaptogenesisState()

    # Co-fire hub and new — should promote and replace weakest (t3 at 0.2)
    record_cofiring(g, ["hub", "new"], state, config)

    assert g.get_edge("hub", "new") is not None
    # t3 should have been evicted (it was weakest)
    assert g.get_edge("hub", "t3") is None


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def test_classify_tiers():
    config = SynaptogenesisConfig()
    assert classify_tier(0.1, config) == "dormant"
    assert classify_tier(0.3, config) == "habitual"
    assert classify_tier(0.5, config) == "habitual"
    assert classify_tier(0.8, config) == "reflex"
    assert classify_tier(1.0, config) == "reflex"


def test_edge_tier_stats():
    g = _graph_with_nodes("A", "B", "C", "D")
    g.add_edge(Edge(source="A", target="B", weight=0.1))  # dormant
    g.add_edge(Edge(source="A", target="C", weight=0.5))  # habitual
    g.add_edge(Edge(source="A", target="D", weight=0.9))  # reflex

    stats = edge_tier_stats(g)
    assert stats["dormant"] == 1
    assert stats["habitual"] == 1
    assert stats["reflex"] == 1


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_state_save_load(tmp_path):
    state = SynaptogenesisState()
    state.proto_edges[("A", "B")] = ProtoEdge(
        source="A",
        target="B",
        credit=2.5,
        first_seen=100.0,
        last_seen=200.0,
        causal_count=3,
        reverse_count=1,
    )

    path = str(tmp_path / "syn.json")
    state.save(path)

    loaded = SynaptogenesisState.load(path)
    pe = loaded.proto_edges[("A", "B")]
    assert pe.credit == 2.5
    assert pe.causal_count == 3


def test_state_load_missing():
    state = SynaptogenesisState.load("/nonexistent.json")
    assert state.proto_edges == {}


# ---------------------------------------------------------------------------
# Multi-node co-firing
# ---------------------------------------------------------------------------


def test_three_way_cofire():
    """Co-firing A, B, C should create proto-edges for all pairs."""
    g = _graph_with_nodes("A", "B", "C")
    state = SynaptogenesisState()

    record_cofiring(g, ["A", "B", "C"], state)

    # Should have proto-edges for all 6 directed pairs
    assert len(state.proto_edges) == 6


def test_mixed_existing_and_new():
    """Some pairs have edges, some don't. Both handled correctly."""
    g = _graph_with_nodes("A", "B", "C")
    g.add_edge(Edge(source="A", target="B", weight=0.4))
    state = SynaptogenesisState()
    config = SynaptogenesisConfig(hebbian_increment=0.05)

    record_cofiring(g, ["A", "B", "C"], state, config)

    # A→B should be reinforced (existing edge)
    assert g.get_edge("A", "B").weight == pytest.approx(0.45)
    # A↔C should be proto-edges
    assert ("A", "C") in state.proto_edges
