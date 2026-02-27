"""Tests for self-regulation: adaptive decay, synaptic scaling, tier hysteresis."""

from __future__ import annotations

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.maintain import compute_adaptive_half_life, apply_synaptic_scaling
from openclawbrain.traverse import TraversalConfig, _tier


# ── Adaptive half-life ──────────────────────────────────────────


def _graph_with_reflex_ratio(n_edges: int, reflex_frac: float) -> Graph:
    """Build a graph where reflex_frac of edges are >= 0.6."""
    g = Graph()
    n_reflex = int(n_edges * reflex_frac)
    for i in range(n_edges + 1):
        g.add_node(Node(f"n{i}", f"node {i}"))
    for i in range(n_edges):
        w = 0.8 if i < n_reflex else 0.3
        g.add_edge(Edge(f"n{i}", f"n{i+1}", weight=w))
    return g


def test_adaptive_half_life_increases_when_few_reflex() -> None:
    """No reflex edges → half-life should increase."""
    g = _graph_with_reflex_ratio(20, 0.0)
    new = compute_adaptive_half_life(g, 140.0)
    assert new > 140.0


def test_adaptive_half_life_decreases_when_many_reflex() -> None:
    """50% reflex edges → half-life should decrease."""
    g = _graph_with_reflex_ratio(20, 0.5)
    new = compute_adaptive_half_life(g, 140.0)
    assert new < 140.0


def test_adaptive_half_life_stable_in_target() -> None:
    """10% reflex edges (within 5-15%) → no change."""
    g = _graph_with_reflex_ratio(100, 0.10)
    new = compute_adaptive_half_life(g, 140.0)
    assert new == 140.0


def test_adaptive_half_life_clamps_low() -> None:
    """Cannot go below 60."""
    g = _graph_with_reflex_ratio(20, 0.9)
    new = compute_adaptive_half_life(g, 61.0)
    # 61 * 0.97 = 59.17 → should clamp to 60
    assert new >= 60.0


def test_adaptive_half_life_clamps_high() -> None:
    """Cannot go above 300."""
    g = _graph_with_reflex_ratio(20, 0.0)
    new = compute_adaptive_half_life(g, 299.0)
    # 299 * 1.03 = 307.97 → should clamp to 300
    assert new <= 300.0


def test_adaptive_half_life_empty_graph() -> None:
    """Empty graph should return clamped current value."""
    g = Graph()
    new = compute_adaptive_half_life(g, 140.0)
    assert 60.0 <= new <= 300.0


# ── Synaptic scaling ───────────────────────────────────────────


def test_synaptic_scaling_reduces_hub() -> None:
    """Node with L1=10 should be scaled down (budget=5)."""
    g = Graph()
    g.add_node(Node("hub", "hub"))
    for i in range(10):
        g.add_node(Node(f"t{i}", f"target {i}"))
        g.add_edge(Edge("hub", f"t{i}", weight=1.0))

    scaled = apply_synaptic_scaling(g, budget=5.0)
    assert scaled == 1

    l1_after = sum(e.weight for _, e in g.outgoing("hub"))
    assert l1_after < 10.0
    assert l1_after > 5.0  # gentle scaling, not hard clamp


def test_synaptic_scaling_skips_under_budget() -> None:
    """Node with L1=1.5 (under budget) should not be touched."""
    g = Graph()
    g.add_node(Node("src", "source"))
    for i in range(3):
        g.add_node(Node(f"t{i}", f"target {i}"))
        g.add_edge(Edge("src", f"t{i}", weight=0.5))

    scaled = apply_synaptic_scaling(g, budget=5.0)
    assert scaled == 0

    weights = [e.weight for _, e in g.outgoing("src")]
    assert all(w == 0.5 for w in weights)


def test_synaptic_scaling_skips_constitutional() -> None:
    """Constitutional nodes should not be scaled."""
    g = Graph()
    g.add_node(Node("const", "constitutional", metadata={"authority": "constitutional"}))
    for i in range(10):
        g.add_node(Node(f"t{i}", f"target {i}"))
        g.add_edge(Edge("const", f"t{i}", weight=1.0))

    constitutional = {n.id for n in g.nodes() if (n.metadata or {}).get("authority") == "constitutional"}
    scaled = apply_synaptic_scaling(g, budget=5.0, skip_node_ids=constitutional)
    assert scaled == 0


def test_synaptic_scaling_ignores_negative_edges() -> None:
    """Inhibitory (negative) edges should not count toward L1 or be scaled."""
    g = Graph()
    g.add_node(Node("src", "source"))
    for i in range(3):
        g.add_node(Node(f"t{i}", f"target {i}"))
        g.add_edge(Edge("src", f"t{i}", weight=1.0))
    g.add_node(Node("inhib", "inhibited"))
    g.add_edge(Edge("src", "inhib", weight=-0.5, kind="inhibitory"))

    scaled = apply_synaptic_scaling(g, budget=5.0)
    assert scaled == 0  # L1=3.0 < 5.0

    inhib_edge = g._edges["src"]["inhib"]
    assert inhib_edge.weight == -0.5  # unchanged


# ── Tier hysteresis ────────────────────────────────────────────


def test_habitual_range_includes_buffer_zone() -> None:
    """With habitual_range=(0.15, 0.6), edge at 0.17 should be habitual."""
    cfg = TraversalConfig()  # defaults
    assert cfg.habitual_range == (0.15, 0.6)
    tier = _tier(0.17, cfg)
    assert tier == "habitual"


def test_edge_at_old_dormant_threshold_is_now_habitual() -> None:
    """Edge at 0.19 was dormant under old threshold (0.2), now habitual."""
    cfg = TraversalConfig()
    tier = _tier(0.19, cfg)
    assert tier == "habitual"


def test_truly_dormant_edge() -> None:
    """Edge at 0.10 is still dormant."""
    cfg = TraversalConfig()
    tier = _tier(0.10, cfg)
    assert tier == "dormant"
