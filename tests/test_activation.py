"""Tests for the CrabPath activation engine."""

from crabpath.activation import ActivationConfig, ActivationEngine, ActivationResult
from crabpath.graph import (
    EdgeType,
    MemoryEdge,
    MemoryGraph,
    MemoryNode,
    NodeType,
)


def _build_test_graph() -> MemoryGraph:
    """Build a small test graph for activation tests."""
    g = MemoryGraph()
    
    # Hub node
    g.add_node(MemoryNode(
        id="hub-deploy",
        node_type=NodeType.HUB,
        content="Deployment triage hub",
        summary="deployment triage",
        tags=["deploy", "triage"],
        prior=0.8,
    ))
    
    # Facts
    g.add_node(MemoryNode(
        id="fact-config",
        node_type=NodeType.FACT,
        content="Config changed on Feb 21, breaking staging",
        summary="config change broke staging",
        tags=["config", "staging"],
        prior=0.5,
    ))
    
    # Action
    g.add_node(MemoryNode(
        id="action-logs",
        node_type=NodeType.ACTION,
        content="tail -n 200 /var/log/service.log",
        summary="check service logs",
        tags=["logs", "debug"],
        prior=0.6,
    ))
    
    # Rule (inhibitory)
    g.add_node(MemoryNode(
        id="rule-no-untested",
        node_type=NodeType.RULE,
        content="Never claim fixed without testing on prod",
        summary="test before claiming fixed",
        tags=["deploy", "verification"],
        prior=0.9,
    ))
    
    # Bad action to inhibit
    g.add_node(MemoryNode(
        id="action-claim-fixed",
        node_type=NodeType.ACTION,
        content="Report fix to user without verification",
        summary="claim fixed without testing",
        tags=["report"],
        prior=0.3,
    ))
    
    # Edges
    g.add_edge(MemoryEdge(
        source="hub-deploy", target="fact-config",
        edge_type=EdgeType.ASSOCIATION, weight=0.9,
    ))
    g.add_edge(MemoryEdge(
        source="hub-deploy", target="action-logs",
        edge_type=EdgeType.SEQUENCE, weight=0.8,
    ))
    g.add_edge(MemoryEdge(
        source="rule-no-untested", target="action-claim-fixed",
        edge_type=EdgeType.INHIBITION, weight=-1.0,
    ))
    g.add_edge(MemoryEdge(
        source="fact-config", target="action-logs",
        edge_type=EdgeType.SEQUENCE, weight=0.7,
    ))
    
    return g


def test_basic_activation():
    g = _build_test_graph()
    engine = ActivationEngine(g)
    
    result = engine.activate("deploy", seed_nodes=["hub-deploy"])
    
    assert len(result.activated_nodes) > 0
    assert result.hops > 0
    # Hub should be in results
    node_ids = [n.id for n, _ in result.activated_nodes]
    assert "hub-deploy" in node_ids


def test_activation_spreads():
    g = _build_test_graph()
    engine = ActivationEngine(g)
    
    result = engine.activate("deploy", seed_nodes=["hub-deploy"])
    
    node_ids = [n.id for n, _ in result.activated_nodes]
    # Activation should spread to connected nodes
    assert "fact-config" in node_ids or "action-logs" in node_ids


def test_inhibition_works():
    g = _build_test_graph()
    engine = ActivationEngine(g)
    
    # Seed from rule node
    result = engine.activate(
        "deploy verification",
        seed_nodes=["rule-no-untested"],
    )
    
    # "action-claim-fixed" should be inhibited
    assert "action-claim-fixed" in result.inhibited_nodes


def test_quarantined_nodes_excluded():
    g = _build_test_graph()
    node = g.get_node("action-logs")
    node.quarantined = True
    
    engine = ActivationEngine(g)
    result = engine.activate("deploy", seed_nodes=["hub-deploy"])
    
    node_ids = [n.id for n, _ in result.activated_nodes]
    assert "action-logs" not in node_ids


def test_empty_activation():
    g = MemoryGraph()
    engine = ActivationEngine(g)
    
    result = engine.activate("anything")
    assert len(result.activated_nodes) == 0
    assert result.tier == "deliberative"


def test_learning_updates_weights():
    g = _build_test_graph()
    engine = ActivationEngine(g)
    
    result = engine.activate("deploy", seed_nodes=["hub-deploy"])
    
    # Get initial weight
    edges = g.get_edges_from("hub-deploy")
    initial_weights = {(e.source, e.target): e.weight for e in edges}
    
    # Learn from success
    engine.learn(result, outcome="success", reward=1.0, learning_rate=0.1)
    
    # Weights along traversal path should increase
    edges_after = g.get_edges_from("hub-deploy")
    for e in edges_after:
        key = (e.source, e.target)
        if key in initial_weights and e.target in result.traversal_path:
            assert e.weight >= initial_weights[key]


def test_config_custom():
    config = ActivationConfig(
        alpha=0.7,
        max_hops=2,
        top_k=5,
        threshold=0.05,
    )
    g = _build_test_graph()
    engine = ActivationEngine(g, config=config)
    
    result = engine.activate("deploy", seed_nodes=["hub-deploy"])
    assert result.hops <= 2
    assert len(result.activated_nodes) <= 5
