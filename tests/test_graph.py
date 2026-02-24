"""Tests for the CrabPath memory graph."""

from crabpath.graph import (
    EdgeType,
    MemoryEdge,
    MemoryGraph,
    MemoryNode,
    NodeType,
)


def test_create_graph():
    g = MemoryGraph()
    assert g.node_count == 0
    assert g.edge_count == 0


def test_add_node():
    g = MemoryGraph()
    node = MemoryNode(
        id="fact-1",
        node_type=NodeType.FACT,
        content="The production database uses PostgreSQL 16.",
        summary="Prod uses Postgres 16",
        tags=["database", "production"],
        prior=0.5,
    )
    g.add_node(node)
    assert g.node_count == 1
    assert g.get_node("fact-1") is node


def test_add_edge():
    g = MemoryGraph()
    g.add_node(MemoryNode(id="a", node_type=NodeType.FACT, content="A"))
    g.add_node(MemoryNode(id="b", node_type=NodeType.FACT, content="B"))
    
    edge = MemoryEdge(
        source="a",
        target="b",
        edge_type=EdgeType.ASSOCIATION,
        weight=0.8,
    )
    g.add_edge(edge)
    assert g.edge_count == 1


def test_inhibitory_edges():
    g = MemoryGraph()
    g.add_node(MemoryNode(id="rule-1", node_type=NodeType.RULE, content="Never clear cache"))
    g.add_node(MemoryNode(id="action-bad", node_type=NodeType.ACTION, content="Tell user to clear cache"))
    
    edge = MemoryEdge(
        source="rule-1",
        target="action-bad",
        edge_type=EdgeType.INHIBITION,
        weight=-1.0,
    )
    g.add_edge(edge)
    
    inhibitors = g.get_inhibitors("action-bad")
    assert len(inhibitors) == 1
    assert inhibitors[0].source == "rule-1"


def test_nodes_by_type():
    g = MemoryGraph()
    g.add_node(MemoryNode(id="f1", node_type=NodeType.FACT, content="fact 1"))
    g.add_node(MemoryNode(id="f2", node_type=NodeType.FACT, content="fact 2"))
    g.add_node(MemoryNode(id="r1", node_type=NodeType.RULE, content="rule 1"))
    
    facts = g.nodes_by_type(NodeType.FACT)
    assert len(facts) == 2
    
    rules = g.nodes_by_type(NodeType.RULE)
    assert len(rules) == 1


def test_quarantine():
    g = MemoryGraph()
    node = MemoryNode(
        id="bad-1",
        node_type=NodeType.ACTION,
        content="rm -rf /",
        quarantined=True,
        quarantine_reason="Caused catastrophic failure",
    )
    g.add_node(node)
    assert g.get_node("bad-1").quarantined is True
    assert g.stats()["quarantined"] == 1


def test_edge_success_rate():
    edge = MemoryEdge(
        source="a", target="b",
        edge_type=EdgeType.SEQUENCE,
        success_count=8,
        failure_count=2,
    )
    assert edge.success_rate == 0.8


def test_graph_stats():
    g = MemoryGraph()
    g.add_node(MemoryNode(id="f1", node_type=NodeType.FACT, content=""))
    g.add_node(MemoryNode(id="t1", node_type=NodeType.TOOL, content=""))
    g.add_edge(MemoryEdge(source="f1", target="t1", edge_type=EdgeType.ASSOCIATION))
    
    stats = g.stats()
    assert stats["nodes"] == 2
    assert stats["edges"] == 1
    assert stats["node_types"]["fact"] == 1
    assert stats["node_types"]["tool"] == 1


def test_remove_node():
    g = MemoryGraph()
    g.add_node(MemoryNode(id="a", node_type=NodeType.FACT, content="A"))
    g.add_node(MemoryNode(id="b", node_type=NodeType.FACT, content="B"))
    g.add_edge(MemoryEdge(source="a", target="b", edge_type=EdgeType.ASSOCIATION))
    
    g.remove_node("a")
    assert g.node_count == 1
    assert g.edge_count == 0
    assert g.get_node("a") is None
