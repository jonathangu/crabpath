"""Tests for the CrabPath graph."""

import json

from crabpath import Edge, Graph, Node


def test_create_graph():
    g = Graph()
    assert g.node_count == 0
    assert g.edge_count == 0


def test_add_node():
    g = Graph()
    g.add_node(Node(id="a", content="hello"))
    assert g.node_count == 1
    assert g.get_node("a").content == "hello"


def test_node_defaults():
    n = Node(id="x", content="test")
    assert n.threshold == 1.0
    assert n.potential == 0.0
    assert n.metadata == {}


def test_node_custom_threshold():
    n = Node(id="x", content="test", threshold=0.5)
    assert n.threshold == 0.5


def test_node_metadata():
    n = Node(id="x", content="test", metadata={"type": "fact", "tags": ["deploy"]})
    assert n.metadata["type"] == "fact"
    assert "deploy" in n.metadata["tags"]


def test_add_edge():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=0.8))
    assert g.edge_count == 1
    assert g.get_edge("a", "b").weight == 0.8


def test_negative_edge():
    g = Graph()
    g.add_node(Node(id="rule", content="don't do X"))
    g.add_node(Node(id="bad", content="do X"))
    g.add_edge(Edge(source="rule", target="bad", weight=-1.0))
    assert g.get_edge("rule", "bad").weight == -1.0


def test_outgoing():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))
    g.add_edge(Edge(source="a", target="b", weight=0.5))
    g.add_edge(Edge(source="a", target="c", weight=0.9))

    out = g.outgoing("a")
    assert len(out) == 2
    ids = {n.id for n, _ in out}
    assert ids == {"b", "c"}


def test_incoming():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=0.5))

    inc = g.incoming("b")
    assert len(inc) == 1
    assert inc[0][0].id == "a"


def test_remove_node():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b"))

    g.remove_node("a")
    assert g.node_count == 1
    assert g.edge_count == 0
    assert g.get_node("a") is None


def test_replace_edge():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=0.5))
    g.add_edge(Edge(source="a", target="b", weight=0.9))
    assert g.edge_count == 1
    assert g.get_edge("a", "b").weight == 0.9


def test_reset_potentials():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.get_node("a").potential = 5.0
    g.reset_potentials()
    assert g.get_node("a").potential == 0.0


def test_repr():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    assert "nodes=1" in repr(g)


def test_consolidate_prunes_weak_edges():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))
    g.add_node(Node(id="d", content="D"))

    g.add_edge(Edge(source="a", target="b", weight=0.04))
    g.add_edge(Edge(source="b", target="c", weight=0.60))
    g.add_edge(Edge(source="c", target="a", weight=0.03))
    g.add_edge(Edge(source="a", target="d", weight=0.10))

    stats = g.consolidate(min_weight=0.05)
    assert stats == {"pruned_edges": 2, "pruned_nodes": 0}
    assert g.edge_count == 2
    assert g.get_edge("b", "c") is not None
    assert g.get_edge("a", "d") is not None


def test_consolidate_prunes_orphans():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))
    g.add_edge(Edge(source="a", target="b", weight=0.8))

    stats = g.consolidate()
    assert stats == {"pruned_edges": 0, "pruned_nodes": 1}
    assert g.node_count == 2
    assert g.get_node("c") is None


def test_consolidate_protects_nodes():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C", metadata={"protected": True}))

    stats = g.consolidate()
    assert stats == {"pruned_edges": 0, "pruned_nodes": 2}
    assert g.node_count == 1
    assert g.get_node("c") is not None
    assert g.get_node("c").metadata["protected"] is True


def test_merge_nodes():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))
    g.add_node(Node(id="d", content="D"))
    g.add_edge(Edge(source="a", target="c", weight=0.2))
    g.add_edge(Edge(source="b", target="c", weight=0.9))
    g.add_edge(Edge(source="d", target="b", weight=0.4))
    g.add_edge(Edge(source="b", target="a", weight=-0.3))

    assert g.merge_nodes("a", "b") is True
    assert g.node_count == 3
    assert g.get_node("b") is None
    assert g.get_edge("a", "c").weight == 0.9
    assert g.get_edge("d", "a").weight == 0.4
    assert g.get_edge("a", "a").weight == -0.3


def test_merge_nodes_edge_conflict():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))
    g.add_edge(Edge(source="a", target="c", weight=0.2))
    g.add_edge(Edge(source="b", target="c", weight=-0.7))

    assert g.merge_nodes("a", "b") is True
    assert g.get_edge("a", "c").weight == -0.7


def test_edge_kind_serialization(tmp_path):
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))

    g.add_edge(
        Edge(source="a", target="b", weight=0.6, kind="excitatory", provenance="unit-test")
    )
    g.add_edge(Edge(source="b", target="c", weight=-0.4, kind="inhibitory"))
    g.add_edge(Edge(source="a", target="c", weight=0.1, kind="neutral"))

    path = tmp_path / "edge_kind.json"
    g.save(str(path))
    restored = Graph.load(str(path))

    assert restored.get_edge("a", "b").kind == "excitatory"
    assert restored.get_edge("b", "c").kind == "inhibitory"
    assert restored.get_edge("a", "c").kind == "neutral"


def test_node_new_fields_serialization(tmp_path):
    g = Graph()
    g.add_node(
        Node(
            id="a",
            content="A",
            access_count=3,
            cluster_id="family-1",
            created_at=1_700_000_000.0,
            failure_count=1,
        )
    )
    g.add_node(Node(id="b", content="B", access_count=4, cluster_id="family-2"))

    path = tmp_path / "node_fields.json"
    g.save(str(path))
    restored = Graph.load(str(path))

    n_a = restored.get_node("a")
    n_b = restored.get_node("b")

    assert n_a.access_count == 3
    assert n_a.cluster_id == "family-1"
    assert n_a.failure_count == 1
    assert n_a.created_at == 1_700_000_000.0
    assert n_b.access_count == 4
    assert n_b.cluster_id == "family-2"


def test_edge_provenance_round_trip(tmp_path):
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=0.8, provenance="correction"))

    path = tmp_path / "edge_provenance.json"
    g.save(str(path))
    restored = Graph.load(str(path))

    assert restored.get_edge("a", "b").provenance == "correction"


def test_graph_load_backward_compatible_schema(tmp_path):
    path = tmp_path / "legacy_graph.json"
    path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")

    graph = Graph.load(str(path))
    assert graph.node_count == 0
    assert graph.edge_count == 0


def test_graph_save_includes_schema_version(tmp_path):
    g = Graph()
    g.add_node(Node(id="a", content="schema"))
    path = tmp_path / "schema_graph.json"
    g.save(str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
