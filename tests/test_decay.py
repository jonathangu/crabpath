from crabpath import Edge, Graph, Node
from crabpath.decay import DecayConfig, apply_decay, decay_factor


def test_decay_reduces_weight():
    graph = Graph()
    graph.add_node(Node(id="root", content="Root"))
    graph.add_node(Node(id="leaf", content="Leaf"))
    graph.add_edge(Edge(source="root", target="leaf", weight=4.0))

    changed = apply_decay(graph, turns_elapsed=10)

    edge = graph.get_edge("root", "leaf")
    assert edge is not None
    assert changed["root->leaf"] == edge.weight
    assert edge.weight < 4.0


def test_decay_respects_half_life():
    config = DecayConfig(half_life_turns=4)
    graph = Graph()
    graph.add_node(Node(id="root", content="Root"))
    graph.add_node(Node(id="leaf", content="Leaf"))
    graph.add_edge(Edge(source="root", target="leaf", weight=2.0))

    changed = apply_decay(graph, turns_elapsed=4, config=config)

    edge = graph.get_edge("root", "leaf")
    assert edge is not None
    assert changed["root->leaf"] == edge.weight
    assert abs(edge.weight - 1.0) < 1e-9


def test_apply_decay_all_edges():
    graph = Graph()
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="b", content="B"))
    graph.add_node(Node(id="c", content="C"))
    graph.add_edge(Edge(source="a", target="b", weight=1.0))
    graph.add_edge(Edge(source="a", target="c", weight=-2.0))

    config = DecayConfig(half_life_turns=1)
    changed = apply_decay(graph, turns_elapsed=2, config=config)

    assert set(changed.keys()) == {"a->b", "a->c"}
    assert changed["a->b"] == graph.get_edge("a", "b").weight
    assert changed["a->c"] == graph.get_edge("a", "c").weight

    expected_factor = decay_factor(1, 2)
    assert expected_factor == 0.25
    assert abs(graph.get_edge("a", "b").weight - 0.25) < 1e-9
    assert abs(graph.get_edge("a", "c").weight + 0.5) < 1e-9
