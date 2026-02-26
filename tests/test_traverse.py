from __future__ import annotations

from crabpath.graph import Edge, Graph, Node
from crabpath.traverse import TraversalConfig, traverse


def test_traverse_prefers_highest_reflex_edge() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.95))
    graph.add_edge(Edge("a", "c", 0.45))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=1, beam_width=1))
    assert result.fired == ["a", "b"]
    assert result.steps[0].to_node == "b"
    assert result.steps[0].tier == "reflex"


def test_reflex_tier_auto_follows_without_route_fn() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.9))
    graph.add_edge(Edge("a", "c", 0.9))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=1, beam_width=2))
    assert result.steps
    assert any(step.tier == "reflex" for step in result.steps)
    assert {step.to_node for step in result.steps} == {"b", "c"}


def test_habitual_tier_calls_route_fn() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.4))
    graph.add_edge(Edge("a", "c", 0.4))

    result = traverse(
        graph,
        [("a", 1.0)],
        config=TraversalConfig(max_hops=1, beam_width=2, reflex_threshold=0.95),
        route_fn=lambda _query, cands: ["c"] if "c" in cands else [],
    )

    assert result.fired == ["a", "c"]
    assert "b" not in result.fired


def test_dormant_tier_is_skipped() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.2))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=2, beam_width=1))
    assert result.fired == ["a"]
    assert len(result.steps) == 0


def test_traverse_edge_damping_reduces_effective_weight() -> None:
    graph = Graph()
    graph.add_node(Node("x", "X"))
    graph.add_node(Node("y", "Y"))
    graph.add_edge(Edge("x", "y", 0.9))
    graph.add_edge(Edge("y", "x", 0.9))

    result = traverse(graph, [("x", 1.0)], config=TraversalConfig(max_hops=4, beam_width=1, edge_damping=0.3))
    assert len(result.steps) == 4
    assert result.steps[0].effective_weight == 0.9
    assert result.steps[1].effective_weight == 0.9
    assert result.steps[2].effective_weight == 0.27
    assert result.steps[3].effective_weight == 0.27


def test_edge_damping_prevents_unbounded_cycles() -> None:
    graph = Graph()
    for node_id in "abcd":
        graph.add_node(Node(node_id, node_id))
    graph.add_edge(Edge("a", "b", 0.9))
    graph.add_edge(Edge("b", "c", 0.9))
    graph.add_edge(Edge("c", "a", 0.9))
    graph.add_edge(Edge("c", "d", 0.01))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=20, beam_width=1, edge_damping=0.2))
    assert len(result.steps) == 20
    assert len(result.fired) >= 3


def test_traverse_beam_width_explores_multiple_paths() -> None:
    graph = Graph()
    for node_id in ["a", "b", "c", "d", "e"]:
        graph.add_node(Node(node_id, node_id))

    graph.add_edge(Edge("a", "b", 0.9))
    graph.add_edge(Edge("a", "c", 0.8))
    graph.add_edge(Edge("b", "d", 0.8))
    graph.add_edge(Edge("c", "e", 0.8))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=2, beam_width=2))
    assert result.fired[0] == "a"
    assert {"b", "c"}.issubset(set(result.fired))
    assert {"d", "e"}.issubset(set(result.fired))


def test_max_hops_is_respected() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.9))
    graph.add_edge(Edge("b", "a", 0.9))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=3, beam_width=1))
    assert len(result.steps) == 3
    assert len(result.fired) == 2


def test_route_fn_empty_culls_all_habitual_choices() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.4))
    graph.add_edge(Edge("a", "c", 0.4))

    result = traverse(
        graph,
        [("a", 1.0)],
        config=TraversalConfig(max_hops=3, beam_width=2, reflex_threshold=0.95),
        route_fn=lambda _query, cands: [],
    )

    assert result.fired == ["a"]
    assert result.steps == []


def test_route_fn_always_picks_same_node_and_damping_applies() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.7))
    graph.add_edge(Edge("b", "a", 0.7))

    result = traverse(
        graph,
        [("a", 1.0)],
        config=TraversalConfig(max_hops=4, beam_width=1, edge_damping=0.2),
        route_fn=lambda _query, cands: [cands[0]] if cands else [],
    )

    assert len(result.steps) == 4
    assert result.steps[0].effective_weight == 0.7
    assert round(result.steps[2].effective_weight, 12) == 0.14


def test_traversal_without_outgoing_edges_terminates() -> None:
    graph = Graph()
    graph.add_node(Node("x", "X"))
    result = traverse(graph, [("x", 1.0)], config=TraversalConfig(max_hops=10, beam_width=3))
    assert result.fired == ["x"]
    assert result.steps == []


def test_traversal_with_non_existent_seed() -> None:
    graph = Graph()
    graph.add_node(Node("x", "X"))
    result = traverse(graph, [("missing", 1.0)], config=TraversalConfig(max_hops=3, beam_width=2))
    assert result.fired == []
    assert result.steps == []
    assert result.context == ""


def test_traversal_with_all_dormant_edges_returns_seed_only() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.01))
    graph.add_edge(Edge("b", "c", 0.02))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=3, beam_width=2, reflex_threshold=0.9))
    assert result.fired == ["a"]
    assert not result.steps


def test_traversal_skips_inhibitory_edges() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", -0.4, kind="inhibitory"))
    graph.add_edge(Edge("a", "c", 0.4))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=2, beam_width=2, reflex_threshold=0.95))
    assert result.fired == ["a", "c"]
    assert result.steps[0].to_node == "c"


def test_edge_damping_factor_one_steps_until_hops() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.9))
    graph.add_edge(Edge("b", "a", 0.9))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=6, beam_width=1, edge_damping=1.0))
    assert len(result.steps) == 6
    assert len(result.fired) == 2


def test_edge_damping_factor_zero_never_reuses_directed_edges() -> None:
    graph = Graph()
    for node_id in ["x", "y", "z"]:
        graph.add_node(Node(node_id, node_id))
    graph.add_edge(Edge("x", "y", 1.0))
    graph.add_edge(Edge("x", "z", 1.0))
    graph.add_edge(Edge("y", "x", 1.0))
    graph.add_edge(Edge("z", "x", 1.0))

    result = traverse(graph, [("x", 1.0)], config=TraversalConfig(max_hops=4, beam_width=1, edge_damping=0.0))
    assert len(result.steps) == 4
    assert len(result.steps) == len(set((step.from_node, step.to_node) for step in result.steps))


def test_traversal_context_includes_all_fired_content() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha content"))
    graph.add_node(Node("b", "beta content"))
    graph.add_node(Node("c", "gamma content"))
    graph.add_edge(Edge("a", "b", 0.95))
    graph.add_edge(Edge("b", "c", 0.95))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=2, beam_width=1))
    assert "alpha content" in result.context
    assert "beta content" in result.context
    assert "gamma content" in result.context


def test_large_graph_traversal_is_responsive() -> None:
    graph = Graph()
    for i in range(50):
        graph.add_node(Node(f"n{i}", f"node {i}"))
        if i > 0:
            graph.add_edge(Edge(f"n{i-1}", f"n{i}", 0.8))

    result = traverse(graph, [("n0", 1.0)], config=TraversalConfig(max_hops=25, beam_width=2))
    assert len(result.steps) == 25
    assert len(result.fired) == 26


def test_seeds_with_higher_weights_are_explored_first() -> None:
    graph = Graph()
    graph.add_node(Node("high", "H"))
    graph.add_node(Node("low", "L"))
    graph.add_node(Node("common", "C"))
    graph.add_edge(Edge("high", "common", 0.6))
    graph.add_edge(Edge("low", "common", 0.6))

    result = traverse(
        graph,
        [("low", 0.1), ("high", 1.0)],
        config=TraversalConfig(max_hops=1, beam_width=2),
    )
    assert result.fired[:2] == ["high", "low"]


def test_visit_penalty_discourages_revisiting_nodes() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_node(Node("d", "D"))

    graph.add_edge(Edge("a", "b", 0.8))
    graph.add_edge(Edge("a", "c", 0.8))
    graph.add_edge(Edge("b", "d", 0.8))
    graph.add_edge(Edge("b", "a", 0.8))
    graph.add_edge(Edge("c", "d", 0.8))

    result = traverse(
        graph,
        [("a", 1.0)],
        config=TraversalConfig(max_hops=2, beam_width=2, visit_penalty=1.5),
    )

    assert result.fired == ["a", "b", "c", "d"]


def test_traversal_with_all_edges_inhibitory_or_low_returns_seed_only() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", -0.4))
    graph.add_edge(Edge("a", "c", -0.1))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=2, beam_width=1, reflex_threshold=0.0))
    assert result.steps == []
    assert result.fired == ["a"]


def test_traversal_tier_classification_is_stable() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.9))
    graph.add_edge(Edge("a", "c", 0.5))

    result = traverse(graph, [("a", 1.0)], config=TraversalConfig(max_hops=1, beam_width=2, reflex_threshold=0.8))
    tiers = [step.tier for step in result.steps]
    assert tiers == ["reflex", "habitual"]
