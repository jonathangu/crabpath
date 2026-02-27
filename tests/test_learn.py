from __future__ import annotations

from math import isclose

from crabpath.graph import Edge, Graph, Node
from crabpath.learn import LearningConfig, apply_outcome, hebbian_update


def test_apply_outcome_positive_strengthens_edges() -> None:
    """test apply outcome positive strengthens edges."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.5))
    graph.add_edge(Edge("b", "c", 0.2))

    apply_outcome(graph, ["a", "b", "c"], outcome=1.0, config=LearningConfig(learning_rate=0.1, discount=1.0))

    assert graph._edges["a"]["b"].weight > 0.5
    assert graph._edges["b"]["c"].weight > 0.2


def test_apply_outcome_negative_weakens_edges() -> None:
    """test apply outcome negative weakens edges."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.5))

    apply_outcome(graph, ["a", "b"], outcome=-1.0, config=LearningConfig(learning_rate=0.2, discount=1.0))
    assert graph._edges["a"]["b"].weight < 0.5
    assert graph._edges["a"]["b"].weight > 0.0
    assert graph._edges["a"]["b"].kind in {"sibling", "inhibitory"}


def test_negative_outcome_creates_inhibitory_edge_if_missing() -> None:
    """test negative outcome creates inhibitory edge if missing."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))

    updates = apply_outcome(graph, ["a", "b"], outcome=-1.0, config=LearningConfig(learning_rate=0.2, discount=1.0))
    assert "a->b" in updates
    assert graph._edges["a"]["b"].kind == "inhibitory"
    assert graph._edges["a"]["b"].weight < 0.0


def test_learning_clips_weights_to_bounds() -> None:
    """test learning clips weights to bounds."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.98))

    apply_outcome(graph, ["a", "b"], outcome=1.0, config=LearningConfig(learning_rate=1.0, discount=1.0))
    assert graph._edges["a"]["b"].weight <= 1.0
    assert graph._edges["a"]["b"].weight == 1.0


def test_hebbian_update_strengthens_shared_edges() -> None:
    """test hebbian update strengthens shared edges."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))

    graph.add_edge(Edge("a", "b", 0.2))
    hebbian_update(graph, ["a", "b", "c"])

    assert graph._edges["a"].get("c").weight == 0.06
    assert graph._edges["a"]["b"].weight == 0.26
    assert graph._edges["b"].get("c").weight == 0.06


def test_hebbian_disconnected_nodes_create_edges() -> None:
    """test hebbian disconnected nodes create edges."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    apply_outcome(graph, ["a", "b", "c"], outcome=1.0)

    assert graph._edges.get("a", {}).get("c") is not None
    assert graph._edges.get("c") is None


def test_learning_rate_controls_magnitude_of_change() -> None:
    """test learning rate controls magnitude of change."""
    graph_slow = Graph()
    graph_fast = Graph()
    for graph in (graph_slow, graph_fast):
        graph.add_node(Node("a", "A"))
        graph.add_node(Node("b", "B"))
        graph.add_edge(Edge("a", "b", 0.2))

    apply_outcome(graph_slow, ["a", "b"], outcome=1.0, config=LearningConfig(learning_rate=0.01, discount=1.0))
    apply_outcome(graph_fast, ["a", "b"], outcome=1.0, config=LearningConfig(learning_rate=0.2, discount=1.0))

    assert graph_fast._edges["a"]["b"].weight > graph_slow._edges["a"]["b"].weight


def test_apply_outcome_uses_per_node_scores() -> None:
    """test apply outcome uses per node scores."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.2))
    graph.add_edge(Edge("b", "c", 0.2))

    apply_outcome(
        graph=graph,
        fired_nodes=["a", "b", "c"],
        outcome=1.0,
        config=LearningConfig(learning_rate=0.4, discount=1.0),
        per_node_outcomes={"a": 1.0, "b": -0.5},
    )

    assert graph._edges["a"]["b"].weight > 0.2
    assert graph._edges["b"]["c"].weight < 0.2


def test_discount_factor_reduces_later_step_credit() -> None:
    """test discount factor reduces later step credit."""
    graph = Graph()
    for node_id in ["a", "b", "c"]:
        graph.add_node(Node(node_id, node_id))
    graph.add_edge(Edge("a", "b", 0.5))
    graph.add_edge(Edge("b", "c", 0.5))

    apply_outcome(graph, ["a", "b", "c"], outcome=1.0, config=LearningConfig(learning_rate=1.0, discount=0.5))
    assert graph._edges["a"]["b"].weight > graph._edges["b"]["c"].weight


def test_apply_outcome_empty_fired_nodes_is_noop() -> None:
    """test apply outcome empty fired nodes is noop."""
    graph = Graph()
    updates = apply_outcome(graph, [], outcome=1.0)
    assert updates == {}


def test_apply_outcome_single_node_does_not_update_edges() -> None:
    """test apply outcome single node does not update edges."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    updates = apply_outcome(graph, ["a"], outcome=1.0)

    assert updates == {}
    assert graph.edge_count() == 0


def test_apply_outcome_uses_full_trajectory_not_last_edge_only() -> None:
    """test apply outcome uses full trajectory not last edge only."""
    graph = Graph()
    for node_id in ["a", "b", "c", "d"]:
        graph.add_node(Node(node_id, node_id))

    graph.add_edge(Edge("a", "b", 0.0))
    graph.add_edge(Edge("b", "c", 0.0))
    graph.add_edge(Edge("c", "d", 0.0))

    updates = apply_outcome(graph, ["a", "b", "c", "d"], outcome=1.0, config=LearningConfig(learning_rate=1.0, discount=0.5))
    assert updates["a->b"] > updates["b->c"] > updates["c->d"]


def test_apply_outcome_multiple_rounds_accumulate() -> None:
    """test apply outcome multiple rounds accumulate."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.1))

    for _ in range(3):
        apply_outcome(graph, ["a", "b"], outcome=1.0, config=LearningConfig(learning_rate=0.1, discount=1.0))

    assert graph._edges["a"]["b"].weight > 0.4
    assert graph._edges["a"]["b"].weight < 1.0


def test_apply_outcome_handles_missing_nodes_gracefully() -> None:
    """test apply outcome handles missing nodes gracefully."""
    graph = Graph()
    apply_outcome(graph, ["ghost", "ghost2"], outcome=1.0)
    graph.add_node(Node("ghost2", "G"))

    assert graph._edges.get("ghost", {}).get("ghost2") is not None


def test_hebbian_update_no_change_for_single_node() -> None:
    """test hebbian update no change for single node."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    hebbian_update(graph, ["a"])
    assert graph.edge_count() == 0


def test_zero_or_negative_outcome_can_flip_kind_to_inhibitory() -> None:
    """test zero or negative outcome can flip kind to inhibitory."""
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.02))

    apply_outcome(graph, ["a", "b"], outcome=-1.0, config=LearningConfig(learning_rate=0.2, discount=1.0))
    assert graph._edges["a"]["b"].kind == "inhibitory"
    assert graph._edges["a"]["b"].weight < 0.0
