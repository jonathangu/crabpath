from __future__ import annotations

from crabpath import Edge, Graph, Node
from crabpath.router import Router, RouterDecision
from crabpath.traversal import (
    TraversalConfig,
    TraversalTrajectory,
    _normalize_seed_nodes,
    render_context,
    traverse,
)


def _node_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node(id="start", content="start content"))
    graph.add_node(Node(id="reflex", content="reflex content"))
    graph.add_node(Node(id="habit", content="habit content"))
    graph.add_node(Node(id="dormant", content="dormant content"))
    graph.add_node(Node(id="alt", content="alt content"))
    graph.add_node(Node(id="loop", content="loop content"))
    return graph


def test_reflex_auto_follows_high_weight():
    graph = _node_graph()
    graph.add_edge(Edge(source="start", target="reflex", weight=0.9))
    graph.add_edge(Edge(source="start", target="habit", weight=0.5))

    class TrackingRouter(Router):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def decide_next(self, *args, **kwargs):
            self.calls += 1
            return super().decide_next(*args, **kwargs)

    router = TrackingRouter()
    trajectory = traverse(
        query="route via reflex",
        graph=graph,
        router=router,
        seed_nodes=["start"],
        config=TraversalConfig(max_hops=3, branch_beam=2),
    )

    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].to_node == "reflex"
    assert trajectory.steps[0].tier == "reflex"
    assert trajectory.steps[0].edge_weight == 0.9
    assert router.calls == 0
    assert trajectory.visit_order == ["start", "reflex"]


def test_habitual_uses_router():
    graph = _node_graph()
    graph.add_edge(Edge(source="start", target="habit", weight=0.5))
    graph.add_edge(Edge(source="start", target="alt", weight=0.5))

    class ForcedRouter(Router):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def decide_next(
            self, query, current_node_id, candidate_nodes, context, tier, previous_reasoning=None
        ):
            self.calls += 1
            # Keep deterministic and explicit for the test.
            return RouterDecision(
                chosen_target="habit",
                rationale="forced",
                confidence=0.99,
                tier=tier,
                alternatives=[],
                raw={},
            )

    router = ForcedRouter()
    trajectory = traverse(
        query="use router",
        graph=graph,
        router=router,
        seed_nodes=["start"],
        config=TraversalConfig(max_hops=3, branch_beam=2),
    )

    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].to_node == "habit"
    assert trajectory.steps[0].tier == "habitual"
    assert router.calls == 1


def test_dormant_skipped():
    graph = _node_graph()
    graph.add_edge(Edge(source="start", target="dormant", weight=0.1))
    graph.add_edge(Edge(source="start", target="alt", weight=0.2))

    router = Router()
    trajectory = traverse(
        query="avoid dormant",
        graph=graph,
        router=router,
        seed_nodes=["start"],
        config=TraversalConfig(max_hops=3),
    )

    assert len(trajectory.steps) == 0
    assert trajectory.visit_order == ["start"]


def test_cycle_prevention():
    graph = _node_graph()
    graph.add_edge(Edge(source="start", target="loop", weight=0.9))
    graph.add_edge(Edge(source="loop", target="start", weight=0.9))

    trajectory = traverse(
        query="no cycle",
        graph=graph,
        router=Router(),
        seed_nodes=["start"],
        config=TraversalConfig(max_hops=5),
    )

    assert trajectory.visit_order == ["start", "loop"]
    assert trajectory.steps[0].to_node == "loop"
    assert len(trajectory.steps) == 1


def test_traversal_returns_all_candidates_per_step():
    graph = _node_graph()
    graph.add_edge(Edge(source="start", target="habit", weight=0.5))
    graph.add_edge(Edge(source="start", target="reflex", weight=0.2))
    graph.add_edge(Edge(source="start", target="dormant", weight=0.1))

    class ForcedHabitualRouter(Router):
        def decide_next(
            self, query, current_node_id, candidate_nodes, context, tier, previous_reasoning=None
        ):
            return RouterDecision(
                chosen_target="habit",
                rationale="forced",
                confidence=0.9,
                tier=tier,
                alternatives=[],
                raw={},
            )

    trajectory = traverse(
        query="all candidates",
        graph=graph,
        router=ForcedHabitualRouter(),
        seed_nodes=["start"],
        config=TraversalConfig(max_hops=3),
    )

    step = trajectory.steps[0]
    assert len(step.candidates) == 3
    assert set(step.candidates) == {("habit", 0.5), ("reflex", 0.2), ("dormant", 0.1)}
    assert step.to_node == "habit"


def test_render_context_truncation():
    graph = Graph()
    graph.add_node(Node(id="a", content="A" * 120))
    graph.add_node(Node(id="b", content="B" * 120))
    graph.add_node(Node(id="c", content="C" * 120))
    trajectory = TraversalTrajectory(
        steps=[],
        visit_order=["a", "b", "c"],
        context_nodes=[],
        raw_context="",
    )

    context = render_context(trajectory, graph, max_chars=150)
    assert len(context) <= 150


def test_normalize_seed_nodes_with_bad_scores():
    assert _normalize_seed_nodes({"a": "bad", "b": None}) == [("a", 1.0), ("b", 1.0)]
    assert _normalize_seed_nodes([("a", "bad"), ("b", None), "plain"]) == [
        ("a", 1.0),
        ("b", 1.0),
        ("plain", 1.0),
    ]
