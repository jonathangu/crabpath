from __future__ import annotations

from crabpath import Edge, Graph, MemoryController, Node
from crabpath.controller import ControllerConfig
from crabpath.decay import DecayConfig
from crabpath.inhibition import InhibitionConfig
from crabpath.learning import LearningConfig
from crabpath.synaptogenesis import SynaptogenesisConfig


def _make_controller(graph: Graph, **updates: object) -> MemoryController:
    cfg_kwargs = dict(
        learning=LearningConfig(),
        synaptogenesis=SynaptogenesisConfig(),
        inhibition=InhibitionConfig(),
        decay=DecayConfig(),
        enable_learning=False,
        enable_synaptogenesis=False,
        enable_inhibition=False,
    )
    cfg_kwargs.update(updates)
    cfg = ControllerConfig(**cfg_kwargs)
    return MemoryController(graph, cfg)


def test_edge_damping_reduces_effective_weight_on_repeated_traversal():
    graph = Graph()
    graph.add_node(Node(id="start", content="loop root"))
    graph.add_node(Node(id="loop", content="loop node"))
    graph.add_edge(Edge(source="start", target="loop", weight=1.0))
    graph.add_edge(Edge(source="loop", target="start", weight=1.0))

    controller = _make_controller(
        graph,
        max_hops=3,
        episode_edge_damping=0.4,
    )

    result = controller.query("loop root")
    steps = result.trajectory
    assert len(steps) >= 3

    assert steps[0]["effective_weight"] == 1.0
    assert steps[1]["effective_weight"] == 1.0
    assert steps[2]["effective_weight"] < steps[1]["effective_weight"]
    assert steps[2]["effective_weight"] < steps[1]["effective_weight"]


def test_inhibitory_edges_are_not_damped():
    graph = Graph()
    graph.add_node(Node(id="start", content="inhibition check"))
    graph.add_node(Node(id="loop", content="loop node"))
    graph.add_node(Node(id="bad", content="inhibitory node"))
    graph.add_edge(Edge(source="start", target="loop", weight=1.0))
    graph.add_edge(Edge(source="loop", target="bad", weight=-1.0))
    graph.add_edge(Edge(source="loop", target="loop", weight=1.0))
    graph.add_edge(Edge(source="bad", target="loop", weight=1.0))

    class InhibitoryPreferringController(MemoryController):
        def _select_next(
            self,
            query_text: str,
            current_node_id: str,
            candidates: list[tuple[str, float]],
            llm_call: object | None,
        ):
            for node_id, _ in candidates:
                if node_id == "bad":
                    return "bad"
            return super()._select_next(query_text, current_node_id, candidates, llm_call)

    base_cfg = ControllerConfig(
        learning=LearningConfig(),
        synaptogenesis=SynaptogenesisConfig(),
        inhibition=InhibitionConfig(suppression_lambda=0.0),
        decay=DecayConfig(),
        enable_learning=False,
        enable_synaptogenesis=False,
        enable_inhibition=False,
        max_hops=4,
        episode_edge_damping=0.5,
    )
    controller = InhibitoryPreferringController(graph, base_cfg)

    result = controller.query("inhibition check")
    steps = result.trajectory

    assert len(steps) >= 4
    assert steps[0]["to_node"] == "loop"
    assert steps[1]["to_node"] == "bad"
    assert steps[3]["to_node"] == "bad"
    assert steps[1]["effective_weight"] == -1.0
    assert steps[3]["effective_weight"] == -1.0


def test_visit_penalty_reduces_score_for_revisited_nodes():
    graph = Graph()
    graph.add_node(Node(id="start", content="visit penalty"))
    graph.add_node(Node(id="fresh", content="fresh candidate"))
    graph.add_node(Node(id="loop", content="loop candidate"))
    graph.add_edge(Edge(source="start", target="loop", weight=1.0))
    graph.add_edge(Edge(source="loop", target="start", weight=1.0))
    graph.add_edge(Edge(source="loop", target="fresh", weight=0.95))

    controller = _make_controller(
        graph,
        max_hops=2,
        episode_visit_penalty=0.6,
    )

    result = controller.query("visit penalty")
    second = result.trajectory[1]

    assert second["to_node"] == "fresh"
    second_candidates = dict(second["candidates"])
    assert second_candidates["start"] < second_candidates["fresh"]


def test_edge_weights_unchanged_after_episode():
    graph = Graph()
    graph.add_node(Node(id="start", content="weight check"))
    graph.add_node(Node(id="loop", content="loop node"))
    graph.add_edge(Edge(source="start", target="loop", weight=0.9))
    graph.add_edge(Edge(source="loop", target="loop", weight=0.8))

    controller = _make_controller(
        graph,
        max_hops=3,
        episode_edge_damping=0.2,
    )
    result = controller.query("weight check")
    assert result.trajectory

    assert graph.get_edge("start", "loop").weight == 0.9
    assert graph.get_edge("loop", "loop").weight == 0.8


def test_max_hops_safety_cap_stops_traversal():
    graph = Graph()
    graph.add_node(Node(id="start", content="hop cap"))
    graph.add_node(Node(id="loop", content="loop node"))
    graph.add_edge(Edge(source="start", target="loop", weight=1.0))
    graph.add_edge(Edge(source="loop", target="loop", weight=1.0))

    controller = _make_controller(graph, max_hops=2)
    result = controller.query("hop cap")

    assert len(result.trajectory) == 2
    assert len(result.selected_nodes) == 3


def test_damping_factor_one_is_identity():
    graph = Graph()
    graph.add_node(Node(id="start", content="damping identity"))
    graph.add_node(Node(id="loop", content="loop node"))
    graph.add_edge(Edge(source="start", target="loop", weight=1.0))
    graph.add_edge(Edge(source="loop", target="loop", weight=1.0))

    controller = _make_controller(
        graph,
        max_hops=3,
        episode_edge_damping=1.0,
    )
    result = controller.query("damping identity")

    assert len(result.trajectory) >= 3
    assert all(step["effective_weight"] == 1.0 for step in result.trajectory)


def test_damping_factor_zero_fully_suppresses_after_first_use():
    graph = Graph()
    graph.add_node(Node(id="start", content="damping zero"))
    graph.add_node(Node(id="loop", content="loop node"))
    graph.add_edge(Edge(source="start", target="loop", weight=1.0))
    graph.add_edge(Edge(source="loop", target="loop", weight=1.0))

    controller = _make_controller(
        graph,
        max_hops=4,
        episode_edge_damping=0.0,
    )
    result = controller.query("damping zero")

    assert len(result.trajectory) == 2
    assert result.trajectory[0]["effective_weight"] == 1.0
    assert result.trajectory[1]["effective_weight"] == 1.0


def test_visit_penalty_zero_is_identity():
    graph = Graph()
    graph.add_node(Node(id="start", content="visit penalty zero"))
    graph.add_node(Node(id="loop", content="loop node"))
    graph.add_edge(Edge(source="start", target="loop", weight=1.0))
    graph.add_edge(Edge(source="loop", target="start", weight=1.0))
    graph.add_edge(Edge(source="loop", target="loop", weight=1.0))

    controller = _make_controller(
        graph,
        max_hops=3,
        episode_visit_penalty=0.0,
    )
    result = controller.query("visit penalty zero")

    assert len(result.trajectory) == 3
    assert result.trajectory[1]["to_node"] in {"start", "loop"}
