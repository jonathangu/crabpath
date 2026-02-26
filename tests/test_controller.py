from __future__ import annotations

from crabpath import Edge, Graph, MemoryController, Node, QueryResult
from crabpath.controller import ControllerConfig
from crabpath.decay import DecayConfig
from crabpath.inhibition import InhibitionConfig
from crabpath.learning import LearningConfig
from crabpath.synaptogenesis import SynaptogenesisConfig


def _graph_chain() -> Graph:
    graph = Graph()
    graph.add_node(Node(id="start", content="alpha root"))
    graph.add_node(Node(id="good", content="alpha good node"))
    graph.add_node(Node(id="bad", content="alpha bad node"))
    graph.add_edge(Edge(source="start", target="good", weight=0.9))
    graph.add_edge(Edge(source="start", target="bad", weight=0.6))
    return graph


def test_query_returns_results():
    graph = _graph_chain()
    controller = MemoryController(graph)

    result = controller.query("alpha")

    assert result.query == "alpha"
    assert result.selected_nodes == ["start", "good"]
    assert result.context
    assert result.context_chars == len(result.context)
    assert result.candidates_considered >= 1


def test_learn_positive_reinforces():
    graph = Graph()
    graph.add_node(Node(id="start", content="alpha root"))
    graph.add_node(Node(id="good", content="alpha good node"))
    graph.add_edge(Edge(source="start", target="good", weight=0.4))

    controller = MemoryController(graph)
    result = controller.query("alpha")
    before = graph.get_edge("start", "good").weight

    summary = controller.learn(result, reward=1.0)

    edge = graph.get_edge("start", "good")
    assert edge is not None
    assert edge.weight > before
    assert summary["reward"] == 1.0


def test_learn_negative_inhibits():
    graph = Graph()
    graph.add_node(Node(id="start", content="alpha root"))
    graph.add_node(Node(id="wrong", content="alpha wrong node"))

    config = ControllerConfig(
        learning=LearningConfig(),
        synaptogenesis=SynaptogenesisConfig(),
        inhibition=InhibitionConfig(),
        decay=DecayConfig(),
        enable_learning=False,
        enable_synaptogenesis=False,
        enable_inhibition=True,
    )
    controller = MemoryController(graph, config)
    result = QueryResult(
        query="alpha",
        selected_nodes=["start", "wrong"],
        context="alpha root\n\nalpha wrong node",
        context_chars=31,
        trajectory=[
            {
                "from_node": "start",
                "to_node": "wrong",
                "edge_weight": 0.0,
                "candidates": [("wrong", 0.0)],
            }
        ],
        candidates_considered=1,
    )

    controller.learn(result, reward=-1.0)

    edge = graph.get_edge("start", "wrong")
    assert edge is not None
    assert edge.kind == "inhibitory"


def test_inhibition_affects_future_queries():
    graph = Graph()
    graph.add_node(Node(id="start", content="alpha root"))
    graph.add_node(Node(id="good", content="alpha good node"))
    graph.add_node(Node(id="bad", content="alpha bad node"))
    graph.add_edge(Edge(source="start", target="good", weight=0.9))
    graph.add_edge(Edge(source="start", target="bad", weight=1.0))

    controller = MemoryController(graph)

    first = controller.query("alpha")
    assert first.selected_nodes == ["start", "bad"]

    controller.learn(first, reward=-1.0)
    second = controller.query("alpha")

    assert second.selected_nodes == ["start", "good"]


def test_decay_fires_on_interval():
    graph = Graph()
    graph.add_node(Node(id="start", content="alpha root"))
    graph.add_node(Node(id="good", content="alpha good node"))
    graph.add_edge(Edge(source="start", target="good", weight=1.0))

    config = ControllerConfig(
        learning=LearningConfig(),
        synaptogenesis=SynaptogenesisConfig(),
        inhibition=InhibitionConfig(),
        decay=DecayConfig(half_life_turns=1),
        enable_learning=False,
        enable_synaptogenesis=False,
        enable_inhibition=False,
        enable_decay=True,
        decay_interval=2,
    )
    controller = MemoryController(graph, config)
    first = controller.query("alpha")
    summary1 = controller.learn(first, reward=0.0)
    assert summary1["decayed"] == {}

    second = controller.query("alpha")
    summary2 = controller.learn(second, reward=0.0)

    assert summary2["decayed"]
    assert "start->good" in summary2["decayed"]


def test_access_count_increments():
    graph = _graph_chain()
    controller = MemoryController(graph)

    start_before = graph.get_node("start").access_count
    good_before = graph.get_node("good").access_count

    controller.query("alpha")

    assert graph.get_node("start").access_count == start_before + 1
    assert graph.get_node("good").access_count == good_before + 1


def test_controller_default_config():
    cfg = ControllerConfig.default()

    assert cfg.traversal_max_hops == 3
    assert isinstance(cfg.learning, LearningConfig)
    assert isinstance(cfg.synaptogenesis, SynaptogenesisConfig)
    assert isinstance(cfg.inhibition, InhibitionConfig)
    assert isinstance(cfg.decay, DecayConfig)
