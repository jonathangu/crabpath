from __future__ import annotations

from pathlib import Path

from crabpath.graph import Edge, Graph, Node
from crabpath.learning import LearningConfig
from crabpath.router import Router
from crabpath.simulator import (
    EpisodeMetrics,
    ScenarioStep,
    load_scenarios,
    render_dashboard,
    run_batch,
    run_episode,
)
from crabpath.traversal import TraversalConfig


def _small_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node(id="start", content="start"))
    graph.add_node(Node(id="left", content="left"))
    graph.add_node(Node(id="right", content="right"))
    graph.add_edge(Edge(source="start", target="left", weight=0.6))
    graph.add_edge(Edge(source="start", target="right", weight=0.4))
    return graph


def test_load_scenarios() -> None:
    scenarios = load_scenarios(Path("scenarios/giraffe_test.jsonl"))
    assert len(scenarios) == 10
    assert all(isinstance(step.query, str) for step in scenarios)
    assert all(isinstance(step.expected_answer_fragments, list) for step in scenarios)


def test_run_episode_returns_metrics() -> None:
    graph = _small_graph()
    router = Router()
    config = TraversalConfig(max_hops=2)
    learning_config = LearningConfig()
    metric = run_episode(
        query="go",
        graph=graph,
        router=router,
        feedback_reward={"reward": 1.0},
        learning_config=learning_config,
        traversal_config=config,
    )

    assert isinstance(metric, EpisodeMetrics)
    assert metric.query == "go"
    assert metric.reward == 1.0
    assert metric.edges_updated >= 1
    assert metric.nodes_created == 0
    assert metric.weight_changes


def test_run_batch_completes() -> None:
    graph = _small_graph()
    router = Router()
    scenarios = [
        ScenarioStep(query="go", feedback={"reward": 1.0}, expected_answer_fragments=["left"]),
        ScenarioStep(query="stay", feedback={"reward": 0.0}, expected_answer_fragments=["right"]),
    ]
    metrics = run_batch(scenarios, graph, router, LearningConfig())

    assert len(metrics) == 2
    assert all(isinstance(metric, EpisodeMetrics) for metric in metrics)


def test_dashboard_output() -> None:
    graph = _small_graph()
    router = Router()
    metrics = run_batch(
        [
            ScenarioStep(query="go", feedback={"reward": 1.0}, expected_answer_fragments=["left"]),
            ScenarioStep(
                query="stay", feedback={"reward": 0.5}, expected_answer_fragments=["right"]
            ),
        ],
        graph=graph,
        router=router,
        learning_config=LearningConfig(),
    )
    dashboard = render_dashboard(metrics, graph)
    assert dashboard["total_episodes"] == 2
    assert dashboard["graph_nodes"] == 3
    assert dashboard["graph_edges"] >= 2
    assert "avg_reward" in dashboard
    assert "total_edges_updated" in dashboard
