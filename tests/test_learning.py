from __future__ import annotations

import math

import pytest

from crabpath import Edge, Graph, Node
from crabpath.learning import (
    LearningConfig,
    RewardSignal,
    apply_weight_updates,
    make_learning_step,
    policy_gradient_update,
    weight_delta,
)


def _make_trajectory() -> list[dict]:
    trajectory = [
        {
            "from_node": "giraffe-codeword",
            "to_node": "codewords",
            "edge_weight": 1.0,
            "candidates": {
                "codewords": 1.0,
                "elephant-codeword": 0.0,
            },
        },
        {
            "from_node": "codewords",
            "to_node": "elephant-codeword",
            "edge_weight": 0.0,
            "candidates": {
                "elephant-codeword": 0.0,
                "stop": 0.0,
            },
        },
    ]
    return trajectory


def _to_delta_map(deltas):
    return {f"{source}->{target}": delta for source, target, delta in deltas}


def _myopic_deltas(trajectory_steps, reward, config):
    last = trajectory_steps[-1]
    candidates = last["candidates"]
    probs = list(candidates.values())
    exp_scores = [math.exp(v / config.temperature) for v in probs]
    total = sum(exp_scores)
    p = exp_scores[1] / total
    grad = 1.0 - p
    return [("codewords", "elephant-codeword", config.learning_rate * reward.final_reward * grad)]


def test_gu_corrected_credits_all_hops():
    trajectory = _make_trajectory()
    cfg = LearningConfig(learning_rate=0.1, discount=1.0, temperature=1.0)
    reward = RewardSignal(episode_id="giraffe-1", final_reward=1.0)

    _, advantages = policy_gradient_update(trajectory, reward, cfg, baseline=0.0)
    deltas = weight_delta(trajectory, advantages, cfg)
    delta_map = _to_delta_map(deltas)

    p_gc = math.exp(1.0 / cfg.temperature) / (
        math.exp(1.0 / cfg.temperature) + math.exp(0.0 / cfg.temperature)
    )
    assert delta_map["giraffe-codeword->codewords"] == pytest.approx((1 - p_gc) * 0.1)
    assert delta_map["giraffe-codeword->elephant-codeword"] == pytest.approx(-(1 - p_gc) * 0.1)
    assert delta_map["codewords->elephant-codeword"] == pytest.approx((1 - 0.5) * 0.1)
    assert delta_map["codewords->stop"] == pytest.approx(-0.5 * 0.1)


def test_myopic_only_credits_last_hop():
    trajectory = _make_trajectory()
    cfg = LearningConfig(learning_rate=0.1, discount=1.0, temperature=1.0)
    reward = RewardSignal(episode_id="giraffe-2", final_reward=1.0)

    _, advantages = policy_gradient_update(trajectory, reward, cfg, baseline=0.0)
    corrected = _to_delta_map(weight_delta(trajectory, advantages, cfg))
    myopic = _to_delta_map(_myopic_deltas(trajectory, reward, cfg))

    assert "giraffe-codeword->codewords" in corrected
    assert corrected["giraffe-codeword->codewords"] > 0.0
    assert "giraffe-codeword->codewords" not in myopic
    assert myopic["codewords->elephant-codeword"] == corrected["codewords->elephant-codeword"]


def test_negative_feedback_punishes_path():
    trajectory = _make_trajectory()
    cfg = LearningConfig(learning_rate=0.1, discount=1.0, temperature=1.0)
    reward = RewardSignal(episode_id="giraffe-3", final_reward=-1.0)

    _, advantages = policy_gradient_update(trajectory, reward, cfg, baseline=0.0)
    deltas = weight_delta(trajectory, advantages, cfg)
    delta_map = _to_delta_map(deltas)

    assert delta_map["giraffe-codeword->codewords"] < 0.0
    assert delta_map["giraffe-codeword->elephant-codeword"] > 0.0
    assert delta_map["codewords->elephant-codeword"] < 0.0
    assert delta_map["codewords->stop"] > 0.0


def test_weight_clamping():
    trajectory = [
        {
            "from_node": "root",
            "to_node": "left",
            "edge_weight": 1.0,
            "candidates": {"left": 1.0, "right": 1.0},
        }
    ]
    graph = Graph()
    graph.add_node(Node(id="root", content="Root"))
    graph.add_node(Node(id="left", content="Left"))
    graph.add_node(Node(id="right", content="Right"))
    graph.add_edge(Edge(source="root", target="left", weight=1.0))
    graph.add_edge(Edge(source="root", target="right", weight=1.0))

    cfg = LearningConfig(learning_rate=100.0, discount=1.0, clip_min=-5, clip_max=5)
    reward = RewardSignal(episode_id="clamp-1", final_reward=10.0)
    _, advantages = policy_gradient_update(trajectory, reward, cfg, baseline=0.0)
    deltas = weight_delta(trajectory, advantages, cfg)
    delta_map = _to_delta_map(deltas)

    assert delta_map["root->left"] == pytest.approx(5.0)
    assert delta_map["root->right"] == pytest.approx(-5.0)

    updates = apply_weight_updates(graph, deltas, cfg)
    updated = {u.target: u.new_weight for u in updates}
    assert updated["left"] == 5.0


def test_baseline_reduces_variance():
    graph = Graph()
    graph.add_node(Node(id="root", content="Root"))
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="b", content="B"))
    graph.add_edge(Edge(source="root", target="a", weight=0.0))
    graph.add_edge(Edge(source="root", target="b", weight=0.0))

    trajectory = [
        {
            "from_node": "root",
            "to_node": "a",
            "edge_weight": 0.0,
            "candidates": {"a": 0.0, "b": 0.0},
        }
    ]
    cfg = LearningConfig(learning_rate=0.1, discount=1.0, baseline_decay=0.5)
    reward = RewardSignal(episode_id="baseline-demo", final_reward=2.0)

    first = make_learning_step(graph, trajectory, reward, cfg)
    second = make_learning_step(graph, trajectory, reward, cfg)

    first_delta = next(u for u in first.updates if u.target == "a").delta
    second_delta = next(u for u in second.updates if u.target == "a").delta
    assert abs(second_delta) < abs(first_delta)
    assert second.baseline < first.avg_reward
    assert second.baseline == pytest.approx(1.5)


def test_empty_trajectory_no_baseline_drift():
    graph = Graph()
    graph.add_node(Node(id="a", content="A"))
    cfg = LearningConfig(learning_rate=0.1, discount=1.0)

    first = make_learning_step(graph, [], RewardSignal(episode_id="empty", final_reward=2.0), cfg)
    second = make_learning_step(graph, [], RewardSignal(episode_id="empty", final_reward=4.0), cfg)

    assert first.updates == []
    assert first.baseline == 0.0
    assert second.baseline == 0.0


def test_empty_trajectory_returns_zero_updates():
    graph = Graph()
    graph.add_node(Node(id="a", content="A"))
    cfg = LearningConfig(learning_rate=0.1, discount=1.0)

    result = make_learning_step(graph, [], RewardSignal(episode_id="empty2", final_reward=1.0), cfg)

    assert result.updates == []
    assert result.baseline == 0.0
    assert result.avg_reward == 1.0


def test_malformed_candidates_in_trajectory():
    graph = Graph()
    graph.add_node(Node(id="root", content="Root"))
    graph.add_node(Node(id="a", content="A"))
    cfg = LearningConfig(learning_rate=0.1, discount=1.0)
    trajectory = [
        {
            "from_node": "root",
            "to_node": "a",
            "edge_weight": 1.0,
            "candidates": "oops",
        }
    ]

    result = make_learning_step(
        graph, trajectory, RewardSignal(episode_id="bad", final_reward=1.0), cfg
    )

    assert result.updates == []


def test_temperature_sharpens_gradient():
    trajectory = [
        {
            "from_node": "q",
            "to_node": "d",
            "edge_weight": 0.0,
            "candidates": {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.0},
        }
    ]

    reward = RewardSignal(episode_id="temp-test", final_reward=1.0)
    warm = LearningConfig(learning_rate=0.1, discount=1.0, temperature=1.0)
    cool = LearningConfig(learning_rate=0.1, discount=1.0, temperature=0.5)

    _, warm_adv = policy_gradient_update(trajectory, reward, warm, baseline=0.0)
    _, cool_adv = policy_gradient_update(trajectory, reward, cool, baseline=0.0)

    warm_delta = _to_delta_map(weight_delta(trajectory, warm_adv, warm))
    cool_delta = _to_delta_map(weight_delta(trajectory, cool_adv, cool))

    assert abs(cool_delta["q->d"]) > abs(warm_delta["q->d"])


def test_query_family_groups_similar_queries():
    graph = Graph()
    graph.add_node(Node(id="root", content="Root"))
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="b", content="B"))
    graph.add_edge(Edge(source="root", target="a", weight=0.0))
    graph.add_edge(Edge(source="root", target="b", weight=0.0))

    trajectory = [
        {
            "from_node": "root",
            "to_node": "a",
            "edge_weight": 0.0,
            "candidates": {"a": 0.0, "b": 0.0},
        }
    ]
    cfg = LearningConfig(learning_rate=0.1, discount=1.0, baseline_decay=0.5, temperature=1.0)
    first = make_learning_step(
        graph,
        trajectory,
        RewardSignal(episode_id="blue machine optimization query", final_reward=1.0),
        cfg,
    )
    second = make_learning_step(
        graph,
        trajectory,
        RewardSignal(episode_id="blue machine optimization strategy", final_reward=1.0),
        cfg,
    )

    first_delta = next(u for u in first.updates if u.target == "a").delta
    second_delta = next(u for u in second.updates if u.target == "a").delta
    assert abs(second_delta) < abs(first_delta)


def test_baseline_state_is_isolated_per_config():
    graph = Graph()
    graph.add_node(Node(id="root", content="Root"))
    graph.add_node(Node(id="a", content="A"))
    graph.add_node(Node(id="b", content="B"))
    graph.add_edge(Edge(source="root", target="a", weight=0.0))
    graph.add_edge(Edge(source="root", target="b", weight=0.0))

    trajectory = [
        {
            "from_node": "root",
            "to_node": "a",
            "edge_weight": 0.0,
            "candidates": {"a": 0.0, "b": 0.0},
        }
    ]
    cfg_one = LearningConfig(learning_rate=0.1, discount=1.0, baseline_decay=0.95)
    cfg_two = LearningConfig(learning_rate=0.1, discount=1.0, baseline_decay=0.95)

    make_learning_step(
        graph,
        trajectory,
        RewardSignal(episode_id="shared-demo", final_reward=2.0),
        cfg_one,
    )
    second = make_learning_step(
        graph,
        trajectory,
        RewardSignal(episode_id="shared-demo", final_reward=0.0),
        cfg_two,
    )

    assert second.baseline == 0.0
