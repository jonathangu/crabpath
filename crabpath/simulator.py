from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

from .decay import apply_decay
from .graph import Graph
from .learning import LearningConfig, RewardSignal, make_learning_step
from .router import Router
from .traversal import TraversalConfig, traverse


@dataclass
class SimulatorConfig:
    max_hops: int = 3
    decay_interval: int = 5


@dataclass
class ScenarioStep:
    query: str
    feedback: dict
    expected_answer_fragments: list[str]


@dataclass
class EpisodeMetrics:
    query: str
    reward: float
    edges_updated: int
    nodes_created: int
    weight_changes: dict


def load_scenarios(path: str | Path) -> list[ScenarioStep]:
    scenarios: list[ScenarioStep] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue

            scenarios.append(
                ScenarioStep(
                    query=str(raw.get("query", "")),
                    feedback=dict(raw.get("feedback") or {}),
                    expected_answer_fragments=list(raw.get("expected_answer_fragments") or []),
                )
            )
    return scenarios


def _extract_reward(feedback_reward: Any) -> float:
    if isinstance(feedback_reward, (int, float)):
        return float(feedback_reward)
    if isinstance(feedback_reward, dict):
        reward = feedback_reward.get("reward", 0.0)
        if isinstance(reward, (int, float)):
            return float(reward)
    return 0.0


def run_episode(
    query: str,
    graph: Graph,
    router: Router,
    feedback_reward: Any,
    learning_config: LearningConfig,
    traversal_config: TraversalConfig,
) -> EpisodeMetrics:
    seed_nodes = [node.id for node in graph.nodes()]
    trajectory = traverse(
        query=query,
        graph=graph,
        router=router,
        seed_nodes=seed_nodes if seed_nodes else None,
        config=traversal_config,
    )
    reward_value = _extract_reward(feedback_reward)
    reward = RewardSignal(episode_id=query, final_reward=reward_value)
    nodes_before = len(graph.nodes())
    result = make_learning_step(graph, trajectory.steps, reward, learning_config)
    nodes_after = len(graph.nodes())

    weight_changes = {
        f"{update.source}->{update.target}": update.delta for update in result.updates
    }
    return EpisodeMetrics(
        query=query,
        reward=float(reward_value),
        edges_updated=len(result.updates),
        nodes_created=max(0, nodes_after - nodes_before),
        weight_changes=weight_changes,
    )


def run_batch(
    scenarios: Sequence[ScenarioStep],
    graph: Graph,
    router: Router,
    learning_config: LearningConfig,
) -> list[EpisodeMetrics]:
    cfg = SimulatorConfig()
    traversal_cfg = TraversalConfig(max_hops=cfg.max_hops)
    results: list[EpisodeMetrics] = []
    for index, step in enumerate(scenarios, start=1):
        results.append(
            run_episode(
                query=step.query,
                graph=graph,
                router=router,
                feedback_reward=step.feedback,
                learning_config=learning_config,
                traversal_config=traversal_cfg,
            )
        )

        if index % cfg.decay_interval == 0:
            apply_decay(graph, turns_elapsed=cfg.decay_interval)

    return results


def render_dashboard(metrics_list: list[EpisodeMetrics], graph: Graph) -> dict:
    total_episodes = len(metrics_list)
    total_reward = sum(metric.reward for metric in metrics_list)
    avg_reward = total_reward / total_episodes if total_episodes else 0.0
    return {
        "total_episodes": total_episodes,
        "avg_reward": avg_reward,
        "total_edges_updated": sum(metric.edges_updated for metric in metrics_list),
        "graph_nodes": len(graph.nodes()),
        "graph_edges": len(graph.edges()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crabpath-sim")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    graph = Graph.load(args.graph)
    scenarios = load_scenarios(args.scenarios)
    router = Router()
    learning_config = LearningConfig()
    metrics = run_batch(scenarios, graph, router, learning_config)
    dashboard = render_dashboard(metrics, graph)

    payload = {
        "episodes": [asdict(metric) for metric in metrics],
        "dashboard": dashboard,
    }

    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    else:
        print(json.dumps(payload, indent=2))

    graph.save(args.graph)
    return 0
