#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crabpath.graph import Graph
from crabpath.learning import LearningConfig, RewardSignal, make_learning_step
from crabpath.router import Router, RouterConfig, RouterDecision
from crabpath.traversal import TraversalConfig, traverse


def _classify_tier(weight: float) -> str:
    if weight > 0.8:
        return "reflex"
    if weight >= 0.3:
        return "habitual"
    return "dormant"


class DeployRouter(Router):
    def _coerce_for_decision(
        self,
        candidates: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        return [(item[0], float(item[1])) for item in candidates if isinstance(item[0], str)]

    def _build_decision(
        self,
        candidates: list[tuple[str, float]],
        chosen_target: str,
        tier: str,
    ) -> RouterDecision:
        normalized = self._coerce_for_decision(candidates)
        if not normalized:
            raise RuntimeError("Cannot make routing decision with empty candidate list")

        chosen = next(
            (node_id for node_id, _ in normalized if node_id == chosen_target),
            None,
        )
        if chosen is None:
            chosen = sorted(normalized, key=lambda item: (item[1], item[0]), reverse=True)[0][0]

        alternatives = [(node_id, weight) for node_id, weight in normalized if node_id != chosen]
        _, confidence_score = next((node_id, weight) for node_id, weight in normalized if node_id == chosen)
        confidence = (confidence_score + 1.0) / 2.0
        if confidence < 0:
            confidence = 0.0
        elif confidence > 1:
            confidence = 1.0

        return RouterDecision(
            chosen_target=chosen,
            rationale="heuristic fallback",
            confidence=confidence,
            tier=tier,
            alternatives=alternatives,
            raw={
                "target": chosen,
                "confidence": confidence,
                "rationale": "heuristic fallback",
            },
        )

    def decide_next(
        self,
        query: str,
        current_node_id: str,
        candidate_nodes: list[tuple[str, float]],
        context: dict,
        tier: str,
        previous_reasoning: str | None = None,
    ) -> RouterDecision:
        del current_node_id, context, previous_reasoning
        text = str(query).lower()
        candidates = self._coerce_for_decision(candidate_nodes)

        # Expedite queries are still allowed to pick the dangerous shortcut.
        if ("out now" in text or "asap" in text) and any(node_id == "skip_tests" for node_id, _ in candidates):
            return self._build_decision(candidates, "skip_tests", tier)

        return self.fallback(candidates, tier)

    def fallback(self, candidates: list[tuple[str, float]], tier: str) -> RouterDecision:  # type: ignore[override]
        normalized = self._coerce_for_decision(candidates)
        if not normalized:
            raise RuntimeError("Cannot fallback with empty candidate list")

        # Favor explicit testing in non-urgent contexts, so check_tests starts as
        # the non-dangerous default when ties exist.
        ranked = sorted(
            normalized,
            key=lambda item: (item[1], item[0] == "check_tests"),
            reverse=True,
        )
        return self._build_decision(ranked, ranked[0][0], tier)


def _load_scenarios(path: Path) -> list[dict]:
    scenarios: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenarios.append(json.loads(line))
    return scenarios


def build_graph(graph_path: Path) -> Graph:
    from experiments import build_deploy_pipeline

    graph = build_deploy_pipeline.build_graph()
    graph.save(str(graph_path))
    return graph


def _urgent_skip_query(query: str) -> bool:
    text = query.lower()
    return ("out now" in text) or ("asap" in text)


def run_episode(
    query: str,
    graph: Graph,
    router: Router,
    learning_cfg: LearningConfig,
    traversal_cfg: TraversalConfig,
    reward: float,
) -> tuple[list[str], float]:
    trajectory = traverse(
        query=query,
        graph=graph,
        router=router,
        config=traversal_cfg,
        seed_nodes=["deploy_root"],
    )
    path = [step.to_node for step in trajectory.steps]
    reward_signal = RewardSignal(episode_id=query, final_reward=reward)
    make_learning_step(graph=graph, trajectory_steps=trajectory.steps, reward=reward_signal, config=learning_cfg)
    return path, reward


def print_weights(graph: Graph, source: str, target_a: str, target_b: str) -> None:
    edge_a = graph.get_edge(source, target_a)
    edge_b = graph.get_edge(source, target_b)
    a_weight = edge_a.weight if edge_a is not None else 0.0
    b_weight = edge_b.weight if edge_b is not None else 0.0
    print(f"  w({source}->{target_a})={a_weight:.3f}")
    print(f"  w({source}->{target_b})={b_weight:.3f}")
    print(f"  skip_tests tier: {_classify_tier(b_weight)}")


def run_simulation(
    graph_path: str = "experiments/deploy_pipeline_graph.json",
    scenario_path: str = "scenarios/deploy_pipeline.jsonl",
) -> None:
    graph_file = Path(graph_path)
    scenario_file = Path(scenario_path)

    graph = build_graph(graph_file)
    scenarios = _load_scenarios(scenario_file)
    router = DeployRouter(config=RouterConfig(fallback_behavior="heuristic"))
    learning_cfg = LearningConfig()
    traversal_cfg = TraversalConfig(max_hops=4, temperature=0.2, branch_beam=3)

    print(f"Loaded {len(scenarios)} episodes from {scenario_file}")
    print(f"Graph saved to {graph_file}\n")

    for i, item in enumerate(scenarios, start=1):
        reward = item.get("feedback", {}).get("reward", 0.0)
        query = str(item.get("query", ""))
        path, actual_reward = run_episode(
            query=query,
            graph=graph,
            router=router,
            learning_cfg=learning_cfg,
            traversal_cfg=traversal_cfg,
            reward=reward,
        )
        print(f"Episode {i}")
        print(f"  Query: {query}")
        print(f"  Chosen path: {path}")
        print(f"  Reward: {actual_reward:+.1f}")
        print_weights(graph, "deploy_root", "check_tests", "skip_tests")
        print("")

    check_edge = graph.get_edge("deploy_root", "check_tests")
    skip_edge = graph.get_edge("deploy_root", "skip_tests")
    check_weight = check_edge.weight if check_edge is not None else 0.0
    skip_weight = skip_edge.weight if skip_edge is not None else 0.0
    safe_path = "YES" if check_weight > 0.8 else "NO"
    dangerous = "YES" if skip_weight < 0.3 else "NO"

    print("SAFE PATH LEARNED:", safe_path)
    print("DANGEROUS PATH SUPPRESSED:", dangerous)
    print("FULL EDGE WEIGHT TABLE:")
    for edge in sorted(graph.edges(), key=lambda e: (e.source, e.target)):
        print(f"  {edge.source}->{edge.target}: {edge.weight:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deploy pipeline learning simulation.")
    parser.add_argument(
        "--graph",
        default="experiments/deploy_pipeline_graph.json",
        help="Path to write deploy pipeline graph JSON",
    )
    parser.add_argument(
        "--scenarios",
        default="scenarios/deploy_pipeline.jsonl",
        help="Path to deploy pipeline scenario JSONL file",
    )
    args = parser.parse_args()

    run_simulation(graph_path=args.graph, scenario_path=args.scenarios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
