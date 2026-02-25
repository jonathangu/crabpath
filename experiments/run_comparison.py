"""Experiment runner for static, RAG, and CrabPath traversal baselines."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import argparse
import json
import copy
import re

from crabpath.embeddings import EmbeddingIndex
from crabpath.graph import Graph
from crabpath.learning import LearningConfig, RewardSignal, make_learning_step
from crabpath.router import Router, RouterConfig
from crabpath.simulator import ScenarioStep, load_scenarios
from crabpath.traversal import TraversalConfig, traverse


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _build_keyword_vectors(graph: Graph) -> tuple[EmbeddingIndex, callable]:
    node_texts = [node.content for node in graph.nodes()]
    vocabulary: dict[str, int] = {}
    for text in node_texts:
        for token in _tokenize(text):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    vocab_items = sorted(vocabulary.items(), key=lambda item: item[1])

    def _embed_fn(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        dim = len(vocab_items)
        for text in texts:
            counts = [0.0] * dim
            for token in _tokenize(text):
                idx = vocabulary.get(token)
                if idx is not None:
                    counts[idx] += 1.0
            vectors.append(counts)
        return vectors

    index = EmbeddingIndex()
    index.build(graph, _embed_fn, batch_size=256)
    return index, _embed_fn


def _contains_expected(context: str, expected_answer_fragments: list[str]) -> bool:
    haystack = context.lower()
    return all(fragment.lower() in haystack for fragment in expected_answer_fragments)


def _reward_from_feedback(feedback: Any) -> float:
    if isinstance(feedback, (int, float)):
        return float(feedback)
    if isinstance(feedback, dict):
        value = feedback.get("reward", 0.0)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _run_static_arm(graph: Graph, scenario: ScenarioStep) -> dict[str, Any]:
    nodes = [node for node in graph.nodes() if node is not None]
    context = "\n\n".join(node.content for node in nodes)
    tokens_loaded = sum(len(node.content) for node in nodes)
    nodes_loaded = len(nodes)
    return {
        "query": scenario.query,
        "tokens_loaded": tokens_loaded,
        "nodes_loaded": nodes_loaded,
        "answer": context,
        "correct": _contains_expected(context, scenario.expected_answer_fragments),
    }


def _run_rag_arm(
    graph: Graph,
    scenario: ScenarioStep,
    top_k: int,
    index: EmbeddingIndex,
    embed_fn,
) -> dict[str, Any]:
    scored = index.raw_scores(scenario.query, embed_fn, top_k=top_k)
    selected_ids = [node_id for node_id, _ in scored[:top_k]]
    nodes = [graph.get_node(node_id) for node_id in selected_ids if graph.get_node(node_id)]
    context = "\n\n".join(node.content for node in nodes)
    tokens_loaded = sum(len(node.content) for node in nodes)
    return {
        "query": scenario.query,
        "tokens_loaded": tokens_loaded,
        "nodes_loaded": len(nodes),
        "selected_nodes": [node.id for node in nodes],
        "answer": context,
        "correct": _contains_expected(context, scenario.expected_answer_fragments),
    }


def _run_crabpath_arm(
    graph: Graph,
    scenario: ScenarioStep,
    top_k: int,
    myopic: bool,
    traversal_config: TraversalConfig,
    index: EmbeddingIndex,
    embed_fn: callable,
) -> tuple[dict[str, Any], float]:
    traversal_config = TraversalConfig(
        max_hops=traversal_config.max_hops,
        temperature=traversal_config.temperature,
        branch_beam=max(1, top_k // 2),
    )
    seed_scores = index.raw_scores(scenario.query, embed_fn, top_k=top_k)
    seed_nodes = [(node_id, float(score)) for node_id, score in seed_scores]
    router = Router(config=RouterConfig(fallback_behavior="heuristic"))
    trajectory = traverse(
        query=scenario.query,
        graph=graph,
        router=router,
        config=traversal_config,
        embedding_index=index,
        seed_nodes=seed_nodes,
    )

    nodes = [graph.get_node(node_id) for node_id in trajectory.context_nodes]
    loaded_nodes = [node for node in nodes if node is not None]
    context = trajectory.raw_context
    tokens_loaded = sum(len(node.content) for node in loaded_nodes)

    reward_value = _reward_from_feedback(scenario.feedback)
    reward = RewardSignal(episode_id=scenario.query, final_reward=reward_value)

    steps_to_apply = trajectory.steps
    if myopic and trajectory.steps:
        steps_to_apply = trajectory.steps[-1:]

    learning_cfg = LearningConfig()
    make_learning_step(graph=graph, trajectory_steps=steps_to_apply, reward=reward, config=learning_cfg)

    return {
        "query": scenario.query,
        "tokens_loaded": tokens_loaded,
        "nodes_loaded": len(loaded_nodes),
        "selected_nodes": [node.id for node in loaded_nodes],
        "visit_order": trajectory.visit_order,
        "answer": context,
        "correct": _contains_expected(context, scenario.expected_answer_fragments),
    }, reward_value


def run_comparison(
    graph: Graph,
    scenario_file: Path,
    top_k: int = 8,
    max_hops: int = 3,
) -> dict[str, Any]:
    scenarios = load_scenarios(scenario_file)
    traversal_config = TraversalConfig(max_hops=max_hops)

    results: dict[str, dict[str, Any]] = {}
    arms = ("static", "rag", "crabpath_corrected", "crabpath_myopic")

    for arm in arms:
        arm_graph = copy.deepcopy(graph)
        episodes = []
        correct_count = 0

        for idx, scenario in enumerate(scenarios, start=1):
            if arm == "static":
                episode = _run_static_arm(arm_graph, scenario)
            elif arm == "rag":
                rag_index, rag_embed_fn = _build_keyword_vectors(arm_graph)
                episode = _run_rag_arm(
                    arm_graph,
                    scenario,
                    top_k=top_k,
                    index=rag_index,
                    embed_fn=rag_embed_fn,
                )
            elif arm == "crabpath_corrected":
                rag_index, rag_embed_fn = _build_keyword_vectors(arm_graph)
                episode, _ = _run_crabpath_arm(
                    arm_graph,
                    scenario,
                    top_k=top_k,
                    myopic=False,
                    traversal_config=traversal_config,
                    index=rag_index,
                    embed_fn=rag_embed_fn,
                )
            else:
                rag_index, rag_embed_fn = _build_keyword_vectors(arm_graph)
                episode, _ = _run_crabpath_arm(
                    arm_graph,
                    scenario,
                    top_k=top_k,
                    myopic=True,
                    traversal_config=traversal_config,
                    index=rag_index,
                    embed_fn=rag_embed_fn,
                )

            if episode["correct"]:
                correct_count += 1
            episode["cumulative_correct"] = correct_count / idx
            episodes.append(episode)

        avg_tokens = 0.0
        if episodes:
            avg_tokens = sum(float(ep["tokens_loaded"]) for ep in episodes) / len(episodes)
        accuracy = correct_count / len(episodes) if episodes else 0.0

        results[arm] = {
            "episodes": episodes,
            "avg_tokens": avg_tokens,
            "accuracy": accuracy,
        }

    return {
        "experiment": scenario_file.stem,
        "arms": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run comparison experiment over a scenario JSONL file.")
    parser.add_argument("--graph", required=True, help="Path to graph JSON file.")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSONL file.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file path.",
    )
    args = parser.parse_args(argv)

    graph = Graph.load(args.graph)
    result = run_comparison(
        graph=graph,
        scenario_file=Path(args.scenario),
        top_k=args.top_k,
        max_hops=args.max_hops,
    )

    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
