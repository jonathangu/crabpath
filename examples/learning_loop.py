"""Runnable end-to-end RL learning-loop example for CrabPath.

This demo:
1) bootstraps a tiny 3-file workspace in-memory,
2) runs 20 queries,
3) scores retrieval with `score_retrieval`,
4) converts scores into a scalar reward for `make_learning_step`,
5) prints before/after edge weights and RL updates.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from crabpath import Edge, Graph, Node
from crabpath.feedback import score_retrieval
from crabpath.learning import LearningConfig, RewardSignal, make_learning_step
from crabpath.router import Router
from crabpath.traversal import TraversalConfig, traverse

WORKSPACE_FILES: dict[str, str] = {
    "deploy_runbook.md": (
        "Deployment runbook: when a deploy fails, check the diff, read service logs, "
        "verify database migration state, then decide between rollback or restart."
    ),
    "incident_playbook.md": (
        "Incident playbook: if users are impacted, create a postmortem note, notify "
        "on-call, and stop risky changes until a rollback window is confirmed."
    ),
    "safety_guidelines.md": (
        "Safety guideline: never claim a fix without validation, never run destructive "
        "commands on production, and always confirm monitoring signals before completion."
    ),
}

QUERIES: list[str] = [
    "deploy failed after merge, what should I do first",
    "where can I check deployment diff changes",
    "how to troubleshoot service startup errors",
    "is rollback required before further testing",
    "tell me safety checks before declaring fixed",
    "service keeps crashing, compare with incident playbook",
    "create a postmortem and notify on call",
    "which commands can I run to restart safely",
    "what is the policy for destructive production changes",
    "did users report errors after migration",
    "what happened before rollback window closed",
    "how do I avoid false confirmation of a fix",
    "incident guidance for a bad deploy",
    "is deployment diff clean and safe",
    "postmortem checklist for recurring failure",
    "restart service then check logs",
    "validation requirements before closing incident",
    "can I skip monitoring confirmation",
    "verify database migration status after deploy",
    "do we need to notify stakeholders on call",
]


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", value.lower()) if token}


def _bootstrap_graph() -> Graph:
    graph = Graph()
    for node_id, content in WORKSPACE_FILES.items():
        graph.add_node(Node(id=node_id, content=content))

    edges = [
        ("deploy_runbook.md", "incident_playbook.md", 1.0),
        ("deploy_runbook.md", "safety_guidelines.md", 0.4),
        ("incident_playbook.md", "safety_guidelines.md", 0.95),
        ("incident_playbook.md", "deploy_runbook.md", 0.5),
        ("safety_guidelines.md", "deploy_runbook.md", 0.9),
        ("safety_guidelines.md", "incident_playbook.md", 0.3),
    ]
    for source, target, weight in edges:
        graph.add_edge(Edge(source=source, target=target, weight=weight))
    return graph


def _seed_nodes(query: str, graph: Graph, top_k: int = 2) -> list[tuple[str, float]]:
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    scored: list[tuple[str, float]] = []
    for node in graph.nodes():
        haystack = _tokenize(f"{node.id} {node.content}")
        overlap = len(query_terms.intersection(haystack))
        if overlap == 0:
            continue
        score = overlap / max(1, len(query_terms))
        scored.append((node.id, score))

    if not scored:
        scored = [(node.id, 0.1) for node in graph.nodes()]

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _score_from_llm(prompt: str, system: str) -> str:
    """
    Deterministic mock scorer that emits strict JSON.

    This keeps the example runnable without external API keys while still
    producing explicit RL labels in {-1, -0.5, 0, 0.5, 1}.
    """
    del system

    query_match = re.search(r"query:\s*(.*)\n", prompt)
    query = (query_match.group(1) if query_match else "").strip().lower()
    query_terms = _tokenize(query or "")

    node_matches = re.findall(r'- "([^"]+)":\s*"(.*?)"\s*$', prompt, flags=re.MULTILINE)
    if not node_matches:
        return json.dumps([])

    scores: list[dict[str, float | str]] = []
    for node_id, snippet in node_matches:
        snippet_terms = _tokenize(snippet)
        overlap = len(query_terms.intersection(snippet_terms))
        if not query_terms:
            overlap_score = 0.0
        else:
            overlap_ratio = overlap / len(query_terms)
            if overlap_ratio >= 0.75:
                overlap_score = 1.0
            elif overlap_ratio >= 0.5:
                overlap_score = 0.5
            elif overlap_ratio >= 0.2:
                overlap_score = 0.0
            elif overlap > 0:
                overlap_score = -0.5
            else:
                overlap_score = -1.0

        scores.append({"node_id": node_id, "score": overlap_score})

    return json.dumps(scores)


def _edge_snapshot(graph: Graph) -> dict[str, float]:
    return {f"{edge.source}->{edge.target}": edge.weight for edge in graph.edges()}


def _print_updates(before: dict[str, float], after: dict[str, float]) -> None:
    changed = False
    for key in sorted(set(before) | set(after)):
        old = before.get(key, 0.0)
        new = after.get(key, 0.0)
        if round(old - new, 9) == 0.0:
            continue
        print(f"  {key}: {old:.4f} -> {new:.4f} ({new - old:+.4f})")
        changed = True
    if not changed:
        print("  (no edge weight changes)")


def main() -> None:
    graph = _bootstrap_graph()
    router = Router()
    config = LearningConfig(learning_rate=0.08, discount=1.0)
    trajectory_cfg = TraversalConfig(max_hops=3, temperature=0.2, branch_beam=3)

    for index, query in enumerate(QUERIES, start=1):
        print(f"\n=== Query {index}/20 ===")
        print(f"query: {query}")

        seeds = _seed_nodes(query, graph, top_k=2)
        print(f"seed nodes: {seeds}")

        trajectory = traverse(
            query=query,
            graph=graph,
            router=router,
            config=trajectory_cfg,
            seed_nodes=seeds,
        )

        retrieved_nodes = [
            (node_id, graph.get_node(node_id).content) for node_id in trajectory.visit_order
            if graph.get_node(node_id) is not None
        ]
        actual_response = "\n\n".join(content for _, content in retrieved_nodes) or query

        print("trajectory nodes:", trajectory.visit_order)
        print("retrieved for scoring:", [node_id for node_id, _ in retrieved_nodes])

        scored_nodes = score_retrieval(
            query=query,
            retrieved_nodes=retrieved_nodes,
            actual_response=actual_response,
            llm_call=_score_from_llm,
        )
        print("scores:", scored_nodes)

        reward_value = sum(score for _, score in scored_nodes) / len(scored_nodes or [(None, 0.0)])
        reward = RewardSignal(episode_id="learning-loop", final_reward=reward_value)
        print(f"reward: {reward_value:.3f}")

        before_weights = _edge_snapshot(graph)
        result = make_learning_step(graph, trajectory.steps, reward, config)
        after_weights = _edge_snapshot(graph)

        print(f"baseline: {result.baseline:.4f}")
        print(f"avg reward: {result.avg_reward:.4f}")
        print("edge-weight updates:")
        _print_updates(before_weights, after_weights)

        if result.updates:
            print("rl updates:")
            for update in result.updates:
                print(
                    "  "
                    f"{update.source}->{update.target} "
                    f"delta={update.delta:+.4f} "
                    f"new={update.new_weight:.4f} "
                    f"rationale={update.rationale}"
                )
        else:
            print("rl updates: (no selected trajectory steps)")

    print("\nFinal graph state:")
    for source in sorted({edge.source for edge in graph.edges()}):
        for _, edge in graph.outgoing(source):
            print(f"{edge.source}->{edge.target}: {edge.weight:.4f}")


if __name__ == "__main__":
    main()
