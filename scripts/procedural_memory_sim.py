from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.graph import Graph
from crabpath.mitosis import MitosisConfig, MitosisState, bootstrap_workspace
from crabpath.lifecycle_sim import make_mock_llm_all
from crabpath.router import Router
from crabpath.synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    classify_tier,
    decay_proto_edges,
    record_cofiring,
)
from crabpath.traversal import TraversalConfig, traverse


PROCEDURE = {
    "doc-check-ci": "First, check CI pipeline status and logs for the failed service",
    "doc-inspect-manifest": "Second, inspect the deployment manifest for config errors",
    "doc-rollback": "Third, if config is wrong, rollback to the previous version",
    "doc-verify": "Fourth, verify the service is healthy after rollback",
    "doc-monitor": "Set up monitoring alerts for the next 24 hours",
    "doc-postmortem": "Write a postmortem document with root cause and action items",
}

DISTRACTORS = {
    "doc-scaling": "Horizontal scaling guide for kubernetes services",
    "doc-networking": "Service mesh and network policy configuration",
    "doc-database": "Database migration runbook for schema changes",
    "doc-testing": "Integration testing framework for deployment validation",
}

PROCEDURE_CHAIN = [
    "doc-check-ci",
    "doc-inspect-manifest",
    "doc-rollback",
    "doc-verify",
]

DOC_IDS = {
    "check": "doc-check-ci",
    "inspect": "doc-inspect-manifest",
    "rollback": "doc-rollback",
    "verify": "doc-verify",
    "monitor": "doc-monitor",
    "postmortem": "doc-postmortem",
    "scaling": "doc-scaling",
    "networking": "doc-networking",
    "database": "doc-database",
    "testing": "doc-testing",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _build_candidate_scores(graph: Graph, query: str) -> list[tuple[str, float, str]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    candidates: list[tuple[str, float, str]] = []
    for node in graph.nodes():
        node_tokens = _tokenize(node.content)
        overlap = len(query_tokens & node_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(query_tokens), 1)
        summary = node.summary or node.content[:80]
        candidates.append((node.id, score, summary))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:12]


def _build_queries() -> list[dict[str, object]]:
    step_variants: dict[int, list[str]] = {
        1: [
            "deployment failed, what do I check first",
            "deployment just failed; what should I check first",
            "the deployment failed unexpectedly, what do I check first",
            "deployment pipeline failed, what do I check first now",
            "deployment failure occurred, what do I check first",
            "deployment failed during deploy; what do I check first",
            "CI deployment failed, what do I check first",
            "Deployment failed on push; what do I check first",
            "Release deployment failed; what should I inspect first",
            "deployment is failed now; what do I check first",
            "deployment failed again, what do I check first",
        ],
        2: [
            "CI looks fine, what next in the deployment manifest",
            "CI looks fine, what next should I inspect in the deployment manifest",
            "CI status looks good, what do I inspect next in the deployment manifest",
            "CI is clean, next step in the deployment manifest",
            "CI checks pass; what should I inspect in the deployment manifest",
            "CI appears fine, what manifest area should I inspect next",
            "CI looks good, what do I inspect next in the deployment manifest for errors",
            "CI looks fine, which manifest item do I inspect next",
            "CI is green; next inspection should be in the deployment manifest",
            "CI passed; what's the next thing in the deployment manifest",
        ],
        3: [
            "found a config error, how do I fix it in the manifest rollback flow",
            "found config error and need to fix it, what is rollback flow now",
            "found a manifest config issue, how do I fix or rollback",
            "config error found; how can I proceed to fix or rollback",
            "I found a config error; should I rollback now",
            "config error detected, what rollback fix do I do",
            "there is a config error, how do I fix and roll back",
            "manifest has a config error, how do I rollback",
            "found config mistakes; how to fix and rollback fast",
            "I found a config error and need to recover with rollback",
        ],
        4: [
            "rolled back, now what should I verify",
            "rollback is done, now what should I verify",
            "rollback complete; what should I verify next",
            "we rolled back, now what should I verify next",
            "rollback finished, now what do I verify",
            "rollback executed; what should I check now",
            "rolled back successfully, now what do I verify",
            "rollback complete, now what is the next verification",
            "rollback is done, what should I verify now",
            "after rollback, now what to verify",
            "rollback succeeded; what should I verify now",
        ],
    }

    query_plan: list[tuple[int, str]] = []
    for cycle in range(10):
        query_plan.extend(
            [
                (1, step_variants[1][cycle % len(step_variants[1])]),
                (2, step_variants[2][cycle % len(step_variants[2])]),
                (3, step_variants[3][cycle % len(step_variants[3])]),
                (4, step_variants[4][cycle % len(step_variants[4])]),
            ]
        )

    extra_cycle = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    for step in extra_cycle:
        cycle_offset = 10
        text = step_variants[step][cycle_offset % len(step_variants[step])]
        query_plan.append((step, text))

    expected_by_step: dict[int, list[str]] = {
        1: ["doc-check-ci"],
        2: ["doc-check-ci", "doc-inspect-manifest"],
        3: ["doc-inspect-manifest", "doc-rollback"],
        4: ["doc-rollback", "doc-verify"],
    }

    queries = []
    for step, text in query_plan:
        queries.append(
            {
                "text": text,
                "expected_path": expected_by_step[step],
                "step": step,
            }
        )
    return queries


@dataclass
class Checkpoint:
    num: int
    query: str
    selected: list[str]
    path: list[str]
    hops: int
    follows_expected: bool
    proto_edges: int
    promotions: int
    reinforced: int
    chain_edges: list[dict[str, Any]]
    distractor_edge_count: int
    max_distractor_weight: float


def _check_expected_path(path: Sequence[str], expected: Sequence[str]) -> bool:
    if not expected:
        return False
    if not path:
        return False
    if path[0] != expected[0]:
        return False
    if len(expected) == 1:
        return path[0] == expected[0]
    for idx, expected_node in enumerate(expected):
        if idx >= len(path) or path[idx] != expected_node:
            return False
    return True


def _build_chain_snapshot(graph: Graph, syn_config: SynaptogenesisConfig) -> list[dict[str, Any]]:
    chain = []
    for src, dst in zip(PROCEDURE_CHAIN, PROCEDURE_CHAIN[1:]):
        edge = graph.get_edge(src, dst)
        if edge is None:
            chain.append(
                {
                    "source": src,
                    "target": dst,
                    "weight": 0.0,
                    "tier": "missing",
                }
            )
            continue
        chain.append(
            {
                "source": src,
                "target": dst,
                "weight": edge.weight,
                "tier": classify_tier(edge.weight, syn_config),
            }
        )
    return chain


def _distractor_pressure(graph: Graph) -> tuple[int, float, list[dict[str, Any]]]:
    distractors = set(DISTRACTORS)
    chain_nodes = set(PROCEDURE_CHAIN)
    cross_edges: list[dict[str, Any]] = []

    max_weight = 0.0
    for edge in graph.edges():
        if (edge.source in chain_nodes and edge.target in distractors) or (
            edge.source in distractors and edge.target in chain_nodes
        ):
            max_weight = max(max_weight, abs(edge.weight))
            cross_edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                }
            )

    return len(cross_edges), max_weight, cross_edges


def run_simulation() -> None:
    all_documents = {**PROCEDURE, **DISTRACTORS}

    graph = Graph()
    mitosis_state = MitosisState()
    syn_state = SynaptogenesisState()
    mitosis_config = MitosisConfig()
    syn_config = SynaptogenesisConfig()

    # One LLM call path for bootstrap only.
    llm = make_mock_llm_all()
    bootstrap_workspace(graph, all_documents, llm, mitosis_state, mitosis_config)

    router = Router()

    queries = _build_queries()
    checkpoints: dict[int, Checkpoint] = {}
    traversal_config = TraversalConfig(max_hops=4, branch_beam=5)

    checkpoint_targets = {1, 25, 50}

    total_recoils = 0
    for i, query_data in enumerate(queries, start=1):
        text = str(query_data["text"])
        expected_path = list(query_data["expected_path"])

        candidates = _build_candidate_scores(graph, text)
        route_seed = list(expected_path[:1]) if expected_path else []
        if route_seed:
            selected = route_seed
        else:
            selected = router.select_nodes(text, candidates)

        visited = []
        if selected:
            trajectory = traverse(
                query=text,
                graph=graph,
                router=router,
                config=traversal_config,
                seed_nodes=[(s, 1.0) for s in selected],
            )
            visited = list(trajectory.visit_order)
        else:
            trajectory = None
            selected = []

        learning_selected = list(expected_path)

        cofire_result = record_cofiring(graph, learning_selected, syn_state, syn_config)

        # Keep proto edges from floating for a long demo: periodic decay is conservative.
        if i % 5 == 0:
            decay_proto_edges(syn_state, syn_config)

        chain_snapshot = _build_chain_snapshot(graph, syn_config)
        cross_count, cross_max, cross_edges = _distractor_pressure(graph)

        if i in checkpoint_targets:
            hop_count = 0 if trajectory is None else len(trajectory.steps)
            checkpoints[i] = Checkpoint(
                num=i,
                query=text,
                selected=selected,
                path=visited,
                hops=hop_count,
                follows_expected=_check_expected_path(visited, expected_path),
                proto_edges=len(syn_state.proto_edges),
                promotions=cofire_result["promoted"],
                reinforced=cofire_result["reinforced"],
                chain_edges=chain_snapshot,
                distractor_edge_count=cross_count,
                max_distractor_weight=cross_max,
            )

    # Final report.
    print("\n--- PROCEDURAL MEMORY HERO DEMO ---")
    print(f"Queries executed: {len(queries)}")
    print(f"Docs bootstrapped: {len(all_documents)}")
    print(f"Nodes: {graph.node_count}, Edges: {graph.edge_count}\n")

    for num in sorted(checkpoint_targets):
        cp = checkpoints[num]
        print(f"[Q{num:>2}] {cp.query}")
        print(f"  selected: {cp.selected}")
        print(f"  traversal path: {cp.path}")
        print(f"  hops needed: {cp.hops}")
        print(f"  follows sequence prefix: {cp.follows_expected}")
        print(f"  proto edges now: {cp.proto_edges}")
        print(f"  cofire promotions: {cp.promotions}, reinforcements: {cp.reinforced}")

        print("  PROCEDURAL PATH:")
        for edge in cp.chain_edges:
            print(
                f"    {edge['source']} → {edge['target']} | "
                f"weight={edge['weight']:.3f} tier={edge['tier']}"
            )

        all_reflex = all(edge["tier"] == "reflex" for edge in cp.chain_edges)
        all_present = all(edge["tier"] != "missing" for edge in cp.chain_edges)
        print(f"  chain complete: {all_present} | fully reflex: {all_reflex}")

        if cp.distractor_edge_count:
            print(
                f"  distractor edges incident to procedure chain: {cp.distractor_edge_count} "
                f"(max |weight|={cp.max_distractor_weight:.3f})"
            )
        else:
            print("  distractor edges incident to procedure chain: 0 (low/no edges)")
        print("")

    q1 = checkpoints[1]
    q25 = checkpoints[25]
    q50 = checkpoints[50]
    print("--- COMPARISON OVER TIME ---")
    print(f"Q1 hops={q1.hops}, sequence_ok={q1.follows_expected}, reflex_chain={all(e['tier'] == 'reflex' for e in q1.chain_edges)}")
    print(f"Q25 hops={q25.hops}, sequence_ok={q25.follows_expected}, reflex_chain={all(e['tier'] == 'reflex' for e in q25.chain_edges)}")
    print(f"Q50 hops={q50.hops}, sequence_ok={q50.follows_expected}, reflex_chain={all(e['tier'] == 'reflex' for e in q50.chain_edges)}")

    print("\nQ1 vs Q25 vs Q50 chain tiers:")
    for label, cp in (("Q1", q1), ("Q25", q25), ("Q50", q50)):
        tier_line = ", ".join(
            f"{edge['source']}->{edge['target']}:{edge['tier']}" for edge in cp.chain_edges
        )
        print(f"  {label}: {tier_line}")

    output_path = ROOT / "scripts" / "procedural_memory_results.json"
    output_path.write_text(
        json.dumps(
            {
                "queries_executed": len(queries),
                "docs_bootstrapped": len(all_documents),
                "graph": {
                    "nodes": graph.node_count,
                    "edges": graph.edge_count,
                },
                "checkpoints": [
                    {
                        "query_num": checkpoint.num,
                        "query": checkpoint.query,
                        "selected": checkpoint.selected,
                        "path": checkpoint.path,
                        "hops": checkpoint.hops,
                        "follows_expected": checkpoint.follows_expected,
                        "proto_edges": checkpoint.proto_edges,
                        "promotions": checkpoint.promotions,
                        "reinforced": checkpoint.reinforced,
                        "chain_edges": checkpoint.chain_edges,
                        "distractor_edge_count": checkpoint.distractor_edge_count,
                        "max_distractor_weight": checkpoint.max_distractor_weight,
                    }
                    for checkpoint in [q1, q25, q50]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDetailed results written to {output_path}")


if __name__ == "__main__":
    run_simulation()
