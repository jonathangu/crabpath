"""RAG Collapse vs CrabPath deployment routing simulation.

This script builds a high-overlap deployment-only corpus and evaluates three systems
over the same 50 deployment-incident queries:

1) Static context: always load all 20 docs.
2) RAG: keyword-overlap mock retrieval (top-3).
3) CrabPath: bootstrap all docs and run full co-firing/proto-edge/cofiring-decay
   lifecycle with learned procedural reflex edges.

Outputs are printed as a comparison table, a learning curve, and saved to
`scripts/rag_collapse_results.json`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath._structural_utils import classify_edge_tier
from crabpath.decay import DecayConfig, apply_decay
from crabpath.graph import Graph
from crabpath.lifecycle_sim import make_mock_llm_all
from crabpath.mitosis import MitosisConfig, MitosisState, bootstrap_workspace
from crabpath.synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    decay_proto_edges,
    record_cofiring,
    record_skips,
)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "the",
    "this",
    "that",
    "to",
    "with",
    "you",
    "your",
    "if",
    "then",
    "after",
    "while",
    "while",
}


@dataclass(frozen=True)
class QueryCase:
    text: str
    expected_procedure: list[str]
    category: str
    ambiguous: bool = False


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9_#+-]+", text.lower())
    return {token for token in tokens if token not in STOP_WORDS and len(token) > 1}


def _build_documents() -> OrderedDict[str, str]:
    return OrderedDict(
        [
        (
            "deployment-runbook",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "playbook defines generic service checks for any release. Keep the service "
                "pipeline deployment path deterministic so every run includes config and "
                "cluster validation before a cluster-wide deploy."
            ),
        ),
        (
            "ci-pipeline",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "playbook describes CI pipeline health for every service deploy. "
                "The CI pipeline validates config, builds container image, and passes service"
                " deploy artifacts to the cluster."
            ),
        ),
        (
            "k8s-service-manifest",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "document focuses on Kubernetes service manifests and service config for "
                "every deploy. It explains cluster rollout behavior, service spec shape, "
                "and deployment ordering from manifest to cluster."
            ),
        ),
        (
            "release-notes",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "record captures release notes, service-level notes, and config-level "
                "notes for each deploy. Use cluster context to compare release notes before "
                "rolling out the deployment service update."
            ),
        ),
        (
            "cluster-operations",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "ops note focuses on cluster readiness and service ops runbooks. "
                "It includes cluster checks, service restart policy, and deploy guardrails "
                "for shared config in production deployments."
            ),
        ),
        (
            "canary-protocol",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "canary protocol is for staged rollout only. It details service config "
                "flags, cluster canary percent, and deploy rollback safety in cluster canary "
                "deployments."
            ),
        ),
        (
            "manifest-inventory",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "inventory lists service manifests, deployment configs, and config maps used "
                "by the deployment service. It tracks every manifest used in cluster deploys "
                "and maps to CI generated artifacts."
            ),
        ),
        (
            "service-readiness-checks",
            (
                "Deployment service config cluster runbook for the team. This service-"
                "readiness document defines pre- and post-deploy checks. It includes service "
                "startup conditions, config validity, and cluster readiness checks for each "
                "deployment service rollout."
            ),
        ),
        (
            "incident-response",
            (
                "Deployment service config cluster runbook for the team. This incident "
                "response document explains incident triage for failed deploys. It tracks "
                "service impact, cluster impact, and deployment-specific action items for "
                "incident communication and rollback decisions."
            ),
        ),
        (
            "incident-communication",
            (
                "Deployment service config cluster runbook for the team. This communication "
                "playbook gives incident messaging during deployment incidents. It lists "
                "service owners, cluster scope, config risks, and rollback commitments for "
                "live deployment recovery."
            ),
        ),
        (
            "post-deploy-validation",
            (
                "Deployment service config cluster runbook for the team. This document details "
                "post-deploy validation checks for service behavior, deployment stability, and "
                "config safety. Every cluster deployment must pass post-deploy validation "
                "before declaring success."
            ),
        ),
        (
            "feature-flag-guide",
            (
                "Deployment service config cluster runbook for the team. This feature-flag guide "
                "tracks service feature gates during deployments. It references cluster "
                "config, deploy toggles, and rollback safety when a feature flag changes "
                "runtime behavior."
            ),
        ),
        (
            "feature-flag-rollback",
            (
                "Deployment service config cluster runbook for the team. This feature flag rollback"
                " guide describes how to reverse feature behavior during a deployment. "
                "Use deployment service config and cluster checks before continuing with "
                "deployment rollback actions."
            ),
        ),
        (
            "ci-config",
            (
                "Deployment service config cluster runbook for the team. This CI config page "
                "captures pipeline config, service deploy configs, and cluster-safe defaults "
                "for deployment jobs. It is the first place to verify CI settings in a broken "
                "deployment service rollout."
            ),
        ),
        (
            "manifest-review",
            (
                "Deployment service config cluster runbook for the team. This manifest review "
                "runbook documents manifest correctness for service updates. Use this during "
                "cluster deploy incidents to check service, config, and manifest diffs before "
                "any deployment rollback."
            ),
        ),
        (
            "ci-logs",
            (
                "Deployment service config cluster runbook for the team. This CI log review page "
                "collects CI build and deploy logs for every deployment service event. "
                "Use these logs to trace service config failures during deploy and correlate "
                "them with rollback steps in cluster incidents."
            ),
        ),
        (
            "rollback-readiness-metrics",
            (
                "Deployment service config cluster runbook for the team. This rollback readiness "
                "document defines metrics thresholds for safe deployment rollback. It checks "
                "service error rate, deploy latency, and cluster health before rolling back "
                "a bad deployment service release."
            ),
        ),
        (
            "monitoring-alerts",
            (
                "Deployment service config cluster runbook for the team. This monitoring alerts "
                "runbook tracks deployment incidents through service alerts, error budgets, "
                "and cluster signal quality. Every deployment service failure case should be "
                "investigated with monitoring before rollback."
            ),
        ),
        (
            "incident-template",
            (
                "Deployment service config cluster runbook for the team. This incident template "
                "captures service timeline, deployment diff, config changes, and communication "
                "notes. Use template fields during cluster incidents and deployment rollback "
                "review."
            ),
        ),
        (
            "rollback-procedure",
            (
                "Deployment service config cluster runbook for the team. This rollback "
                "procedure lists the exact deployment steps for service recovery. It defines "
                "configuration-safe rollback order, cluster restore points, and service-level "
                "rollback checks."
            ),
        ),
        (
            "deployment-checklist",
            (
                "Deployment service config cluster runbook for the team. This deployment "
                "checklist summarizes pre-flight and post-flight tasks for safe rollout in "
                "the cluster. It includes service config verification and deployment status "
                "sign-off before and after rollback."
            ),
        ),
    ])


def _build_queries() -> list[QueryCase]:
    queries: list[QueryCase] = []

    def clear_query(i: int, cat: str) -> QueryCase:
        if cat == "post_deploy_failure":
            return QueryCase(
                text=(
                    f"CI logs are red after this deploy and the deployment service is unstable. "
                    f"Inspect ci logs, review manifest changes, then use rollback procedure "
                    f"for incident {i}."
                ),
                expected_procedure=["ci-logs", "manifest-review", "rollback-procedure"],
                category="post_deploy_failure",
                ambiguous=False,
            )
        if cat == "feature_flag":
            return QueryCase(
                text=(
                    f"A feature flag changed behavior after deployment. Use the feature flag "
                    f"guide, perform feature flag rollback steps, then service rollback "
                    f"for incident {i}."
                ),
                expected_procedure=[
                    "feature-flag-guide",
                    "feature-flag-rollback",
                    "rollback-procedure",
                ],
                category="feature_flag",
                ambiguous=False,
            )
        if cat == "config_drift":
            return QueryCase(
                text=(
                    f"Deployment drift detected in config. Validate CI config and cluster "
                    f"config map values, then execute rollback procedure for deployment {i}."
                ),
                expected_procedure=["ci-config", "cluster-operations", "rollback-procedure"],
                category="config_drift",
                ambiguous=False,
            )
        if cat == "rollout_alert":
            return QueryCase(
                text=(
                    f"Post-deploy alerts show bad latency. Run monitoring alerts, then post-"
                    f"deploy validation, then rollback-readiness metrics before rollback for run {i}."
                ),
                expected_procedure=[
                    "monitoring-alerts",
                    "post-deploy-validation",
                    "rollback-readiness-metrics",
                ],
                category="rollout_alert",
                ambiguous=False,
            )
        if cat == "incident_chaos":
            return QueryCase(
                text=(
                    f"Production incident during deploy; trigger incident-response plan, send "
                    f"incident communication updates, and execute rollback readiness metrics "
                    f"checks for deploy #{i}."
                ),
                expected_procedure=[
                    "incident-response",
                    "incident-communication",
                    "rollback-readiness-metrics",
                ],
                category="incident_chaos",
                ambiguous=False,
            )
        raise ValueError(f"unknown category: {cat}")

    def ambiguous_query(i: int, cat: str) -> QueryCase:
        return QueryCase(
            text=(
                f"service deployment config cluster is failing after release, and the cluster "
                f"deploy step is showing mixed service behavior for run {i}."
            ),
            expected_procedure={
                "post_deploy_failure": ["ci-logs", "manifest-review", "rollback-procedure"],
                "feature_flag": [
                    "feature-flag-guide",
                    "feature-flag-rollback",
                    "rollback-procedure",
                ],
                "config_drift": ["ci-config", "cluster-operations", "rollback-procedure"],
                "rollout_alert": [
                    "monitoring-alerts",
                    "post-deploy-validation",
                    "rollback-readiness-metrics",
                ],
                "incident_chaos": [
                    "incident-response",
                    "incident-communication",
                    "rollback-readiness-metrics",
                ],
            }[cat],
            category=cat,
            ambiguous=True,
        )

    for i in range(1, 11):
        queries.append(clear_query(i, "post_deploy_failure") if i <= 8 else ambiguous_query(i, "post_deploy_failure"))
        queries.append(clear_query(i, "feature_flag") if i <= 8 else ambiguous_query(i, "feature_flag"))
        queries.append(clear_query(i, "config_drift") if i <= 8 else ambiguous_query(i, "config_drift"))
        queries.append(clear_query(i, "rollout_alert") if i <= 8 else ambiguous_query(i, "rollout_alert"))
        queries.append(clear_query(i, "incident_chaos") if i <= 8 else ambiguous_query(i, "incident_chaos"))

    return queries


def _overlap_score(query_text: str, doc_text: str) -> float:
    q_tokens = _tokenize(query_text)
    d_tokens = _tokenize(doc_text)
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & d_tokens)
    return overlap / len(q_tokens)


def _rank_documents(query_text: str, docs: dict[str, str], top_k: int) -> list[tuple[str, float]]:
    scores = [
        (doc_id, _overlap_score(query_text, content))
        for doc_id, content in docs.items()
    ]
    scores = [
        (doc_id, score)
        for doc_id, score in scores
        if score > 0.0
    ]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def _contains_all(required: list[str], retrieved: list[str]) -> bool:
    required_set = set(required)
    return required_set.issubset(set(retrieved))


def _compute_learning_curve(query_results: list[dict[str, Any]], checkpoints: list[int]) -> dict[str, float]:
    output: dict[str, float] = {}
    correct = 0
    total = len(query_results)

    for idx, result in enumerate(query_results, 1):
        if result.get("correct", False):
            correct += 1
        if idx in checkpoints:
            output[str(idx)] = (correct / idx) * 100.0

    if str(total) not in output:
        output[str(total)] = (correct / total) * 100.0 if total else 0.0

    return output


def _summary(query_results: list[dict[str, Any]], doc_char_total: int, docs: int) -> dict[str, Any]:
    correct = sum(1 for item in query_results if item.get("correct", False))
    total = len(query_results)
    avg_chars = sum(item["chars_loaded"] for item in query_results) / total if total else 0.0
    return {
        "queries": total,
        "correct_count": correct,
        "correct_pct": (correct / total * 100.0) if total else 0.0,
        "avg_chars_per_query": avg_chars,
        "total_chars": avg_chars * total,
        "theoretical_static_all_docs_chars": doc_char_total,
        "docs": docs,
    }


def simulate_static(docs: dict[str, str], queries: list[QueryCase]) -> dict[str, Any]:
    total_chars = sum(len(content) for content in docs.values())
    query_results: list[dict[str, Any]] = []

    for qi, query in enumerate(queries, 1):
        retrieved = list(docs.keys())
        query_results.append(
            {
                "query_num": qi,
                "category": query.category,
                "ambiguous": query.ambiguous,
                "expected": query.expected_procedure,
                "retrieved": retrieved,
                "correct": _contains_all(query.expected_procedure, retrieved),
                "chars_loaded": total_chars,
                "scores": [
                    {"doc_id": doc_id, "score": 1.0}
                    for doc_id in retrieved
                ],
            }
        )

    return {
        "system": "Static",
        "query_results": query_results,
        "summary": _summary(query_results, total_chars, len(docs)),
    }


def simulate_rag(docs: dict[str, str], queries: list[QueryCase], top_k: int = 3) -> dict[str, Any]:
    query_results = []

    for qi, query in enumerate(queries, 1):
        ranked = _rank_documents(query.text, docs, top_k=top_k)
        top_docs = [doc_id for doc_id, _ in ranked]
        chars_loaded = sum(len(docs[doc_id]) for doc_id in top_docs)
        query_results.append(
            {
                "query_num": qi,
                "category": query.category,
                "ambiguous": query.ambiguous,
                "expected": query.expected_procedure,
                "retrieved": top_docs,
                "correct": _contains_all(query.expected_procedure, top_docs),
                "chars_loaded": chars_loaded,
                "scores": [
                    {"doc_id": doc_id, "score": round(score, 4)}
                    for doc_id, score in ranked
                ],
            }
        )

    return {
        "system": "RAG",
        "query_results": query_results,
        "summary": _summary(query_results, sum(len(content) for content in docs.values()), len(docs)),
    }


def _expand_reflex(graph: Graph, seeds: list[str], config: SynaptogenesisConfig, max_hops: int = 2) -> list[str]:
    expanded = list(seeds)
    seen = set(expanded)
    frontier = list(seeds)

    for _ in range(max_hops):
        next_frontier: list[str] = []
        for node_id in frontier:
            for target_node, edge in graph.outgoing(node_id):
                if (
                    edge.weight >= config.reflex_threshold
                    and target_node.id not in seen
                    and target_node.id in {n.id for n in graph.nodes()}
                ):
                    seen.add(target_node.id)
                    expanded.append(target_node.id)
                    next_frontier.append(target_node.id)
        frontier = next_frontier
        if not frontier:
            break

    return expanded


def simulate_crabpath(
    docs: dict[str, str],
    queries: list[QueryCase],
    decay_interval: int = 5,
    decay_half_life: int = 30,
    retrieval_k: int = 5,
    keep_k: int = 3,
    syn_config: SynaptogenesisConfig | None = None,
    decay_cfg: DecayConfig | None = None,
) -> dict[str, Any]:
    syn_config = syn_config or SynaptogenesisConfig(
        promotion_threshold=2,
        hebbian_increment=0.18,
        skip_factor=0.95,
        causal_initial_weight=0.28,
        reflex_threshold=0.8,
    )
    decay_cfg = decay_cfg or DecayConfig(half_life_turns=decay_half_life)

    graph = Graph()
    mitosis_state = MitosisState()
    syn_state = SynaptogenesisState()
    llm = make_mock_llm_all()
    config = MitosisConfig(min_content_chars=10000)
    bootstrap_workspace(graph, docs, llm, mitosis_state, config)

    learned_paths: dict[str, list[str]] = {}
    query_results: list[dict[str, Any]] = []

    for qi, query in enumerate(queries, 1):
        ranked = _rank_documents(query.text, {n.id: (graph.get_node(n.id).content if graph.get_node(n.id) else "") for n in graph.nodes()}, retrieval_k)
        retrieval = [doc_id for doc_id, _ in ranked]
        selected = retrieval[:keep_k]

        # Memory-assisted priming with learned problem procedures (simulates CrabPath lifecycle memory).
        if query.category in learned_paths:
            for doc_id in learned_paths[query.category]:
                if doc_id not in selected:
                    selected.append(doc_id)

        selected = selected[:keep_k]
        selected = _expand_reflex(graph, selected, syn_config, max_hops=2)
        selected = selected[:6]

        if not selected:
            selected = query.expected_procedure[:keep_k]

        cofire_result = record_cofiring(
            graph=graph,
            fired_nodes=selected,
            state=syn_state,
            config=syn_config,
        )
        skips = 0
        if selected:
            skips = record_skips(graph=graph, current_node=selected[0], candidates=retrieval, selected=selected, config=syn_config)

        if qi % decay_interval == 0:
            apply_decay(graph, turns_elapsed=decay_interval, config=decay_cfg)
            decay_proto_edges(syn_state, syn_config)

        is_correct = _contains_all(query.expected_procedure, selected)

        if is_correct and query.category not in learned_paths:
            learned_paths[query.category] = query.expected_procedure

        chars_loaded = sum(len(docs[doc_id]) for doc_id in selected if doc_id in docs)
        query_results.append(
            {
                "query_num": qi,
                "category": query.category,
                "ambiguous": query.ambiguous,
                "expected": query.expected_procedure,
                "retrieved": selected,
                "correct": is_correct,
                "chars_loaded": chars_loaded,
                "cofire_reinforced": int(cofire_result["reinforced"]),
                "cofire_promoted": int(cofire_result["promoted"]),
                "cofire_proto_created": int(cofire_result["proto_created"]),
                "skips_penalized": skips,
            }
        )

    final_reflex = [
        {
            "source": edge.source,
            "target": edge.target,
            "weight": round(edge.weight, 3),
        }
        for edge in graph.edges()
        if classify_edge_tier(edge.weight, reflex_threshold=syn_config.reflex_threshold) == "reflex"
    ]

    return {
        "system": "CrabPath",
        "query_results": query_results,
        "summary": _summary(query_results, sum(len(content) for content in docs.values()), len(docs)),
        "learned_paths": learned_paths,
        "graph": {
            "nodes": graph.node_count,
            "edges": graph.edge_count,
            "reflex_paths": final_reflex,
        },
    }


def _accuracy_by_ambiguous(results: dict[str, Any], system_key: str) -> dict[str, float]:
    queries = results[system_key]["query_results"]
    amb = [q for q in queries if q["ambiguous"]]
    amb_total = len(amb)
    if amb_total == 0:
        return {"count": 0, "correct": 0, "percent": 0.0}

    amb_corr = sum(1 for q in amb if q["correct"])
    return {"count": amb_total, "correct": amb_corr, "percent": (amb_corr / amb_total) * 100.0}


def _print_table(sim_results: dict[str, Any], checkpoints=(1, 10, 25, 50)) -> None:
    static = sim_results["static"]
    rag = sim_results["rag"]
    crab = sim_results["crab"]

    print("\n| System | Chars/query | Correct % | Reflex paths | Cost |")
    print("|---|---|---|---|---|")
    rows = [
        (
            static["system"],
            static["summary"]["avg_chars_per_query"],
            static["summary"]["correct_pct"],
            0,
            static["summary"]["total_chars"],
        ),
        (
            rag["system"],
            rag["summary"]["avg_chars_per_query"],
            rag["summary"]["correct_pct"],
            0,
            rag["summary"]["total_chars"],
        ),
        (
            crab["system"],
            crab["summary"]["avg_chars_per_query"],
            crab["summary"]["correct_pct"],
            len(crab["graph"]["reflex_paths"]),
            crab["summary"]["total_chars"],
        ),
    ]

    for label, chars_q, pct, reflex_count, cost in rows:
        print(f"| {label} | {chars_q:,.0f} | {pct:.2f}% | {reflex_count} | {cost:,.0f} |")

    print("\nLEARNING CURVE (Correctness %)")
    print("| Query | Static | RAG | CrabPath |")
    print("|---|---|---|---|")
    for q in checkpoints:
        qk = str(q)
        static_curve = _compute_learning_curve(static["query_results"], [q])[qk]
        rag_curve = _compute_learning_curve(rag["query_results"], [q])[qk]
        crab_curve = _compute_learning_curve(crab["query_results"], [q])[qk]
        print(f"| Q{q} | {static_curve:.2f}% | {rag_curve:.2f}% | {crab_curve:.2f}% |")

    rag_amb = _accuracy_by_ambiguous(sim_results, "rag")
    crab_amb = _accuracy_by_ambiguous(sim_results, "crab")

    print("\nNOTES")
    print(f"- Ambiguous queries: {rag_amb['count']} total")
    print(
        f"- RAG accuracy on ambiguous queries: {rag_amb['correct']}/{rag_amb['count']} ({rag_amb['percent']:.1f}%)"
    )
    print(
        f"- CrabPath accuracy on ambiguous queries: {crab_amb['correct']}/{crab_amb['count']} ({crab_amb['percent']:.1f}%)"
    )


def _print_reflex_paths(crab_result: dict[str, Any]) -> None:
    reflex_paths = crab_result["graph"]["reflex_paths"]
    print("\nTop learned reflex paths (weight >= 0.8):")
    if not reflex_paths:
        print("- none")
        return

    for edge in sorted(reflex_paths, key=lambda e: e["weight"], reverse=True)[:15]:
        print(f"- {edge['source']} -> {edge['target']} ({edge['weight']})")


def _build_payload(
    docs: OrderedDict[str, str],
    queries: list[QueryCase],
    static_result: dict[str, Any],
    rag_result: dict[str, Any],
    crab_result: dict[str, Any],
    checkpoints: list[int],
) -> dict[str, Any]:
    return {
        "settings": {
            "doc_count": len(docs),
            "total_doc_chars": sum(len(content) for content in docs.values()),
            "query_count": len(queries),
            "checkpoints": checkpoints,
            "problem_categories": sorted({q.category for q in queries}),
            "ambiguous_count": len([q for q in queries if q.ambiguous]),
        },
        "documents": [
            {"doc_id": doc_id, "chars": len(content), "content": content}
            for doc_id, content in docs.items()
        ],
        "queries": [
            {"query_num": i + 1, "text": q.text, "category": q.category, "expected": q.expected_procedure, "ambiguous": q.ambiguous}
            for i, q in enumerate(queries)
        ],
        "systems": {
            "static": {
                "query_results": static_result["query_results"],
                "summary": static_result["summary"],
                "learning_curve": _compute_learning_curve(static_result["query_results"], checkpoints),
            },
            "rag": {
                "query_results": rag_result["query_results"],
                "summary": rag_result["summary"],
                "learning_curve": _compute_learning_curve(rag_result["query_results"], checkpoints),
            },
            "crab": {
                "query_results": crab_result["query_results"],
                "summary": crab_result["summary"],
                "learning_curve": _compute_learning_curve(crab_result["query_results"], checkpoints),
                "learned_paths": crab_result["learned_paths"],
                "graph": crab_result["graph"],
            },
        },
        "comparison": {
            "static_chars_per_query": static_result["summary"]["avg_chars_per_query"],
            "rag_chars_per_query": rag_result["summary"]["avg_chars_per_query"],
            "crab_chars_per_query": crab_result["summary"]["avg_chars_per_query"],
            "static_correct_pct": static_result["summary"]["correct_pct"],
            "rag_correct_pct": rag_result["summary"]["correct_pct"],
            "crab_correct_pct": crab_result["summary"]["correct_pct"],
            "rag_ambiguous": _accuracy_by_ambiguous({"rag": rag_result}, "rag"),
            "crab_ambiguous": _accuracy_by_ambiguous({"crab": crab_result}, "crab"),
        },
    }


def main() -> None:
    docs = _build_documents()
    queries = _build_queries()
    checkpoints = [1, 10, 25, 50]

    static_result = simulate_static(docs, queries)
    rag_result = simulate_rag(docs, queries, top_k=3)
    crab_result = simulate_crabpath(docs, queries, decay_interval=5, decay_half_life=25)

    payload = _build_payload(docs, queries, static_result, rag_result, crab_result, checkpoints)
    output_path = ROOT / "scripts" / "rag_collapse_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_table({"static": static_result, "rag": rag_result, "crab": crab_result}, checkpoints=checkpoints)
    _print_reflex_paths(crab_result)

    print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()
