"""A/B simulation: Hebbian-only vs Hebbian+RL scoring under ambiguity."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from crabpath.autotune import HEALTH_TARGETS, measure_health
from crabpath.decay import DecayConfig, apply_decay
from crabpath.graph import Edge, Graph, Node
from crabpath.learning import LearningConfig, RewardSignal, _BASELINE_STATE, make_learning_step
from crabpath.mitosis import MitosisConfig, MitosisState, bootstrap_workspace
from crabpath.synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    decay_proto_edges,
    edge_tier_stats,
    record_correction,
    record_cofiring,
    record_skips,
)
from crabpath._structural_utils import count_cross_file_edges


CHECKPOINTS = (10, 20, 30, 40, 50)
CORRECT_DEPLOY_IDS = [f"deploy_correct_{i}" for i in range(1, 6)]
WRONG_DEPLOY_IDS = [f"deploy_wrong_{i}" for i in range(1, 6)]
HOTFIX_IDS = [f"hotfix_{i}" for i in range(1, 6)]
TOPIC_NODES = {
    "deploy": "topic::deploy",
    "hotfix": "topic::hotfix",
    "scale": "topic::scale",
    "network": "topic::networking",
    "database": "topic::database",
    "testing": "topic::testing",
    "monitor": "topic::monitoring",
    "default": "topic::default",
}


@dataclass
class QuerySpec:
    text: str
    correct_node_ids: list[str]
    wrong_node_ids: list[str] | None = None


def _build_workspace() -> dict[str, str]:
    return {
        "deploy_correct_1": (
            "Standard deploy procedure starts by checking CI status and commit tags, "
            "then inspect the deployment manifest for service and cluster config changes."
        ),
        "deploy_correct_2": (
            "After approval, deploy through the normal pipeline only after manifest integrity check. "
            "Keep config review mandatory, verify rollout plan, and do not skip checkpoints."
        ),
        "deploy_correct_3": (
            "If deploy validation fails, rollback immediately to the last good manifest, "
            "revert service config, and capture traceability for the cluster event."
        ),
        "deploy_correct_4": (
            "Post-deploy verification should run smoke checks against key service endpoints, "
            "ensure config reconciliation settled, then validate cluster stability."
        ),
        "deploy_correct_5": (
            "During rollout monitoring, track error budget, CPU spikes, restart counters, "
            "and rollback if safety thresholds are crossed before finishing verification."
        ),
        "deploy_wrong_1": (
            "Legacy deploy flow: skip CI checks if they are slow and deploy directly to cluster, "
            "assuming the service config will reconcile by itself."
        ),
        "deploy_wrong_2": (
            "Old rollout habit: deploy from local branch, ignore manifest details, and rely on hope that "
            "deploy and config will pass in production."
        ),
        "deploy_wrong_3": (
            "Outdated procedure says no rollback unless outage is severe; run deploy directly, "
            "monitor later, and avoid touching CI because it delays service."
        ),
        "deploy_wrong_4": (
            "When service is healthy enough, bypass preflight checks and push changes; "
            "if config drifts, redeploy once the cluster is quiet."
        ),
        "deploy_wrong_5": (
            "Quick and simple deploy: skip manifest inspection, skip verify gates, "
            "and hope the service rollout does not break the cluster."
        ),
        "hotfix_1": (
            "Hotfix procedure starts with a minimal change package and an explicit emergency patch tag "
            "for service config and deployment manifest."
        ),
        "hotfix_2": (
            "Run a narrow hotfix rollout only for the affected service and keep standard CI optional "
            "when incident response requires speed with explicit operator notice."
        ),
        "hotfix_3": (
            "Hotfix validation is targeted: run the regression for the impacted flow, check monitor alerts, "
            "and confirm no config drift across affected clusters."
        ),
        "hotfix_4": (
            "After a hotfix, document root cause and temporary config changes, "
            "then schedule normal deploy procedure for a full clean release."
        ),
        "hotfix_5": (
            "If hotfix causes production regression, rollback via hotfix tag and notify on-call "
            "while keeping monitoring strict on service behavior."
        ),
        "scaling_node": (
            "Scaling guidance: update autoscaling policy, tune CPU and memory thresholds, "
            "and test node drain strategies across cluster zones."
        ),
        "networking_node": (
            "Networking playbook covers ingress routing, service mesh policy, DNS cache updates, "
            "and port-level firewall rules for stable request paths."
        ),
        "database_node": (
            "Database recovery plan includes backup cadence, replica lag checks, index maintenance, "
            "and connection pool cleanup after failover."
        ),
        "testing_node": (
            "Testing discipline covers unit and integration suites, flake triage, "
            "and release criteria before merging service changes."
        ),
        "monitoring_node": (
            "Monitoring baseline defines alert thresholds, dashboards for request latency and errors, "
            "and paging rules for cluster incidents."
        ),
    }


def _build_queries() -> list[QuerySpec]:
    queries: list[QuerySpec] = []

    def add(text: str, correct: list[str], wrong: list[str] | None = None) -> None:
        queries.append(QuerySpec(text=text, correct_node_ids=correct, wrong_node_ids=wrong))

    add(
        "deploy failed what do I do",
        correct=CORRECT_DEPLOY_IDS,
        wrong=WRONG_DEPLOY_IDS,
    )
    for i in range(1, 10):
        add(f"deployment failed while releasing service now {i}", CORRECT_DEPLOY_IDS, WRONG_DEPLOY_IDS)
    for i in range(1, 3):
        add(f"what is the standard deploy path for the cluster update {i}", CORRECT_DEPLOY_IDS, WRONG_DEPLOY_IDS)
    add(
        "quick fix needed now",
        correct=HOTFIX_IDS,
        wrong=CORRECT_DEPLOY_IDS,
    )
    add(
        "quick fix needed now for production bug",
        correct=HOTFIX_IDS,
        wrong=CORRECT_DEPLOY_IDS,
    )
    for i in range(1, 5):
        add(f"quick fix needed now before release {i}", HOTFIX_IDS, CORRECT_DEPLOY_IDS)
    add("skip ci and deploy directly because build is late", CORRECT_DEPLOY_IDS, WRONG_DEPLOY_IDS)
    add("skip tests and deploy directly to cluster", CORRECT_DEPLOY_IDS, WRONG_DEPLOY_IDS)
    add("deploy directly then see if it works in production", CORRECT_DEPLOY_IDS, WRONG_DEPLOY_IDS)
    add("just deploy service config now and monitor", CORRECT_DEPLOY_IDS, WRONG_DEPLOY_IDS)
    for i in range(1, 4):
        add(
            f"should I skip ci for this deploy {i}",
            CORRECT_DEPLOY_IDS,
            WRONG_DEPLOY_IDS,
        )
    for i in range(1, 6):
        add(f"scale node pools for traffic spike {i}", ["scaling_node"])
    for i in range(1, 6):
        add(f"change networking policy for new endpoint {i}", ["networking_node"])
    for i in range(1, 6):
        add(f"database failover and backup sanity check {i}", ["database_node"])
    for i in range(1, 6):
        add(f"testing strategy for release confidence {i}", ["testing_node"])
    for i in range(1, 6):
        add(f"monitoring dashboard thresholds after changes {i}", ["monitoring_node"])

    assert len(queries) == 50, f"Expected 50 queries, built {len(queries)}"
    return queries


def _build_candidates(graph: Graph, query: str) -> list[tuple[str, float, str]]:
    q_words = set(query.lower().split())
    candidates: list[tuple[str, float, str]] = []

    for node in graph.nodes():
        if node.id.startswith("topic::"):
            continue
        n_words = set(node.content.lower().split())
        overlap = len(q_words & n_words)
        score = min(overlap / max(len(q_words), 1), 1.0)
        if score > 0.1:
            candidates.append((node.id, score, node.summary or node.content[:80]))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:10]


def _rank_candidates(
    query: QuerySpec,
    candidates: list[tuple[str, float, str]],
    graph: Graph,
    topic_node: str,
) -> list[tuple[str, float, float, float]]:
    ranked: list[tuple[str, float, float, float]] = []
    for node_id, score, _ in candidates:
        edge = graph.get_edge(topic_node, node_id)
        edge_weight = edge.weight if edge is not None else 0.0
        combined_score = score + (edge_weight * 0.30)
        ranked.append((node_id, score, edge_weight, combined_score))

    ranked.sort(key=lambda item: item[3], reverse=True)
    return ranked[:10]


def _route_nodes(
    query: QuerySpec,
    ranked_candidates: list[tuple[str, float, float, float]],
    rng: random.Random,
) -> list[str]:
    if not ranked_candidates:
        return []

    threshold = ranked_candidates[0][3] * 0.5
    selected = [node_id for node_id, _, _, combined in ranked_candidates if combined >= threshold]
    if not selected:
        selected = [ranked_candidates[0][0]]
    selected = selected[:5]

    wrong_ids = set(query.wrong_node_ids or [])
    correct_ids = set(query.correct_node_ids)
    edge_wrong_max = max(
        (edge_weight for node_id, _, edge_weight, _ in ranked_candidates if node_id in wrong_ids),
        default=0.0,
    )
    edge_correct_max = max(
        (
            edge_weight
            for node_id, _, edge_weight, _ in ranked_candidates
            if node_id in correct_ids
        ),
        default=0.0,
    )

    misroute_probability = 0.30
    if wrong_ids and correct_ids:
        edge_delta = edge_correct_max - edge_wrong_max
        misroute_probability = 0.30 - max(0.0, min(0.30, edge_delta * 0.08))
        misroute_probability = max(0.0, misroute_probability)

    if wrong_ids and correct_ids and rng.random() < misroute_probability:
        if any(n in correct_ids for n in selected):
            wrong_candidates = [
                node_id for node_id, _, _, _ in ranked_candidates if node_id in wrong_ids
            ]
            if wrong_candidates:
                replacement = wrong_candidates[0]
                selected = list(selected)
                for idx, node_id in enumerate(selected):
                    if node_id in correct_ids:
                        selected[idx] = replacement
                        break

    # Dedupe while keeping order
    deduped: list[str] = []
    for node_id in selected:
        if node_id not in deduped:
            deduped.append(node_id)
    return deduped[:1]


def _score_retrieval(
    selected_node_ids: list[str],
    correct_node_ids: list[str],
    wrong_node_ids: list[str] | None = None,
) -> tuple[float, bool]:
    correct = set(correct_node_ids)
    wrong = set(wrong_node_ids or [])

    selected_set = set(selected_node_ids)
    wrong_overlap = bool(selected_set & wrong)
    if wrong_overlap:
        return -1.0, False

    correct_overlap = bool(selected_set & correct)
    if correct_overlap:
        return 1.0, True
    return 0.0, False


def _build_trajectory(
    topic_node: str,
    selected_nodes: list[str],
    candidate_pairs: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    if not selected_nodes:
        return []

    trajectory = [
        {
            "from_node": topic_node,
            "to_node": selected_nodes[0],
            "candidates": candidate_pairs,
        }
    ]
    return trajectory


def _topic_node(query: str) -> str:
    text = query.lower()
    if "quick fix" in text or "hotfix" in text:
        return TOPIC_NODES["hotfix"]
    if "scale" in text or "scaling" in text:
        return TOPIC_NODES["scale"]
    if "network" in text or "ingress" in text:
        return TOPIC_NODES["network"]
    if "database" in text or "db" in text:
        return TOPIC_NODES["database"]
    if "testing" in text:
        return TOPIC_NODES["testing"]
    if "monitor" in text:
        return TOPIC_NODES["monitor"]
    if "deploy" in text:
        return TOPIC_NODES["deploy"]
    return TOPIC_NODES["default"]


def _compute_health_summary(graph: Graph, mitosis_state: MitosisState, query_stats: dict[str, Any]) -> dict[str, Any]:
    health = measure_health(graph, mitosis_state, query_stats)
    metric_rows: dict[str, dict[str, Any]] = {}
    in_range_count = 0

    for metric, target in HEALTH_TARGETS.items():
        value = getattr(health, metric)
        lo, hi = target
        in_range = (lo is None or value >= lo) and (hi is None or value <= hi)
        metric_rows[metric] = {
            "value": value,
            "target": target,
            "in_range": in_range,
        }
        if in_range:
            in_range_count += 1

    return {
        "raw": health,
        "in_range_count": in_range_count,
        "in_range_total": len(HEALTH_TARGETS),
        "metrics": metric_rows,
    }


def run_arm(name: str, queries: list[QuerySpec], use_rl: bool = False, seed: int = 1337) -> dict[str, Any]:
    workspace_files = _build_workspace()
    graph = Graph()
    mitosis_state = MitosisState()
    syn_state = SynaptogenesisState()
    syn_config = SynaptogenesisConfig()
    decay_config = DecayConfig()
    mitosis_config = MitosisConfig()

    bootstrap_workspace(
        graph=graph,
        workspace_files=workspace_files,
        llm_call=lambda *args, **kwargs: "",  # keep bootstrap deterministic
        state=mitosis_state,
        config=mitosis_config,
    )

    # Ensure each node has a short content-based summary for traceability.
    for node in graph.nodes():
        if not node.summary:
            node.summary = node.content[:80].replace("\n", " ")

    # Topic routers provide a learnable routing prior for ambiguous intent classes.
    for topic_id in set(TOPIC_NODES.values()):
        if graph.get_node(topic_id) is None:
            graph.add_node(
                Node(
                    id=topic_id,
                    content=topic_id,
                    summary=topic_id,
                    type="topic_router",
                    metadata={"created_ts": 0.0, "fired_count": 0, "last_fired_ts": 0.0},
                )
            )

    for topic_id in TOPIC_NODES.values():
        for node in graph.nodes():
            if node.id.startswith("topic::") or node.id == topic_id:
                continue
            if graph.get_edge(topic_id, node.id) is None:
                graph.add_edge(
                    Edge(source=topic_id, target=node.id, weight=0.22, created_by="manual")
                )

    rng = random.Random(seed)
    _BASELINE_STATE.clear()

    query_stats: dict[str, Any] = {
        "queries": 0,
        "promotions": 0,
        "proto_created": 0,
        "avg_nodes_fired_per_query": 0.0,
        "context_chars": 0.0,
    }
    total_nodes_fired = 0
    total_context_chars = 0
    total_promotions = 0
    total_proto_created = 0

    full_accuracy_count = 0
    wrong_seen = 0
    wrong_hits = 0
    candidate_overlap_count = 0
    learning_config = LearningConfig(learning_rate=0.35 if use_rl else 0.05)

    checkpoints: dict[int, dict[str, Any]] = {}
    last_skips_penalized = 0

    for qi, qspec in enumerate(queries, start=1):
        topic = _topic_node(qspec.text)
        candidates = _build_candidates(graph, qspec.text)
        ranked_candidates = _rank_candidates(qspec, candidates, graph, topic)
        candidate_pairs = [
            (node_id, combined_score) for node_id, _, _, combined_score in ranked_candidates
        ]
        selected_nodes = _route_nodes(qspec, ranked_candidates, rng)

        reward, correct_hit = _score_retrieval(selected_nodes, qspec.correct_node_ids, qspec.wrong_node_ids)
        if reward > 0.0:
            full_accuracy_count += 1
        if correct_hit:
            candidate_overlap_count += 1

        if qspec.wrong_node_ids:
            wrong_seen += 1
            if set(selected_nodes) & set(qspec.wrong_node_ids):
                wrong_hits += 1

        cofire_result = record_cofiring(graph, selected_nodes, syn_state, syn_config)
        total_promotions += cofire_result["promoted"]
        total_proto_created += cofire_result["proto_created"]
        last_skips_penalized = 0
        if candidates and selected_nodes:
            candidate_ids = [node_id for node_id, _, _ in candidates]
            last_skips_penalized = record_skips(
                graph=graph,
                current_node=selected_nodes[0],
                candidates=candidate_ids,
                selected=selected_nodes,
                config=syn_config,
            )

        if use_rl and reward < 0.0:
            record_correction(graph, selected_nodes, reward=reward, config=syn_config)

        if use_rl and selected_nodes:
            trajectory = _build_trajectory(topic, selected_nodes, candidate_pairs)
            if trajectory:
                make_learning_step(
                    graph=graph,
                    trajectory_steps=trajectory,
                    reward=RewardSignal(
                        episode_id=f"{name.replace(' ', '_')}-q-{qi % 10}",
                        final_reward=reward,
                    ),
                    config=learning_config,
                )

        if qi % 5 == 0:
            apply_decay(graph, turns_elapsed=5, config=decay_config)
            decay_proto_edges(syn_state, syn_config)

        total_nodes_fired += len(selected_nodes)
        selected_node_text_len = sum(
            len(graph.get_node(node_id).content) for node_id in selected_nodes if graph.get_node(node_id)
        )
        total_context_chars += selected_node_text_len

        query_stats["queries"] = qi
        query_stats["promotions"] = total_promotions
        query_stats["proto_created"] = total_proto_created
        query_stats["avg_nodes_fired_per_query"] = total_nodes_fired / qi
        query_stats["context_chars"] = total_context_chars

        if qi in CHECKPOINTS:
            learning_signal = "rl update" if use_rl else "hebbian only"

            health_summary = _compute_health_summary(graph, mitosis_state, query_stats)
            tiers = edge_tier_stats(graph, syn_config)
            wrong_rate = (wrong_hits / wrong_seen * 100.0) if wrong_seen else 0.0
            checkpoints[qi] = {
                "accuracy_full_pct": (full_accuracy_count / qi) * 100.0,
                "cross_file_edges": count_cross_file_edges(graph),
                "tiers": tiers,
                "health": {
                    "in_range": f"{health_summary['in_range_count']}/{health_summary['in_range_total']}",
                    "metrics": health_summary["metrics"],
                },
                "wrong_seen": wrong_seen,
                "wrong_hits": wrong_hits,
                "wrong_path_hit_rate_pct": wrong_rate,
                "skips_penalized": last_skips_penalized,
                "promotions": total_promotions,
                "proto_created": total_proto_created,
                "learning_signal": learning_signal,
                "candidate_overlap_pct": (candidate_overlap_count / qi) * 100.0,
            }

    final = checkpoints[CHECKPOINTS[-1]]
    final_tiers = edge_tier_stats(graph, syn_config)
    return {
        "name": name,
        "use_rl": use_rl,
        "checkpoints": checkpoints,
        "query_count": len(queries),
        "metrics": {
            "accuracy_full_pct": final["accuracy_full_pct"],
            "cross_file_edges": final["cross_file_edges"],
            "health_in_range": final["health"]["in_range"],
            "wrong_path_hit_rate_pct": final["wrong_path_hit_rate_pct"],
            "wrong_seen": final["wrong_seen"],
            "wrong_hits": final["wrong_hits"],
            "final_tiers": final_tiers,
            "candidate_overlap_pct": final["candidate_overlap_pct"],
        },
        "query_stats": query_stats,
        "learning_enabled": use_rl,
    }


def main() -> None:
    queries = _build_queries()
    arm_a = run_arm("Arm A", queries, use_rl=False)
    arm_b = run_arm("Arm B", queries, use_rl=True)

    print("\nLearning curve (Q / full retrieval accuracy)")
    print("Q   | Arm A accuracy | Arm B accuracy | Arm A wrong-hit % | Arm B wrong-hit %")
    for q in CHECKPOINTS:
        a = arm_a["checkpoints"][q]["accuracy_full_pct"]
        b = arm_b["checkpoints"][q]["accuracy_full_pct"]
        a_wrong = arm_a["checkpoints"][q]["wrong_path_hit_rate_pct"]
        b_wrong = arm_b["checkpoints"][q]["wrong_path_hit_rate_pct"]
        print(f"{q:<3}|{a:14.2f}%|{b:14.2f}%|{a_wrong:16.2f}%|{b_wrong:16.2f}%")

    a_final = arm_a["metrics"]
    b_final = arm_b["metrics"]
    a_health_n, a_health_d = map(int, a_final["health_in_range"].split("/"))
    b_health_n, b_health_d = map(int, b_final["health_in_range"].split("/"))

    def _delta(a: float, b: float) -> str:
        return f"{b - a:+.2f}"

    def _delta_ratio(a_raw: str, b_raw: str) -> str:
        a_num, a_den = map(int, a_raw.split("/"))
        b_num, b_den = map(int, b_raw.split("/"))
        return f"{(b_num / b_den) - (a_num / a_den):.2f}"

    print("\nComparison table")
    print(
        f"Retrieval accuracy | {a_final['accuracy_full_pct']:.2f}% | "
        f"{b_final['accuracy_full_pct']:.2f}% | {_delta(a_final['accuracy_full_pct'], b_final['accuracy_full_pct'])}"
    )
    print(
        f"Wrong-path hit rate | {a_final['wrong_path_hit_rate_pct']:.2f}% | "
        f"{b_final['wrong_path_hit_rate_pct']:.2f}% | {_delta(a_final['wrong_path_hit_rate_pct'], b_final['wrong_path_hit_rate_pct'])}"
    )
    print(
        f"Wrong nodes hit      | {a_final['wrong_hits']} of {a_final['wrong_seen']} | "
        f"{b_final['wrong_hits']} of {b_final['wrong_seen']}"
    )
    print(
        f"Final tiers         | {a_final['final_tiers']} | {b_final['final_tiers']} | n/a"
    )
    print(
        f"Health score        | {a_final['health_in_range']} | {b_final['health_in_range']} | "
        f"{_delta_ratio(a_final['health_in_range'], b_final['health_in_range'])}"
    )
    print(
        f"Cross-file edges    | {a_final['cross_file_edges']} | "
        f"{b_final['cross_file_edges']} | {_delta(a_final['cross_file_edges'], b_final['cross_file_edges'])}"
    )

    result_path = "scripts/ab_scoring_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoints": CHECKPOINTS,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "queries": [q.__dict__ for q in queries],
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
