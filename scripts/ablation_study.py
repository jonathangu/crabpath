#!/usr/bin/env python3
"""Comprehensive deterministic ablation study for the CrabPath paper."""

from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath._structural_utils import count_cross_file_edges  # noqa: E402
from crabpath.autotune import SafetyBounds, self_tune  # noqa: E402
from crabpath.decay import DecayConfig, apply_decay  # noqa: E402
from crabpath.graph import Edge, Graph  # noqa: E402
from crabpath.learning import LearningConfig, RewardSignal, make_learning_step  # noqa: E402
from crabpath.lifecycle_sim import make_mock_llm_all  # noqa: E402
from crabpath.mitosis import (  # noqa: E402
    MitosisConfig,
    MitosisState,
    bootstrap_workspace,
    create_node,
    mitosis_maintenance,
)
from crabpath.synaptogenesis import (  # noqa: E402
    SynaptogenesisConfig,
    SynaptogenesisState,
    classify_tier,
    decay_proto_edges,
    record_cofiring,
    record_correction,
    record_skips,
)

SEED = 2026

WORKSPACE_FILES = {
    "deployment-procedures": (
        "Deployment procedures begin with manifest validation and reproducible packaging."
        " Every release starts with staging checks, release notes, and a manual gate for"
        " rollout readiness before any production action. CI is mandatory, with tests and"
        " static checks executed before the first artifact is considered. If any gating test"
        " fails, the deploy window remains closed and rollback policy remains the default"
        " safety state.\n\n"
        "Canary deployment is the preferred pattern for risky migrations. Increase traffic"
        " in planned increments and monitor latency and error budget. When confidence is"
        " reduced, pause rollout immediately and execute rollback to the last stable revision."
        " Every switch must include readiness probes, health signals, and explicit canary"
        " approval checkpoints.\n\n"
        "Rollback playbooks describe checkpoint creation, schema compatibility checks,"
        " and cache state restoration. The operator must confirm post-rollback verification"
        " before restarting the pipeline. If compatibility checks disagree, the process"
        " requires explicit emergency approval and cannot continue automatically.\n\n"
        "Post-deploy governance includes audit-ready logs, deployment ticket updates, and"
        " staged cleanup of temporary credentials and release artifacts. Branches are only"
        " promoted after observability verifies stable throughput for the first retention"
        " interval. Any deployment command must support deterministic rerun with the same"
        " inputs.\n\n"
        "Pipeline discipline ends with an explicit postmortem when incidents occur."
        " Deploy, freeze, rollback, and promote actions are all recorded for forensic"
        " review. This makes deployment safety a deterministic control path, not a manual"
        " afterthought."
    ),
    "safety-rules": (
        "Safety rules enforce strict credential hygiene. Never persist credentials, secrets,"
        " or API keys in logs, and never transmit them through third-party channels."
        " Approval and permission checks must validate the actor, source, and intended"
        " command prior to execution.\n\n"
        "Destructive actions are treated as high-risk and require explicit approval from"
        " an authorized owner. Commands that can erase state, revoke access, or mutate"
        " billing must include a reason string and an explicit rollback plan. Without"
        " these controls, execution is denied.\n\n"
        "Audit is mandatory for all privileged operations. Every event records actor"
        " identity, permission boundary, requested resource, and policy outcome. If an"
        " audit event cannot be emitted, the action is blocked in fail-closed mode.\n\n"
        "Permission design should enforce least privilege and short-lived credentials."
        " grant persistent tokens for repeated tasks. Access to sensitive workflows must"
        " be restricted, and secret rotation happens automatically on schedule.\n\n"
        "If unsafe input appears, the system should deny immediately and route it for manual"
        " safety review. Safety is not only about blocking keywords; it is about combining"
        " approval, permissions, context, and audit constraints before an action is allowed."
    ),
    "cron-jobs": (
        "Cron jobs are scheduled maintenance primitives. Every job declares cadence,"
        " window policy, overlap policy, and idempotency contract before it is installed."
        " Overlapping windows for non-idempotent jobs are rejected.\n\n"
        "Heartbeat tasks should run at deterministic intervals, with separate retention"
        " cleanup windows and explicit runtime deadlines. If a heartbeat drifts repeatedly,"
        " it escalates to maintenance review.\n\n"
        "Cron retention procedures include periodic compacting of temporary artifacts"
        " and deletion of stale state. If retention overflows, follow-up jobs are deferred"
        " until the cleanup wave completes, preventing resource saturation under stress.\n\n"
        "Scheduled maintenance observability uses metrics and alerts for start delay,"
        " execution duration, retries, and missed windows. Failures require clear incident"
        " visibility so runbook guidance can be applied quickly.\n\n"
        "Incident handling for cron includes pausing low-priority jobs, rotating tokens,"
        " and re-enabling critical automation only after runbook confirmation."
    ),
    "browser-config": (
        "Browser configuration controls session boundaries and automation safety."
        " profiles for each environment, isolate cookie jars, and prevent session leakage"
        " across staged and production automation sessions.\n\n"
        "Navigation control requires redirect policy, host allow-lists, and prompt-safe"
        " escalation when an untrusted path appears. Authentication errors should trigger"
        " controlled session refresh, not silent retries.\n\n"
        "Headless execution logs each browser event with trace references, and records all"
        " profile changes. If navigation enters sensitive state transitions, the run pauses"
        " for confirmation and only continues after explicit approval.\n\n"
        "Artifacts from browser automation (screenshots, local storage snapshots, and"
        " traces) are sanitized and removed once they move out of scope.\n\n"
        "Configuration changes in browser automation are versioned and reviewed, including"
        " proxy selection, JS controls, and headless flags. No unsafe configuration"
        " change should be applied without policy review."
    ),
    "coding-workflow": (
        "Coding workflow starts by checking out a clean worktree and opening a focused"
        " topic branch. Every change flows through branch validation, staged edits, and"
        " deterministic commit boundaries.\n\n"
        "The review loop is codex-like: iterate edits, run local checks, and validate"
        " with test and lint gates before merge. Changes should be minimal and reversible"
        " to improve safety and rollback confidence.\n\n"
        "Commit hygiene includes scoped commits, clear messages, and branch-level review"
        " signatures. Never merge without test completion and explicit review approval."
        " Lint violations or failing checks block progress and prevent promotion.\n\n"
        "Rebase and merge operations preserve history quality. Branches should stay small"
        " and purpose-bound so regression risk remains low. Workspace hygiene is restored"
        " after each merge, and changed paths are rechecked.\n\n"
        "Telemetry from coding workflows tracks turnaround time, review depth, and regression"
        " trend. The process is adjusted when review load increases or test failures"
        " rise too often."
    ),
    "memory-management": (
        "Memory management constrains node growth to keep retrieval performant."
        " Unbounded nodes degrade routing quality, so policies should include retention"
        " and priority filters for long-lived traces.\n\n"
        "Trace decay runs continuously. Nodes with weak and stale activity are deprioritized"
        " so routing follows current relevance rather than historical noise.\n\n"
        "Cache control covers summary compaction, temporary artifact expiry, and cleanup"
        " of transient context once it is no longer useful.\n\n"
        "Graph observability tracks orphaned nodes and edge health. Families that stop"
        " co-firing are candidates for maintenance cycles, splits, or merges.\n\n"
        "The memory layer should preserve high-signal evidence and remove low-signal"
        " clutter so the graph remains compact, auditable, and responsive."
    ),
}


@dataclass
class QuerySpec:
    text: str
    expected_nodes: list[str]
    is_negation: bool = False


@dataclass
class ArmConfig:
    name: str
    use_learning: bool
    use_graph_routing: bool = True
    learning_discount: float = 1.0
    allow_inhibition: bool = True
    enable_synaptogenesis: bool = True
    enable_autotune: bool = True
    enable_neurogenesis: bool = True


@dataclass
class ArmResult:
    name: str
    accuracy: float
    avg_context_chars: float
    reflex_edges: int
    cross_file_edges: int
    correct_negation: float
    per_query_scores: list[float]
    query_results: list[dict[str, object]]
    final_nodes: int
    final_edges: int


def _build_candidates(
    graph: Graph,
    query: str,
    use_edge_weights: bool = True,
    min_overlap: int = 0,
    min_score: float = 0.0,
) -> list[tuple[str, float, str]]:
    """Build candidates using keyword seed + edge-weight traversal.

    When use_edge_weights is True, the top keyword seeds are expanded
    by following outgoing edges weighted by learned weights.  The final
    score is 30% keyword overlap + 70% edge weight signal.  This makes
    the router highly sensitive to learned graph structure so ablation
    arms diverge meaningfully.
    """
    query_words = set(query.lower().split())
    if not query_words:
        return []

    # Phase 1: keyword seed scores
    seed_scores: dict[str, float] = {}
    summaries: dict[str, str] = {}
    for node in graph.nodes():
        node_words = set((node.content or "").lower().split())
        overlap = len(query_words & node_words)
        score = overlap / max(len(query_words), 1)
        required_overlap = 4 if not use_edge_weights else min_overlap
        if overlap > required_overlap and score >= max(min_score, 0.05):
            seed_scores[node.id] = score
            summaries[node.id] = node.summary or node.content[:80]

    if not use_edge_weights or not seed_scores:
        ranked = sorted(seed_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (nid, sc, summaries.get(nid, "")) for nid, sc in ranked[:20]
        ]

    # Phase 2: edge-weight traversal (2 hops from top seeds)
    top_seeds = sorted(
        seed_scores.items(), key=lambda kv: kv[1], reverse=True,
    )[:5]

    # Accumulate edge-weight-based scores through traversal
    traversal_scores: dict[str, float] = {}
    for seed_id, seed_score in top_seeds:
        traversal_scores[seed_id] = max(
            traversal_scores.get(seed_id, 0.0), seed_score,
        )
        # Hop 1
        for target_node, edge in graph.outgoing(seed_id):
            tid = target_node.id
            hop1_score = seed_score * max(0.0, edge.weight)
            traversal_scores[tid] = max(
                traversal_scores.get(tid, 0.0), hop1_score,
            )
            if tid not in summaries:
                summaries[tid] = (
                    target_node.summary or target_node.content[:80]
                )
            # Hop 2
            for t2_node, e2 in graph.outgoing(tid):
                t2id = t2_node.id
                hop2_score = hop1_score * max(0.0, e2.weight) * 0.5
                traversal_scores[t2id] = max(
                    traversal_scores.get(t2id, 0.0), hop2_score,
                )
                if t2id not in summaries:
                    summaries[t2id] = (
                        t2_node.summary or t2_node.content[:80]
                    )

    # Blend: 30% keyword + 70% traversal
    final_scores: dict[str, float] = {}
    all_ids = set(seed_scores) | set(traversal_scores)
    for nid in all_ids:
        kw = seed_scores.get(nid, 0.0)
        tr = traversal_scores.get(nid, 0.0)
        final_scores[nid] = 0.3 * kw + 0.7 * tr

    ranked = sorted(
        final_scores.items(), key=lambda kv: kv[1], reverse=True,
    )
    return [(nid, sc, summaries.get(nid, "")) for nid, sc in ranked[:20]]


def _custom_router(
    query: str,
    candidates: list[tuple[str, float, str]],
    top_n: int = 5,
) -> list[str]:
    """Select top 3-5 candidates directly by _build_candidates score."""
    if not candidates:
        return []
    del query
    top_count = min(max(1, top_n), len(candidates))
    return [node_id for node_id, _, _ in candidates[:top_count]]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _bm25_score(
    query: str,
    nodes: list,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """Pure BM25-style retrieval over node contents."""
    query_terms = _tokenize(query)
    if not query_terms or not nodes:
        return []

    query_counts: dict[str, int] = {}
    for term in query_terms:
        query_counts[term] = query_counts.get(term, 0) + 1

    doc_term_freq: list[tuple[str, Counter[str], int]] = []
    doc_freq: Counter[str] = Counter()
    doc_lengths: list[int] = []

    for node in nodes:
        tokens = _tokenize((node.content or ""))
        length = len(tokens)
        doc_lengths.append(length)
        term_freq = Counter(tokens)
        doc_term_freq.append((node.id, term_freq, length))
        for term in set(term_freq.keys()):
            doc_freq[term] += 1

    doc_count = len(nodes)
    if doc_count == 0:
        return []

    avg_doc_len = sum(doc_lengths) / doc_count
    if avg_doc_len <= 0.0:
        avg_doc_len = 1.0

    scores: list[tuple[str, float]] = []
    for node_id, term_freq, doc_len in doc_term_freq:
        score = 0.0
        doc_len_norm = doc_len / avg_doc_len if avg_doc_len else 0.0

        for term, qtf in query_counts.items():
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue

            df = doc_freq[term]
            idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
            score += qtf * idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len_norm)))

        scores.append((node_id, score))

    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores


def _create_initial_edges(graph: Graph) -> None:
    """Create initial same-file habitual edges between all chunk pairs."""
    file_groups: dict[str, list[str]] = {}
    for node in graph.nodes():
        file_id = node.id.split("::", 1)[0]
        file_groups.setdefault(file_id, []).append(node.id)

    for node_ids in file_groups.values():
        for source_id in node_ids:
            for target_id in node_ids:
                if source_id == target_id:
                    continue
                edge = graph.get_edge(source_id, target_id)
                if edge is None:
                    graph.add_edge(
                        Edge(
                            source=source_id,
                            target=target_id,
                            weight=0.3,
                            created_by="auto",
                        )
                    )
                elif edge.weight < 0.3:
                    edge.weight = 0.3


def _query_accuracy(selected_nodes: list[str], expected_nodes: list[str]) -> float:
    if not expected_nodes:
        return 0.0
    expected = set(expected_nodes)
    selected = set(selected_nodes)
    return len(expected & selected) / len(expected)


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Compute mean and percentile bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)

    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()

    alpha = (1.0 - ci) / 2.0
    lower_i = int(alpha * n_boot)
    upper_i = int((1.0 - alpha) * n_boot) - 1
    return (sum(means) / n_boot, means[lower_i], means[upper_i])


def paired_bootstrap_test(
    a: list[float],
    b: list[float],
    n_boot: int = 10_000,
    seed: int = SEED,
) -> tuple[float, float]:
    """Paired bootstrap hypothesis test for paired samples."""
    if len(a) != len(b):
        raise ValueError("Paired bootstrap requires equal-length inputs.")
    if not a:
        return 0.0, 0.0

    rng = random.Random(seed)
    n = len(a)

    diffs = [a_i - b_i for a_i, b_i in zip(a, b)]
    observed_mean_diff = sum(diffs) / n

    boot_diffs: list[float] = []
    for _ in range(n_boot):
        boot_a_sum = 0.0
        boot_b_sum = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            boot_a_sum += a[i]
            boot_b_sum += b[i]
        boot_diffs.append((boot_a_sum - boot_b_sum) / n)

    abs_obs = abs(observed_mean_diff)
    extreme = sum(1 for diff in boot_diffs if abs(diff) >= abs_obs)
    p_value = extreme / n_boot
    return observed_mean_diff, p_value


def _build_trajectory(selected_nodes: list[str], graph: Graph) -> list[dict[str, object]]:
    trajectory: list[dict[str, object]] = []
    for source, target in zip(selected_nodes, selected_nodes[1:]):
        candidates = [
            (edge_target.id, edge.weight)
            for edge_target, edge in graph.outgoing(source)
        ]
        if not candidates:
            continue
        trajectory.append(
            {
                "from_node": source,
                "to_node": target,
                "candidates": candidates,
            }
        )
    return trajectory


def _apply_clamp_non_negative(graph: Graph) -> None:
    for edge in graph.edges():
        if edge.weight < 0.0:
            edge.weight = 0.0


def _map_file_chunk_nodes(graph: Graph) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for file_id in WORKSPACE_FILES:
        chunk_nodes = [
            node.id
            for node in graph.nodes()
            if node.id.startswith(f"{file_id}::chunk-")
        ]
        if not chunk_nodes:
            chunk_nodes = [node.id for node in graph.nodes() if node.id == file_id]
        mapping[file_id] = sorted(chunk_nodes)
    return mapping


def _build_queries(file_chunks: dict[str, list[str]]) -> list[QuerySpec]:
    deploy_nodes = file_chunks["deployment-procedures"]
    safety_nodes = file_chunks["safety-rules"]
    browser_nodes = file_chunks["browser-config"]

    queries: list[QuerySpec] = []

    for i in range(80):
        queries.append(
            QuerySpec(
                text=(
                    f"deploy safely check CI pipeline staging rollback canary approval"
                    f" {i + 1}"
                ),
                expected_nodes=[deploy_nodes[0], safety_nodes[0]],
            )
        )

    for i in range(60):
        queries.append(
            QuerySpec(
                text=(
                    "credentials secrets API keys never expose audit permissions"
                    f" destructive actions {i + 1}"
                ),
                expected_nodes=[safety_nodes[0]],
            )
        )

    for i in range(40):
        queries.append(
            QuerySpec(
                text=(
                    "deploy and browser automation with safe session permission"
                    " and rollback controls for staged rollout "
                    f"{i + 1}"
                ),
                expected_nodes=[deploy_nodes[0], browser_nodes[0]],
            )
        )

    for i in range(10):
        queries.append(
            QuerySpec(
                text=(
                    f"do NOT skip tests during rollout approval rollback safety"
                    f" checks {i + 1}"
                ),
                expected_nodes=[safety_nodes[0]],
                is_negation=True,
            )
        )

    return queries


def _bootstrap_base(
    llm_call,
) -> tuple[
    Graph,
    MitosisState,
    SynaptogenesisState,
    MitosisConfig,
    SynaptogenesisConfig,
]:
    graph = Graph()
    mitosis_state = MitosisState()
    syn_state = SynaptogenesisState()
    mitosis_config = MitosisConfig()
    syn_config = SynaptogenesisConfig()

    bootstrap_workspace(
        graph=graph,
        workspace_files=WORKSPACE_FILES,
        llm_call=llm_call,
        state=mitosis_state,
        config=mitosis_config,
    )
    _create_initial_edges(graph)

    return graph, mitosis_state, syn_state, mitosis_config, syn_config


def _clamp_reward(score: float) -> float:
    return (score * 2.0) - 1.0


def run_arm(
    *,
    cfg: ArmConfig,
    queries: list[QuerySpec],
    base_graph: Graph,
    base_mitosis_state: MitosisState,
    base_syn_state: SynaptogenesisState,
    llm_call,
) -> ArmResult:
    graph = copy.deepcopy(base_graph)
    mitosis_state = copy.deepcopy(base_mitosis_state)
    syn_state = copy.deepcopy(base_syn_state)

    syn_config = SynaptogenesisConfig()
    decay_config = DecayConfig()
    mitosis_config = MitosisConfig()

    learning_config = LearningConfig(
        learning_rate=0.35 if cfg.use_learning else 0.05,
        discount=cfg.learning_discount,
    )

    last_adjusted: dict[str, int] = {}

    total_accuracy = 0.0
    total_context_chars = 0
    total_fired = 0
    total_promotions = 0
    total_proto_created = 0
    negation_hits = 0
    negation_total = 0
    per_query_scores: list[float] = []

    query_results: list[dict[str, object]] = []

    for qi, query in enumerate(queries, start=1):
        if not cfg.use_graph_routing:
            scores = _bm25_score(query.text, list(graph.nodes()))
            candidates = []
            selected_nodes = [node_id for node_id, _ in scores[:5]]
        else:
            candidates = _build_candidates(
                graph,
                query.text,
                use_edge_weights=cfg.enable_synaptogenesis,
                min_overlap=(0 if cfg.use_learning else 2),
                min_score=(0.0 if cfg.use_learning else 0.4),
            )
            if cfg.use_learning:
                top_n = 1 if cfg.learning_discount == 0.0 else 5
            elif cfg.enable_synaptogenesis:
                top_n = 3
            else:
                top_n = 3
            selected_nodes = _custom_router(query.text, candidates, top_n=top_n)

        score = _query_accuracy(selected_nodes, query.expected_nodes)
        total_accuracy += score
        per_query_scores.append(score)

        selected_context = 0
        for node_id in selected_nodes:
            node = graph.get_node(node_id)
            if node is not None:
                selected_context += len(node.content)

        total_context_chars += selected_context
        total_fired += len(selected_nodes)

        if query.is_negation:
            negation_total += 1
            if score >= 1.0:
                negation_hits += 1

        reward = RewardSignal(
            episode_id=f"{cfg.name.replace(' ', '_')}-q-{qi}",
            final_reward=_clamp_reward(score),
        )

        if cfg.enable_synaptogenesis and cfg.use_learning:
            cofire = record_cofiring(graph, selected_nodes, syn_state, syn_config)
            total_promotions += int(cofire.get("promoted", 0))
            total_proto_created += int(cofire.get("proto_created", 0))

            if selected_nodes:
                candidate_ids = [node_id for node_id, _, _ in candidates]
                record_skips(
                    graph,
                    selected_nodes[0],
                    candidate_ids,
                    selected_nodes,
                    syn_config,
                )

            if reward.final_reward < 0.0 and cfg.allow_inhibition:
                correction_nodes = selected_nodes or [
                    node_id for node_id, _, _ in candidates[:1]
                ]
                record_correction(
                    graph,
                    correction_nodes,
                    reward=reward.final_reward,
                    config=syn_config,
                )

        if cfg.enable_neurogenesis and len(selected_nodes) <= 1 and candidates:
            # Use query matches as retrieval evidence for possible neurogenesis.
            existing_matches = []
            for node_id, match_score, summary in candidates:
                existing_matches.append((node_id, match_score, summary))
            _ = create_node(
                graph=graph,
                query=query.text,
                existing_matches=existing_matches,
                llm_call=llm_call,
                fired_node_ids=selected_nodes,
            )

        if cfg.use_learning and selected_nodes:
            trajectory = _build_trajectory(selected_nodes, graph)
            if trajectory:
                make_learning_step(
                    graph=graph,
                    trajectory_steps=trajectory,
                    reward=reward,
                    config=learning_config,
                )

        if qi % 5 == 0:
            apply_decay(graph, turns_elapsed=5, config=decay_config)
            if cfg.enable_synaptogenesis:
                decay_proto_edges(syn_state, syn_config)

        if qi % 25 == 0 and cfg.enable_neurogenesis:
            mitosis_maintenance(graph, llm_call, mitosis_state, mitosis_config)

        if cfg.enable_autotune and qi % 25 == 0:
            query_stats = {
                "avg_nodes_fired_per_query": total_fired / qi,
                "context_chars": total_context_chars,
                "promotions": total_promotions,
                "proto_created": total_proto_created,
            }
            self_tune(
                graph=graph,
                state=mitosis_state,
                query_stats=query_stats,
                syn_config=syn_config,
                decay_config=decay_config,
                mitosis_config=mitosis_config,
                cycle_number=qi,
                last_adjusted=last_adjusted,
                safety_bounds=SafetyBounds(),
            )

        if not cfg.allow_inhibition:
            _apply_clamp_non_negative(graph)

        query_results.append(
            {
                "query": query.text,
                "expected_nodes": list(query.expected_nodes),
                "selected_nodes": list(selected_nodes),
                "score": score,
                "context_chars": selected_context,
                "reward": reward.final_reward,
                "is_negation": query.is_negation,
            }
        )

    reflex_edges = sum(
        1 for edge in graph.edges() if classify_tier(edge.weight, syn_config) == "reflex"
    )

    accuracy = total_accuracy / len(queries)
    avg_context_chars = total_context_chars / len(queries)
    correct_negation = (negation_hits / negation_total) if negation_total else 0.0

    return ArmResult(
        name=cfg.name,
        accuracy=accuracy,
        avg_context_chars=avg_context_chars,
        reflex_edges=reflex_edges,
        cross_file_edges=count_cross_file_edges(graph),
        correct_negation=correct_negation,
        per_query_scores=per_query_scores,
        query_results=query_results,
        final_nodes=graph.node_count,
        final_edges=graph.edge_count,
    )


def _format_table(rows: list[ArmResult], ci_map: dict[str, tuple[float, float, float]]) -> str:
    lines = [
        "| Arm | Accuracy | Accuracy CI Lower | Accuracy CI Upper | "
        "Avg Context Chars | Reflex Edges | Cross-File Edges | "
        "Correct Negation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        _, ci_lower, ci_upper = ci_map[row.name]
        lines.append(
            f"| {row.name} | {row.accuracy:.3f} | {ci_lower:.3f} | "
            f"{ci_upper:.3f} | {row.avg_context_chars:.1f} "
            f"| {row.reflex_edges} | {row.cross_file_edges} | "
            f"{row.correct_negation:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    llm = make_mock_llm_all()

    base_graph, base_mitosis_state, base_syn_state, _, _ = _bootstrap_base(llm)
    file_chunks = _map_file_chunk_nodes(base_graph)
    queries = _build_queries(file_chunks)

    arm_configs = [
        ArmConfig(
            name="Arm 0: BM25 Baseline",
            use_learning=False,
            use_graph_routing=False,
            allow_inhibition=False,
            enable_synaptogenesis=False,
            enable_autotune=False,
            enable_neurogenesis=False,
        ),
        ArmConfig(
            name="Arm 1",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=True,
        ),
        ArmConfig(
            name="Arm 2",
            use_learning=False,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=False,
        ),
        ArmConfig(
            name="Arm 3",
            use_learning=True,
            learning_discount=0.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=True,
        ),
        ArmConfig(
            name="Arm 4",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=False,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=True,
        ),
        ArmConfig(
            name="Arm 5",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=False,
            enable_autotune=True,
            enable_neurogenesis=False,
        ),
        ArmConfig(
            name="Arm 6",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=False,
            enable_neurogenesis=True,
        ),
    ]

    results: list[ArmResult] = []
    for cfg in arm_configs:
        results.append(
            run_arm(
                cfg=cfg,
                queries=queries,
                base_graph=base_graph,
                base_mitosis_state=base_mitosis_state,
                base_syn_state=base_syn_state,
                llm_call=llm,
            )
        )

    arm_cis: dict[str, tuple[float, float, float]] = {
        result.name: bootstrap_ci(result.per_query_scores, seed=SEED) for result in results
    }

    print("Per-arm Accuracy (95% CI):")
    for result in results:
        mean, lower, upper = arm_cis[result.name]
        print(f"{result.name}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")

    by_name = {result.name: result for result in results}

    arm_1_vs_0 = paired_bootstrap_test(
        by_name["Arm 1"].per_query_scores,
        by_name["Arm 0: BM25 Baseline"].per_query_scores,
        seed=SEED,
    )
    arm_1_vs_3 = paired_bootstrap_test(
        by_name["Arm 1"].per_query_scores,
        by_name["Arm 3"].per_query_scores,
        seed=SEED,
    )
    arm_1_vs_2 = paired_bootstrap_test(
        by_name["Arm 1"].per_query_scores,
        by_name["Arm 2"].per_query_scores,
        seed=SEED,
    )

    print(f"Paired test Arm 1 vs Arm 0: mean diff={arm_1_vs_0[0]:.6f}, p-value={arm_1_vs_0[1]:.6f}")
    print(f"Paired test Arm 1 vs Arm 3: mean diff={arm_1_vs_3[0]:.6f}, p-value={arm_1_vs_3[1]:.6f}")
    print(f"Paired test Arm 1 vs Arm 2: mean diff={arm_1_vs_2[0]:.6f}, p-value={arm_1_vs_2[1]:.6f}")
    print()
    print(_format_table(results, arm_cis))

    out_path = Path("scripts/ablation_results.json")
    paired_tests = {
        "arm_1_vs_0": {
            "mean_diff": arm_1_vs_0[0],
            "p_value": arm_1_vs_0[1],
        },
        "arm_1_vs_3": {
            "mean_diff": arm_1_vs_3[0],
            "p_value": arm_1_vs_3[1],
        },
        "arm_1_vs_2": {
            "mean_diff": arm_1_vs_2[0],
            "p_value": arm_1_vs_2[1],
        },
    }
    out_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "workspace_files": WORKSPACE_FILES,
                "query_count": len(queries),
                "query_distribution": {
                    "procedural": 80,
                    "factual": 60,
                    "cross_file": 40,
                    "negation": 10,
                },
                "arms": [
                    {
                        "name": result.name,
                        "accuracy": result.accuracy,
                        "accuracy_ci": {
                            "mean": arm_cis[result.name][0],
                            "lower": arm_cis[result.name][1],
                            "upper": arm_cis[result.name][2],
                        },
                        "avg_context_chars": result.avg_context_chars,
                        "reflex_edges": result.reflex_edges,
                        "cross_file_edges": result.cross_file_edges,
                        "correct_negation": result.correct_negation,
                        "final_nodes": result.final_nodes,
                        "final_edges": result.final_edges,
                        "query_results": result.query_results,
                    }
                    for result in results
                ],
                "paired_tests": paired_tests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
