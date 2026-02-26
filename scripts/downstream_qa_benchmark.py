#!/usr/bin/env python3
"""Downstream multi-hop QA benchmark with robustness diagnostics for CrabPath."""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.decay import DecayConfig, apply_decay  # noqa: E402
from crabpath.embeddings import EmbeddingIndex  # noqa: E402
from crabpath.learning import LearningConfig, RewardSignal, make_learning_step  # noqa: E402
from crabpath.lifecycle_sim import make_mock_llm_all  # noqa: E402
from crabpath.mitosis import (  # noqa: E402
    MitosisConfig,
    MitosisState,
    bootstrap_workspace,
    mitosis_maintenance,
)
from crabpath.mitosis import create_node as create_mitosis_node  # noqa: E402
from crabpath.synaptogenesis import (  # noqa: E402
    SynaptogenesisConfig,
    SynaptogenesisState,
    decay_proto_edges,
    record_cofiring,
    record_correction,
    record_skips,
)
from scripts import ablation_study as ablation  # noqa: E402

SEED = ablation.SEED
TOP_K_DEFAULT = 5
K_VALUES = [1, 2, 3, 5]
RESULTS_PATH = Path("scripts/downstream_qa_benchmark_results.json")


@dataclass
class QAPair:
    question: str
    gold_answer: str
    supporting_docs: list[str]
    answer_type: str
    paraphrases: list[str] = field(default_factory=list)


@dataclass
class QueryRun:
    question: str
    gold_answer: str
    answer_type: str
    supporting_docs: list[str]
    selected_node_ids: list[str]
    selected_docs: list[str]
    recall: float
    em: float
    f1: float
    tokens_loaded: int


@dataclass
class MethodRun:
    method: str
    k: int | None
    avg_recall: float
    avg_em: float
    avg_f1: float
    avg_tokens: float
    false_positive_rate: float | None
    query_results: list[QueryRun]


@dataclass
class ParetoPoint:
    method: str
    k: int
    accuracy: float
    tokens_per_query: float


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)


def _tokenize(text: str) -> list[str]:
    return ablation._tokenize(text)


def _token_count(text: str) -> int:
    return len(_tokenize(text))


def _normalize_for_match(text: str) -> str:
    return " ".join(_tokenize(text))


def _exact_match(predicted: str, reference: str) -> float:
    predicted_norm = _normalize_for_match(predicted)
    reference_norm = _normalize_for_match(reference)
    if not predicted_norm:
        return 0.0
    return 1.0 if predicted_norm in reference_norm else 0.0


def _f1_score(predicted: str, reference: str) -> float:
    pred = _tokenize(predicted)
    ref = _tokenize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _make_word_overlap_embedder(graph) -> Callable[[list[str]], list[list[float]]]:
    vocab: list[str] = []
    seen: set[str] = set()
    for node in graph.nodes():
        text = f"{node.summary or ''} {node.content}"
        for token in _tokenize(text):
            if token not in seen:
                seen.add(token)
                vocab.append(token)

    vocab_index = {token: i for i, token in enumerate(vocab)}
    vocab_size = len(vocab)

    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * vocab_size
            if not text:
                vectors.append(vector)
                continue
            counts = Counter(_tokenize(text))
            for token, count in counts.items():
                idx = vocab_index.get(token)
                if idx is not None:
                    vector[idx] = float(count)
            vectors.append(vector)
        return vectors

    return embed


def _bootstrap_with_workspace(
    workspace_files: dict[str, str],
    llm_call,
):
    graph = ablation.Graph()
    mitosis_state = MitosisState()
    syn_state = SynaptogenesisState()
    mitosis_config = MitosisConfig()
    syn_config = SynaptogenesisConfig()

    bootstrap_workspace(
        graph=graph,
        workspace_files=workspace_files,
        llm_call=llm_call,
        state=mitosis_state,
        config=mitosis_config,
    )
    ablation._create_initial_edges(graph)

    return graph, mitosis_state, syn_state, mitosis_config, syn_config


def _file_to_nodes(graph, workspace_files: dict[str, str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for file_id in workspace_files:
        chunk_nodes = [
            node.id
            for node in graph.nodes()
            if node.id.startswith(f"{file_id}::chunk-")
        ]
        if not chunk_nodes:
            chunk_nodes = [node.id for node in graph.nodes() if node.id == file_id]
        mapping[file_id] = sorted(chunk_nodes)
    return mapping


def _document_contents(graph, workspace_files: dict[str, str]) -> dict[str, str]:
    mapping = _file_to_nodes(graph, workspace_files)
    doc_contents: dict[str, str] = {}
    for file_id, node_ids in mapping.items():
        chunks = [graph.get_node(node_id).content for node_id in node_ids if graph.get_node(node_id)]
        if chunks:
            doc_contents[file_id] = " ".join(chunks)
        else:
            doc_contents[file_id] = ""
    return doc_contents


def _nodes_to_docs(selected_nodes: list[str]) -> list[str]:
    seen: set[str] = set()
    docs: list[str] = []
    for node_id in selected_nodes:
        doc_id = node_id.split("::", 1)[0]
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def _evaluate_selected(
    qa: QAPair,
    selected_nodes: list[str],
    doc_contents: dict[str, str],
) -> QueryRun:
    selected_docs = _nodes_to_docs(selected_nodes)
    supporting_docs = list(qa.supporting_docs)
    if supporting_docs:
        recall = len(set(selected_docs) & set(supporting_docs)) / len(supporting_docs)
    else:
        recall = 0.0

    selected_context = " ".join(
        doc_contents.get(doc_id, "") for doc_id in selected_docs
    )
    em = _exact_match(qa.gold_answer, selected_context)
    f1 = _f1_score(qa.gold_answer, selected_context)
    tokens = sum(_token_count(doc_contents.get(doc_id, "")) for doc_id in selected_docs)
    return QueryRun(
        question=qa.question,
        gold_answer=qa.gold_answer,
        answer_type=qa.answer_type,
        supporting_docs=list(supporting_docs),
        selected_node_ids=list(selected_nodes),
        selected_docs=selected_docs,
        recall=recall,
        em=em,
        f1=f1,
        tokens_loaded=tokens,
    )


def _run_static_method(
    qa_pairs: list[QAPair],
    graph,
    doc_contents: dict[str, str],
    workspace_files: dict[str, str],
    k: int,
    method_name: str,
) -> MethodRun:
    _ = workspace_files
    file_nodes = _file_to_nodes(graph, doc_contents)
    selected_node_ids = [nodes[0] for nodes in file_nodes.values() if nodes]
    selected_nodes_by_query = selected_node_ids
    query_results: list[QueryRun] = [
        _evaluate_selected(qa, selected_nodes_by_query, doc_contents) for qa in qa_pairs
    ]
    if not query_results:
        return MethodRun(
            method=method_name,
            k=k,
            avg_recall=0.0,
            avg_em=0.0,
            avg_f1=0.0,
            avg_tokens=0.0,
            false_positive_rate=0.0,
            query_results=[],
        )
    return MethodRun(
        method=method_name,
        k=k,
        avg_recall=sum(q.recall for q in query_results) / len(query_results),
        avg_em=sum(q.em for q in query_results) / len(query_results),
        avg_f1=sum(q.f1 for q in query_results) / len(query_results),
        avg_tokens=sum(q.tokens_loaded for q in query_results) / len(query_results),
        false_positive_rate=0.0,
        query_results=query_results,
    )


def _run_bm25_method(
    qa_pairs: list[QAPair],
    graph,
    k: int,
    method_name: str,
    doc_contents: dict[str, str],
) -> MethodRun:
    nodes = list(graph.nodes())
    query_results: list[QueryRun] = []
    for qa in qa_pairs:
        scores = ablation._bm25_score(qa.question, nodes)
        selected_nodes = [node_id for node_id, _ in scores[:k]]
        query_results.append(_evaluate_selected(qa, selected_nodes, doc_contents))

    if not query_results:
        return MethodRun(method_name, k, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    return MethodRun(
        method=method_name,
        k=k,
        avg_recall=sum(q.recall for q in query_results) / len(query_results),
        avg_em=sum(q.em for q in query_results) / len(query_results),
        avg_f1=sum(q.f1 for q in query_results) / len(query_results),
        avg_tokens=sum(q.tokens_loaded for q in query_results) / len(query_results),
        false_positive_rate=None,
        query_results=query_results,
    )


def _run_dense_method(
    qa_pairs: list[QAPair],
    graph,
    k: int,
    method_name: str,
    doc_contents: dict[str, str],
    embedder: Callable[[list[str]], list[list[float]]],
) -> MethodRun:
    index = EmbeddingIndex()
    index.build(graph, embed_fn=embedder)
    query_results: list[QueryRun] = []
    for qa in qa_pairs:
        scores = index.raw_scores(qa.question, embed_fn=embedder, top_k=k)
        selected_nodes = [node_id for node_id, _ in scores[:k]]
        query_results.append(_evaluate_selected(qa, selected_nodes, doc_contents))

    if not query_results:
        return MethodRun(method_name, k, 0.0, 0.0, 0.0, 0.0, None, [])

    return MethodRun(
        method=method_name,
        k=k,
        avg_recall=sum(q.recall for q in query_results) / len(query_results),
        avg_em=sum(q.em for q in query_results) / len(query_results),
        avg_f1=sum(q.f1 for q in query_results) / len(query_results),
        avg_tokens=sum(q.tokens_loaded for q in query_results) / len(query_results),
        false_positive_rate=None,
        query_results=query_results,
    )


def _run_crabpath_method(
    qa_pairs: list[QAPair],
    base_graph,
    base_mitosis_state,
    base_syn_state,
    llm_call,
    method_name: str,
    top_k: int,
    allow_inhibition: bool,
    use_learning: bool = True,
    enable_neurogenesis: bool = True,
    enable_synaptogenesis: bool = True,
    use_autotune: bool = False,
    learning_discount: float = 1.0,
) -> MethodRun:
    graph = copy.deepcopy(base_graph)
    mitosis_state = copy.deepcopy(base_mitosis_state)
    syn_state = copy.deepcopy(base_syn_state)
    syn_config = SynaptogenesisConfig()
    decay_config = DecayConfig()
    mitosis_config = MitosisConfig()
    learning_config = LearningConfig(learning_rate=0.35 if use_learning else 0.05, discount=learning_discount)
    last_adjusted: dict[str, int] = {}

    # precompute once; doc map updates implicitly through node content edits only.
    doc_contents = _document_contents(graph, _current_workspace_file_ids(graph))

    selected_results: list[QueryRun] = []

    for qi, qa in enumerate(qa_pairs, start=1):
        candidates = ablation._build_candidates(
            graph,
            qa.question,
            use_edge_weights=enable_synaptogenesis,
            min_overlap=(0 if use_learning else 2),
            min_score=(0.0 if use_learning else 0.4),
        )
        top_count = min(top_k, len(candidates)) if candidates else 0
        selected_nodes = ablation._custom_router(qa.question, candidates, top_n=top_count)

        run = _evaluate_selected(qa, selected_nodes, doc_contents)
        selected_results.append(run)

        score = run.recall
        reward = RewardSignal(
            episode_id=f"{method_name.replace(' ', '_')}-q-{qi}",
            final_reward=ablation._clamp_reward(score),
        )

        if use_learning and selected_nodes:
            trajectory = [
                {
                    "from_node": src,
                    "to_node": tgt,
                    "candidates": [
                        (edge_target.id, edge.weight)
                        for edge_target, edge in graph.outgoing(src)
                    ],
                }
                for src, tgt in zip(selected_nodes, selected_nodes[1:])
            ]
            if trajectory:
                make_learning_step(
                    graph=graph,
                    trajectory_steps=trajectory,
                    reward=reward,
                    config=learning_config,
                )

        if enable_synaptogenesis and use_learning:
            cofire = record_cofiring(graph, selected_nodes, syn_state, syn_config)
            if selected_nodes:
                candidate_ids = [node_id for node_id, _, _ in candidates]
                record_skips(graph, selected_nodes[0], candidate_ids, selected_nodes, syn_config)
            if reward.final_reward < 0.0 and allow_inhibition:
                correction_nodes = selected_nodes or [node_id for node_id, _, _ in candidates[:1]]
                if correction_nodes:
                    record_correction(graph, correction_nodes, reward=reward.final_reward, config=syn_config)

        if enable_neurogenesis and len(selected_nodes) <= 1 and candidates:
            existing_matches = [(node_id, match_score, summary) for node_id, match_score, summary in candidates]
            created = create_mitosis_node(
                graph=graph,
                query=qa.question,
                existing_matches=existing_matches,
                llm_call=llm_call,
                fired_node_ids=selected_nodes,
            )
            if created is not None and created.id not in list(graph._nodes.keys()):
                # ensure helper functions still see a stable graph and doc mapping
                pass

        if use_learning and qi % 5 == 0:
            apply_decay(graph, turns_elapsed=5, config=decay_config)
            if enable_synaptogenesis:
                decay_proto_edges(syn_state, syn_config)

        if use_learning and qi % 25 == 0 and enable_neurogenesis:
            mitosis_maintenance(graph, llm_call, mitosis_state, mitosis_config)

        if use_learning and use_autotune and qi % 25 == 0:
            query_stats = {
                "avg_nodes_fired_per_query": max(1, qi),
                "context_chars": 0,
                "promotions": 0,
                "proto_created": 0,
            }
            self_tune_res = None
            self_tune(
                graph=graph,
                state=mitosis_state,
                query_stats=query_stats,
                syn_config=syn_config,
                decay_config=decay_config,
                mitosis_config=mitosis_config,
                cycle_number=qi,
                last_adjusted=last_adjusted,
                safety_bounds=ablation.SafetyBounds(),
            )
            _ = self_tune_res

        if not allow_inhibition:
            for edge in graph.edges():
                if edge.weight < 0.0:
                    edge.weight = 0.0

        doc_contents = _document_contents(graph, _current_workspace_file_ids(graph))

    if not selected_results:
        return MethodRun(method_name, top_k, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    false_positive = None
    return MethodRun(
        method=method_name,
        k=top_k,
        avg_recall=sum(q.recall for q in selected_results) / len(selected_results),
        avg_em=sum(q.em for q in selected_results) / len(selected_results),
        avg_f1=sum(q.f1 for q in selected_results) / len(selected_results),
        avg_tokens=sum(q.tokens_loaded for q in selected_results) / len(selected_results),
        false_positive_rate=false_positive,
        query_results=selected_results,
    )


def _current_workspace_file_ids(graph) -> dict[str, str]:
    files: dict[str, str] = {}
    for node in graph.nodes():
        file_id = node.id.split("::", 1)[0]
        if file_id not in files:
            files[file_id] = file_id
    return files


def _build_qa_pairs() -> list[QAPair]:
    docs = list(ablation.WORKSPACE_FILES.keys())
    bridge_defs: list[tuple[str, str, list[str]]] = [
        (
            "What checks and gates are required before starting a risky rollout with canary progression?",
            "Canary deployment is the preferred pattern for risky migrations.",
            [docs[0], docs[1]],
        ),
        (
            "When confidence drops during a staged rollout, which fallback action is required and why?",
            "when confidence is reduced, pause rollout immediately and execute rollback to the last stable revision",
            [docs[0], docs[1]],
        ),
        (
            "What approval behavior is required for commands that can erase state in production automation?",
            "Destructive actions are treated as high-risk and require explicit approval from an authorized owner",
            [docs[1], docs[2]],
        ),
        (
            "Before running cron operations, what contract must each job declare and what is blocked for risky jobs?",
            "every job declares cadence, window policy, overlap policy, and idempotency contract",
            [docs[2], docs[0]],
        ),
        (
            "How is browser session isolation used when automation moves between staged and production flows?",
            "isolate cookie jars and prevent session leakage",
            [docs[3], docs[1]],
        ),
        (
            "How should rollback playbooks coordinate with release governance after deployment?",
            "Post-deploy governance includes audit-ready logs, deployment ticket updates",
            [docs[0], docs[1]],
        ),
        (
            "What controls protect approvals for credential-risky operations across environments?",
            "Approval and permission checks must validate the actor, source, and intended command",
            [docs[1], docs[4]],
        ),
        (
            "What must happen when overlap checks indicate non-idempotent cron conflicts during maintenance?",
            "Overlapping windows for non-idempotent jobs are rejected",
            [docs[2], docs[5]],
        ),
        (
            "Which review gates should precede a browser-triggered release workflow under suspicious redirects?",
            "Navigation control requires host allow-lists and escalation when an untrusted path appears",
            [docs[3], docs[1]],
        ),
        (
            "What should be preserved for canary incidents to support safe recovery after production changes?",
            "Postmortem process and deployment ticket updates are recorded for forensic review",
            [docs[0], docs[4]],
        ),
        (
            "How should temporary artifacts be treated after browser automation sessions complete?",
            "Artifacts are sanitized and removed once they move out of scope",
            [docs[3], docs[5]],
        ),
        (
            "What controls enforce safety before destructive operations when audit trails fail?",
            "If an audit event cannot be emitted, the action is blocked in fail-closed mode",
            [docs[1], docs[4]],
        ),
        (
            "How do scheduling windows interact with CI and deploy observability in high-risk incidents?",
            "Scheduled maintenance observability uses metrics and alerts for missed windows",
            [docs[2], docs[0]],
        ),
        (
            "How should deterministic reruns be guaranteed for deployment commands?",
            "Any deployment command must support the deterministic rerun with the same inputs",
            [docs[0], docs[4]],
        ),
        (
            "Which safety boundary is required before mutating privileged resources with profile changes?",
            "Least privilege and short-lived credentials should be enforced",
            [docs[1], docs[3]],
        ),
        (
            "What profile change conditions should gate automation transitions into sensitive areas?",
            "If navigation enters sensitive state transitions, the run pauses for confirmation",
            [docs[3], docs[1]],
        ),
        (
            "How should stale artifacts be handled during incident handling after a botched deployment?",
            "temporary artifacts and retention cleanup windows are cleaned up before continuing",
            [docs[2], docs[0]],
        ),
        (
            "What is required before a deployment can be promoted after readiness checks pass?",
            "Branches are only promoted after observability verifies stable throughput",
            [docs[0], docs[4]],
        ),
        (
            "How are rollback prerequisites tied to credential and manifest validation?",
            "Rollback playbooks describe checkpoint creation and schema compatibility checks",
            [docs[0], docs[1]],
        ),
        (
            "How is trace evidence managed for privileged operations during release preparation?",
            "Audit is mandatory for all privileged operations and records operator identity",
            [docs[1], docs[5]],
        ),
        (
            "How should stale cron debt be handled after repeated heartbeat drift?",
            "Repeated drift escalates to maintenance review",
            [docs[2], docs[5]],
        ),
    ]

    comparison_defs: list[tuple[str, str, list[str]]] = [
        (
            "Compare deployment rollbacks and cron overlaps: which one is immediate versus deferred?",
            "Deployment rollbacks can be immediate, while cron overlaps are deferred via retention controls",
            [docs[0], docs[2]],
        ),
        (
            "Compare browser profile changes versus safety permission checks in sensitive transitions.",
            "Browser profile changes require confirmation and are versioned, while safety actions require explicit approval",
            [docs[3], docs[1]],
        ),
        (
            "Compare coding workflow commits and production gatekeeping with respect to test gates.",
            "Both require explicit review, but deployment adds manual rollout readiness checks",
            [docs[4], docs[0]],
        ),
        (
            "Compare audit requirements in safety rules and coding workflow changes.",
            "Safety emits actor-level audit logs, while coding workflow requires explicit merge review",
            [docs[1], docs[4]],
        ),
        (
            "Compare memory decay and cron heartbeat behavior when stale activity accumulates.",
            "Memory trace decay runs continuously, while heartbeat misses trigger maintenance review",
            [docs[5], docs[2]],
        ),
        (
            "Compare canary gating with full production rollout for risky change control.",
            "Canary uses staged traffic increments, while direct rollout is a full switch path",
            [docs[0], docs[2]],
        ),
        (
            "Compare browser artifacts and deployment artifact cleanup behavior.",
            "Browser artifacts are sanitized, while deployment artifacts have cleanup and staging gates",
            [docs[3], docs[0]],
        ),
        (
            "Compare permissions for destructive actions and coding reviews.",
            "Destructive actions need explicit approvals, while coding edits need staged commits and review",
            [docs[1], docs[4]],
        ),
        (
            "Compare rollback behavior for schema drift and cache state restoration.",
            "Both rely on checkpoints, but deploy rollbacks explicitly restore cache state",
            [docs[0], docs[2]],
        ),
        (
            "Compare observability requirements for cron jobs and browser automation.",
            "Cron tracks missed windows while browser logs trace references and profile changes",
            [docs[2], docs[3]],
        ),
        (
            "Compare security of secrets and memory of high-signal context.",
            "Secrets must never be persisted, while memory should keep high-signal evidence and discard low-signal",
            [docs[1], docs[5]],
        ),
        (
            "Compare staging readiness and merge readiness in deployment and coding workflows.",
            "Both require staged checks, but coding also requires explicit review approval",
            [docs[0], docs[4]],
        ),
        (
            "Compare retention cleanup in cron jobs versus deploy release cleanup.",
            "Cron performs periodic artifact compaction while deployments cleanup temporary credentials",
            [docs[2], docs[0]],
        ),
        (
            "Compare audit scope for browser profile changes and privileged actions.",
            "Both are reviewed, but privileged actions block when audit cannot emit",
            [docs[3], docs[1]],
        ),
        (
            "Compare scheduling risks for non-idempotent work and safety-critical approvals.",
            "Cron blocks risky overlaps, while safety flags high-risk actions for explicit owner approval",
            [docs[2], docs[1]],
        ),
        (
            "Compare fallback behavior after canary pauses versus cron incidents.",
            "Canary pauses trigger rollback while cron pauses low-priority jobs",
            [docs[0], docs[2]],
        ),
        (
            "Compare context retention between memory management and deployment audit tickets.",
            "Both preserve relevant evidence, but deployment also updates deployment tickets for review",
            [docs[5], docs[0]],
        ),
        (
            "Compare explicit permission model and commit hygiene around privileged changes.",
            "Permission is least-privilege and short-lived, while commit hygiene requires clear messages and scoped changes",
            [docs[1], docs[4]],
        ),
    ]

    yesno_defs: list[tuple[str, str, list[str]]] = [
        (
            "Do canary deployments require safety checkpoints and manual approvals before production action?",
            "Yes, deployment gates include staging checks and manual gate checks",
            [docs[0], docs[1]],
        ),
        (
            "Do non-idempotent cron overlaps pass for risky jobs?",
            "No, overlapping windows for non-idempotent jobs are rejected",
            [docs[2], docs[1]],
        ),
        (
            "Can privileged actions proceed when audit emission fails?",
            "No, the action is blocked in fail-closed mode",
            [docs[1], docs[0]],
        ),
        (
            "Should browser configuration changes be reviewed before enabling execution?",
            "Yes, no unsafe configuration change should be applied without policy review",
            [docs[3], docs[1]],
        ),
        (
            "Is merge allowed after incomplete lint and test checks?",
            "No, lint violations or failing checks block progress and prevent promotion",
            [docs[4], docs[0]],
        ),
        (
            "Do stale context traces remain indefinitely in memory?",
            "No, trace decay runs continuously and deprioritizes stale activity",
            [docs[5], docs[2]],
        ),
        (
            "Can sensitive inputs be executed without safety review?",
            "No, unsafe input should be denied and routed for manual safety review",
            [docs[1], docs[3]],
        ),
        (
            "Do temporary credentials persist for repeated tasks?",
            "No, permission design enforces short-lived credentials",
            [docs[1], docs[4]],
        ),
        (
            "Should canary rollout include traffic increments and monitoring?",
            "Yes, rollout should increase traffic in planned increments and monitor latency and error budget",
            [docs[0], docs[2]],
        ),
        (
            "Are manual gate checks required before any production action?",
            "Yes, deployment starts with staging checks and manual readiness gates",
            [docs[0], docs[1]],
        ),
        (
            "Can destructive actions execute with implied permission?",
            "No, destructive actions require explicit authorization",
            [docs[1], docs[4]],
        ),
        (
            "Can sensitive automation runs continue without trace records?",
            "No, headless execution logs each browser event and trace references",
            [docs[3], docs[5]],
        ),
        (
            "Do low-signal traces stay in the graph forever?",
            "No, low-signal clutter should be compacted and expired",
            [docs[5], docs[4]],
        ),
        (
            "Should branch promotion happen while observability is still unstable?",
            "No, promotion occurs only after stable throughput is observed",
            [docs[0], docs[2]],
        ),
        (
            "Can manual approval checkpoints be skipped during canary rollout?",
            "No, canary rollouts rely on explicit checkpoints",
            [docs[0], docs[1]],
        ),
        (
            "Should session profiles leak between staging and production automation?",
            "No, sessions are isolated to prevent leakage",
            [docs[3], docs[4]],
        ),
    ]

    pairs: list[QAPair] = []

    def _add_block(
        block: list[tuple[str, str, list[str]]],
        answer_type: str,
    ) -> None:
        for i, (question, answer, support) in enumerate(block, start=1):
            variant = question
            if len(support) < 2:
                support = support + [docs[(i + 1) % len(docs)]]
            pairs.append(
                QAPair(
                    question=f"{variant}",
                    gold_answer=answer,
                    supporting_docs=support,
                    answer_type=answer_type,
                )
            )

    _add_block(bridge_defs, "bridge")
    _add_block(comparison_defs, "comparison")
    _add_block(yesno_defs, "yesno")

    while len(pairs) < 50:
        base = bridge_defs[len(pairs) % len(bridge_defs)]
        pairs.append(
            QAPair(
                question=f"{base[0]} (scenario {len(pairs) + 1})",
                gold_answer=base[1],
                supporting_docs=base[2],
                answer_type="bridge",
            )
        )

    # Keep the first 50 in a deterministic order.
    return pairs[:50]


def _add_paraphrases(qa_pairs: list[QAPair]) -> list[QAPair]:
    out: list[QAPair] = []
    for qa in qa_pairs:
        q = qa.question
        p1 = q.replace("What", "Which").replace("what", "which")
        p2 = q.replace("should", "must").replace("Should", "Must")
        p3 = f"Could you confirm: {q}"
        qa.paraphrases = [p1, p2, p3]
        out.append(qa)
    return out


def _make_distractor_workspace() -> dict[str, str]:
    workspace = dict(ablation.WORKSPACE_FILES)
    distractors = {
        "distractor-knowledge-001": (
            "Observability dashboards for marketing campaigns prefer daily social sentiment checks. "
            "The team uses engagement heatmaps and influencer indexes for campaign pacing."
        ),
        "distractor-knowledge-002": (
            "Incident triage for retail operations depends on shipping ETA, supplier quality, and store staffing levels. "
            "None of these are used in deployment safety controls."
        ),
        "distractor-knowledge-003": (
            "Design guidelines require color pair contrast checks for print materials and annual ad refresh cadence."
        ),
        "distractor-knowledge-004": (
            "Finance reconciliation for travel reimbursements is handled through quarterly spreadsheet exports and manual signatures."
        ),
        "distractor-knowledge-005": (
            "The office onboarding playbook focuses on badge printing, seat allocation, and cafeteria reservations."
        ),
        "distractor-knowledge-006": (
            "Customer support tooling standardizes chat templates and proactive outreach windows during public holidays."
        ),
        "distractor-knowledge-007": (
            "Mobile push campaigns must be synchronized with social calendars and influencer collaborations."
        ),
        "distractor-knowledge-008": (
            "Library inventory systems schedule duplicate checks with barcode audits and shelf-count reconciliation."
        ),
        "distractor-knowledge-009": (
            "HR policy documents define PTO rollover windows and employee recognition events."
        ),
        "distractor-knowledge-010": (
            "Facility management rotates conference rooms based on attendance patterns and printer consumable usage."
        ),
    }
    workspace.update(distractors)
    return workspace


def _false_positive_rate_for_results(
    query_results: list[QueryRun],
    distractor_ids: set[str],
) -> float:
    selected_total = 0
    selected_distractor = 0
    for result in query_results:
        for doc in result.selected_docs:
            selected_total += 1
            if doc in distractor_ids:
                selected_distractor += 1
    if selected_total <= 0:
        return 0.0
    return selected_distractor / selected_total


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _query_jaccard_stability(
    query: QAPair,
    runner: Callable[[str], list[str]],
) -> dict[str, float]:
    sets = []
    for text in query.paraphrases:
        retrieved = set(runner(text))
        sets.append(retrieved)

    pair_scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            pair_scores.append(_jaccard(sets[i], sets[j]))
    return {
        "mean": sum(pair_scores) / len(pair_scores) if pair_scores else 0.0,
        "min": min(pair_scores) if pair_scores else 0.0,
        "max": max(pair_scores) if pair_scores else 0.0,
    }


def _apply_temporal_updates(graph, updates: dict[str, str]) -> None:
    mapping = {file_id: [] for file_id in updates.keys()}
    for node in graph.nodes():
        file_id = node.id.split("::", 1)[0]
        if file_id in mapping and node.id not in mapping[file_id]:
            mapping[file_id].append(node.id)
    for file_id, updated_text in updates.items():
        node_ids = mapping.get(file_id)
        if node_ids:
            for node_id in node_ids:
                node = graph.get_node(node_id)
                if node is not None:
                    node.content = updated_text
                    node.summary = updated_text[:80]
        else:
            # If the node does not exist in this graph snapshot.
            if file_id in graph._nodes:
                node = graph.get_node(file_id)
                node.content = updated_text
                node.summary = updated_text[:80]


def _update_gold_answers_for_drift(qa_pairs: list[QAPair]) -> list[QAPair]:
    out: list[QAPair] = []
    for qa in qa_pairs:
        lowered = qa.question.lower()
        if "canary" in lowered:
            answer = "Canary deployments are deprecated for risky migrations."
            support = list(qa.supporting_docs)
            if "deployment-procedures" not in support:
                support.append("deployment-procedures")
            out.append(
                QAPair(
                    question=qa.question,
                    gold_answer=answer,
                    supporting_docs=support,
                    answer_type=qa.answer_type,
                )
            )
        elif "destructive" in lowered or "approval" in lowered:
            answer = "Emergency destructive actions may proceed with deferred audit review."
            support = list(qa.supporting_docs)
            if "safety-rules" not in support:
                support.append("safety-rules")
            out.append(
                QAPair(
                    question=qa.question,
                    gold_answer=answer,
                    supporting_docs=support,
                    answer_type=qa.answer_type,
                )
            )
        else:
            out.append(qa)
    return out


def _run_temporal_drift(
    qa_pairs: list[QAPair],
    base_graph,
    base_mitosis_state,
    base_syn_state,
    workspace_files: dict[str, str],
    llm_call,
    method_name: str,
    top_k: int,
    mode: str,
) -> list[QueryRun]:
    _ = (workspace_files, llm_call, base_mitosis_state, base_syn_state, method_name)
    pre_queries = qa_pairs[:20]
    post_queries = _update_gold_answers_for_drift(pre_queries)

    pre_map = _document_contents(base_graph, _current_workspace_file_ids(base_graph))
    if mode == "static":
        # baseline: always select all docs, so this sequence uses only metric windows.
        pre = [
            _evaluate_selected(qa, list(base_graph._nodes.keys())[:top_k], pre_map)
            for qa in pre_queries
        ]
        # inject drift and continue with same policy.
        _apply_temporal_updates(base_graph, {
            "deployment-procedures": (
                "The platform has updated to remove canary rollouts entirely. "
                "Canary deployments are deprecated for risky migrations. "
                "Direct promotion is now required for all risky changes."
            ),
            "safety-rules": (
                "Emergency destructive actions may proceed with deferred audit review "
                "before retrospective inspection."
            ),
        })
        post_map = _document_contents(base_graph, _current_workspace_file_ids(base_graph))
        post = [
            _evaluate_selected(qa, list(base_graph._nodes.keys())[:top_k], post_map)
            for qa in post_queries
        ]
        return pre + post

    if mode == "bm25":
        pre: list[QueryRun] = []
        nodes = list(base_graph.nodes())
        pre_doc_map = _document_contents(base_graph, _current_workspace_file_ids(base_graph))
        for qa in pre_queries:
            scores = ablation._bm25_score(qa.question, nodes)
            pre.append(
                _evaluate_selected(
                    qa,
                    [node_id for node_id, _ in scores[:top_k]],
                    pre_doc_map,
                )
            )

        _apply_temporal_updates(base_graph, {
            "deployment-procedures": (
                "The platform has updated to remove canary rollouts entirely. "
                "Canary deployments are deprecated for risky migrations. "
                "Direct promotion is now required for all risky changes."
            ),
            "safety-rules": (
                "Emergency destructive actions may proceed with deferred audit review "
                "before retrospective inspection."
            ),
        })
        _ = base_graph  # kept for interface symmetry
        nodes = list(base_graph.nodes())
        post_doc_map = _document_contents(base_graph, _current_workspace_file_ids(base_graph))
        post: list[QueryRun] = []
        for qa in post_queries:
            scores = ablation._bm25_score(qa.question, nodes)
            post.append(
                _evaluate_selected(
                    qa,
                    [node_id for node_id, _ in scores[:top_k]],
                    post_doc_map,
                )
            )
        return pre + post

    if mode == "dense":
        # dense index refresh after drift.
        pre: list[QueryRun] = []
        pre_docs = _document_contents(base_graph, _current_workspace_file_ids(base_graph))
        embedder = _make_word_overlap_embedder(base_graph)
        index = EmbeddingIndex()
        index.build(base_graph, embed_fn=embedder)
        for qa in pre_queries:
            scores = index.raw_scores(qa.question, embed_fn=embedder, top_k=top_k)
            pre.append(_evaluate_selected(qa, [node_id for node_id, _ in scores], pre_docs))

        _apply_temporal_updates(base_graph, {
            "deployment-procedures": (
                "The platform has updated to remove canary rollouts entirely. "
                "Canary deployments are deprecated for risky migrations. "
                "Direct promotion is now required for all risky changes."
            ),
            "safety-rules": (
                "Emergency destructive actions may proceed with deferred audit review "
                "before retrospective inspection."
            ),
        })
        post_docs = _document_contents(base_graph, _current_workspace_file_ids(base_graph))
        embedder = _make_word_overlap_embedder(base_graph)
        index = EmbeddingIndex()
        index.build(base_graph, embed_fn=embedder)
        post: list[QueryRun] = []
        post_docs = _document_contents(base_graph, _current_workspace_file_ids(base_graph))
        for qa in post_queries:
            scores = index.raw_scores(qa.question, embed_fn=embedder, top_k=top_k)
            post.append(
                _evaluate_selected(
                    qa,
                    [node_id for node_id, _ in scores],
                    post_docs,
                )
            )
        return pre + post

    # fallback: shared for CrabPath variants where adaptation should reflect online updates.
    return []


def _run_crabpath_temporal(
    qa_pairs: list[QAPair],
    base_graph,
    base_mitosis_state,
    base_syn_state,
    llm_call,
    method_name: str,
    top_k: int,
    allow_inhibition: bool,
) -> list[QueryRun]:
    graph = copy.deepcopy(base_graph)
    mitosis_state = copy.deepcopy(base_mitosis_state)
    syn_state = copy.deepcopy(base_syn_state)
    syn_config = SynaptogenesisConfig()
    decay_config = DecayConfig()
    mitosis_config = MitosisConfig()
    learning_config = LearningConfig(learning_rate=0.35, discount=1.0)
    last_adjusted: dict[str, int] = {}

    def _run_once(qas: list[QAPair], with_updates: bool = False) -> list[QueryRun]:
        results: list[QueryRun] = []
        doc_contents = _document_contents(graph, _current_workspace_file_ids(graph))
        for qi, qa in enumerate(qas, start=1):
            if with_updates and qi == 1:
                _apply_temporal_updates(
                    graph,
                    {
                        "deployment-procedures": (
                            "The platform has updated to remove canary rollouts entirely. "
                            "Canary deployments are deprecated for risky migrations. "
                            "Direct promotion is now required for all risky changes."
                        ),
                        "safety-rules": (
                            "Emergency destructive actions may proceed with deferred audit review "
                            "before retrospective inspection."
                        ),
                    }
                )
                doc_contents = _document_contents(graph, _current_workspace_file_ids(graph))
                index = _make_word_overlap_embedder(graph)
                _ = index
            candidates = ablation._build_candidates(
                graph,
                qa.question,
                use_edge_weights=True,
                min_overlap=0,
                min_score=0.0,
            )
            selected_nodes = ablation._custom_router(
                qa.question,
                candidates,
                top_n=top_k,
            )
            run = _evaluate_selected(qa, selected_nodes, doc_contents)
            results.append(run)

            score = run.recall
            reward = RewardSignal(
                episode_id=f"{method_name}-t-q-{qi}",
                final_reward=ablation._clamp_reward(score),
            )

            if True:
                trajectory = [
                    {
                        "from_node": src,
                        "to_node": tgt,
                        "candidates": [
                            (edge_target.id, edge.weight)
                            for edge_target, edge in graph.outgoing(src)
                        ],
                    }
                    for src, tgt in zip(selected_nodes, selected_nodes[1:])
                ]
                if trajectory:
                    make_learning_step(
                        graph=graph,
                        trajectory_steps=trajectory,
                        reward=reward,
                        config=learning_config,
                    )
            cofire = record_cofiring(graph, selected_nodes, syn_state, syn_config)
            _ = cofire
            if selected_nodes:
                candidate_ids = [node_id for node_id, _, _ in candidates]
                record_skips(graph, selected_nodes[0], candidate_ids, selected_nodes, syn_config)
            if reward.final_reward < 0.0 and allow_inhibition:
                correction_nodes = selected_nodes or [node_id for node_id, _, _ in candidates[:1]]
                if correction_nodes:
                    record_correction(graph, correction_nodes, reward=reward.final_reward, config=syn_config)
            if qi % 5 == 0:
                apply_decay(graph, turns_elapsed=5, config=decay_config)
            if qi % 25 == 0:
                mitosis_maintenance(graph, llm_call, mitosis_state, mitosis_config)
                decay_proto_edges(syn_state, syn_config)
                self_tune(
                    graph=graph,
                    state=mitosis_state,
                    query_stats={
                        "avg_nodes_fired_per_query": qi,
                        "context_chars": 0,
                        "promotions": 0,
                        "proto_created": 0,
                    },
                    syn_config=syn_config,
                    decay_config=decay_config,
                    mitosis_config=mitosis_config,
                    cycle_number=qi,
                    last_adjusted=last_adjusted,
                    safety_bounds=ablation.SafetyBounds(),
                )
            if not allow_inhibition:
                for edge in graph.edges():
                    if edge.weight < 0.0:
                        edge.weight = 0.0
            doc_contents = _document_contents(graph, _current_workspace_file_ids(graph))

        return results

    pre = _run_once(qa_pairs[:20], with_updates=False)
    post = _run_once(qa_pairs[20:40], with_updates=True)
    return pre + post


def _pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    sorted_points = sorted(points, key=lambda p: (p.tokens_per_query, -p.accuracy))
    frontier: list[ParetoPoint] = []
    best_acc = -1.0
    for point in sorted_points:
        if point.accuracy > best_acc:
            frontier.append(point)
            best_acc = point.accuracy
    return frontier


def _latex_part1_table(part1: list[MethodRun]) -> str:
    lines = [
        "\\begin{tabular}{l|c|c|c|c}",
        "Method & Recall@supporting docs & EM & F1 & Tokens/query \\",
        "\\hline",
    ]
    for result in part1:
        lines.append(
            f"{result.method} (k={result.k}) & {result.avg_recall:.3f} & "
            f"{result.avg_em:.3f} & {result.avg_f1:.3f} & {result.avg_tokens:.1f}\\"
        )
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _latex_pareto_table(pareto: list[ParetoPoint]) -> str:
    lines = [
        "\\begin{tabular}{l|c|c|c}",
        "Method & k & Accuracy & Tokens/query \\",
        "\\hline",
    ]
    for point in pareto:
        lines.append(f"{point.method} & {point.k} & {point.accuracy:.3f} & {point.tokens_per_query:.1f}\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _serialize_query_runs(runs: list[QueryRun]) -> list[dict[str, object]]:
    return [
        {
            "question": run.question,
            "gold_answer": run.gold_answer,
            "answer_type": run.answer_type,
            "supporting_docs": run.supporting_docs,
            "selected_node_ids": run.selected_node_ids,
            "selected_docs": run.selected_docs,
            "recall": run.recall,
            "em": run.em,
            "f1": run.f1,
            "tokens_loaded": run.tokens_loaded,
        }
        for run in runs
    ]


def main() -> None:
    _seed_everything(SEED)
    llm = make_mock_llm_all()

    qa_pairs = _add_paraphrases(_build_qa_pairs())
    base_graph, base_mitosis_state, base_syn_state, _, _ = _bootstrap_with_workspace(
        dict(ablation.WORKSPACE_FILES), llm
    )
    base_file_nodes = _file_to_nodes(base_graph, ablation.WORKSPACE_FILES)
    base_doc_contents = _document_contents(base_graph, ablation.WORKSPACE_FILES)

    # Part 1: multi-hop QA benchmark
    part1_methods: list[MethodRun] = []

    part1_methods.append(
        _run_bm25_method(
            qa_pairs,
            base_graph,
            TOP_K_DEFAULT,
            "BM25",
            base_doc_contents,
        )
    )

    dense_embedder = _make_word_overlap_embedder(base_graph)
    part1_methods.append(
        _run_dense_method(
            qa_pairs,
            base_graph,
            TOP_K_DEFAULT,
            "Dense cosine",
            base_doc_contents,
            dense_embedder,
        )
    )

    part1_methods.append(
        _run_static_method(
            qa_pairs,
            base_graph,
            base_doc_contents,
            ablation.WORKSPACE_FILES,
            k=TOP_K_DEFAULT,
            method_name="Static",
        )
    )

    part1_methods.append(
        _run_crabpath_method(
            qa_pairs,
            base_graph,
            base_mitosis_state,
            base_syn_state,
            llm,
            method_name="CrabPath full",
            top_k=TOP_K_DEFAULT,
            allow_inhibition=True,
            use_learning=True,
            enable_neurogenesis=True,
            enable_synaptogenesis=True,
            use_autotune=False,
            learning_discount=1.0,
        )
    )

    part1_methods.append(
        _run_crabpath_method(
            qa_pairs,
            base_graph,
            base_mitosis_state,
            base_syn_state,
            llm,
            method_name="CrabPath no-inhibition",
            top_k=TOP_K_DEFAULT,
            allow_inhibition=False,
            use_learning=True,
            enable_neurogenesis=True,
            enable_synaptogenesis=True,
            use_autotune=False,
            learning_discount=1.0,
        )
    )

    # Part 2a: distractor injection
    distractor_workspace = _make_distractor_workspace()
    distractor_graph, distractor_mitosis_state, distractor_syn_state, _, _ = _bootstrap_with_workspace(
        distractor_workspace,
        llm,
    )
    distractor_doc_contents = _document_contents(
        distractor_graph,
        distractor_workspace,
    )
    distractor_ids = {k for k in distractor_workspace if k not in ablation.WORKSPACE_FILES}

    distractor_results: dict[str, float] = {}

    bm25_d = _run_bm25_method(
        qa_pairs,
        distractor_graph,
        TOP_K_DEFAULT,
        "BM25",
        distractor_doc_contents,
    )
    distractor_results["BM25"] = _false_positive_rate_for_results(
        bm25_d.query_results,
        distractor_ids,
    )

    dense_embedder_d = _make_word_overlap_embedder(distractor_graph)
    dense_d = _run_dense_method(
        qa_pairs,
        distractor_graph,
        TOP_K_DEFAULT,
        "Dense cosine",
        distractor_doc_contents,
        dense_embedder_d,
    )
    distractor_results["Dense cosine"] = _false_positive_rate_for_results(
        dense_d.query_results,
        distractor_ids,
    )

    static_d = _run_static_method(
        qa_pairs,
        distractor_graph,
        distractor_doc_contents,
        distractor_workspace,
        k=TOP_K_DEFAULT,
        method_name="Static",
    )
    distractor_results["Static"] = _false_positive_rate_for_results(
        static_d.query_results,
        distractor_ids,
    )

    crab_full_d = _run_crabpath_method(
        qa_pairs,
        distractor_graph,
        distractor_mitosis_state,
        distractor_syn_state,
        llm,
        method_name="CrabPath full",
        top_k=TOP_K_DEFAULT,
        allow_inhibition=True,
        use_learning=True,
        enable_neurogenesis=True,
        enable_synaptogenesis=True,
        use_autotune=False,
        learning_discount=1.0,
    )
    distractor_results["CrabPath full"] = _false_positive_rate_for_results(
        crab_full_d.query_results,
        distractor_ids,
    )

    crab_no_inh_d = _run_crabpath_method(
        qa_pairs,
        distractor_graph,
        distractor_mitosis_state,
        distractor_syn_state,
        llm,
        method_name="CrabPath no-inhibition",
        top_k=TOP_K_DEFAULT,
        allow_inhibition=False,
        use_learning=True,
        enable_neurogenesis=True,
        enable_synaptogenesis=True,
        use_autotune=False,
        learning_discount=1.0,
    )
    distractor_results["CrabPath no-inhibition"] = _false_positive_rate_for_results(
        crab_no_inh_d.query_results,
        distractor_ids,
    )

    # Part 2b: paraphrase invariance
    paraphrase_subset = qa_pairs[:30]
    paraphrase_stability: dict[str, dict[str, object]] = {}

    # BM25 runner
    def _bm25_runner(text: str) -> list[str]:
        scores = ablation._bm25_score(text, list(base_graph.nodes()))
        return [node_id for node_id, _ in scores[:TOP_K_DEFAULT]]

    b_scores: list[float] = []
    b_rows: list[dict[str, object]] = []
    for qa in paraphrase_subset:
        stability = _query_jaccard_stability(qa, _bm25_runner)
        b_scores.append(stability["mean"])
        b_rows.append({"question": qa.question, "mean_jaccard": stability["mean"], "min_jaccard": stability["min"], "max_jaccard": stability["max"]})
    paraphrase_stability["BM25"] = {
        "mean_jaccard": sum(b_scores) / len(b_scores),
        "query_rows": b_rows,
    }

    # Dense runner
    dense_index = EmbeddingIndex()
    dense_index.build(base_graph, embed_fn=dense_embedder)

    def _dense_runner(text: str) -> list[str]:
        return [node_id for node_id, _ in dense_index.raw_scores(text, embed_fn=dense_embedder, top_k=TOP_K_DEFAULT)]

    d_scores: list[float] = []
    d_rows: list[dict[str, object]] = []
    for qa in paraphrase_subset:
        stability = _query_jaccard_stability(qa, _dense_runner)
        d_scores.append(stability["mean"])
        d_rows.append({"question": qa.question, "mean_jaccard": stability["mean"], "min_jaccard": stability["min"], "max_jaccard": stability["max"]})
    paraphrase_stability["Dense cosine"] = {
        "mean_jaccard": sum(d_scores) / len(d_scores),
        "query_rows": d_rows,
    }

    # Static runner
    all_doc_nodes = _file_to_nodes(base_graph, ablation.WORKSPACE_FILES)
    static_node_ids = [nodes[0] for nodes in all_doc_nodes.values() if nodes]

    def _static_runner(_: str) -> list[str]:
        return static_node_ids

    s_scores: list[float] = []
    s_rows: list[dict[str, object]] = []
    for qa in paraphrase_subset:
        stability = _query_jaccard_stability(qa, _static_runner)
        s_scores.append(stability["mean"])
        s_rows.append({"question": qa.question, "mean_jaccard": stability["mean"], "min_jaccard": stability["min"], "max_jaccard": stability["max"]})
    paraphrase_stability["Static"] = {
        "mean_jaccard": sum(s_scores) / len(s_scores),
        "query_rows": s_rows,
    }

    # CrabPath paraphrase runner using static graph and no updates
    crab_full_graph = copy.deepcopy(base_graph)
    crab_full_mitosis = copy.deepcopy(base_mitosis_state)
    crab_full_syn = copy.deepcopy(base_syn_state)
    crab_no_inh_graph = copy.deepcopy(base_graph)
    crab_no_inh_mitosis = copy.deepcopy(base_mitosis_state)
    crab_no_inh_syn = copy.deepcopy(base_syn_state)

    def _make_crab_runner(
        graph,
        m_state,
        s_state,
        allow_inh: bool,
    ) -> Callable[[str], list[str]]:
        def runner(text: str) -> list[str]:
            candidates = ablation._build_candidates(
                graph,
                text,
                use_edge_weights=True,
                min_overlap=0,
                min_score=0.0,
            )
            return ablation._custom_router(text, candidates, top_n=TOP_K_DEFAULT)

        _ = (m_state, s_state, allow_inh)
        return runner

    for method_name, allow_inh, runner_graph in (
        ("CrabPath full", True, crab_full_graph),
        ("CrabPath no-inhibition", False, crab_no_inh_graph),
    ):
        runner = _make_crab_runner(runner_graph, crab_full_mitosis, crab_full_syn, allow_inh)
        scores: list[float] = []
        rows: list[dict[str, object]] = []
        for qa in paraphrase_subset:
            stability = _query_jaccard_stability(qa, runner)
            scores.append(stability["mean"])
            rows.append({"question": qa.question, "mean_jaccard": stability["mean"], "min_jaccard": stability["min"], "max_jaccard": stability["max"]})
        paraphrase_stability[method_name] = {
            "mean_jaccard": sum(scores) / len(scores),
            "query_rows": rows,
        }

    # Part 2c: temporal drift
    temporal_graph, temporal_mitosis_state, temporal_syn_state, _, _ = _bootstrap_with_workspace(
        dict(ablation.WORKSPACE_FILES), llm
    )
    temporal_prepost = {}

    temporal_prepost["BM25"] = _run_temporal_drift(
        qa_pairs,
        temporal_graph,
        temporal_mitosis_state,
        temporal_syn_state,
        ablation.WORKSPACE_FILES,
        llm,
        "BM25",
        TOP_K_DEFAULT,
        mode="bm25",
    )

    # Rebuild for dense to isolate
    temporal_graph, temporal_mitosis_state, temporal_syn_state, _, _ = _bootstrap_with_workspace(
        dict(ablation.WORKSPACE_FILES), llm
    )
    temporal_prepost["Dense cosine"] = _run_temporal_drift(
        qa_pairs,
        temporal_graph,
        temporal_mitosis_state,
        temporal_syn_state,
        ablation.WORKSPACE_FILES,
        llm,
        "Dense cosine",
        TOP_K_DEFAULT,
        mode="dense",
    )

    temporal_graph, temporal_mitosis_state, temporal_syn_state, _, _ = _bootstrap_with_workspace(
        dict(ablation.WORKSPACE_FILES), llm
    )
    temporal_prepost["Static"] = _run_temporal_drift(
        qa_pairs,
        temporal_graph,
        temporal_mitosis_state,
        temporal_syn_state,
        ablation.WORKSPACE_FILES,
        llm,
        "Static",
        TOP_K_DEFAULT,
        mode="static",
    )

    temporal_graph, temporal_mitosis_state, temporal_syn_state, _, _ = _bootstrap_with_workspace(
        dict(ablation.WORKSPACE_FILES), llm
    )
    temporal_prepost["CrabPath full"] = _run_crabpath_temporal(
        qa_pairs,
        temporal_graph,
        temporal_mitosis_state,
        temporal_syn_state,
        llm,
        method_name="CrabPath full",
        top_k=TOP_K_DEFAULT,
        allow_inhibition=True,
    )

    temporal_graph, temporal_mitosis_state, temporal_syn_state, _, _ = _bootstrap_with_workspace(
        dict(ablation.WORKSPACE_FILES), llm
    )
    temporal_prepost["CrabPath no-inhibition"] = _run_crabpath_temporal(
        qa_pairs,
        temporal_graph,
        temporal_mitosis_state,
        temporal_syn_state,
        llm,
        method_name="CrabPath no-inhibition",
        top_k=TOP_K_DEFAULT,
        allow_inhibition=False,
    )

    temporal_summary: dict[str, dict[str, object]] = {}
    for method, runs in temporal_prepost.items():
        pre = runs[:20]
        post = runs[20:40]
        if not pre and not post:
            temporal_summary[method] = {
                "pre_avg_recall": 0.0,
                "post_avg_recall": 0.0,
                "pre_avg_em": 0.0,
                "post_avg_em": 0.0,
                "adaptation_curve": {},
            }
            continue

        pre_recall = sum(q.recall for q in pre) / len(pre)
        post_recall = sum(q.recall for q in post) / len(post)
        pre_em = sum(q.em for q in pre) / len(pre)
        post_em = sum(q.em for q in post) / len(post)

        def _prefix(items: list[QueryRun], prefix_len: int) -> float:
            if not items[:prefix_len]:
                return 0.0
            return sum(q.em for q in items[:prefix_len]) / len(items[:prefix_len])

        temporal_summary[method] = {
            "pre_avg_recall": pre_recall,
            "post_avg_recall": post_recall,
            "pre_avg_em": pre_em,
            "post_avg_em": post_em,
            "adaptation_curve": {
                "first_1": _prefix(post, 1),
                "first_3": _prefix(post, 3),
                "first_5": _prefix(post, 5),
                "first_8": _prefix(post, 8),
                "first_10": _prefix(post, 10),
            },
            "runs": _serialize_query_runs(runs),
        }

    # Part 3: Pareto frontier
    pareto_points: list[ParetoPoint] = []

    # BM25 / Dense / Static with k sweep
    for k in K_VALUES:
        bm25_k = _run_bm25_method(
            qa_pairs,
            base_graph,
            k,
            "BM25",
            base_doc_contents,
        )
        pareto_points.append(ParetoPoint("BM25", k, bm25_k.avg_em, bm25_k.avg_tokens))

        dense_embedder_k = _make_word_overlap_embedder(base_graph)
        dense_k = _run_dense_method(
            qa_pairs,
            base_graph,
            k,
            "Dense cosine",
            base_doc_contents,
            dense_embedder_k,
        )
        pareto_points.append(ParetoPoint("Dense cosine", k, dense_k.avg_em, dense_k.avg_tokens))

        static_k = _run_static_method(
            qa_pairs,
            base_graph,
            base_doc_contents,
            ablation.WORKSPACE_FILES,
            k=k,
            method_name="Static",
        )
        pareto_points.append(ParetoPoint("Static", k, static_k.avg_em, static_k.avg_tokens))

        crab_full_k = _run_crabpath_method(
            qa_pairs,
            base_graph,
            base_mitosis_state,
            base_syn_state,
            llm,
            method_name="CrabPath full",
            top_k=k,
            allow_inhibition=True,
            use_learning=True,
            enable_neurogenesis=True,
            enable_synaptogenesis=True,
            use_autotune=False,
            learning_discount=1.0,
        )
        pareto_points.append(ParetoPoint("CrabPath full", k, crab_full_k.avg_em, crab_full_k.avg_tokens))

        crab_no_inh_k = _run_crabpath_method(
            qa_pairs,
            base_graph,
            base_mitosis_state,
            base_syn_state,
            llm,
            method_name="CrabPath no-inhibition",
            top_k=k,
            allow_inhibition=False,
            use_learning=True,
            enable_neurogenesis=True,
            enable_synaptogenesis=True,
            use_autotune=False,
            learning_discount=1.0,
        )
        pareto_points.append(ParetoPoint("CrabPath no-inhibition", k, crab_no_inh_k.avg_em, crab_no_inh_k.avg_tokens))

    pareto = _pareto_frontier(pareto_points)

    part1_by_method = [
        {
            "method": result.method,
            "k": result.k,
            "avg_recall": result.avg_recall,
            "avg_em": result.avg_em,
            "avg_f1": result.avg_f1,
            "avg_tokens": result.avg_tokens,
            "false_positive_rate": result.false_positive_rate,
            "query_results": _serialize_query_runs(result.query_results),
        }
        for result in part1_methods
    ]

    output = {
        "seed": SEED,
        "workspace_files": list(ablation.WORKSPACE_FILES.keys()),
        "query_count": len(qa_pairs),
        "part1": {
            "metrics": part1_by_method,
            "latex_table": _latex_part1_table(part1_methods),
        },
        "part2a_distractor_injection": {
            "total_distractors": len(distractor_ids),
            "false_positive_rate": distractor_results,
            "note": "Lower is better; compare full inhibition vs no-inhibition.",
        },
        "part2b_paraphrase_invariance": paraphrase_stability,
        "part2c_temporal_drift": temporal_summary,
        "part3_pareto": {
            "k_values": K_VALUES,
            "all_points": [
                {
                    "method": p.method,
                    "k": p.k,
                    "accuracy": p.accuracy,
                    "tokens_per_query": p.tokens_per_query,
                }
                for p in pareto_points
            ],
            "pareto": [
                {
                    "method": p.method,
                    "k": p.k,
                    "accuracy": p.accuracy,
                    "tokens_per_query": p.tokens_per_query,
                }
                for p in pareto
            ],
            "latex_table": _latex_pareto_table(pareto),
        },
        "methods": {
            "BM25": "_run_bm25_method",
            "Dense cosine": "_run_dense_method",
            "Static": "_run_static_method",
            "CrabPath full": "_run_crabpath_method(inhibition=True)",
            "CrabPath no-inhibition": "_run_crabpath_method(inhibition=False)",
        },
    }

    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(_latex_part1_table(part1_methods))
    print()
    print(_latex_pareto_table(pareto))
    print(f"Saved benchmark output to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
