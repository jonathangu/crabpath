#!/usr/bin/env python3
"""Downstream accuracy + RULER + external benchmark stubs for CrabPath."""

from __future__ import annotations

import copy
import json
import random
import re
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.graph import Edge, Graph, Node
from crabpath.legacy.activation import Firing, activate, learn
from scripts import ablation_study as ablation  # noqa: E402

SEED = 2026
RESULTS_PATH = ROOT / "scripts" / "downstream_accuracy_results.json"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
TOP_K_DOWNSTREAM = 8
TOP_K_RULER = 20
TOP_K_EXTERNAL = 3


@dataclass
class NodeSpec:
    node_id: str
    cluster: str
    marker: str
    content: str
    is_fact: bool = False


@dataclass
class QueryCase:
    query: str
    gold_node_ids: list[str]


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _token_count(text: str) -> int:
    return len(_tokenize(text))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _f1_overlap(answer: str, gold: str) -> float:
    ans_tokens = _tokenize(answer)
    gold_tokens = _tokenize(gold)
    if not ans_tokens and not gold_tokens:
        return 1.0
    if not ans_tokens or not gold_tokens:
        return 0.0

    pred = Counter(ans_tokens)
    ref = Counter(gold_tokens)
    overlap = sum((pred & ref).values())
    precision = _safe_divide(overlap, len(ans_tokens))
    recall = _safe_divide(overlap, len(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _coverage(answer: str, gold: str) -> float:
    gold_tokens = set(_tokenize(gold))
    if not gold_tokens:
        return 0.0
    ans_tokens = set(_tokenize(answer))
    return len(ans_tokens & gold_tokens) / len(gold_tokens)


def _noise_ratio(answer: str, gold: str) -> float:
    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return 0.0
    gold_tokens = set(_tokenize(gold))
    noise = sum(1 for token in ans_tokens if token not in gold_tokens)
    return _safe_divide(noise, len(ans_tokens))


def _recall_at_k(selected: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    expected_set = set(expected)
    return _safe_divide(len(set(selected[:k]) & expected_set), len(expected_set))


def _dcg_gain(rank: int) -> float:
    return 1.0 / math.log2(rank + 2.0)


def _ndcg_at_k(selected: list[str], expected: list[str], k: int) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    hits = [1.0 if node_id in expected_set else 0.0 for node_id in selected[:k]]
    dcg = sum(hit * _dcg_gain(i) for i, hit in enumerate(hits))
    ideal_hits = min(len(expected_set), k)
    ideal_dcg = sum(_dcg_gain(i) for i in range(ideal_hits))
    return _safe_divide(dcg, ideal_dcg)


def _mrr_at_k(selected: list[str], expected: list[str], k: int) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    for idx, node_id in enumerate(selected[:k], start=1):
        if node_id in expected_set:
            return 1.0 / idx
    return 0.0


def _add_edge(graph: Graph, source: str, target: str, weight: float) -> None:
    if source == target:
        return
    if graph.get_edge(source, target) is not None:
        return
    graph.add_edge(
        Edge(
            source=source,
            target=target,
            weight=weight,
        )
    )


def _build_cluster_nodes(
    size: int,
    rng: random.Random,
    clusters: dict[str, tuple[str, list[str]]],
) -> list[NodeSpec]:
    base_count = size // len(clusters)
    extra = size % len(clusters)
    specs: list[NodeSpec] = []
    node_idx = 0

    for i, (cluster_id, (topic, terms)) in enumerate(clusters.items()):
        count = base_count + (1 if i < extra else 0)
        for local_idx in range(count):
            marker = f"{cluster_id}_{local_idx + 1:03d}_{rng.randint(100, 999)}"
            term_a = terms[local_idx % len(terms)]
            term_b = terms[(local_idx + 1) % len(terms)]
            node_id = f"{cluster_id}-{node_idx + 1:03d}"
            content = (
                f"{marker} documents a {topic} report about {term_a} and {term_b}."
                f" The node links {terms[(local_idx + 2) % len(terms)]} events to"
                f" {terms[(local_idx + 3) % len(terms)]}."
            )
            specs.append(
                NodeSpec(
                    node_id=node_id,
                    cluster=cluster_id,
                    marker=marker,
                    content=content,
                )
            )
            node_idx += 1

    return specs


def _build_sparse_graph(size: int, seed: int, clusters: dict[str, tuple[str, list[str]]]) -> tuple[Graph, list[NodeSpec], dict[str, NodeSpec]]:
    rng = random.Random(seed)
    specs = _build_cluster_nodes(size=size, rng=rng, clusters=clusters)
    graph = Graph()

    for spec in specs:
        graph.add_node(
            Node(
                id=spec.node_id,
                content=spec.content,
                summary=spec.content[:120],
                cluster_id=spec.cluster,
                metadata={"marker": spec.marker},
            )
        )

    cluster_node_ids: dict[str, list[str]] = {cluster: [] for cluster in clusters}
    for spec in specs:
        cluster_node_ids[spec.cluster].append(spec.node_id)

    node_ids = [spec.node_id for spec in specs]
    shuffled = node_ids.copy()
    rng.shuffle(shuffled)
    for i in range(len(shuffled) - 1):
        _add_edge(graph, shuffled[i], shuffled[i + 1], weight=0.55 + rng.uniform(-0.08, 0.08))
        _add_edge(graph, shuffled[i + 1], shuffled[i], weight=0.50 + rng.uniform(-0.08, 0.08))

    cluster_keys = list(clusters)
    for i in range(len(cluster_keys) - 1):
        source_cluster = cluster_node_ids[cluster_keys[i]]
        target_cluster = cluster_node_ids[cluster_keys[i + 1]]
        if not source_cluster or not target_cluster:
            continue
        _add_edge(
            graph,
            rng.choice(source_cluster),
            rng.choice(target_cluster),
            weight=0.46 + rng.uniform(-0.06, 0.06),
        )
        _add_edge(
            graph,
            rng.choice(target_cluster),
            rng.choice(source_cluster),
            weight=0.46 + rng.uniform(-0.06, 0.06),
        )

    # Add sparse intra- and cross-cluster jumpers.
    for source_spec in specs:
        source_id = source_spec.node_id
        wanted = max(1, int(0.05 * size))
        same_cluster_candidates = [
            nid
            for nid in cluster_node_ids[source_spec.cluster]
            if nid != source_id
            and graph.get_edge(source_id, nid) is None
        ]
        other_candidates = [
            nid
            for nid in node_ids
            if nid != source_id and graph.get_edge(source_id, nid) is None
        ]
        pool = same_cluster_candidates * 3 + other_candidates
        if not pool:
            continue
        for target_id in rng.sample(pool, k=min(wanted, len(pool))):
            _add_edge(
                graph,
                source_id,
                target_id,
                weight=(
                    0.60 + rng.uniform(-0.08, 0.08)
                    if target_id in cluster_node_ids[source_spec.cluster]
                    else 0.35 + rng.uniform(-0.08, 0.08)
                ),
            )

    spec_by_id = {spec.node_id: spec for spec in specs}
    return graph, specs, spec_by_id


def _bm25_ranking(graph: Graph, query: str, top_k: int) -> list[str]:
    scores = ablation._bm25_score(query, list(graph.nodes()))
    return [node_id for node_id, _ in scores[:top_k]]


def _crabpath_ranking(
    graph: Graph,
    query: str,
    top_k: int,
    seed_count: int = 5,
    max_steps: int = 3,
) -> tuple[list[str], Firing]:
    bm25 = _bm25_ranking(graph, query, top_k=seed_count * 2)
    seeds = {node_id: 1.0 - 0.15 * i for i, node_id in enumerate(bm25)}
    if not seeds:
        return [], Firing(fired=[], inhibited=[])

    firing = activate(
        graph=graph,
        seeds=seeds,
        max_steps=max_steps,
        top_k=top_k,
        reset=True,
    )
    selected = [node.id for node, _ in firing.fired]
    if not selected:
        selected = bm25[:top_k]
    return selected[:top_k], firing


def _evaluate_downstream_case(
    query: QueryCase,
    selected: list[str],
    node_contents: dict[str, str],
) -> dict[str, float | int | str | list[str]]:
    answer = " ".join(node_contents.get(nid, "") for nid in selected)
    gold = " ".join(node_contents[nid] for nid in query.gold_node_ids if nid in node_contents)
    answer_tokens = _token_count(answer)
    return {
        "query": query.query,
        "gold_node_ids": list(query.gold_node_ids),
        "selected_node_ids": list(selected),
        "f1": _f1_overlap(answer, gold),
        "coverage": _coverage(answer, gold),
        "noise_ratio": _noise_ratio(answer, gold),
        "answer_length": answer_tokens,
    }


def _evaluate_downstream_benchmark(
    graph: Graph,
    queries: list[QueryCase],
    use_crabpath: bool,
    do_learning: bool = False,
    top_k: int = TOP_K_DOWNSTREAM,
) -> dict[str, Any]:
    node_contents = {node.id: node.content for node in graph.nodes()}
    warmup_count = 40 if len(queries) >= 40 else len(queries) // 2
    warmup_queries = queries[:warmup_count]
    eval_queries = queries[warmup_count:]

    eval_results: list[dict[str, Any]] = []
    for i, query in enumerate(warmup_queries, start=1):
        if not use_crabpath:
            continue
        selected, firing = _crabpath_ranking(graph=graph, query=query.query, top_k=top_k)
        metrics = _evaluate_downstream_case(query, selected, node_contents)
        reward = max(-1.0, min(1.0, 2.0 * metrics["f1"] - 1.0))
        if do_learning:
            learn(graph=graph, result=firing, outcome=reward, rate=0.12)

    for query in eval_queries:
        if use_crabpath:
            selected, _ = _crabpath_ranking(graph=graph, query=query.query, top_k=top_k)
        else:
            selected = _bm25_ranking(graph=graph, query=query.query, top_k=top_k)
        eval_results.append(_evaluate_downstream_case(query, selected, node_contents))

    if not eval_results:
        return {
            "f1": 0.0,
            "coverage": 0.0,
            "noise_ratio": 0.0,
            "answer_length": 0.0,
            "query_count": 0,
            "queries": [],
        }

    return {
        "f1": statistics.mean(item["f1"] for item in eval_results),
        "coverage": statistics.mean(item["coverage"] for item in eval_results),
        "noise_ratio": statistics.mean(item["noise_ratio"] for item in eval_results),
        "answer_length": statistics.mean(item["answer_length"] for item in eval_results),
        "query_count": len(eval_results),
        "queries": eval_results,
    }


def _build_downstream_data(rng: random.Random, spec_by_id: dict[str, NodeSpec]) -> list[QueryCase]:
    node_ids = list(spec_by_id)
    queries: list[QueryCase] = []

    for idx in range(80):
        count = 1 if idx % 2 == 0 else 2
        gold_ids = rng.sample(node_ids, k=min(count, len(node_ids)))
        markers = [spec_by_id[node_id].marker for node_id in gold_ids]
        distractor = rng.choice(
            [spec.marker for spec in spec_by_id.values() if spec.node_id not in gold_ids]
        )
        if len(markers) == 1:
            query = (
                f"Which document discusses {markers[0]} and also references {distractor}?"
            )
        else:
            query = (
                f"Find notes combining {markers[0]} with {markers[1]} while touching"
                f" {distractor}."
            )
        queries.append(QueryCase(query=query, gold_node_ids=gold_ids))

    return queries


def _build_ruler_queries(
    rng: random.Random,
    fact_specs: list[NodeSpec],
    n_facts: int,
) -> list[QueryCase]:
    fact_ids = [spec.node_id for spec in fact_specs]
    queries: list[QueryCase] = []
    for i in range(20):
        selected = rng.sample(fact_ids, k=n_facts)
        query_parts = ", ".join(selected)
        queries.append(
            QueryCase(
                query=f"Locate all required markers in this fact set: {query_parts}. Query {i + 1}.",
                gold_node_ids=selected,
            )
        )
    return queries


def _evaluate_ruler_case(
    query: QueryCase,
    selected: list[str],
    node_contents: dict[str, str],
) -> dict[str, Any]:
    selected_set = set(selected)
    target_set = set(query.gold_node_ids)
    found = target_set & selected_set
    partial_recall = _safe_divide(len(found), len(target_set))

    prefix_tokens = 0
    remaining = set(target_set)
    for node_id in selected:
        prefix_tokens += _token_count(node_contents.get(node_id, ""))
        remaining.discard(node_id)
        if not remaining:
            break
    tokens_to_find_all = prefix_tokens if not remaining else sum(
        _token_count(node_contents.get(node_id, "")) for node_id in selected
    )

    return {
        "query": query.query,
        "gold_node_ids": list(target_set),
        "selected_node_ids": list(selected),
        "all_found": len(found) == len(target_set),
        "partial_recall": partial_recall,
        "tokens_to_find_all": tokens_to_find_all,
    }


def _evaluate_ruler_benchmark(
    graph: Graph,
    queries: list[QueryCase],
    use_crabpath: bool,
    do_learning: bool = False,
    top_k: int = TOP_K_RULER,
) -> dict[str, Any]:
    node_contents = {node.id: node.content for node in graph.nodes()}
    warmup_queries = queries[:10]
    eval_queries = queries[10:]
    eval_results: list[dict[str, Any]] = []

    for query in warmup_queries:
        if not use_crabpath:
            continue
        selected, firing = _crabpath_ranking(
            graph=graph,
            query=query.query,
            top_k=top_k,
            seed_count=8,
        )
        metrics = _evaluate_ruler_case(query, selected, node_contents)
        reward = max(-1.0, min(1.0, 2.0 * metrics["partial_recall"] - 1.0))
        if do_learning:
            learn(graph=graph, result=firing, outcome=reward, rate=0.1)

    for query in eval_queries:
        if use_crabpath:
            selected, _ = _crabpath_ranking(
                graph=graph,
                query=query.query,
                top_k=top_k,
                seed_count=8,
            )
        else:
            selected = _bm25_ranking(graph=graph, query=query.query, top_k=top_k)
        eval_results.append(_evaluate_ruler_case(query, selected, node_contents))

    if not eval_results:
        return {
            "all_found": 0.0,
            "partial_recall": 0.0,
            "tokens_to_find_all": 0.0,
            "query_count": 0,
            "queries": [],
        }

    all_found_count = sum(1 for item in eval_results if item["all_found"])
    return {
        "all_found": _safe_divide(all_found_count, len(eval_results)),
        "partial_recall": statistics.mean(item["partial_recall"] for item in eval_results),
        "tokens_to_find_all": statistics.mean(item["tokens_to_find_all"] for item in eval_results),
        "query_count": len(eval_results),
        "queries": eval_results,
    }


def _build_part1_query_set(rng: random.Random, spec_by_id: dict[str, NodeSpec]) -> list[QueryCase]:
    return _build_downstream_data(rng=rng, spec_by_id=spec_by_id)


def _build_fact_graph(seed: int, size: int = 200) -> tuple[Graph, list[NodeSpec], list[NodeSpec]]:
    clusters = {
        "ops": ("operations guidance", ["deploy", "pipeline", "rollback", "canary", "runbook"]),
        "infra": ("infrastructure control", ["node", "service", "boundary", "latency", "throttle"]),
        "security": ("security policy", ["credential", "audit", "policy", "threat", "credential"]),
        "finance": ("finance operations", ["invoice", "risk", "portfolio", "budget", "hedge"]),
        "health": ("health systems", ["patient", "triage", "diagnostic", "treatment", "recovery"]),
    }
    graph, specs, spec_by_id = _build_sparse_graph(size=size, seed=seed, clusters=clusters)

    fact_specs: list[NodeSpec] = []
    fact_count = 40
    for idx, node_id in enumerate(rng := random.Random(seed).sample([spec.node_id for spec in specs], fact_count)):
        spec = spec_by_id[node_id]
        spec.is_fact = True
        spec.marker = f"fact_{idx + 1:03d}_{random.Random(seed + idx).randint(1000, 9999)}"
        fact_text = f"{spec.content} This fact is tagged {spec.marker}. "
        spec.content = fact_text
        node = graph.get_node(spec.node_id)
        if node:
            node.content = fact_text
        fact_specs.append(spec)

    fact_node_ids = [spec.node_id for spec in fact_specs]
    for i in range(len(fact_node_ids) - 1):
        _add_edge(graph, fact_node_ids[i], fact_node_ids[i + 1], 0.72 + random.Random(seed + i).uniform(-0.1, 0.1))
        if random.Random(seed + i + 100).random() < 0.4:
            _add_edge(graph, fact_node_ids[i + 1], fact_node_ids[i], 0.72 + random.Random(seed + i + 200).uniform(-0.1, 0.1))

    return graph, specs, fact_specs


def _build_narrative_nodes() -> list[NodeSpec]:
    narratives = {
        "moonwatch": [
            "At Duskpoint, the observatory detects a flare over sector twelve.",
            "Control logs show pressure drift in the atmospheric array after sunset.",
            "A junior analyst correlates the flare timestamps with a cooling anomaly.",
            "The team delays public reporting until backup sensors confirm the pattern.",
            "A senior engineer reroutes telemetry from array two to protect storage.",
            "The reroute reveals the first node is actually a software saturation event.",
            "Operators trigger a failover window to isolate the affected intake pump.",
            "A clean signal returns; the team restores normal sampling and opens the audit trail.",
            "Later, the post-incident report ties the flare to a firmware timing loop.",
            "The observatory publishes a safety memo and upgrades firmware checks.",
        ],
        "riverline": [
            "At Riverline Clinic, a resident books a same-day visit after a late symptom report.",
            "The triage chatbot forwards the case to a cardiology nurse with high-priority tags.",
            "Nurse notes show elevated pulse and intermittent dizziness for several hours.",
            "Lab results return normal for infection but abnormal rhythm persists.",
            "A remote wearable stream confirms arrhythmia during mild activity.",
            "The physician asks for a second ECG and bedside echo before discharge.",
            "Data from the echo shows mild valve leakage requiring follow-up monitoring.",
            "The patient is sent home with timed medication and a follow-up alarm plan.",
            "Within two days, the follow-up check confirms stabilization after dose adjustment.",
            "The clinic updates its triage policy to include smartwatch rhythm flags.",
        ],
        "ledgerline": [
            "Ledgerline Finance opens the month with rising short-term yields.",
            "The treasury desk notices an unusual spread between two regional bonds.",
            "Risk analytics flags a concentration of derivative exposure from legacy deals.",
            "A compliance officer asks the desk to freeze automatic hedging rules.",
            "Operations validates that collateral calls were settled without mismatch.",
            "The CFO authorizes manual stress tests on the top-risk portfolio bucket.",
            "Back-testing shows one fund strategy underperforms during volatility spikes.",
            "The trading desk rebalances toward shorter duration notes at quarter close.",
            "Portfolio guardrails are re-applied and a reporting memo is sent to clients.",
            "The month ends with drawdown containment and an updated risk playbook.",
        ],
    }

    specs: list[NodeSpec] = []
    for nidx, (story_key, paragraphs) in enumerate(narratives.items()):
        for pidx, paragraph in enumerate(paragraphs):
            marker = f"{story_key}_p{pidx + 1:02d}"
            node_id = f"{story_key}-{pidx + 1:02d}"
            specs.append(
                NodeSpec(
                    node_id=node_id,
                    cluster=story_key,
                    marker=marker,
                    content=f"{paragraph} (marker: {marker}).",
                )
            )
    return specs


def _build_narrative_graph() -> tuple[Graph, list[NodeSpec]]:
    specs = _build_narrative_nodes()
    rng = random.Random(SEED)
    graph = Graph()
    for spec in specs:
        graph.add_node(
            Node(
                id=spec.node_id,
                content=spec.content,
                summary=spec.content[:100],
                cluster_id=spec.cluster,
            )
        )
    story_names = ["moonwatch", "riverline", "ledgerline"]
    for story_name in story_names:
        story_nodes = [spec.node_id for spec in specs if spec.cluster == story_name]
        for source, target in zip(story_nodes, story_nodes[1:]):
            _add_edge(graph, source, target, 0.72 + rng.uniform(-0.08, 0.08))
            _add_edge(graph, target, source, 0.66 + rng.uniform(-0.08, 0.08))
        for _ in range(2):
            if len(story_nodes) >= 2:
                _add_edge(
                    graph,
                    rng.choice(story_nodes),
                    rng.choice(story_nodes),
                    0.32 + rng.uniform(-0.08, 0.08),
                )

    all_nodes = [spec.node_id for spec in specs]
    for _ in range(6):
        a = rng.choice(all_nodes)
        b = rng.choice(all_nodes)
        if a != b:
            _add_edge(graph, a, b, 0.25 + rng.uniform(-0.04, 0.08))

    return graph, specs


def _build_narrative_queries() -> list[QueryCase]:
    queries: list[QueryCase] = []
    story_to_nodes = {
        "moonwatch": ["moonwatch-01", "moonwatch-03", "moonwatch-07", "moonwatch-10"],
        "riverline": ["riverline-02", "riverline-05", "riverline-08", "riverline-10"],
        "ledgerline": ["ledgerline-01", "ledgerline-04", "ledgerline-07", "ledgerline-10"],
    }

    for story, nodes in story_to_nodes.items():
        queries.append(QueryCase(query=f"In {story}, connect the first warning, response, and resolution.", gold_node_ids=[nodes[0], nodes[1], nodes[2]]))
        queries.append(QueryCase(query=f"How does {story} progress from signal to mitigation and final documentation?", gold_node_ids=[nodes[0], nodes[1], nodes[3]]))
        queries.append(QueryCase(query=f"Explain {story}'s policy update after the root-cause discovery.", gold_node_ids=[nodes[2], nodes[3]]))
        queries.append(QueryCase(query=f"What links {story}-01 and {nodes[1]} to the final outcome?", gold_node_ids=[nodes[0], nodes[1], nodes[3]]))
        queries.append(QueryCase(query=f"Describe the chain that starts with the initial anomaly in {story}.", gold_node_ids=[nodes[0], nodes[2], nodes[1]]))

    while len(queries) < 20:
        idx = len(queries) % 3
        extra_nodes = {
            0: ["moonwatch-02", "riverline-03", "ledgerline-05"],
            1: ["riverline-04", "ledgerline-02", "moonwatch-06"],
            2: ["ledgerline-03", "moonwatch-04", "riverline-07"],
        }[idx]
        queries.append(
            QueryCase(
                query=(
                    f"Synthesize how {extra_nodes[0]}, {extra_nodes[1]}, and {extra_nodes[2]}"
                    " combine into an outcome-focused arc."
                ),
                gold_node_ids=extra_nodes[:2],
            )
        )

    return queries[:20]


def _build_ms_marco_stubs() -> tuple[list[NodeSpec], list[QueryCase]]:
    passages = [
        ("t1-01", "technology", "edge inference", "A local edge cache cut API latency after rolling out a 12-hour TTL."),
        ("t1-02", "technology", "auth protocol", "OAuth tokens now rotate every 12 hours with scoped refresh windows."),
        ("t1-03", "technology", "container", "Container cold starts dropped after prewarming worker pools at deploy."),
        ("t1-04", "technology", "telemetry", "Structured telemetry now includes dependency DAG timing in each event."),
        ("t1-05", "technology", "consistency", "Event consistency checks gate writes when quorum drops below three nodes."),
        ("h1-01", "health", "arrhythmia", "Wearables can report high-confidence arrhythmia episodes at 30-second resolution."),
        ("h1-02", "health", "diet", "Dietary sodium reduction helps reduce average blood pressure in 8-week follow-ups."),
        ("h1-03", "health", "triage", "Clinical triage systems prioritize chest-pain flags and unstable vitals."),
        ("h1-04", "health", "sleep", "Short sleep windows increase risk scores in cognitive fatigue screening."),
        ("h1-05", "health", "rehab", "Post-acute rehab outcomes improved with two-week follow-up reminders."),
        ("h2-01", "health", "vaccination", "Booster scheduling systems reduce missed appointments by alerting caregivers."),
        ("h2-02", "health", "imaging", "MRI pre-processing can filter motion artifacts without lowering lesion visibility."),
        ("f1-01", "finance", "budget", "Budget reforecasting uses scenario stress bands before quarterly close."),
        ("f1-02", "finance", "hedging", "Interest-rate hedging often shifts from long duration to barbell exposure in spikes."),
        ("f1-03", "finance", "fraud", "Card fraud detection needs velocity checks and device-consistency signals."),
        ("f1-04", "finance", "credit", "Credit policy flags applications with unexplained income volatility."),
        ("f1-05", "finance", "retirement", "Retirement modeling now blends inflation and longevity assumptions."),
        ("f2-01", "finance", "market", "Market liquidity dips can force temporary lot-size rules in execution engines."),
        ("f2-02", "finance", "tax", "Tax-advantaged rebalance windows can reduce realization friction."),
        ("f2-03", "finance", "reporting", "Monthly reporting now requires reconciliation of cash equivalents and accrued fees."),
    ]

    specs = [
        NodeSpec(
            node_id=passage_id,
            cluster=domain,
            marker=f"{topic}_{passage_id}",
            content=f"{topic}: {text} ({passage_id})",
        )
        for passage_id, domain, topic, text in passages
    ]

    rng = random.Random(SEED + 99)
    queries: list[QueryCase] = []
    for idx in range(30):
        if idx % 5 == 0:
            chosen = rng.sample(specs, k=2)
            relevant = [chosen[0].node_id, chosen[1].node_id]
            query_text = (
                f"Which passages explain both {chosen[0].marker} and {chosen[1].marker}?"
            )
        else:
            chosen = [rng.choice(specs)]
            relevant = [chosen[0].node_id]
            query_text = f"Which passage best covers {chosen[0].marker}?"
        queries.append(QueryCase(query=query_text, gold_node_ids=relevant))

    return specs, queries


def _build_ms_marco_graph(specs: list[NodeSpec]) -> Graph:
    graph = Graph()
    for spec in specs:
        graph.add_node(Node(id=spec.node_id, content=spec.content, cluster_id=spec.cluster))

    graph_node_ids = [spec.node_id for spec in specs]
    rng = random.Random(SEED + 7)
    for source in graph_node_ids:
        targets = [tid for tid in graph_node_ids if tid != source]
        rng.shuffle(targets)
        for target in targets[:6]:
            weight = 0.7 if source.split("-")[0] == target.split("-")[0] else 0.4
            _add_edge(graph, source=source, target=target, weight=weight + rng.uniform(-0.1, 0.08))
    return graph


def _evaluate_ir_case(
    selected: list[str],
    query: QueryCase,
) -> dict[str, float]:
    return {
        "recall@3": _recall_at_k(selected, query.gold_node_ids, k=3),
        "ndcg@3": _ndcg_at_k(selected, query.gold_node_ids, k=3),
        "mrr@3": _mrr_at_k(selected, query.gold_node_ids, k=3),
    }


def _run_external_suite(
    graph: Graph,
    queries: list[QueryCase],
    use_crabpath: bool,
    top_k: int = TOP_K_EXTERNAL,
) -> dict[str, float]:
    per_query: list[dict[str, float]] = []
    for query in queries:
        selected = _bm25_ranking(graph=graph, query=query.query, top_k=top_k)
        if use_crabpath:
            selected, _ = _crabpath_ranking(graph=graph, query=query.query, top_k=top_k)
        per_query.append(_evaluate_ir_case(selected, query))

    if not per_query:
        return {"recall@3": 0.0, "ndcg@3": 0.0, "mrr@3": 0.0}

    return {
        "recall@3": statistics.mean(item["recall@3"] for item in per_query),
        "ndcg@3": statistics.mean(item["ndcg@3"] for item in per_query),
        "mrr@3": statistics.mean(item["mrr@3"] for item in per_query),
    }


def _print_summary_table(rows: list[tuple[str, Any]], header: str) -> None:
    print(f"\n{header}")
    width = max(18, max(len(str(row[0])) for row in rows) if rows else 18)
    print("-" * (width + 52))
    print(f"{'Method':<{width}} | F1 | Coverage | Noise | AvgLen")
    print("-" * (width + 52))
    for method, payload in rows:
        print(
            f"{method:<{width}} | {payload.get('f1', 0.0):5.3f} |"
            f" {payload.get('coverage', 0.0):8.3f} | {payload.get('noise_ratio', 0.0):5.3f} |"
            f" {payload.get('answer_length', 0.0):6.1f}"
        )


def _print_ruler_summary(rows: list[tuple[int, dict[str, Any], dict[str, Any]]]) -> None:
    print("\nRULER Multi-Fact (N = facts needed)")
    print("N | BM25 all_found | CRAB all_found | BM25 partial | CRAB partial")
    print("-------------------------------------------------------------")
    for n_value, bm25_metrics, cp_metrics in rows:
        print(
            f"{n_value:2d} | {bm25_metrics['all_found']:13.3f} |"
            f" {cp_metrics['all_found']:13.3f} |"
            f" {bm25_metrics['partial_recall']:12.3f} |"
            f" {cp_metrics['partial_recall']:11.3f}"
        )


def _print_external_summary(name: str, metrics: dict[str, dict[str, float]], include_mrr: bool = False) -> None:
    print(f"\n{name}")
    if include_mrr:
        print("Method | Recall@3 | NDCG@3 | MRR@3")
    else:
        print("Method | Recall@3 | NDCG@3")
    print("--------------------------------------------------")
    for method, payload in metrics.items():
        if include_mrr:
            print(
                f"{method:<6} | {payload['recall@3']:8.3f} |"
                f" {payload['ndcg@3']:8.3f} | {payload['mrr@3']:8.3f}"
            )
        else:
            print(f"{method:<6} | {payload['recall@3']:8.3f} | {payload['ndcg@3']:8.3f}")


def _build_ruler_suite() -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for n_facts in (2, 3, 5, 8):
        graph, specs, fact_specs = _build_fact_graph(seed=SEED + n_facts * 11, size=200)
        query_set = _build_ruler_queries(random.Random(SEED + n_facts), fact_specs, n_facts=n_facts)

        bm25 = _evaluate_ruler_benchmark(graph=copy.deepcopy(graph), queries=query_set, use_crabpath=False)
        cp_graph = copy.deepcopy(graph)
        cp = _evaluate_ruler_benchmark(
            graph=cp_graph,
            queries=query_set,
            use_crabpath=True,
            do_learning=True,
        )
        results[n_facts] = {
            "bm25": bm25,
            "crabpath": cp,
            "queries": [q.__dict__ for q in query_set],
            "graph": {
                "nodes": graph.node_count,
                "edges": graph.edge_count,
            },
        }
    return results


def _build_downstream_benchmark() -> dict[str, Any]:
    clusters = {
        "deploy": ("deployment operations", ["artifact", "canary", "release", "rollback", "pipeline"]),
        "incident": ("incident response", ["alert", "outage", "severity", "playbook", "mitigation"]),
        "security": ("security controls", ["credential", "audit", "policy", "threat", "compliance"]),
        "monitoring": ("monitoring", ["latency", "telemetry", "threshold", "dashboard", "anomaly"]),
        "architecture": ("architecture", ["service", "boundary", "contract", "dependency", "throughput"]),
    }
    graph, specs, spec_by_id = _build_sparse_graph(size=100, seed=SEED, clusters=clusters)
    node_contents = {node.id: node.content for node in graph.nodes()}
    _ = node_contents
    query_set = _build_part1_query_set(random.Random(SEED + 1), spec_by_id=spec_by_id)

    bm25 = _evaluate_downstream_benchmark(
        graph=copy.deepcopy(graph),
        queries=query_set,
        use_crabpath=False,
    )
    cp_graph = copy.deepcopy(graph)
    cp = _evaluate_downstream_benchmark(
        graph=cp_graph,
        queries=query_set,
        use_crabpath=True,
        do_learning=True,
    )

    return {
        "bm25": bm25,
        "crabpath": cp,
        "queries": [q.__dict__ for q in query_set],
    }


def _build_narrative_suite() -> dict[str, Any]:
    graph, specs = _build_narrative_graph()
    queries = _build_narrative_queries()

    bm25 = _run_external_suite(graph=copy.deepcopy(graph), queries=queries, use_crabpath=False)
    cp = _run_external_suite(graph=copy.deepcopy(graph), queries=queries, use_crabpath=True)
    return {
        "bm25": bm25,
        "crabpath": cp,
        "queries": [q.__dict__ for q in queries],
    }


def _build_ms_marco_suite() -> dict[str, Any]:
    specs, queries = _build_ms_marco_stubs()
    graph = _build_ms_marco_graph(specs)
    bm25 = _run_external_suite(graph=copy.deepcopy(graph), queries=queries, use_crabpath=False)
    cp = _run_external_suite(graph=copy.deepcopy(graph), queries=queries, use_crabpath=True)
    return {
        "bm25": bm25,
        "crabpath": cp,
        "queries": [q.__dict__ for q in queries],
    }


def run() -> dict[str, Any]:
    random.seed(SEED)
    downstream = _build_downstream_benchmark()
    ruler = _build_ruler_suite()
    narrative = _build_narrative_suite()
    ms_marco = _build_ms_marco_suite()

    results: dict[str, Any] = {
        "seed": SEED,
        "downstream_accuracy": downstream,
        "ruler_multi_fact": ruler,
        "narrative_qa": narrative,
        "ms_marco": ms_marco,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    _print_summary_table(
        [
            ("BM25", downstream["bm25"]),
            ("CrabPath", downstream["crabpath"]),
        ],
        header="Downstream Accuracy",
    )

    _print_ruler_summary(
        [
            (n_value, metrics["bm25"], metrics["crabpath"])
            for n_value, metrics in sorted(ruler.items())
        ]
    )
    _print_external_summary(
        "NarrativeQA",
        {"BM25": narrative["bm25"], "CRAB": narrative["crabpath"]},
        include_mrr=False,
    )
    _print_external_summary(
        "MS MARCO",
        {"BM25": ms_marco["bm25"], "CRAB": ms_marco["crabpath"]},
        include_mrr=True,
    )

    print(f"\nSaved results: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run()
