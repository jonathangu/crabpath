#!/usr/bin/env python3
"""Context utilization, noise sensitivity, and temporal drift benchmark for CrabPath."""

from __future__ import annotations

import copy
import json
import random
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.graph import Edge, Graph, Node  # noqa: E402
from crabpath.legacy.activation import activate, learn  # noqa: E402
from scripts import ablation_study as ablation  # noqa: E402

SEED = 2026
TOP_K = 3
WARMUP_QUERIES = 25
RESULTS_PATH = ROOT / "scripts" / "context_noise_drift_results.json"
WINDOW_SIZE = 5


CLUSTERS = {
    "deploy": {
        "topic": "deployment operations",
        "anchor_terms": [
            "artifact",
            "canary",
            "rollout",
            "release",
            "pipeline",
            "rollback",
        ],
    },
    "incident": {
        "topic": "incident response",
        "anchor_terms": [
            "alert",
            "outage",
            "severity",
            "playbook",
            "mitigation",
            "postmortem",
        ],
    },
    "security": {
        "topic": "security controls",
        "anchor_terms": [
            "hardening",
            "audit",
            "credential",
            "policy",
            "threat",
            "compliance",
        ],
    },
    "monitoring": {
        "topic": "system monitoring",
        "anchor_terms": [
            "latency",
            "telemetry",
            "threshold",
            "dashboards",
            "anomaly",
            "incident",
        ],
    },
    "architecture": {
        "topic": "system architecture",
        "anchor_terms": [
            "boundary",
            "service",
            "contract",
            "dependency",
            "storage",
            "throughput",
        ],
    },
}


@dataclass
class DocSpec:
    node_id: str
    cluster: str
    keywords: list[str]
    content: str


@dataclass
class QueryCase:
    text: str
    expected_node_ids: list[str]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _token_count(text: str) -> int:
    return len(_tokenize(text))


def _char_count(text: str) -> int:
    return len(text)


def _build_cluster_distribution(size: int) -> dict[str, int]:
    base = size // len(CLUSTERS)
    extra = size % len(CLUSTERS)
    counts: dict[str, int] = {}
    for i, cluster in enumerate(CLUSTERS):
        counts[cluster] = base + (1 if i < extra else 0)
    return counts


def _synthesize_doc(cluster: str, idx: int, term_a: str, term_b: str, rng: random.Random) -> str:
    topic = CLUSTERS[cluster]["topic"]
    terms = CLUSTERS[cluster]["anchor_terms"]
    base = [
        (
            f"{term_a} provides {topic} notes with guidance on {terms[idx % len(terms)]}"
            f" and {terms[(idx + 1) % len(terms)]}."
        ),
        (
            f"Operational notes show {term_a} linking to {term_b} across"
            f" {cluster} boundaries and service workflows."
        ),
        (
            f"This entry references {term_b}, and highlights interactions under"
            f" {terms[(idx + 2) % len(terms)]} and {terms[(idx + 3) % len(terms)]}."
        ),
    ]
    if rng.random() < 0.6:
        base.append(
            f"Cross-file review tracks {term_a} through {terms[(idx + 4) % len(terms)]}"
            f" while keeping observability in scope."
        )
    return " ".join(base)


def _build_documents(size: int, rng: random.Random) -> tuple[dict[str, DocSpec], list[DocSpec]]:
    counts = _build_cluster_distribution(size)
    specs: dict[str, DocSpec] = {}
    ordered: list[DocSpec] = []
    for cluster, count in counts.items():
        terms = CLUSTERS[cluster]["anchor_terms"]
        for i in range(count):
            node_idx = f"{cluster}-{i + 1:03d}"
            node_id = f"{cluster}_{node_idx}"
            keywords = [
                f"{cluster}_signal_{node_idx}",
                f"{cluster}_{terms[i % len(terms)]}_{rng.randint(10, 99)}",
                f"{cluster}_link_{rng.randint(100, 999)}",
            ]
            content = _synthesize_doc(cluster, i, keywords[0], keywords[1], rng)
            spec = DocSpec(node_id=node_id, cluster=cluster, keywords=keywords, content=content)
            specs[node_id] = spec
            ordered.append(spec)
    return specs, ordered


def _add_edge(graph: Graph, source: str, target: str, *, weight_hint: float, rng: random.Random) -> None:
    if source == target:
        return
    if graph.get_edge(source, target) is not None:
        return
    weight = max(0.05, min(0.95, weight_hint + rng.uniform(-0.08, 0.08)))
    graph.add_edge(Edge(source=source, target=target, weight=weight))


def _build_graph(size: int, edge_ratio: float, seed: int) -> tuple[Graph, dict[str, DocSpec], list[DocSpec]]:
    rng = random.Random(seed)
    specs, ordered = _build_documents(size, rng)
    graph = Graph()
    for spec in ordered:
        graph.add_node(
            Node(
                id=spec.node_id,
                content=spec.content,
                summary=spec.content[:120],
                cluster_id=spec.cluster,
            )
        )

    node_ids = [spec.node_id for spec in ordered]
    if not node_ids:
        return graph, specs, ordered

    by_cluster: dict[str, list[str]] = {cluster: [] for cluster in CLUSTERS}
    for spec in ordered:
        by_cluster.setdefault(spec.cluster, []).append(spec.node_id)

    chain = node_ids.copy()
    rng.shuffle(chain)
    for i in range(len(chain) - 1):
        _add_edge(graph, chain[i], chain[i + 1], weight_hint=0.60, rng=rng)
        _add_edge(graph, chain[i + 1], chain[i], weight_hint=0.60, rng=rng)

    cluster_order = list(CLUSTERS)
    for i in range(len(cluster_order) - 1):
        left = by_cluster.get(cluster_order[i], [])
        right = by_cluster.get(cluster_order[i + 1], [])
        if left and right:
            _add_edge(graph, rng.choice(left), rng.choice(right), weight_hint=0.50, rng=rng)
            _add_edge(graph, rng.choice(right), rng.choice(left), weight_hint=0.50, rng=rng)

    desired_out = max(1, int(round(edge_ratio * (size - 1))))
    for source in node_ids:
        source_cluster = next(spec.cluster for spec in ordered if spec.node_id == source)
        existing = {edge_target.id for edge_target, _ in graph.outgoing(source)}
        needed = max(0, desired_out - len(existing))
        if needed <= 0:
            continue

        same_cluster = [nid for nid in by_cluster.get(source_cluster, []) if nid != source and nid not in existing]
        other_cluster = [nid for nid in node_ids if nid != source and nid not in existing and nid not in same_cluster]
        weighted_candidates = same_cluster * 3 + other_cluster
        if weighted_candidates:
            random_candidates = random.Random(rng.randint(0, 2**31 - 1)).choices(
                weighted_candidates,
                k=min(len(weighted_candidates), needed),
            )
            for target in random_candidates:
                if target in existing:
                    continue
                _add_edge(
                    graph,
                    source,
                    target,
                    weight_hint=0.45 if target not in same_cluster else 0.70,
                    rng=rng,
                )
                needed -= 1
                if needed <= 0:
                    break

        if needed <= 0:
            continue
        remaining = [nid for nid in node_ids if nid != source and nid not in {edge.target.id for edge in graph.outgoing(source)}]
        if remaining:
            for target in random.Random(rng.randint(0, 2**31 - 1)).sample(
                remaining,
                k=min(len(remaining), needed),
            ):
                _add_edge(graph, source, target, weight_hint=0.40, rng=rng)

    return graph, specs, ordered


def _build_queries(
    specs: dict[str, DocSpec],
    num_queries: int,
    rng: random.Random,
    node_pool: list[str] | None = None,
) -> list[QueryCase]:
    pool = node_pool or list(specs.keys())
    queries: list[QueryCase] = []
    if not pool:
        return queries

    for qidx in range(num_queries):
        expected_count = rng.choice((2, 3))
        expected_count = min(expected_count, len(pool))
        expected = rng.sample(pool, k=expected_count)
        terms = [rng.choice(specs[node_id].keywords) for node_id in expected]
        query = (
            f"Locate the set of {expected_count} related notes that reference "
            f"{terms[0]} with corroborating context around "
            f"{' ,'.join(terms[1:])}. Query {qidx + 1}."
        )
        queries.append(QueryCase(text=query, expected_node_ids=expected))

    return queries


def _query_from_expected(
    expected: list[str],
    specs: dict[str, DocSpec],
    rng: random.Random,
) -> QueryCase:
    expected = list(expected)
    terms: list[str] = [rng.choice(specs[node_id].keywords) for node_id in expected]
    return QueryCase(
        text=(
            f"Locate the set of {len(expected)} correlated notes for "
            f"{', '.join(terms)} across the same control plane. "
            "Query alias."
        ),
        expected_node_ids=expected,
    )


def _bm25_top_ids(graph: Graph, query: str) -> list[str]:
    scores = ablation._bm25_score(query, list(graph.nodes()))
    return [node_id for node_id, _ in scores[:TOP_K]]


def _seed_from_top(top_ids: list[str], graph: Graph) -> dict[str, float]:
    seeds: dict[str, float] = {}
    for idx, node_id in enumerate(top_ids):
        if graph.get_node(node_id):
            seeds[node_id] = 1.0 - 0.2 * idx
    return seeds


def _recall_at_k(selected: list[str], expected: list[str], k: int = TOP_K) -> float:
    if not expected:
        return 0.0
    expected_set = set(expected)
    hits = sum(1 for node_id in selected[:k] if node_id in expected_set)
    return _safe_divide(hits, len(expected_set))


def _run_bm25_step(graph: Graph, query: QueryCase, distractor_set: set[str] | None = None) -> dict[str, Any]:
    retrieved = _bm25_top_ids(graph, query.text)
    recall = _recall_at_k(retrieved, query.expected_node_ids, TOP_K)
    loaded_nodes = [_node for _node in (graph.get_node(nid) for nid in retrieved) if _node is not None]
    tokens_loaded = sum(_char_count(node.content) for node in loaded_nodes)
    tokens_relevant = sum(
        _char_count(node.content)
        for node in loaded_nodes
        if node.id in set(query.expected_node_ids)
    )
    distractor_selected = 0
    if distractor_set:
        distractor_selected = sum(1 for nid in retrieved if nid in distractor_set)
    return {
        "selected": retrieved,
        "recall": recall,
        "tokens_loaded": tokens_loaded,
        "tokens_relevant": tokens_relevant,
        "distractor_selected": distractor_selected,
        "retrieved_count": len(retrieved),
    }


def _run_crabpath_step(
    graph: Graph,
    query: QueryCase,
    *,
    learn_feedback: float | None,
    max_steps: int = 3,
    distractor_set: set[str] | None = None,
) -> dict[str, Any]:
    bm25_ids = _bm25_top_ids(graph, query.text)
    seeds = _seed_from_top(bm25_ids, graph)
    if not seeds:
        seeds = {bm25_ids[0]: 1.0} if bm25_ids else {}

    firing = None
    if seeds:
        firing = activate(
            graph=graph,
            seeds=seeds,
            max_steps=max_steps,
            top_k=TOP_K,
            reset=True,
        )
        selected = [node.id for node, _ in firing.fired]
    else:
        selected = []

    if not selected:
        selected = bm25_ids

    recall = _recall_at_k(selected, query.expected_node_ids, TOP_K)
    loaded_nodes = [_node for _node in (graph.get_node(nid) for nid in selected) if _node is not None]
    tokens_loaded = sum(_char_count(node.content) for node in loaded_nodes)
    tokens_relevant = sum(
        _char_count(node.content)
        for node in loaded_nodes
        if node.id in set(query.expected_node_ids)
    )
    distractor_selected = 0
    if distractor_set:
        distractor_selected = sum(1 for nid in selected if nid in distractor_set)
    if learn_feedback is not None and firing is not None:
        learn(graph=graph, result=firing, outcome=learn_feedback, rate=0.12)

    return {
        "selected": selected,
        "recall": recall,
        "tokens_loaded": tokens_loaded,
        "tokens_relevant": tokens_relevant,
        "distractor_selected": distractor_selected,
        "retrieved_count": len(selected),
    }


def _precision_relevant_metrics(
    tokens_loaded: int,
    tokens_relevant: int,
) -> tuple[float, float, float]:
    precision = _safe_divide(tokens_relevant, tokens_loaded)
    waste = 1.0 - precision if tokens_loaded > 0 else 0.0
    efficiency = precision
    return precision, waste, efficiency


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _run_context_utilization() -> dict[str, Any]:
    rng = random.Random(SEED + 1)
    graph, specs, _ = _build_graph(size=100, edge_ratio=0.1, seed=rng.randint(0, 2**31 - 1))
    queries = _build_queries(specs, num_queries=100, rng=rng)

    bm25_per_query: list[dict[str, Any]] = []
    cp_per_query: list[dict[str, Any]] = []

    for qidx, query in enumerate(queries, start=1):
        bm25_result = _run_bm25_step(graph=graph, query=query)
        cp_result = _run_crabpath_step(graph=copy.deepcopy(graph), query=query, learn_feedback=None)

        bm_precision, bm_waste, bm_eff = _precision_relevant_metrics(
            bm25_result["tokens_loaded"],
            bm25_result["tokens_relevant"],
        )
        cp_precision, cp_waste, cp_eff = _precision_relevant_metrics(
            cp_result["tokens_loaded"],
            cp_result["tokens_relevant"],
        )

        bm25_per_query.append(
            {
                "query_index": qidx,
                "query": query.text,
                "expected": query.expected_node_ids,
                "selected": bm25_result["selected"],
                "tokens_loaded": bm25_result["tokens_loaded"],
                "tokens_relevant": bm25_result["tokens_relevant"],
                "context_precision": bm_precision,
                "context_waste": bm_waste,
                "token_efficiency": bm_eff,
            }
        )
        cp_per_query.append(
            {
                "query_index": qidx,
                "query": query.text,
                "expected": query.expected_node_ids,
                "selected": cp_result["selected"],
                "tokens_loaded": cp_result["tokens_loaded"],
                "tokens_relevant": cp_result["tokens_relevant"],
                "context_precision": cp_precision,
                "context_waste": cp_waste,
                "token_efficiency": cp_eff,
            }
        )

    bm_summary = {
        "avg_tokens_loaded": _safe_divide(
            sum(item["tokens_loaded"] for item in bm25_per_query),
            len(bm25_per_query),
        ),
        "avg_precision": statistics.mean([item["context_precision"] for item in bm25_per_query]),
        "avg_waste": statistics.mean([item["context_waste"] for item in bm25_per_query]),
        "precision_distribution": _summary_stats([item["context_precision"] for item in bm25_per_query]),
        "waste_distribution": _summary_stats([item["context_waste"] for item in bm25_per_query]),
        "efficiency_distribution": _summary_stats([item["token_efficiency"] for item in bm25_per_query]),
        "query_results": bm25_per_query,
    }
    cp_summary = {
        "avg_tokens_loaded": _safe_divide(
            sum(item["tokens_loaded"] for item in cp_per_query),
            len(cp_per_query),
        ),
        "avg_precision": statistics.mean([item["context_precision"] for item in cp_per_query]),
        "avg_waste": statistics.mean([item["context_waste"] for item in cp_per_query]),
        "precision_distribution": _summary_stats([item["context_precision"] for item in cp_per_query]),
        "waste_distribution": _summary_stats([item["context_waste"] for item in cp_per_query]),
        "efficiency_distribution": _summary_stats([item["token_efficiency"] for item in cp_per_query]),
        "query_results": cp_per_query,
    }

    return {
        "queries": [query.__dict__ for query in queries],
        "bm25": bm_summary,
        "crabpath": cp_summary,
    }


def _inject_distractors(
    base_graph: Graph,
    base_specs: dict[str, DocSpec],
    *,
    distractor_count: int,
    rng: random.Random,
) -> tuple[Graph, dict[str, DocSpec], set[str]]:
    graph = copy.deepcopy(base_graph)
    specs = copy.deepcopy(base_specs)
    if distractor_count <= 0:
        return graph, specs, set()

    pool = [term for spec in specs.values() for term in spec.keywords]
    distractor_nodes: set[str] = set()
    distractor_ids = [
        f"distractor-{i + 1:03d}" for i in range(distractor_count)
    ]
    for idx, node_id in enumerate(distractor_ids):
        anchor = rng.choice(pool) if pool else f"anchor_{idx}"
        keywords = [f"distractor_{node_id}_{idx}", f"{anchor}_{idx}", f"noise_{rng.randint(1, 999)}"]
        content = (
            f"Decoy note {node_id} references {anchor} for topic overlap,"
            f" but remains a synthetic distractor isolated from real answer paths."
        )
        graph.add_node(
            Node(
                id=node_id,
                content=content,
                summary=content[:120],
                cluster_id="distractor",
                metadata={"is_distractor": True},
            )
        )
        specs[node_id] = DocSpec(
            node_id=node_id,
            cluster="distractor",
            keywords=keywords,
            content=content,
        )
        distractor_nodes.add(node_id)

    # Keep distractors isolated from real nodes; connect only amongst themselves.
    for i, node_id in enumerate(distractor_ids):
        next_index = (i + 1) % len(distractor_ids)
        _add_edge(graph, node_id, distractor_ids[next_index], weight_hint=0.40, rng=rng)
        if i % 2 == 0:
            prev_index = (i - 1) % len(distractor_ids)
            _add_edge(graph, node_id, distractor_ids[prev_index], weight_hint=0.35, rng=rng)

    return graph, specs, distractor_nodes


def _run_noise_experiment() -> dict[str, Any]:
    base_seed = SEED + 2
    base_graph, base_specs, _ = _build_graph(size=50, edge_ratio=0.1, seed=base_seed)
    query_rng = random.Random(base_seed + 100)
    base_queries = _build_queries(base_specs, num_queries=50, rng=query_rng)
    levels = [0, 10, 20, 50, 100]
    level_results: list[dict[str, Any]] = []

    for distractor_count in levels:
        injection_seed = query_rng.randint(0, 2**31 - 1)
        injection_rng = random.Random(injection_seed)
        noisy_graph, noisy_specs, distractor_ids = _inject_distractors(
            base_graph=base_graph,
            base_specs=base_specs,
            distractor_count=distractor_count,
            rng=injection_rng,
        )

        bm25_queries = _run_bm25_sequence(
            graph=copy.deepcopy(noisy_graph),
            queries=base_queries,
            learn=False,
            warmup=WARMUP_QUERIES,
            distractor_ids=distractor_ids,
        )
        crab_queries = _run_bm25_sequence(
            graph=copy.deepcopy(noisy_graph),
            queries=base_queries,
            learn=True,
            warmup=WARMUP_QUERIES,
            distractor_ids=distractor_ids,
            method="crabpath",
            specs=noisy_specs,
        )
        level_results.append(
            {
                "distractor_count": distractor_count,
                "distractor_ids": sorted(distractor_ids),
                "bm25": bm25_queries,
                "crabpath": crab_queries,
            }
        )

    return {
        "noise_levels": level_results,
        "rng_seed": base_seed,
        "queries": [query.__dict__ for query in base_queries],
    }


def _run_bm25_sequence(
    graph: Graph,
    queries: list[QueryCase],
    *,
    learn: bool,
    warmup: int = 0,
    distractor_ids: set[str] | None = None,
    method: str = "bm25",
    specs: dict[str, DocSpec] | None = None,
) -> dict[str, Any]:
    del specs
    per_query: list[dict[str, Any]] = []
    eval_recalls: list[float] = []
    total_distractor_selected = 0
    total_selected = 0
    eval_distractor_selected = 0
    eval_selected = 0

    for idx, query in enumerate(queries):
        if method == "bm25":
            result = _run_bm25_step(graph=graph, query=query, distractor_set=distractor_ids)
            step_result = result
            learn_feedback = None
        else:
            learn_feedback = 1.0 if _recall_at_k(_bm25_top_ids(graph, query.text), query.expected_node_ids) > 0.0 else -1.0
            result = _run_crabpath_step(
                graph=graph,
                query=query,
                learn_feedback=learn_feedback if learn else None,
                distractor_set=distractor_ids,
            )

        selected_set = set(result["selected"])
        retrieved = result["retrieved_count"]
        total_distractor_selected += result["distractor_selected"]
        total_selected += retrieved

        recall = result["recall"]
        if idx >= warmup:
            eval_recalls.append(recall)
            eval_distractor_selected += result["distractor_selected"]
            eval_selected += retrieved

        per_query.append(
            {
                "query_index": idx + 1,
                "query": query.text,
                "expected": query.expected_node_ids,
                "selected": result["selected"],
                "recall": recall,
                "tokens_loaded": result["tokens_loaded"],
                "tokens_relevant": result["tokens_relevant"],
                "distractor_selected": result["distractor_selected"],
                "retrieved_count": retrieved,
                "learn_feedback": learn_feedback,
            }
        )

    eval_count = max(1, len(queries) - warmup)
    recall_eval = _safe_divide(sum(eval_recalls), eval_count)
    fpr = _safe_divide(eval_distractor_selected, eval_selected)
    return {
        "recall_eval": recall_eval,
        "fpr_eval": fpr,
        "queries": per_query,
        "eval_count": eval_count,
        "distractor_selected": eval_distractor_selected,
        "selected_total": eval_selected,
        "all_recall_mean": _safe_divide(sum(item["recall"] for item in per_query), len(per_query)),
    }


def _mutate_documents(
    graph: Graph,
    specs: dict[str, DocSpec],
    *,
    ratio: float,
    rng: random.Random,
) -> set[str]:
    total = len(specs)
    mutate_count = max(1, int(total * ratio))
    mutable_ids = list(specs.keys())
    changed_ids = set(rng.sample(mutable_ids, k=min(mutate_count, total)))

    for idx, node_id in enumerate(sorted(changed_ids)):
        spec = specs[node_id]
        new_cluster = rng.choice([cluster for cluster in CLUSTERS if cluster != spec.cluster])
        changed_terms = CLUSTERS[new_cluster]["anchor_terms"]
        new_keywords = [
            f"{new_cluster}_shift_{idx + 1:02d}_{rng.randint(10, 999)}",
            f"{changed_terms[idx % len(changed_terms)]}_{rng.randint(10, 99)}",
            f"{new_cluster}_drift_{rng.randint(100, 999)}",
        ]
        new_content = (
            f"{new_keywords[0]} now encodes {new_cluster} operations, replacing {spec.keywords[0]}"
            f" and referencing {changed_terms[(idx + 1) % len(changed_terms)]}."
            f" This doc has drifted from its original topic."
        )
        node = graph.get_node(node_id)
        if node is not None:
            node.content = new_content
            node.summary = new_content[:120]
        spec.keywords = new_keywords
        spec.content = new_content

    return changed_ids


def _run_temporal_drift() -> dict[str, Any]:
    rng = random.Random(SEED + 3)
    graph_bm25, specs_bm25, _ = _build_graph(size=50, edge_ratio=0.1, seed=rng.randint(0, 2**31 - 1))
    graph_cp = copy.deepcopy(graph_bm25)
    specs_cp = copy.deepcopy(specs_bm25)

    # Phase 1
    phase1_queries = _build_queries(specs_bm25, num_queries=50, rng=rng)
    phase1_bm25 = _run_bm25_sequence(
        graph=graph_bm25,
        queries=phase1_queries,
        learn=False,
        warmup=0,
        method="bm25",
    )
    phase1_crabpath = _run_bm25_sequence(
        graph=graph_cp,
        queries=phase1_queries,
        learn=True,
        warmup=0,
        method="crabpath",
        specs=specs_cp,
    )

    # Phase 2
    specs_bm25_stale = copy.deepcopy(specs_bm25)
    changed_ids = _mutate_documents(
        graph=graph_bm25,
        specs=specs_bm25,
        ratio=0.30,
        rng=rng,
    )
    changed_ids_cp = _mutate_documents(
        graph=graph_cp,
        specs=specs_cp,
        ratio=0.30,
        rng=rng,
    )
    stale_queries = [_query_from_expected(q.expected_node_ids, specs_bm25_stale, rng) for q in phase1_queries[:25]]

    phase2_bm25 = _run_bm25_sequence(
        graph=graph_bm25,
        queries=stale_queries,
        learn=False,
        warmup=0,
        method="bm25",
    )
    phase2_crabpath = _run_bm25_sequence(
        graph=graph_cp,
        queries=stale_queries,
        learn=True,
        warmup=0,
        method="crabpath",
        specs=specs_cp,
    )

    # phase2 adaptive windows after drift
    phase2_bm25_windows: list[dict[str, Any]] = []
    phase2_cp_windows: list[dict[str, Any]] = []
    for start in range(0, len(phase2_bm25["queries"]), WINDOW_SIZE):
        end = min(start + WINDOW_SIZE, len(phase2_bm25["queries"]))
        bm_window = [phase2_bm25["queries"][i]["recall"] for i in range(start, end)]
        cp_window = [phase2_crabpath["queries"][i]["recall"] for i in range(start, end)]
        phase2_bm25_windows.append(
            {
                "window": f"{start + 1}-{end}",
                "recall": _safe_divide(sum(bm_window), len(bm_window)),
            }
        )
        phase2_cp_windows.append(
            {
                "window": f"{start + 1}-{end}",
                "recall": _safe_divide(sum(cp_window), len(cp_window)),
            }
        )

    # Phase 3: new queries for the changed content
    changed_query_pool = list(changed_ids)
    if not changed_query_pool:
        changed_query_pool = list(specs_bm25.keys())
    phase3_queries = _build_queries(specs_bm25, num_queries=25, rng=rng, node_pool=changed_query_pool)
    phase3_bm25 = _run_bm25_sequence(
        graph=graph_bm25,
        queries=phase3_queries,
        learn=False,
        warmup=0,
        method="bm25",
    )
    phase3_cp = _run_bm25_sequence(
        graph=graph_cp,
        queries=phase3_queries,
        learn=True,
        warmup=0,
        method="crabpath",
        specs=specs_cp,
    )

    return {
        "phase1": {
            "bm25_queries": phase1_bm25,
            "crabpath_queries": phase1_crabpath,
            "recall": {
                "bm25": phase1_bm25["all_recall_mean"],
                "crabpath": phase1_crabpath["all_recall_mean"],
            },
        },
        "phase2": {
            "bm25_queries": phase2_bm25,
            "crabpath_queries": phase2_crabpath,
            "adaptation_windows": {
                "bm25": phase2_bm25_windows,
                "crabpath": phase2_cp_windows,
            },
            "query_count": len(stale_queries),
            "changed_nodes": sorted(changed_ids),
            "queries": [query.__dict__ for query in stale_queries],
            "recall": {
                "bm25": phase2_bm25["all_recall_mean"],
                "crabpath": phase2_crabpath["all_recall_mean"],
            },
        },
        "phase3": {
            "bm25_queries": phase3_bm25,
            "crabpath_queries": phase3_cp,
            "queries": [query.__dict__ for query in phase3_queries],
            "recall": {
                "bm25": phase3_bm25["all_recall_mean"],
                "crabpath": phase3_cp["all_recall_mean"],
            },
        },
    }


def _print_context_table(context_summary: dict[str, Any]) -> None:
    print("\nContext utilization")
    print(f"{'method':8} | {'avg tokens loaded':16} | {'avg precision':13} | {'avg waste':9}")
    print("-" * 60)
    for label, data in (("BM25", context_summary["bm25"]), ("CrabPath", context_summary["crabpath"])):
        print(
            f"{label:8} | {data['avg_tokens_loaded']:16.2f}"
            f" | {data['avg_precision']:13.3f} | {data['avg_waste']:9.3f}"
        )


def _print_noise_table(noise_summary: dict[str, Any]) -> None:
    print("\nNoise sensitivity")
    print("distractors | BM25 R@3 | CRAB R@3 | BM25 FPR | CRAB FPR")
    print("-" * 60)
    for entry in noise_summary["noise_levels"]:
        bm = entry["bm25"]
        cr = entry["crabpath"]
        print(
            f"{entry['distractor_count']:10d} | {bm['recall_eval']:8.3f} |"
            f" {cr['recall_eval']:8.3f} | {bm['fpr_eval']:8.3f} | {cr['fpr_eval']:8.3f}"
        )


def _print_temporal_table(temporal_summary: dict[str, Any]) -> None:
    print("\nTemporal drift")
    print("phase | BM25 R@3 | CRAB R@3")
    print("-" * 32)
    print(f"1     | {temporal_summary['phase1']['recall']['bm25']:8.3f} | {temporal_summary['phase1']['recall']['crabpath']:8.3f}")
    phase2_bm = temporal_summary["phase2"]["recall"]["bm25"]
    phase2_cp = temporal_summary["phase2"]["recall"]["crabpath"]
    print(f"2     | {phase2_bm:8.3f} | {phase2_cp:8.3f}")
    print(f"3     | {temporal_summary['phase3']['recall']['bm25']:8.3f} | {temporal_summary['phase3']['recall']['crabpath']:8.3f}")
    print("\nPhase 2 adaptation windows (Recall@3)")
    print("window | BM25 R@3 | CRAB R@3")
    print("-" * 34)
    for bm, cp in zip(
        temporal_summary["phase2"]["adaptation_windows"]["bm25"],
        temporal_summary["phase2"]["adaptation_windows"]["crabpath"],
    ):
        print(f"{bm['window']:>6} | {bm['recall']:8.3f} | {cp['recall']:8.3f}")


def run_experiment() -> dict[str, Any]:
    context_summary = _run_context_utilization()
    noise_summary = _run_noise_experiment()
    temporal_summary = _run_temporal_drift()

    payload = {
        "seed": SEED,
        "top_k": TOP_K,
        "context_utilization": context_summary,
        "noise_sensitivity": noise_summary,
        "temporal_drift": temporal_summary,
    }
    return payload


def main() -> None:
    payload = run_experiment()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _print_context_table(payload["context_utilization"])
    _print_noise_table(payload["noise_sensitivity"])
    _print_temporal_table(payload["temporal_drift"])
    print(f"\nWrote results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
