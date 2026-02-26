#!/usr/bin/env python3
"""NIAH multi-needle and scaling benchmark for CrabPath vs BM25."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.graph import Edge, Graph, Node
from crabpath.legacy.activation import activate, learn
from scripts import ablation_study

SEED = 2026
RESULTS_PATH = ROOT / "scripts" / "niah_scaling_results.json"

NIAH_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]
SCALING_SIZES = [20, 50, 100, 200, 500, 1000, 2000, 5000]
QUICK_NIAH_SIZES = [50, 200, 1000]
QUICK_SCALING_SIZES = [20, 100, 500]
DEFAULT_NIAH_QUERY_COUNT = 30
QUICK_NIAH_QUERY_COUNT = 10
DEFAULT_SCALING_QUERY_COUNT = 100
QUICK_SCALING_QUERY_COUNT = 30
DEFAULT_NIAH_WARMUP = 15
QUICK_NIAH_WARMUP = 5
DEFAULT_SCALING_WARMUP = 50
QUICK_SCALING_WARMUP = 15
DEFAULT_SCALING_QUERY_DISTRIBUTION = {1: 60, 2: 30, 3: 10}
QUICK_SCALING_QUERY_DISTRIBUTION = {1: 15, 2: 10, 3: 5}
NIAH_K_VALUES = [1, 2, 3, 5]
CLUSTERS = {
    "deploy": {
        "topic": "deployment operations",
        "anchor_terms": [
            "artifact",
            "canary",
            "rollout",
            "release",
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
            "dashboard",
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
    max_hops: int


def _safe_divide(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _build_cluster_distribution(size: int) -> dict[str, int]:
    base = size // len(CLUSTERS)
    extra = size % len(CLUSTERS)
    counts: dict[str, int] = {}
    for i, cluster in enumerate(CLUSTERS):
        counts[cluster] = base + (1 if i < extra else 0)
    return counts


def _make_documents(size: int, rng: random.Random) -> tuple[dict[str, DocSpec], list[DocSpec]]:
    counts = _build_cluster_distribution(size)
    node_specs: dict[str, DocSpec] = {}
    ordered_docs: list[DocSpec] = []

    for cluster, count in counts.items():
        terms = CLUSTERS[cluster]["anchor_terms"]
        for i in range(count):
            node_index = f"{cluster}_{i + 1:05d}"
            keywords = [
                f"{cluster}_topic_{i + 1:05d}_{rng.randint(10, 999)}",
                f"{cluster}_signal_{i + 1:05d}_{rng.randint(10, 999)}",
            ]

            sentences = [
                (
                    f"{keywords[0]} captures {CLUSTERS[cluster]['topic']} guidance in sentence one."
                    f" It mentions {terms[i % len(terms)]} and {terms[(i + 1) % len(terms)]}."
                ),
                (
                    f"The note links {keywords[1]} to operations evidence around"
                    f" {cluster} flow and practical follow-on action."
                ),
            ]

            if rng.random() < 0.75:
                sibling_term = terms[(i + 2) % len(terms)]
                sentences.append(
                    f"It references {sibling_term} in a downstream handoff."
                )

            if rng.random() < 0.55:
                sentences.append(
                    f"Cross-cluster review expects {keywords[0]} to align with"
                    f" {keywords[1]} under service pressure."
                )

            content = " ".join(sentences)
            node_id = f"{cluster}_{node_index}"
            spec = DocSpec(
                node_id=node_id,
                cluster=cluster,
                keywords=keywords,
                content=content,
            )
            node_specs[node_id] = spec
            ordered_docs.append(spec)

    return node_specs, ordered_docs


def _add_edge(graph: Graph, source: str, target: str, *, weight_hint: float, rng: random.Random) -> None:
    if source == target:
        return
    existing = graph.get_edge(source, target)
    weight = weight_hint + rng.uniform(-0.06, 0.06)
    if existing is None:
        graph.add_edge(Edge(source=source, target=target, weight=max(0.05, min(0.95, weight))))


def _build_cluster_graph(size: int, edge_ratio: float, rng: random.Random) -> tuple[Graph, dict[str, DocSpec], list[DocSpec]]:
    node_specs, ordered_docs = _make_documents(size, rng)
    graph = Graph()

    for spec in ordered_docs:
        graph.add_node(
            Node(
                id=spec.node_id,
                content=spec.content,
                cluster_id=spec.cluster,
            )
        )

    node_ids = [spec.node_id for spec in ordered_docs]
    cluster_nodes: dict[str, list[str]] = {}
    for cluster in CLUSTERS:
        cluster_nodes[cluster] = [spec.node_id for spec in ordered_docs if spec.cluster == cluster]

    # Baseline connectivity chain.
    ordered = node_ids.copy()
    rng.shuffle(ordered)
    for i in range(len(ordered) - 1):
        _add_edge(graph, ordered[i], ordered[i + 1], weight_hint=0.60, rng=rng)
        _add_edge(graph, ordered[i + 1], ordered[i], weight_hint=0.60, rng=rng)

    # Ensure at least one inter-cluster bridge in the chain of clusters.
    cluster_names = list(CLUSTERS.keys())
    for i in range(len(cluster_names) - 1):
        source_candidates = cluster_nodes[cluster_names[i]]
        target_candidates = cluster_nodes[cluster_names[i + 1]]
        if source_candidates and target_candidates:
            _add_edge(graph, rng.choice(source_candidates), rng.choice(target_candidates), weight_hint=0.50, rng=rng)
            _add_edge(graph, rng.choice(target_candidates), rng.choice(source_candidates), weight_hint=0.50, rng=rng)

    if edge_ratio < 1.0:
        desired_out = max(1, int(round(edge_ratio * (size - 1))))
    else:
        desired_out = size - 1

    for source in node_ids:
        source_cluster = node_specs[source].cluster
        existing = set(graph._outgoing.get(source, []))
        needed = max(0, desired_out - len(existing))
        if needed <= 0:
            continue

        same_cluster = [n for n in cluster_nodes[source_cluster] if n != source and n not in existing]
        other_cluster = [n for n in node_ids if n != source and n not in existing and node_specs[n].cluster != source_cluster]

        if edge_ratio == 1.0:
            candidates = [n for n in node_ids if n != source and n not in existing]
        else:
            weighted_candidates = same_cluster * 3 + other_cluster
            if weighted_candidates:
                max_picks = min(len(weighted_candidates), needed)
                candidates = []
                seen: set[str] = set(existing)
                for candidate in rng.sample(weighted_candidates, k=max_picks):
                    if candidate not in seen:
                        candidates.append(candidate)
                        seen.add(candidate)
                    if len(candidates) >= needed:
                        break
            else:
                candidates = []

        if len(candidates) < needed:
            remaining = [n for n in node_ids if n != source and n not in existing and n not in candidates]
            if remaining:
                candidates.extend(
                        rng.sample(
                            remaining,
                            k=min(needed - len(candidates), len(remaining)),
                        )
                )

        for target in candidates[:needed]:
            weight_hint = 0.72 if node_specs[target].cluster == source_cluster else 0.48
            _add_edge(graph, source, target, weight_hint=weight_hint, rng=rng)

    return graph, node_specs, ordered_docs


def _draw_path(graph: Graph, rng: random.Random, start: str, length: int) -> list[str]:
    if length <= 0:
        return [start]
    path = [start]
    for _ in range(length):
        outgoing = [target.id for target, _ in graph.outgoing(path[-1])]
        if not outgoing:
            return []
        path.append(rng.choice(outgoing))
    return path


def _query_keywords(specs: dict[str, DocSpec], node_id: str, rng: random.Random) -> str:
    return rng.choice(specs[node_id].keywords)


def _seed_from_bm25(ranking: list[str], top_n: int) -> dict[str, float]:
    seeds: dict[str, float] = {}
    for idx, node_id in enumerate(ranking[:top_n]):
        seeds[node_id] = max(0.15, 1.0 - 0.2 * idx)
    return seeds


def _metric_multi(retrieved: list[str], expected: list[str]) -> dict[str, float]:
    if not expected:
        return {"recall@K": 0.0, "partial_recall": 0.0, "mrr": 0.0}

    expected_set = set(expected)
    hits: list[str] = []
    hit_set = set()
    for rank, node_id in enumerate(retrieved, start=1):
        if node_id in expected_set and node_id not in hit_set:
            hit_set.add(node_id)
            hits.append(node_id)
            if len(hits) == 1:
                first_rank = rank

    if len(hits) == 0:
        first_rank = None
    found_count = len(hits)

    return {
        "recall@K": 1.0 if found_count == len(expected_set) else 0.0,
        "partial_recall": _safe_divide(found_count, len(expected_set)),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
    }


def _metric_at_3(ranking: list[str], expected: list[str]) -> dict[str, float]:
    expected_set = set(expected)
    if not expected_set:
        return {"recall@3": 0.0, "ndcg@3": 0.0, "mrr@3": 0.0}

    top3 = ranking[:3]
    hits = 0
    for node_id in top3:
        if node_id in expected_set:
            hits += 1
    recall3 = _safe_divide(hits, len(expected_set))

    ideal_hits = min(len(expected_set), 3)
    ideal_dcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal_hits))
    dcg = 0.0
    for idx, node_id in enumerate(top3):
        if node_id in expected_set:
            dcg += 1.0 / math.log2(idx + 2.0)
    ndcg3 = _safe_divide(dcg, ideal_dcg)

    mrr3 = 0.0
    for idx, node_id in enumerate(top3, start=1):
        if node_id in expected_set:
            mrr3 = 1.0 / idx
            break

    return {"recall@3": recall3, "ndcg@3": ndcg3, "mrr@3": mrr3}


def _build_multi_needle_configuration(
    size: int,
    k: int,
    rng: random.Random,
    query_count: int,
) -> tuple[Graph, dict[str, DocSpec], list[QueryCase], list[str]]:
    graph, specs, ordered_docs = _build_cluster_graph(size, edge_ratio=0.1, rng=rng)
    base_node_ids = list(specs.keys())

    cluster_names = list(CLUSTERS.keys())
    needle_clusters = (
        cluster_names
        if k >= len(cluster_names)
        else rng.sample(cluster_names, k=k)
    )
    if k > len(cluster_names):
        needle_clusters = [cluster_names[i % len(cluster_names)] for i in range(k)]

    needle_ids: list[str] = []
    needle_paths: list[list[str]] = []

    for idx in range(k):
        depth = rng.randint(1, 3)
        needle_cluster = needle_clusters[idx]
        path: list[str]
        for _ in range(300):
            start = rng.choice(base_node_ids)
            path = _draw_path(graph, rng, start, depth)
            if path and len(path) == depth + 1:
                break
        else:
            path = [rng.choice(base_node_ids)]

        needle_id = f"needle_{size}_{idx + 1:02d}_{rng.randint(1000, 9999)}"
        needle_keywords = [
            f"needle_{needle_id}_anchor",
            f"needle_{needle_id}_evidence_{rng.randint(100, 999)}",
        ]
        needle_content = (
            "Needle artifact for multi-target retrieval stress."
            f" {needle_keywords[0]} indicates a hidden signal tied to {needle_cluster}."
            f" Include token {needle_keywords[1]} for uniqueness."
        )
        graph.add_node(
            Node(
                id=needle_id,
                content=needle_content,
                cluster_id=needle_cluster,
                metadata={"is_needle": True},
            )
        )
        specs[needle_id] = DocSpec(
            node_id=needle_id,
            cluster=needle_cluster,
            keywords=needle_keywords,
            content=needle_content,
        )
        ordered_docs.append(specs[needle_id])

        predecessor = path[-1] if path else start
        _add_edge(graph=graph, source=predecessor, target=needle_id, weight_hint=0.64, rng=rng)
        needle_ids.append(needle_id)
        needle_paths.append(path + [needle_id])

    expected = list(needle_ids)
    queries: list[QueryCase] = []

    for q_idx in range(query_count):
        clauses = []
        for path in needle_paths:
            if not path:
                continue
            # Reference keywords from the chain into this needle, not the needle content itself.
            chain_terms = [_query_keywords(specs, node_id, rng) for node_id in path[:-1] if node_id in specs]
            if not chain_terms:
                continue
            clauses.append(
                f"Trace {chain_terms[0]} -> {' -> '.join(chain_terms[1:])} through each handoff."
            )
        if not clauses:
            clauses = ["Locate the final nodes for the requested chain targets."]
        text = (
            f"For the scenario set {q_idx + 1}, recover all target outcomes: "
            f"{' Also '.join(clauses)} Return the full target set."
        )
        queries.append(QueryCase(text=text, expected_node_ids=expected, max_hops=3))

    return graph, specs, queries, expected


def _run_multi_needle_queries(
    graph: Graph,
    specs: dict[str, DocSpec],
    queries: list[QueryCase],
    k: int,
    warmup: int,
    *,
    use_crab: bool,
) -> dict[str, Any]:
    initial_inhibitory = sum(1 for edge in graph.edges() if edge.weight < 0.0)

    per_query: list[dict[str, Any]] = []
    total_tokens_bm25 = 0
    total_tokens_cp = 0
    eval_tokens_bm25 = 0
    eval_tokens_cp = 0
    nodes_visited_bm25 = 0
    nodes_visited_cp = 0

    bm25_recall_scores: list[float] = []
    bm25_partial_scores: list[float] = []
    bm25_mrr_scores: list[float] = []
    cp_recall_scores: list[float] = []
    cp_partial_scores: list[float] = []
    cp_mrr_scores: list[float] = []

    for idx, query in enumerate(queries, start=1):
        bm25_ranking = [node_id for node_id, _ in ablation_study._bm25_score(query.text, list(graph.nodes()))[:k]]
        bm25_metrics = _metric_multi(bm25_ranking, query.expected_node_ids)
        bm25_tokens = 0
        for node_id in bm25_ranking:
            node = graph.get_node(node_id)
            if node is not None:
                bm25_tokens += _token_count(node.content)
        total_tokens_bm25 += bm25_tokens
        nodes_visited_bm25 += len(bm25_ranking)

        cp_ranking: list[str]
        cp_nodes: int
        if use_crab:
            seeds = _seed_from_bm25(bm25_ranking, max(1, min(5, k)))
            if not seeds:
                fallback_nodes = [node.id for node in graph.nodes()]
                seeds = {fallback_nodes[0]: 1.0} if fallback_nodes else {}
            firing = activate(
                graph=graph,
                seeds=seeds,
                max_steps=max(2, query.max_hops + 1),
                top_k=max(5, k),
            )
            cp_ranking = [node.id for node, _ in firing.fired]
            cp_nodes = len(cp_ranking)
            if not cp_ranking:
                cp_ranking = bm25_ranking[:k]
            reward = max(0.0, _metric_multi(cp_ranking, query.expected_node_ids)["partial_recall"]) * 2.0 - 1.0
            learn(graph=graph, result=firing, outcome=reward, rate=0.12)
            nodes_visited_cp += cp_nodes
        else:
            cp_ranking = bm25_ranking[:k]
            cp_nodes = len(cp_ranking)
            nodes_visited_cp += cp_nodes

        cp_ranking = cp_ranking[:k]
        cp_metrics = _metric_multi(cp_ranking, query.expected_node_ids)
        cp_tokens = 0
        for node_id in cp_ranking:
            node = graph.get_node(node_id)
            if node is not None:
                cp_tokens += _token_count(node.content)
        total_tokens_cp += cp_tokens

        if idx > warmup:
            eval_tokens_bm25 += bm25_tokens
            eval_tokens_cp += cp_tokens
            bm25_recall_scores.append(bm25_metrics["recall@K"])
            bm25_partial_scores.append(bm25_metrics["partial_recall"])
            bm25_mrr_scores.append(bm25_metrics["mrr"])
            cp_recall_scores.append(cp_metrics["recall@K"])
            cp_partial_scores.append(cp_metrics["partial_recall"])
            cp_mrr_scores.append(cp_metrics["mrr"])

        per_query.append(
            {
                "query": query.text,
                "expected": query.expected_node_ids,
                "bm25_ranking": bm25_ranking,
                "crabpath_ranking": cp_ranking,
                "bm25_metrics": bm25_metrics,
                "crabpath_metrics": cp_metrics,
                "tokens": {"bm25": bm25_tokens, "crabpath": cp_tokens},
            }
        )

    eval_count = max(1, len(per_query) - warmup if warmup < len(per_query) else 0)
    if warmup >= len(per_query):
        eval_count = len(per_query)

    return {
        "metrics": {
            "recall@K": {
                "mean": sum(cp_recall_scores) / eval_count if use_crab else sum(bm25_recall_scores) / eval_count,
                "ci": ablation_study.bootstrap_ci(cp_recall_scores if use_crab else bm25_recall_scores),
            },
            "partial_recall": {
                "mean": sum(cp_partial_scores) / eval_count if use_crab else sum(bm25_partial_scores) / eval_count,
                "ci": ablation_study.bootstrap_ci(cp_partial_scores if use_crab else bm25_partial_scores),
            },
            "mrr": {
                "mean": sum(cp_mrr_scores) / eval_count if use_crab else sum(bm25_mrr_scores) / eval_count,
                "ci": ablation_study.bootstrap_ci(cp_mrr_scores if use_crab else bm25_mrr_scores),
            },
        },
        "avg_tokens_loaded": {
            "bm25": _safe_divide(eval_tokens_bm25, eval_count),
            "crabpath": _safe_divide(eval_tokens_cp, eval_count),
        },
        "nodes_visited": {
            "bm25": nodes_visited_bm25,
            "crabpath": nodes_visited_cp,
        },
        "inhibitory_edges": {
            "initial": initial_inhibitory,
            "final": sum(1 for edge in graph.edges() if edge.weight < 0.0),
            "formed": max(0, sum(1 for edge in graph.edges() if edge.weight < 0.0) - initial_inhibitory),
        },
        "query_results": per_query,
    }

def _run_niah_benchmarks(
    sizes: list[int],
    query_count: int,
    warmup_queries: int,
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    all_results: list[dict[str, Any]] = []

    for size in sizes:
        for k in NIAH_K_VALUES:
            run_seed = rng.randint(0, 2**31 - 1)
            graph, specs, queries, expected = _build_multi_needle_configuration(
                size=size,
                k=k,
                rng=random.Random(run_seed),
                query_count=query_count,
            )

            bm25 = _run_multi_needle_queries(
                graph=copy.deepcopy(graph),
                specs=copy.deepcopy(specs),
                queries=queries,
                k=k,
                warmup=warmup_queries,
                use_crab=False,
            )
            crab = _run_multi_needle_queries(
                graph=copy.deepcopy(graph),
                specs=copy.deepcopy(specs),
                queries=queries,
                k=k,
                warmup=warmup_queries,
                use_crab=True,
            )

            all_results.append(
                {
                    "size": size,
                    "k": k,
                    "seed": run_seed,
                    "num_needles": k,
                    "expected": expected,
                    "query_count": len(queries),
                    "bm25": bm25,
                    "crabpath": crab,
                }
            )

    return all_results


def _build_scaling_queries(
    graph: Graph,
    specs: dict[str, DocSpec],
    rng: random.Random,
    query_distribution: dict[int, int],
) -> list[QueryCase]:
    query_counts = dict(query_distribution)
    node_ids = [node_id for node_id in specs.keys()]
    queries: list[QueryCase] = []
    seen: set[tuple[str, ...]] = set()

    for hops, count in query_counts.items():
        produced = 0
        attempts = 0
        while produced < count and attempts < 20_000:
            attempts += 1
            start = rng.choice(node_ids)
            path = _draw_path(graph, rng, start, hops)
            if not path or len(path) != hops + 1:
                continue

            path_key = tuple(path)
            if path_key in seen:
                continue

            if hops == 1:
                anchors = [_query_keywords(specs, path[0], rng)]
                text = (
                    f"Find the handoff document that follows {anchors[0]}."
                )
            elif hops == 2:
                anchors = [_query_keywords(specs, path[0], rng), _query_keywords(specs, path[1], rng)]
                text = (
                    f"From {anchors[0]}, continue to {anchors[1]} and return the final follow-up node."
                )
            else:
                anchors = [
                    _query_keywords(specs, path[0], rng),
                    _query_keywords(specs, path[1], rng),
                    _query_keywords(specs, path[2], rng),
                ]
                text = (
                    f"Trace {anchors[0]} to {anchors[1]} then to {anchors[2]}, then return the target."
                )

            queries.append(
                QueryCase(
                    text=text,
                    expected_node_ids=[path[-1]],
                    max_hops=hops,
                )
            )
            seen.add(path_key)
            produced += 1

    return queries[:100]


def _run_scale_configuration(
    size: int,
    seed: int,
    query_distribution: dict[int, int],
    query_count: int,
    warmup_queries: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    graph, specs, _ = _build_cluster_graph(size, edge_ratio=0.1, rng=rng)
    queries = _build_scaling_queries(
        graph=graph,
        specs=specs,
        rng=rng,
        query_distribution=query_distribution,
    )
    queries = queries[:query_count]

    bm25 = _run_scaling_suite(
        graph=copy.deepcopy(graph),
        queries=queries,
        warmup=0,
        use_crab=False,
    )
    crab = _run_scaling_suite(
        graph=copy.deepcopy(graph),
        queries=queries,
        warmup=warmup_queries,
        use_crab=True,
    )

    return {
        "size": size,
        "seed": seed,
        "query_count": len(queries),
        "query_distribution": query_distribution,
        "bm25": bm25,
        "crabpath": crab,
    }


def _run_scaling_suite(
    graph: Graph,
    queries: list[QueryCase],
    *,
    use_crab: bool,
    warmup: int,
) -> dict[str, Any]:
    initial_inhibitory = sum(1 for edge in graph.edges() if edge.weight < 0.0)

    per_query: list[dict[str, Any]] = []
    total_tokens_bm25 = 0
    total_tokens_cp = 0
    eval_tokens_bm25 = 0
    eval_tokens_cp = 0

    bm25_recall: list[float] = []
    bm25_ndcg: list[float] = []
    bm25_mrr: list[float] = []
    cp_recall: list[float] = []
    cp_ndcg: list[float] = []
    cp_mrr: list[float] = []
    nodes_visited_bm25 = 0
    nodes_visited_cp = 0

    for idx, query in enumerate(queries, start=1):
        bm25_ranking = [node_id for node_id, _ in ablation_study._bm25_score(query.text, list(graph.nodes()))[:3]]
        bm25_metrics = _metric_at_3(bm25_ranking, query.expected_node_ids)
        bm25_tokens = 0
        for node_id in bm25_ranking:
            node = graph.get_node(node_id)
            if node is not None:
                bm25_tokens += _token_count(node.content)
        total_tokens_bm25 += bm25_tokens
        nodes_visited_bm25 += len(bm25_ranking)

        cp_ranking: list[str]
        cp_nodes = 0
        if use_crab:
            seeds = _seed_from_bm25(bm25_ranking, top_n=3)
            if not seeds:
                fallback_nodes = [node.id for node in graph.nodes()]
                seeds = {fallback_nodes[0]: 1.0} if fallback_nodes else {}
            firing = activate(
                graph=graph,
                seeds=seeds,
                max_steps=max(2, query.max_hops + 1),
                top_k=3,
            )
            cp_ranking = [node.id for node, _ in firing.fired]
            cp_nodes = len(cp_ranking)
            nodes_visited_cp += cp_nodes

            if not cp_ranking:
                cp_ranking = bm25_ranking
            cp_metrics = _metric_at_3(cp_ranking, query.expected_node_ids)
            reward = (1.0 if cp_metrics["recall@3"] > 0.0 else 0.0) * 2.0 - 1.0
            learn(graph=graph, result=firing, outcome=reward, rate=0.12)
        else:
            cp_ranking = bm25_ranking
            cp_nodes = len(cp_ranking)
            nodes_visited_cp += cp_nodes
            cp_metrics = bm25_metrics

        cp_ranking = cp_ranking[:3]
        cp_tokens = 0
        for node_id in cp_ranking:
            node = graph.get_node(node_id)
            if node is not None:
                cp_tokens += _token_count(node.content)
        total_tokens_cp += cp_tokens

        if idx > warmup:
            eval_tokens_bm25 += bm25_tokens
            eval_tokens_cp += cp_tokens
            bm25_recall.append(bm25_metrics["recall@3"])
            bm25_ndcg.append(bm25_metrics["ndcg@3"])
            bm25_mrr.append(bm25_metrics["mrr@3"])
            cp_recall.append(cp_metrics["recall@3"])
            cp_ndcg.append(cp_metrics["ndcg@3"])
            cp_mrr.append(cp_metrics["mrr@3"])

        per_query.append(
            {
                "query": query.text,
                "expected": query.expected_node_ids,
                "bm25_ranking": bm25_ranking,
                "crabpath_ranking": cp_ranking,
                "bm25_metrics": bm25_metrics,
                "crabpath_metrics": cp_metrics,
                "tokens": {"bm25": bm25_tokens, "crabpath": cp_tokens},
            }
        )

    eval_count = max(1, len(per_query) - warmup if warmup < len(per_query) else 0)
    if warmup >= len(per_query):
        eval_count = len(per_query)

    return {
        "metrics": {
            "recall@3": {
                "mean": sum(cp_recall if use_crab else bm25_recall) / eval_count,
                "ci": ablation_study.bootstrap_ci(cp_recall if use_crab else bm25_recall),
            },
            "ndcg@3": {
                "mean": sum(cp_ndcg if use_crab else bm25_ndcg) / eval_count,
                "ci": ablation_study.bootstrap_ci(cp_ndcg if use_crab else bm25_ndcg),
            },
            "mrr@3": {
                "mean": sum(cp_mrr if use_crab else bm25_mrr) / eval_count,
                "ci": ablation_study.bootstrap_ci(cp_mrr if use_crab else bm25_mrr),
            },
        },
        "avg_tokens_loaded": {
            "bm25": _safe_divide(eval_tokens_bm25, eval_count),
            "crabpath": _safe_divide(eval_tokens_cp, eval_count),
        },
        "nodes_visited": {
            "bm25": nodes_visited_bm25,
            "crabpath": nodes_visited_cp,
        },
        "inhibitory_edges": {
            "initial": initial_inhibitory,
            "final": sum(1 for edge in graph.edges() if edge.weight < 0.0),
            "formed": max(0, sum(1 for edge in graph.edges() if edge.weight < 0.0) - initial_inhibitory),
        },
        "query_results": per_query,
    }


def _run_scaling_benchmarks(
    sizes: list[int],
    query_distribution: dict[int, int],
    query_count: int,
    warmup_queries: int,
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    results: list[dict[str, Any]] = []
    for size in sizes:
        seed = rng.randint(0, 2**31 - 1)
        results.append(
            _run_scale_configuration(
                size=size,
                seed=seed,
                query_distribution=query_distribution,
                query_count=query_count,
                warmup_queries=warmup_queries,
            )
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NIAH and scaling benchmarks for CrabPath vs BM25."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced problem sizes and query counts for faster execution.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum graph size to include for both NIAH and scaling runs.",
    )
    return parser.parse_args()


def _select_sizes(base_sizes: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return base_sizes
    return [size for size in base_sizes if size <= limit]


def _normalize_scaling_distribution(total_queries: int) -> dict[int, int]:
    if total_queries <= 0:
        return {1: 0, 2: 0, 3: 0}
    # Keep a simple 1-hop / 2-hop / 3-hop workload shape from 60/30/10.
    one_hop = math.floor(total_queries * 0.6)
    two_hop = math.floor(total_queries * 0.3)
    three_hop = max(0, total_queries - one_hop - two_hop)
    return {1: one_hop, 2: two_hop, 3: three_hop}


def _print_niah_summary(results: list[dict[str, Any]], sizes: list[int]) -> None:
    print("\nNeedle-in-a-Haystack (Multi-Needle) Recall@K")
    header = ["Size"]
    for k in NIAH_K_VALUES:
        header.extend([f"K={k} BM25", f"K={k} CRAB"])
    print(" | ".join(header))
    print("-" * (len(header) * 12))

    by_size: dict[int, dict[int, dict[str, float]]] = {}
    for result in results:
        size_bucket = by_size.setdefault(result["size"], {})
        k_bucket = size_bucket.setdefault(result["k"], {})
        k_bucket["bm25"] = result["bm25"]["metrics"]["recall@K"]["mean"]
        k_bucket["crab"] = result["crabpath"]["metrics"]["recall@K"]["mean"]

    for size in sizes:
        row = [str(size)]
        for k in NIAH_K_VALUES:
            values = by_size.get(size, {}).get(k, {})
            row.append(f"{values.get('bm25', 0.0):.3f}")
            row.append(f"{values.get('crab', 0.0):.3f}")
        print(" | ".join(row))

    print("\nNeedle-in-a-Haystack (Multi-Needle) partial recall and MRR (last 15 eval queries)")
    print("Size | K=1 Partial BM25 | K=1 Partial CRAB | K=1 MRR BM25 | K=1 MRR CRAB | ...")
    print("-" * 95)
    for size in sizes:
        row = [str(size)]
        for k in NIAH_K_VALUES:
            values = by_size.get(size, {}).get(k, {})
            # Locate original records for detailed metrics.
            target = next(
                (
                    item for item in results
                    if item["size"] == size and item["k"] == k
                ),
                None,
            )
            if target is None:
                row.extend(["0.000", "0.000", "0.000", "0.000"])
                continue
            row.append(f"{target['bm25']['metrics']['partial_recall']['mean']:.3f}")
            row.append(f"{target['crabpath']['metrics']['partial_recall']['mean']:.3f}")
            row.append(f"{target['bm25']['metrics']['mrr']['mean']:.3f}")
            row.append(f"{target['crabpath']['metrics']['mrr']['mean']:.3f}")
        print(" | ".join(row))


def _print_scaling_summary(results: list[dict[str, Any]]) -> None:
    print("\nScaling Curves (Recall@3 @ edge_ratio=0.1)")
    print("Size | BM25 Recall@3 | CRAB Recall@3 | BM25 NDCG@3 | CRAB NDCG@3 | BM25 MRR@3 | CRAB MRR@3 | CRAB Tokens/query")
    print("-" * 125)
    for item in results:
        size = item["size"]
        bm = item["bm25"]["metrics"]["recall@3"]["mean"]
        cp = item["crabpath"]["metrics"]["recall@3"]["mean"]
        bm_ndcg = item["bm25"]["metrics"]["ndcg@3"]["mean"]
        cp_ndcg = item["crabpath"]["metrics"]["ndcg@3"]["mean"]
        bm_mrr = item["bm25"]["metrics"]["mrr@3"]["mean"]
        cp_mrr = item["crabpath"]["metrics"]["mrr@3"]["mean"]
        cp_tokens = item["crabpath"]["avg_tokens_loaded"]["crabpath"]
        print(
            f"{size} | {bm:.3f} | {cp:.3f} | {bm_ndcg:.3f} | {cp_ndcg:.3f} |"
            f" {bm_mrr:.3f} | {cp_mrr:.3f} | {cp_tokens:.1f}"
        )


def main() -> None:
    args = _parse_args()

    if args.quick:
        niah_sizes = QUICK_NIAH_SIZES
        scaling_sizes = QUICK_SCALING_SIZES
        niah_query_count = QUICK_NIAH_QUERY_COUNT
        scaling_query_count = QUICK_SCALING_QUERY_COUNT
        niah_warmup = QUICK_NIAH_WARMUP
        scaling_warmup = QUICK_SCALING_WARMUP
        scaling_distribution = QUICK_SCALING_QUERY_DISTRIBUTION
    else:
        niah_sizes = NIAH_SIZES
        scaling_sizes = SCALING_SIZES
        niah_query_count = DEFAULT_NIAH_QUERY_COUNT
        scaling_query_count = DEFAULT_SCALING_QUERY_COUNT
        niah_warmup = DEFAULT_NIAH_WARMUP
        scaling_warmup = DEFAULT_SCALING_WARMUP
        scaling_distribution = DEFAULT_SCALING_QUERY_DISTRIBUTION

    scaling_distribution = {
        hop: count
        for hop, count in scaling_distribution.items()
        if count > 0
    }
    if sum(scaling_distribution.values()) != scaling_query_count:
        scaling_distribution = _normalize_scaling_distribution(scaling_query_count)

    niah_sizes = _select_sizes(niah_sizes, args.max_size)
    scaling_sizes = _select_sizes(scaling_sizes, args.max_size)

    niah_results = _run_niah_benchmarks(
        sizes=niah_sizes,
        query_count=niah_query_count,
        warmup_queries=niah_warmup,
    )
    scaling_results = _run_scaling_benchmarks(
        sizes=scaling_sizes,
        query_distribution=scaling_distribution,
        query_count=scaling_query_count,
        warmup_queries=scaling_warmup,
    )

    payload = {
        "seed": SEED,
        "niah": {
            "sizes": niah_sizes,
            "k_values": NIAH_K_VALUES,
            "query_count": niah_query_count,
            "warmup_queries": niah_warmup,
            "evaluate_last": niah_query_count - niah_warmup,
            "results": niah_results,
        },
        "scaling": {
            "sizes": scaling_sizes,
            "query_distribution": {f"{k}-hop": v for k, v in scaling_distribution.items()},
            "query_count": scaling_query_count,
            "warmup_queries": scaling_warmup,
            "evaluate_last": scaling_query_count - scaling_warmup,
            "results": scaling_results,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_niah_summary(niah_results, niah_sizes)
    _print_scaling_summary(scaling_results)
    print(f"\nWrote detailed benchmark output to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
