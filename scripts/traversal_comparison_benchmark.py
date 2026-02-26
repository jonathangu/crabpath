#!/usr/bin/env python3
"""Traversal comparison benchmark for damping/penalty mechanisms and BM25 baseline."""

from __future__ import annotations

import argparse
import copy
import math
import random
import re
from dataclasses import dataclass
from collections import Counter
from typing import Any

from crabpath import Edge, Graph, Node, MemoryController
from crabpath.controller import ControllerConfig
from crabpath.inhibition import inhibition_stats


SEED = 2026
EDGE_RATIO = 0.1
SIZES = [50, 100, 200, 500, 1000]
QUICK_SIZES = [50, 200]
WARMUP_QUERIES = 50
EVAL_QUERIES = 50
TOTAL_QUERIES = WARMUP_QUERIES + EVAL_QUERIES


@dataclass
class QueryCase:
    text: str
    expected: list[str]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_graph(size: int, edge_ratio: float, rng: random.Random) -> tuple[Graph, dict[str, str]]:
    graph = Graph()
    node_tokens: dict[str, str] = {}
    for idx in range(size):
        node_id = f"n{idx:05d}"
        token = f"token_{idx:05d}"
        content = (
            f"{token} {node_id} this node discusses token {token} "
            f"with additional signal {rng.randint(10, 99)}."
        )
        graph.add_node(Node(id=node_id, content=content))
        node_tokens[node_id] = token

    if size <= 1:
        return graph, node_tokens

    desired_out = max(1, int(round(edge_ratio * (size - 1))))
    all_nodes = list(node_tokens)

    for source in all_nodes:
        existing: set[str] = {edge.target for _, edge in graph.outgoing(source)}
        candidates = [node for node in all_nodes if node != source and node not in existing]
        if not candidates:
            continue

        k = min(desired_out, len(candidates))
        targets = rng.sample(candidates, k=k)
        for target in targets:
            graph.add_edge(
                Edge(
                    source=source,
                    target=target,
                    weight=max(0.05, min(0.95, rng.uniform(0.35, 0.95))),
                )
            )

    return graph, node_tokens


def _random_queries(
    graph: Graph,
    node_tokens: dict[str, str],
    rng: random.Random,
    count: int,
) -> list[QueryCase]:
    cases: list[QueryCase] = []
    node_ids = list(node_tokens)

    for _ in range(count):
        start = rng.choice(node_ids)
        current = start
        hops = rng.randint(1, 3)
        for _ in range(hops):
            outgoing = [target.id for target, _ in graph.outgoing(current)]
            if not outgoing:
                break
            current = rng.choice(outgoing)

        cases.append(
            QueryCase(
                text=f"query about {node_tokens[start]} and related topic",
                expected=[current],
            )
        )

    return cases


def _metric_at_3(
    ranking: list[str],
    expected: list[str],
) -> tuple[float, float, float]:
    expected_set = set(expected)
    if not expected_set:
        return 0.0, 0.0, 0.0

    top3 = ranking[:3]
    hit_positions = [idx + 1 for idx, node_id in enumerate(top3) if node_id in expected_set]
    recall = 1.0 if any(node_id in expected_set for node_id in top3) else 0.0
    if not hit_positions:
        return recall, 0.0, 0.0

    first = hit_positions[0]
    mrr = 1.0 / first

    ndcg = sum((1.0 / math.log2(idx + 2.0)) for idx, node_id in enumerate(top3) if node_id in expected_set)
    ideal = 1.0 / math.log2(2.0)
    return recall, ndcg / ideal, mrr


def _bm25_score(
    query: str,
    nodes: list[Node],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    query_terms = _tokenize(query)
    if not query_terms or not nodes:
        return []

    query_counts: dict[str, int] = {}
    for term in query_terms:
        query_counts[term] = query_counts.get(term, 0) + 1

    doc_term_freq: list[tuple[str, Counter[str], int]] = []
    doc_freq: dict[str, int] = {}
    doc_lengths: list[int] = []

    for node in nodes:
        tokens = _tokenize(node.content or "")
        length = max(1, len(tokens))
        doc_lengths.append(length)
        term_freq: Counter[str] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        doc_term_freq.append((node.id, term_freq, length))
        for token in set(term_freq):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    doc_count = len(nodes)
    avg_dl = sum(doc_lengths) / doc_count
    score_by_node: dict[str, float] = {}

    for node_id, term_freq, doc_len in doc_term_freq:
        score = 0.0
        for term, qf in query_counts.items():
            f = term_freq.get(term, 0)
            if f <= 0:
                continue
            nq = doc_freq.get(term, 0)
            if nq <= 0:
                continue
            idf = math.log((doc_count - nq + 0.5) / (nq + 0.5) + 1.0)
            denom = f + k1 * (1 - b + b * (doc_len / avg_dl))
            score += idf * (f * (k1 + 1) / denom) * qf
        score_by_node[node_id] = score

    ranked = sorted(score_by_node.items(), key=lambda item: item[1], reverse=True)
    return [item for item in ranked if item[1] > 0.0]


def _run_bm25(
    graph: Graph,
    queries: list[QueryCase],
) -> dict[str, float]:
    nodes = list(graph.nodes())
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []

    for idx, query in enumerate(queries):
        if idx < WARMUP_QUERIES:
            continue

        scores = _bm25_score(query.text, nodes)
        ranking = [node_id for node_id, _ in scores[:3]]
        recall, ndcg, mrr = _metric_at_3(ranking, query.expected)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)

    count = max(len(recall_scores), 1)
    return {
        "recall": sum(recall_scores) / count,
        "ndcg": sum(ndcg_scores) / count,
        "mrr": sum(mrr_scores) / count,
    }


def _run_config(
    base_graph: Graph,
    queries: list[QueryCase],
    config: ControllerConfig,
) -> dict[str, Any]:
    graph = copy.deepcopy(base_graph)
    controller = MemoryController(graph, config)

    pre = inhibition_stats(graph).get("total_inhibitory_edges", 0)
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []
    hop_counts: list[int] = []
    node_counts: list[int] = []

    for idx, query in enumerate(queries):
        result = controller.query(query.text)
        if idx >= WARMUP_QUERIES:
            ranking = result.selected_nodes
            recall, ndcg, mrr = _metric_at_3(ranking, query.expected)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            hop_counts.append(max(0, len(result.selected_nodes) - 1))
            node_counts.append(len(result.selected_nodes))

        reward = 1.0 if set(query.expected) & set(result.selected_nodes) else -1.0
        controller.learn(result, reward)

    eval_count = max(len(recall_scores), 1)
    final_inhibitory = inhibition_stats(graph).get("total_inhibitory_edges", 0)
    return {
        "recall": sum(recall_scores) / eval_count,
        "ndcg": sum(ndcg_scores) / eval_count,
        "mrr": sum(mrr_scores) / eval_count,
        "avg_hops": sum(hop_counts) / eval_count,
        "avg_nodes": sum(node_counts) / eval_count,
        "inhibitory_formed": final_inhibitory - pre,
    }


def _print_table(rows: list[dict[str, float | int | str]]) -> None:
    header = (
        f"{'Size':>4} | {'Config':<6} | {'R@3':>6} | "
        f"{'NDCG@3':>8} | {'MRR@3':>7} | {'Avg Hops':>8} | {'Avg Nodes Visited':>17}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    for row in rows:
        print(
            f"{str(row['size']):>4} | "
            f"{str(row['config']):<6} | "
            f"{row['recall']:.3f} | "
            f"{row['ndcg']:.3f} | "
            f"{row['mrr']:.3f} | "
            f"{row['avg_hops']:.3f} | "
            f"{row['avg_nodes']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run subset of sizes for quick checks.")
    args = parser.parse_args()

    sizes = QUICK_SIZES if args.quick else SIZES
    rng = random.Random(SEED)

    configs = []

    # Baseline A/B preserve legacy behavior for explicit comparison:
    # A: no damping, shallow traversal (3 hops)
    legacy_a = ControllerConfig.default()
    legacy_a.max_hops = 3
    legacy_a.episode_edge_damping = 1.0
    legacy_a.episode_visit_penalty = 0.0
    legacy_a.enable_learning = True
    legacy_a.enable_synaptogenesis = False
    configs.append(("A", legacy_a))

    # B: no damping, visit-penalty path selection
    legacy_b = ControllerConfig.default()
    legacy_b.episode_edge_damping = 1.0
    legacy_b.episode_visit_penalty = 0.5
    legacy_b.enable_learning = True
    legacy_b.enable_synaptogenesis = False
    configs.append(("B", legacy_b))

    # C/D use current production defaults (max_hops=30, episode_edge_damping=0.3),
    # with optional visit penalty.
    modern_c = ControllerConfig.default()
    modern_c.enable_learning = True
    modern_c.enable_synaptogenesis = False
    configs.append(("C", modern_c))

    modern_d = ControllerConfig.default()
    modern_d.episode_visit_penalty = 0.3
    modern_d.enable_learning = True
    modern_d.enable_synaptogenesis = False
    configs.append(("D", modern_d))

    rows: list[dict[str, float | int | str]] = []
    for size in sizes:
        graph, node_tokens = _build_graph(size=size, edge_ratio=EDGE_RATIO, rng=rng)
        queries = _random_queries(graph, node_tokens, rng=rng, count=TOTAL_QUERIES)

        bm25_result = _run_bm25(graph, queries)
        print(f"\nSize={size} BM25: R@3={bm25_result['recall']:.3f}, NDCG@3={bm25_result['ndcg']:.3f}, MRR@3={bm25_result['mrr']:.3f}")

        for label, config in configs:
            result = _run_config(graph, queries, config)
            rows.append(
                {
                    "size": size,
                    "config": f"{label}",
                    "recall": result["recall"],
                    "ndcg": result["ndcg"],
                    "mrr": result["mrr"],
                    "avg_hops": result["avg_hops"],
                    "avg_nodes": result["avg_nodes"],
                }
            )
            print(
                f"Config {label}: inhibitory edges formed={result['inhibitory_formed']}"
            )

    _print_table(rows)


if __name__ == "__main__":
    main()
