#!/usr/bin/env python3
"""Standard IR metrics evaluation for all CrabPath ablation arms + new baselines."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.embeddings import EmbeddingIndex  # noqa: E402
from scripts import ablation_study as ablation  # noqa: E402

SEED = ablation.SEED


RECALL_KS = [1, 3, 5, 10]
NDCG_KS = [3, 5, 10]
MRR_KS = [5, 10]
PRECISION_KS = [1, 3, 5]
HIT_KS = [1, 3, 5]

TOP_K = 10


@dataclass
class ArmIRResult:
    name: str
    metrics: dict[str, float]
    metric_cis: dict[str, tuple[float, float, float]]
    per_query_metrics: dict[str, list[float]]
    query_results: list[dict[str, object]]


def _tokenize(text: str) -> list[str]:
    return ablation._tokenize(text)


def _safe_divide(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _dcg_gain(rank: int) -> float:
    return 1.0 / math.log2(rank + 2.0)


def _compute_binary_metrics(
    ranking: list[str],
    expected_nodes: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    expected = set(expected_nodes)

    query_metrics: dict[str, float] = {}
    if not expected:
        for k in RECALL_KS:
            query_metrics[f"recall@{k}"] = 0.0
        for k in NDCG_KS:
            query_metrics[f"ndcg@{k}"] = 0.0
        for k in MRR_KS:
            query_metrics[f"mrr@{k}"] = 0.0
        for k in PRECISION_KS:
            query_metrics[f"precision@{k}"] = 0.0
        for k in HIT_KS:
            query_metrics[f"hit@{k}"] = 0.0
        return query_metrics, {}

    num_relevant = len(expected)

    for k in RECALL_KS:
        top_k = ranking[:k]
        hit_count = sum(1 for node_id in top_k if node_id in expected)
        query_metrics[f"recall@{k}"] = _safe_divide(hit_count, num_relevant)

    for k in PRECISION_KS:
        top_k = ranking[:k]
        hit_count = sum(1 for node_id in top_k if node_id in expected)
        query_metrics[f"precision@{k}"] = _safe_divide(hit_count, k)

    for k in HIT_KS:
        query_metrics[f"hit@{k}"] = 1.0 if set(ranking[:k]) & expected else 0.0

    ideal_hits = min(num_relevant, max(NDCG_KS))
    ideal_dcg = sum(_dcg_gain(i) for i in range(ideal_hits))

    for k in NDCG_KS:
        dcg = 0.0
        for idx, node_id in enumerate(ranking[:k]):
            if node_id in expected:
                dcg += _dcg_gain(idx)
        query_metrics[f"ndcg@{k}"] = _safe_divide(dcg, ideal_dcg)

    for k in MRR_KS:
        value = 0.0
        for idx, node_id in enumerate(ranking[:k]):
            if node_id in expected:
                value = 1.0 / (idx + 1)
                break
        query_metrics[f"mrr@{k}"] = value

    return query_metrics, {}


def _compute_metric_aggregates(
    per_query: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, tuple[float, float, float]]]:
    metrics: dict[str, float] = {}
    cis: dict[str, tuple[float, float, float]] = {}

    for metric_name, values in per_query.items():
        if not values:
            metrics[metric_name] = 0.0
            cis[metric_name] = (0.0, 0.0, 0.0)
        else:
            metrics[metric_name] = sum(values) / len(values)
            cis[metric_name] = ablation.bootstrap_ci(values, seed=SEED)
    return metrics, cis


def _run_ablation_arm(cfg: ablation.ArmConfig, queries: list[ablation.QuerySpec], base_graph, base_mitosis_state, base_syn_state, llm_call) -> ArmIRResult:
    arm_result = ablation.run_arm(
        cfg=cfg,
        queries=queries,
        base_graph=base_graph,
        base_mitosis_state=base_mitosis_state,
        base_syn_state=base_syn_state,
        llm_call=llm_call,
    )

    per_query_metrics: dict[str, list[float]] = {
        f"recall@{k}": [] for k in RECALL_KS
    }
    per_query_metrics.update({f"ndcg@{k}": [] for k in NDCG_KS})
    per_query_metrics.update({f"mrr@{k}": [] for k in MRR_KS})
    per_query_metrics.update({f"precision@{k}": [] for k in PRECISION_KS})
    per_query_metrics.update({f"hit@{k}": [] for k in HIT_KS})

    query_results: list[dict[str, object]] = []

    for qres in arm_result.query_results:
        ranking = list(qres.get("selected_nodes", []))
        expected = list(qres.get("expected_nodes", []))
        metrics, _ = _compute_binary_metrics(ranking=ranking, expected_nodes=expected)
        for name, value in metrics.items():
            per_query_metrics[name].append(float(value))
        query_results.append(
            {
                "query": qres.get("query", ""),
                "expected_nodes": expected,
                "selected_nodes": ranking,
            }
        )

    metrics, cis = _compute_metric_aggregates(per_query_metrics)
    return ArmIRResult(
        name=arm_result.name,
        metrics=metrics,
        metric_cis=cis,
        per_query_metrics=per_query_metrics,
        query_results=query_results,
    )


def _make_word_overlap_embedder(graph):
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


def _run_dense_retriever_arm(
    graph, queries: list[ablation.QuerySpec],
) -> ArmIRResult:
    embed = _make_word_overlap_embedder(graph)
    index = EmbeddingIndex()
    index.build(graph, embed_fn=embed)

    per_query_metrics: dict[str, list[float]] = {
        f"recall@{k}": [] for k in RECALL_KS
    }
    per_query_metrics.update({f"ndcg@{k}": [] for k in NDCG_KS})
    per_query_metrics.update({f"mrr@{k}": [] for k in MRR_KS})
    per_query_metrics.update({f"precision@{k}": [] for k in PRECISION_KS})
    per_query_metrics.update({f"hit@{k}": [] for k in HIT_KS})

    query_results: list[dict[str, object]] = []

    for query in queries:
        scores = index.raw_scores(query.text, embed_fn=embed, top_k=TOP_K)
        ranking = [node_id for node_id, _ in scores]
        metric_values, _ = _compute_binary_metrics(ranking, query.expected_nodes)
        for metric_name, value in metric_values.items():
            per_query_metrics[metric_name].append(float(value))
        query_results.append(
            {
                "query": query.text,
                "expected_nodes": list(query.expected_nodes),
                "selected_nodes": ranking,
            }
        )

    metrics, cis = _compute_metric_aggregates(per_query_metrics)
    return ArmIRResult(
        name="Arm 7: Dense Retriever",
        metrics=metrics,
        metric_cis=cis,
        per_query_metrics=per_query_metrics,
        query_results=query_results,
    )


def _build_transition(graph):
    transition: dict[str, list[tuple[str, float]]] = {}
    for node in graph.nodes():
        edges: list[tuple[str, float]] = []
        total = 0.0
        for target, edge in graph.outgoing(node.id):
            weight = max(0.0, edge.weight)
            if weight <= 0.0:
                continue
            edges.append((target.id, weight))
            total += weight

        if not edges or total <= 0.0:
            continue

        transition[node.id] = [(target_id, weight / total) for target_id, weight in edges]
    return transition


def _run_personalized_pagerank_arm(
    graph,
    queries: list[ablation.QuerySpec],
    seed_count: int = 5,
    alpha: float = 0.15,
    iters: int = 30,
) -> ArmIRResult:
    node_ids = [node.id for node in graph.nodes()]
    node_count = len(node_ids)
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    transition = _build_transition(graph)
    all_nodes = graph.nodes()

    per_query_metrics: dict[str, list[float]] = {
        f"recall@{k}": [] for k in RECALL_KS
    }
    per_query_metrics.update({f"ndcg@{k}": [] for k in NDCG_KS})
    per_query_metrics.update({f"mrr@{k}": [] for k in MRR_KS})
    per_query_metrics.update({f"precision@{k}": [] for k in PRECISION_KS})
    per_query_metrics.update({f"hit@{k}": [] for k in HIT_KS})

    query_results: list[dict[str, object]] = []

    for query in queries:
        bm25_scores = ablation._bm25_score(query.text, all_nodes)
        seeds = bm25_scores[:seed_count]
        if not seeds and all_nodes:
            seeds = [(all_nodes[i].id, 1.0) for i in range(min(seed_count, len(all_nodes)))]

        p0 = [0.0] * node_count
        if node_count and seeds:
            total = sum(max(0.0, score) for _, score in seeds)
            if total <= 0.0:
                for idx in range(node_count):
                    p0[idx] = 1.0 / node_count
            else:
                for node_id, score in seeds:
                    idx = id_to_idx.get(node_id)
                    if idx is None:
                        continue
                    p0[idx] = max(0.0, score) / total
            p_norm = sum(p0)
            if p_norm <= 0.0:
                for idx in range(node_count):
                    p0[idx] = 1.0 / node_count
            else:
                p0 = [p / p_norm for p in p0]

        if node_count == 0:
            ranking: list[str] = []
            metric_values, _ = _compute_binary_metrics(ranking, query.expected_nodes)
            for metric_name, value in metric_values.items():
                per_query_metrics[metric_name].append(float(value))
            query_results.append(
                {
                    "query": query.text,
                    "expected_nodes": list(query.expected_nodes),
                    "selected_nodes": ranking,
                }
            )
            continue

        p = p0[:]
        for _ in range(iters):
            next_p = [0.0] * node_count
            for src_id, src_prob in zip(node_ids, p):
                if src_prob <= 0.0:
                    continue
                outgoing = transition.get(src_id)
                if not outgoing:
                    next_p[id_to_idx[src_id]] += (1.0 - alpha) * src_prob
                    continue
                for target_id, prob in outgoing:
                    target_idx = id_to_idx.get(target_id)
                    if target_idx is None:
                        continue
                    next_p[target_idx] += (1.0 - alpha) * src_prob * prob

            for i, base_mass in enumerate(p0):
                if base_mass > 0.0:
                    next_p[i] += alpha * base_mass

            norm = sum(next_p)
            if norm <= 0.0:
                p = [1.0 / node_count] * node_count
            else:
                p = [val / norm for val in next_p]

        ranked = sorted(
            [(node_ids[idx], score) for idx, score in enumerate(p)],
            key=lambda item: item[1],
            reverse=True,
        )
        ranking = [node_id for node_id, _ in ranked[:TOP_K]]

        metric_values, _ = _compute_binary_metrics(ranking, query.expected_nodes)
        for metric_name, value in metric_values.items():
            per_query_metrics[metric_name].append(float(value))

        query_results.append(
            {
                "query": query.text,
                "expected_nodes": list(query.expected_nodes),
                "selected_nodes": ranking,
            }
        )

    metrics, cis = _compute_metric_aggregates(per_query_metrics)
    return ArmIRResult(
        name="Arm 8: Personalized PageRank",
        metrics=metrics,
        metric_cis=cis,
        per_query_metrics=per_query_metrics,
        query_results=query_results,
    )


def _run_memgpt_baseline_arm(
    graph,
    queries: list[ablation.QuerySpec],
    window_size: int = 5,
) -> ArmIRResult:
    nodes = list(graph.nodes())

    per_query_metrics: dict[str, list[float]] = {
        f"recall@{k}": [] for k in RECALL_KS
    }
    per_query_metrics.update({f"ndcg@{k}": [] for k in NDCG_KS})
    per_query_metrics.update({f"mrr@{k}": [] for k in MRR_KS})
    per_query_metrics.update({f"precision@{k}": [] for k in PRECISION_KS})
    per_query_metrics.update({f"hit@{k}": [] for k in HIT_KS})

    recent_nodes: deque[str] = deque(maxlen=window_size)
    query_results: list[dict[str, object]] = []

    for query in queries:
        keyword_scores = ablation._bm25_score(query.text, nodes)[:2]
        keyword_nodes = [node_id for node_id, _ in keyword_scores]

        ranking: list[str] = []
        ranking.extend(reversed(list(recent_nodes)))
        ranking.extend(keyword_nodes)

        deduped: list[str] = []
        seen_nodes: set[str] = set()
        for node_id in ranking:
            if node_id in seen_nodes:
                continue
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                deduped.append(node_id)

        ranking = deduped[:TOP_K]

        metric_values, _ = _compute_binary_metrics(ranking, query.expected_nodes)
        for metric_name, value in metric_values.items():
            per_query_metrics[metric_name].append(float(value))

        for node_id in ranking:
            if node_id in recent_nodes:
                recent_nodes.remove(node_id)
            recent_nodes.append(node_id)

        query_results.append(
            {
                "query": query.text,
                "expected_nodes": list(query.expected_nodes),
                "selected_nodes": ranking,
            }
        )

    metrics, cis = _compute_metric_aggregates(per_query_metrics)
    return ArmIRResult(
        name="Arm 9: MemGPT Baseline",
        metrics=metrics,
        metric_cis=cis,
        per_query_metrics=per_query_metrics,
        query_results=query_results,
    )


def _build_ablation_arms() -> list[ablation.ArmConfig]:
    return [
        ablation.ArmConfig(
            name="Arm 0: BM25 Baseline",
            use_learning=False,
            use_graph_routing=False,
            allow_inhibition=False,
            enable_synaptogenesis=False,
            enable_autotune=False,
            enable_neurogenesis=False,
        ),
        ablation.ArmConfig(
            name="Arm 1",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=True,
        ),
        ablation.ArmConfig(
            name="Arm 2",
            use_learning=False,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=False,
        ),
        ablation.ArmConfig(
            name="Arm 3",
            use_learning=True,
            learning_discount=0.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=True,
        ),
        ablation.ArmConfig(
            name="Arm 4",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=False,
            enable_synaptogenesis=True,
            enable_autotune=True,
            enable_neurogenesis=True,
        ),
        ablation.ArmConfig(
            name="Arm 5",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=False,
            enable_autotune=True,
            enable_neurogenesis=False,
        ),
        ablation.ArmConfig(
            name="Arm 6",
            use_learning=True,
            learning_discount=1.0,
            allow_inhibition=True,
            enable_synaptogenesis=True,
            enable_autotune=False,
            enable_neurogenesis=True,
        ),
    ]

def _render_latex_table(arm_results: list[ArmIRResult]) -> str:
    header = [
        "Arm",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "nDCG@3",
        "nDCG@5",
        "nDCG@10",
        "MRR@5",
        "MRR@10",
        "Precision@1",
        "Precision@3",
        "Precision@5",
        "Hit@1",
        "Hit@3",
        "Hit@5",
    ]

    lines = ["\\begin{tabular}{l" + "r" * (len(header) - 1) + "}", "\\hline"]
    lines.append(" & ".join(header) + " " + "\\" * 2)
    lines.append("\\hline")

    for result in arm_results:
        values = [
            result.metrics.get("recall@1", 0.0),
            result.metrics.get("recall@3", 0.0),
            result.metrics.get("recall@5", 0.0),
            result.metrics.get("recall@10", 0.0),
            result.metrics.get("ndcg@3", 0.0),
            result.metrics.get("ndcg@5", 0.0),
            result.metrics.get("ndcg@10", 0.0),
            result.metrics.get("mrr@5", 0.0),
            result.metrics.get("mrr@10", 0.0),
            result.metrics.get("precision@1", 0.0),
            result.metrics.get("precision@3", 0.0),
            result.metrics.get("precision@5", 0.0),
            result.metrics.get("hit@1", 0.0),
            result.metrics.get("hit@3", 0.0),
            result.metrics.get("hit@5", 0.0),
        ]

        latex_name = result.name.replace("&", "\\&")
        values_text = [f"{v:.3f}" for v in values]
        lines.append(" & ".join([latex_name, *values_text]) + " " + "\\" * 2)

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)

def _pairwise_bootstrap_for_crabpath(
    arms: list[ArmIRResult],
    crab_name: str,
    baselines: Iterable[str],
    metric_names: Iterable[str],
) -> dict[str, dict[str, dict[str, float]]]:
    by_name = {arm.name: arm for arm in arms}
    crab = by_name[crab_name]
    out: dict[str, dict[str, dict[str, float]]] = {}

    for baseline_name in baselines:
        baseline = by_name[baseline_name]
        tests: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            mean_diff, p_value = ablation.paired_bootstrap_test(
                a=crab.per_query_metrics.get(metric_name, []),
                b=baseline.per_query_metrics.get(metric_name, []),
                seed=SEED,
            )
            tests[metric_name] = {
                "mean_diff": mean_diff,
                "p_value": p_value,
            }
        out[f"crab_vs_{baseline_name}"] = tests
    return out


def main() -> None:
    random.seed(SEED)

    llm = ablation.make_mock_llm_all()
    base_graph, base_mitosis_state, base_syn_state, _, _ = ablation._bootstrap_base(llm)

    file_chunks = ablation._map_file_chunk_nodes(base_graph)
    queries = ablation._build_queries(file_chunks)

    arm_results: list[ArmIRResult] = []

    for cfg in _build_ablation_arms():
        arm_results.append(
            _run_ablation_arm(
                cfg=cfg,
                queries=queries,
                base_graph=base_graph,
                base_mitosis_state=base_mitosis_state,
                base_syn_state=base_syn_state,
                llm_call=llm,
            )
        )

    arm_results.append(_run_dense_retriever_arm(base_graph, queries))
    arm_results.append(_run_personalized_pagerank_arm(base_graph, queries))
    arm_results.append(_run_memgpt_baseline_arm(base_graph, queries))

    latex_table = _render_latex_table(arm_results)
    print(latex_table)

    baselines = [
        "Arm 0: BM25 Baseline",
        "Arm 7: Dense Retriever",
        "Arm 8: Personalized PageRank",
        "Arm 9: MemGPT Baseline",
    ]

    metric_names: list[str] = [
        *[f"recall@{k}" for k in RECALL_KS],
        *[f"ndcg@{k}" for k in NDCG_KS],
        *[f"mrr@{k}" for k in MRR_KS],
        *[f"precision@{k}" for k in PRECISION_KS],
        *[f"hit@{k}" for k in HIT_KS],
    ]

    paired = _pairwise_bootstrap_for_crabpath(
        arms=arm_results,
        crab_name="Arm 1",
        baselines=baselines,
        metric_names=metric_names,
    )

    out_path = Path("scripts/standard_ir_metrics.json")
    out_path.write_text(
        json.dumps(
            {
                "seed": SEED,
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
                        "metrics": {
                            metric_name: {
                                "mean": result.metrics[metric_name],
                                "ci": {
                                    "mean": result.metric_cis[metric_name][0],
                                    "lower": result.metric_cis[metric_name][1],
                                    "upper": result.metric_cis[metric_name][2],
                                },
                            }
                            for metric_name in metric_names
                        },
                        "query_results": result.query_results,
                        "per_query_metrics": result.per_query_metrics,
                    }
                    for result in arm_results
                ],
                "paired_bootstrap_tests": paired,
                "latex_table": latex_table,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
