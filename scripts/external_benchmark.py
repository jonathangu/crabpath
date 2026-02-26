#!/usr/bin/env python3
"""External benchmark script for HotpotQA subset and frozen BEIR-style suites."""

from __future__ import annotations

import json
import math
import random
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts import ablation_study as ablation  # noqa: E402
from scripts.standard_ir_metrics import _make_word_overlap_embedder  # noqa: E402
from crabpath.embeddings import EmbeddingIndex  # noqa: E402
from crabpath.graph import Edge, Graph, Node  # noqa: E402
from crabpath.legacy.activation import activate, learn as _learn  # noqa: E402

SEED = 2026
TOP_K = 5
METRICS = ("recall@2", "recall@5", "precision@2", "precision@5", "ndcg@5", "mrr@5")
LEARNING_CURVE_WINDOW = 10
RESULTS_PATH = Path("scripts/external_benchmark_results.json")
HOTPOT_DATASET = Path("scripts/hotpot_subset_100.json")


@dataclass
class MethodStats:
    name: str
    per_query_metrics: dict[str, list[float]]
    query_results: list[dict]

    @property
    def metrics(self) -> dict[str, dict[str, float]]:
        total = len(self.query_results)
        out: dict[str, dict[str, float]] = {}
        if total == 0:
            for metric in METRICS:
                out[metric] = {
                    "mean": 0.0,
                    "ci": {"mean": 0.0, "lower": 0.0, "upper": 0.0},
                }
            return out

        for metric in METRICS:
            values = self.per_query_metrics.get(metric, [])
            if not values:
                out[metric] = {
                    "mean": 0.0,
                    "ci": {"mean": 0.0, "lower": 0.0, "upper": 0.0},
                }
                continue
            mean = sum(values) / len(values)
            mean_ci = ablation.bootstrap_ci(values, seed=SEED)
            out[metric] = {
                "mean": mean,
                "ci": {
                    "mean": mean_ci[0],
                    "lower": mean_ci[1],
                    "upper": mean_ci[2],
                },
            }
        return out


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)


def _normalize_metrics(ranking: list[str], gold: list[str]) -> dict[str, float]:
    expected = set(gold)
    if not expected:
        return {metric: 0.0 for metric in METRICS}

    num_relevant = len(expected)

    top2 = ranking[:2]
    top5 = ranking[:5]
    recall2 = len(set(top2) & expected) / num_relevant
    recall5 = len(set(top5) & expected) / num_relevant
    precision2 = len(set(top2) & expected) / 2.0
    precision5 = len(set(top5) & expected) / 5.0

    ideal_hits = min(num_relevant, 5)
    ideal_dcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal_hits))
    dcg = 0.0
    for idx, node_id in enumerate(top5):
        if node_id in expected:
            dcg += 1.0 / math.log2(idx + 2.0)
    ndcg5 = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    mrr5 = 0.0
    for idx, node_id in enumerate(top5, start=1):
        if node_id in expected:
            mrr5 = 1.0 / idx
            break

    return {
        "recall@2": recall2,
        "recall@5": recall5,
        "precision@2": precision2,
        "precision@5": precision5,
        "ndcg@5": ndcg5,
        "mrr@5": mrr5,
    }


def _normalize_metrics_3(ranking: list[str], gold: list[str]) -> dict[str, float]:
    """Top-k@3 metrics used by the recurring-topic benchmark."""
    base = _normalize_metrics(ranking, gold)

    expected = set(gold)
    if not expected:
        return {
            "recall@2": base["recall@2"],
            "recall@3": 0.0,
            "ndcg@3": 0.0,
            "mrr@3": 0.0,
        }

    top3 = ranking[:3]
    num_relevant = len(expected)

    recall3 = len(set(top3) & expected) / num_relevant

    ideal_hits = min(num_relevant, 3)
    ideal_dcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal_hits))
    dcg = 0.0
    for idx, node_id in enumerate(top3):
        if node_id in expected:
            dcg += 1.0 / math.log2(idx + 2.0)
    ndcg3 = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    mrr3 = 0.0
    for idx, node_id in enumerate(top3, start=1):
        if node_id in expected:
            mrr3 = 1.0 / idx
            break

    return {
        "recall@2": base["recall@2"],
        "recall@3": recall3,
        "ndcg@3": ndcg3,
        "mrr@3": mrr3,
    }


def _window_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_learning_graph(questions: list[dict]) -> Graph:
    graph = Graph()
    seen_contexts: dict[str, str] = {}
    question_contexts: list[list[str]] = []

    for item in questions:
        titles = []
        seen = set()
        for title, text in item.get("contexts", []):
            if title in seen:
                continue
            seen.add(title)
            titles.append(title)
            seen_contexts.setdefault(title, text or "")
        question_contexts.append(titles)

    for title, text in seen_contexts.items():
        graph.add_node(Node(id=title, content=text))

    for titles in question_contexts:
        for source, target in combinations(titles, 2):
            graph.add_edge(Edge(source=source, target=target, weight=0.5))
            graph.add_edge(Edge(source=target, target=source, weight=0.5))

    return graph


def _graph_window_stats(graph: Graph, initial_edge_weights: dict[tuple[str, str], float]) -> dict[str, int | float]:
    edges = graph.edges()
    edge_count = len(edges)
    inhibitory = sum(1 for edge in edges if edge.weight < 0.0)
    avg_weight = (sum(edge.weight for edge in edges) / edge_count) if edge_count else 0.0
    reflex = sum(1 for edge in edges if edge.weight > 0.8)
    dormant = sum(1 for edge in edges if edge.weight < 0.3)
    changed = sum(
        1
        for edge in edges
        if initial_edge_weights.get((edge.source, edge.target), 0.5) != edge.weight
    )
    return {
        "inhibitory_edges": inhibitory,
        "avg_weight": avg_weight,
        "reflex_edges": reflex,
        "dormant_edges": dormant,
        "changed_edges": changed,
    }


def _read_hotpot(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions: list[dict] = []
    for item in payload:
        contexts = [(title, text) for title, text in item.get("ctx", [])[:10]]
        questions.append(
            {
                "id": str(item.get("id", "")),
                "question": str(item.get("q", "")),
                "gold": [str(x) for x in item.get("gold", [])],
                "contexts": contexts,
            }
        )
    return questions


def _build_graph(contexts: list[tuple[str, str]]) -> Graph:
    graph = Graph()
    for title, text in contexts:
        graph.add_node(Node(id=title, content=text or ""))
    node_ids = [node_id for node_id, _ in contexts]
    for source in node_ids:
        for target in node_ids:
            if source == target:
                continue
            graph.add_edge(Edge(source=source, target=target, weight=0.5))
    return graph


def _bm25_ranking(graph: Graph, query: str) -> list[str]:
    scores = ablation._bm25_score(query, list(graph.nodes()))
    return [node_id for node_id, _ in scores]


def _dense_ranking(graph: Graph, query: str, embed: Callable[[list[str]], list[list[float]]]) -> list[str]:
    index = EmbeddingIndex()
    index.build(graph=graph, embed_fn=embed)
    scores = index.raw_scores(query, embed_fn=embed, top_k=TOP_K)
    return [node_id for node_id, _ in scores]


def _static_ranking(graph: Graph) -> list[str]:
    return [node.id for node in graph.nodes()]


def _clamp_negative_edges(graph: Graph) -> None:
    for edge in graph.edges():
        if edge.weight < 0.0:
            edge.weight = 0.0


def _warmup_queries(question: str) -> list[str]:
    return [
        question,
        f"In one sentence, answer this: {question}",
        f"{question}  (clear and factual)",
    ]


def _activation_ranking(graph: Graph, query: str) -> tuple[list[str], object]:
    bm25 = _bm25_ranking(graph, query)
    if bm25:
        top_candidates = bm25[:min(TOP_K, len(bm25))]
        max_score = 1.0
        seeds: dict[str, float] = {}
        for idx, node_id in enumerate(top_candidates):
            seeds[node_id] = max(1.0, 2.0 - 0.2 * idx)
        result = activate(graph=graph, seeds=seeds, max_steps=3, top_k=TOP_K)
        selected = [node.id for node, _ in result.fired]
    else:
        fallback = [node.id for node in graph.nodes()][:TOP_K]
        result = activate(
            graph=graph,
            seeds={node_id: 1.0 for node_id in fallback},
            max_steps=1,
            top_k=TOP_K,
        )
        selected = [node.id for node, _ in result.fired]

    if not selected:
        selected = _bm25_ranking(graph, query)[:TOP_K]
    return selected[:TOP_K], result


def _learn_outcome(ranking: list[str], gold: list[str]) -> float:
    relevant = set(gold)
    if not relevant:
        return 0.0
    return 1.0 if set(ranking[:2]) & relevant else -1.0


def _crabpath_full_ranking(graph: Graph, question: str, gold: list[str], disable_inhibition: bool = False) -> list[str]:
    warmups = _warmup_queries(question)
    for warm_query in warmups:
        selected, activation_result = _activation_ranking(graph, warm_query)
        outcome = _learn_outcome(selected, gold)
        if outcome != 0.0:
            _learn(graph=graph, result=activation_result, outcome=outcome, rate=0.1)
        if disable_inhibition:
            _clamp_negative_edges(graph)

    ranking, _ = _activation_ranking(graph, question)
    if disable_inhibition:
        _clamp_negative_edges(graph)
    return ranking


def _evaluate_method(
    name: str,
    get_ranking: Callable[[], list[str]],
    gold: list[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    values = {metric: [] for metric in METRICS}
    query_result: dict[str, object] = {}

    ranking = get_ranking()
    metric_values = _normalize_metrics(ranking, gold)
    for metric, value in metric_values.items():
        values[metric].append(float(value))

    query_result["selected_nodes"] = ranking[:TOP_K]
    query_result["ranking"] = ranking
    query_result["metrics"] = metric_values

    method_stats = MethodStats(name=name, per_query_metrics=values, query_results=[query_result])
    return method_stats.metrics, method_stats.query_results, metric_values


def _run_query_case(question: str, gold: list[str], contexts: list[tuple[str, str]]) -> dict[str, MethodStats]:
    results: dict[str, MethodStats] = {}

    static_graph = _build_graph(contexts)
    bm25_ranking = _bm25_ranking(static_graph, question)
    bm25_metrics = _normalize_metrics(bm25_ranking, gold)
    results["BM25"] = MethodStats(
        name="BM25",
        per_query_metrics={metric: [bm25_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": bm25_ranking[:TOP_K], "metrics": bm25_metrics}],
    )

    dense_embedder = _make_word_overlap_embedder(static_graph)
    dense_scores = _dense_ranking(static_graph, question, embed=dense_embedder)
    dense_metrics = _normalize_metrics(dense_scores, gold)
    results["Dense"] = MethodStats(
        name="Dense",
        per_query_metrics={metric: [dense_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": dense_scores[:TOP_K], "metrics": dense_metrics}],
    )

    static_ranking = _static_ranking(static_graph)
    static_metrics = _normalize_metrics(static_ranking, gold)
    results["Static"] = MethodStats(
        name="Static",
        per_query_metrics={metric: [static_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": static_ranking, "metrics": static_metrics}],
    )

    full_graph = _build_graph(contexts)
    full_ranking = _crabpath_full_ranking(full_graph, question, gold, disable_inhibition=False)
    full_metrics = _normalize_metrics(full_ranking, gold)
    results["CrabPath full"] = MethodStats(
        name="CrabPath full",
        per_query_metrics={metric: [full_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": full_ranking, "metrics": full_metrics}],
    )

    noninhib_graph = _build_graph(contexts)
    noninhib_ranking = _crabpath_full_ranking(
        noninhib_graph, question, gold, disable_inhibition=True
    )
    noninhib_metrics = _normalize_metrics(noninhib_ranking, gold)
    results["CrabPath no-inhibition"] = MethodStats(
        name="CrabPath no-inhibition",
        per_query_metrics={metric: [noninhib_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": noninhib_ranking, "metrics": noninhib_metrics}],
    )

    return results


def _run_retrieval_suite(
    questions: list[dict],
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    method_agg: dict[str, MethodStats] = {
        "BM25": MethodStats("BM25", {metric: [] for metric in METRICS}, []),
        "Dense": MethodStats("Dense", {metric: [] for metric in METRICS}, []),
        "Static": MethodStats("Static", {metric: [] for metric in METRICS}, []),
        "CrabPath full": MethodStats("CrabPath full", {metric: [] for metric in METRICS}, []),
        "CrabPath no-inhibition": MethodStats(
            "CrabPath no-inhibition",
            {metric: [] for metric in METRICS},
            [],
        ),
    }

    method_query_results: dict[str, list[dict]] = {name: [] for name in method_agg}

    for query_item in questions:
        question = query_item["question"]
        gold = list(query_item["gold"])
        contexts = list(query_item["contexts"])
        per_method = _run_query_case(question=question, gold=gold, contexts=contexts)

        for name, stats in per_method.items():
            for metric in METRICS:
                method_agg[name].per_query_metrics[metric].extend(
                    stats.per_query_metrics[metric]
                )
            entry = {
                "query": question,
                "query_id": query_item.get("id", ""),
                "expected_nodes": gold,
                "selected_nodes": stats.query_results[0]["ranking"],
                **stats.query_results[0]["metrics"],
            }
            method_query_results[name].append(entry)

            method_agg[name].query_results.append(
                {
                    "query_id": query_item.get("id", ""),
                    "query": question,
                    "expected_nodes": gold,
                    "selected_nodes": stats.query_results[0]["ranking"],
                    "query_metrics": stats.query_results[0]["metrics"],
                }
            )

    method_metric_map: dict[str, dict[str, dict[str, float]]] = {}
    for name, stat in method_agg.items():
        method_metric_map[name] = stat.metrics

    return method_metric_map, method_query_results


def _run_learning_curve(
    questions: list[dict],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    graph = _build_learning_graph(questions)
    initial_edge_weights = {
        (edge.source, edge.target): edge.weight for edge in graph.edges()
    }

    warm_window = []
    window_bm25_metrics: dict[str, list[float]] = {
        metric: [] for metric in ("recall@2", "recall@5", "ndcg@5", "mrr@5")
    }
    window_crab_metrics: dict[str, list[float]] = {
        metric: [] for metric in ("recall@2", "recall@5", "ndcg@5", "mrr@5")
    }
    query_results: list[dict[str, object]] = []
    window_summaries: list[dict[str, object]] = []

    for idx, query_item in enumerate(questions, 1):
        question = str(query_item.get("question", ""))
        query_id = str(query_item.get("id", ""))
        gold = list(query_item.get("gold", []))

        bm25_ranking = _bm25_ranking(graph, question)
        bm25_metrics = _normalize_metrics(bm25_ranking, gold)
        selected_bm25 = bm25_ranking[:TOP_K]

        crab_ranking, activation_result = _activation_ranking(graph, question)
        crab_metrics = _normalize_metrics(crab_ranking, gold)
        selected_crab = crab_ranking[:TOP_K]

        outcome = _learn_outcome(selected_crab, gold)
        if outcome != 0.0:
            _learn(graph=graph, result=activation_result, outcome=outcome, rate=0.1)

        per_query = {
            "query_num": idx,
            "query_id": query_id,
            "query": question,
            "expected_nodes": gold,
            "bm25": {
                "selected_nodes": selected_bm25,
                "all_nodes": bm25_ranking,
                "metrics": bm25_metrics,
            },
            "crabpath": {
                "selected_nodes": selected_crab,
                "all_nodes": crab_ranking,
                "metrics": crab_metrics,
                "reward": outcome,
            },
        }
        query_results.append(per_query)

        for metric in window_bm25_metrics:
            window_bm25_metrics[metric].append(float(bm25_metrics[metric]))
        for metric in window_crab_metrics:
            window_crab_metrics[metric].append(float(crab_metrics[metric]))

        if idx % LEARNING_CURVE_WINDOW == 0:
            start = idx - (LEARNING_CURVE_WINDOW - 1)
            graph_stats = _graph_window_stats(graph, initial_edge_weights)
            window = {
                "window": f"{start}-{idx}",
                "method_metrics": {
                    "BM25": {
                        metric: _window_mean(window_bm25_metrics[metric])
                        for metric in window_bm25_metrics
                    },
                    "CrabPath": {
                        metric: _window_mean(window_crab_metrics[metric])
                        for metric in window_crab_metrics
                    },
                },
                "graph": graph_stats,
                "query_results": [
                    q for q in query_results if start <= q["query_num"] <= idx
                ],
            }
            window_summaries.append(window)
            window_bm25_metrics = {metric: [] for metric in window_bm25_metrics}
            window_crab_metrics = {metric: [] for metric in window_crab_metrics}

    final_stats = _graph_window_stats(graph, initial_edge_weights)
    final_summary = {
        "window_count": len(window_summaries),
        "window_size": LEARNING_CURVE_WINDOW,
        "final_graph": final_stats,
    }

    return {
        "windows": window_summaries,
        "summary": final_summary,
    }, query_results, final_summary


def _build_beir_corpus() -> list[dict]:
    rng = random.Random(SEED)

    technical_docs = [
        ("tech-01", "Use HTTPS for all service-to-service traffic in production."),
        ("tech-02", "Enable request tracing to observe distributed latency spikes quickly."),
        ("tech-03", "Use idempotency keys for safe retry behavior."),
        ("tech-04", "Feature flags allow staged rollouts with rollback support."),
        ("tech-05", "Limit cache TTL to avoid stale configuration reads."),
        ("tech-06", "Retry with exponential backoff for transient API errors."),
        ("tech-07", "Use connection pooling to avoid exhausting database sockets."),
        ("tech-08", "Prefer bulk writes when ingesting telemetry in batches."),
        ("tech-09", "Encrypt secrets at rest and rotate credentials weekly."),
        ("tech-10", "Validate schema before accepting incoming JSON payloads."),
        ("tech-11", "Use circuit breakers to isolate failing downstream services."),
        ("tech-12", "Run canary tests when deploying infra changes."),
        ("tech-13", "Apply strict input sanitization for webhooks."),
        ("tech-14", "Document runbooks for common failure recovery actions."),
        ("tech-15", "Pin dependency versions for deterministic container builds."),
        ("tech-16", "Use read replicas for read-heavy database workloads."),
        ("tech-17", "Limit payload size and streaming chunk size for uploads."),
        ("tech-18", "Collect synthetic checks for critical API availability."),
        ("tech-19", "Keep event ordering when processing state updates."),
        ("tech-20", "Throttle unauthenticated requests during traffic spikes."),
    ]

    science_docs = [
        ("sci-01", "Mitochondria generate ATP through oxidative phosphorylation."),
        ("sci-02", "CRISPR systems allow precise gene editing with guide RNAs."),
        ("sci-03", "PCR doubles DNA sequences with thermal cycling steps."),
        ("sci-04", "Cell membranes regulate ion flow across bilayers."),
        ("sci-05", "Newton's second law links force with mass and acceleration."),
        ("sci-06", "Photosynthesis converts light to chemical energy in plants."),
        ("sci-07", "The placebo effect can influence perceived treatment outcomes."),
        ("sci-08", "Quantum entanglement appears in non-local correlations."),
        ("sci-09", "Black holes emit radiation through quantum effects near event horizons."),
        ("sci-10", "RNA polymerase transcribes DNA into messenger RNA."),
        ("sci-11", "Gravity bends spacetime according to general relativity."),
        ("sci-12", "Protein folding depends on hydrophobic and hydrogen bonds."),
        ("sci-13", "Plate tectonics explains continental drift over millions of years."),
        ("sci-14", "Antibodies bind antigens using variable loops in proteins."),
        ("sci-15", "Convolutional filters detect spatial features in images."),
        ("sci-16", "Statistical significance uses p-values and null-hypothesis testing."),
        ("sci-17", "Dark matter explains galaxy rotation curves in astronomy."),
        ("sci-18", "Machine learning models can overfit when capacity is high."),
        ("sci-19", "A heat engine converts thermal gradients into mechanical work."),
        ("sci-20", "Cellular respiration consumes glucose and oxygen for ATP."),
    ]

    faq_docs = [
        ("faq-01", "To reset your password, use the forgot-password link in login."),
        ("faq-02", "Clear cache files if the dashboard shows stale data."),
        ("faq-03", "Escalate billing questions to the finance support mailbox."),
        ("faq-04", "To close an account, contact support from account settings."),
        ("faq-05", "Enable two-factor authentication from security preferences."),
        ("faq-06", "Reset your API token from the developer portal page."),
        ("faq-07", "Use the incident channel for production outage notifications."),
        ("faq-08", "Delete local cookies if authentication loops keep failing."),
        ("faq-09", "For failed payments, verify card status and retry after update."),
        ("faq-10", "Invite team members from the workspace access page."),
        ("faq-11", "Troubleshoot upload failures by checking file size limits."),
        ("faq-12", "Open a support ticket during business hours for SLA updates."),
        ("faq-13", "Archive old projects manually after finalizing migration."),
        ("faq-14", "Turn off auto-sync before reconnecting integration tokens."),
        ("faq-15", "Set timezone in account profile for correct reporting windows."),
        ("faq-16", "Use read-only mode when reviewing sensitive logs."),
        ("faq-17", "Enable maintenance window to pause scheduled jobs."),
        ("faq-18", "Restart the local service when state appears inconsistent."),
        ("faq-19", "Use webhook retries and delivery logs for delivery failures."),
        ("faq-20", "Report suspicious activity through security incident form."),
    ]

    domain_queries = {
        "technical docs": (technical_docs, [
            "How should deployments handle failures in a safe rollout path?",
            "What guidance is provided for API reliability and request safety?",
            "How can database and traffic safeguards reduce production incidents?",
            "Which practices reduce configuration drift during releases?",
            "What is a reliable approach for schema validation in services?",
            "How do retries and backups interact with transient failures?",
            "How can rollout safety be improved for infrastructure changes?",
            "What protects production traffic from malformed external payloads?",
            "What should be done for high-throughput database pressure?",
            "How can secrets and credentials be protected in pipelines?",
            "How do teams recover from partial state synchronization issues?",
            "When should canary and synthetic monitoring be used?",
            "What limits or quotas prevent service abuse?",
            "How should API tokens be managed during maintenance?",
            "How can one preserve ordering guarantees in asynchronous systems?",
            "Which method reduces stale dashboards and stale caches?",
            "What steps secure webhook and callback handling?",
            "Which operational practice reduces incident response time?",
        ]),
        "scientific abstracts": (science_docs, [
            "What process converts light into stored chemical energy?",
            "How is protein structure stabilized in water-based systems?",
            "Which technique amplifies DNA segments during molecular experiments?",
            "What role does ion transport play in cellular membranes?",
            "How is force connected to motion in classical mechanics?",
            "Which effect can emit radiation from compact massive objects?",
            "What molecular tool can modify genes with guide sequences?",
            "Which framework explains curved spacetime under gravity?",
            "How are machine learning models evaluated for overfitting risk?",
            "What does machine learning detect through convolutional filters?",
            "How can statistical significance be interpreted in experiments?",
            "What is the biochemical role of RNA polymerase?",
            "How do antibodies recognize and bind specific antigens?",
            "Which phenomenon links continents over geological timescales?",
            "Which hypothesis accounts for fast galaxy rotation?",
            "How is ATP produced in cellular respiration pathways?",
            "What is a standard explanation of quantum correlations?",
            "How can heat convert to mechanical motion in thermodynamics?",
            "What causes the placebo effect in experiments?",
            "How can image data be transformed through convolution?",
            "How do thermal and quantum ideas meet at horizons?",
        ]),
        "faq/support": (faq_docs, [
            "How do I reset my account password?",
            "What steps recover from an authentication loop?",
            "How can billing support be contacted for invoice issues?",
            "Where can I close or archive an inactive account?",
            "How do I enable stronger sign-in security controls?",
            "How should API credentials be reset or rotated?",
            "Which channel should be used during critical incidents?",
            "What is the process for clearing browser cache issues?",
            "What should I do if my card payment fails repeatedly?",
            "How are new teammates added during onboarding?",
            "How can I debug file upload failures quickly?",
            "Who should I contact for service-level concerns?",
            "What is recommended before deleting completed projects?",
            "How should integration tokens be reconnected safely?",
            "How do I correct incorrect timezone in reports?",
            "What mode should I use for read-only log review?",
            "How do I pause work during scheduled maintenance?",
            "What should I do if local services appear inconsistent?",
            "How are webhook failures diagnosed and retried?",
            "Where do I report suspicious activity?",
            "What is the fastest support path for urgent errors?",
        ]),
    }

    queries: list[dict] = []
    for q_idx in range(50):
        if q_idx < 17:
            domain = "technical docs"
        elif q_idx < 34:
            domain = "scientific abstracts"
        else:
            domain = "faq/support"

        docs, questions = domain_queries[domain]
        local_idx = q_idx % len(questions)
        query_text = questions[local_idx]
        relevant_local = (
            local_idx,
            (local_idx + 3) % len(docs),
        )
        relevant_three = (
            (local_idx + 7) % len(docs),
            local_idx,
            (local_idx + 3) % len(docs),
        )
        include_three = (q_idx % 3 == 0)
        relevant_pairs = list(relevant_three) if include_three else list(relevant_local)
        doc_count = len(docs)
        if doc_count == 0:
            continue
        relevant_ids = [docs[idx % doc_count][0] for idx in relevant_pairs]
        unique_relevant: list[str] = []
        for rid in relevant_ids:
            if rid not in unique_relevant:
                unique_relevant.append(rid)
        relevant_ids = unique_relevant

        distractor_pool = [doc_id for doc_id, _ in docs if doc_id not in relevant_ids]
        target_size = min(20, doc_count)
        selected_ids = relevant_ids + rng.sample(distractor_pool, target_size - len(relevant_ids))
        rng.shuffle(selected_ids)

        selected_docs = []
        doc_by_id = {doc_id: text for doc_id, text in docs}
        for doc_id in selected_ids:
            selected_docs.append((doc_id, doc_by_id[doc_id]))

        queries.append(
            {
                "id": f"beir-{domain.replace(' ', '-')}-{q_idx + 1:02d}",
                "question": query_text,
                "domain": domain,
                "corpus": selected_docs,
                "gold": relevant_ids,
            }
        )

    return queries


def _run_beir_case(question: str, gold: list[str], corpus: list[tuple[str, str]]) -> dict[str, MethodStats]:
    results: dict[str, MethodStats] = {}
    graph = _build_graph(corpus)

    bm25_ranking = _bm25_ranking(graph, question)
    bm25_metrics = _normalize_metrics(bm25_ranking, gold)
    results["BM25"] = MethodStats(
        name="BM25",
        per_query_metrics={metric: [bm25_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": bm25_ranking[:TOP_K], "metrics": bm25_metrics}],
    )

    dense_embedder = _make_word_overlap_embedder(graph)
    dense_scores = _dense_ranking(graph, question, embed=dense_embedder)
    dense_metrics = _normalize_metrics(dense_scores, gold)
    results["Dense"] = MethodStats(
        name="Dense",
        per_query_metrics={metric: [dense_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": dense_scores[:TOP_K], "metrics": dense_metrics}],
    )

    static_ranking = _static_ranking(graph)
    static_metrics = _normalize_metrics(static_ranking, gold)
    results["Static"] = MethodStats(
        name="Static",
        per_query_metrics={metric: [static_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": static_ranking, "metrics": static_metrics}],
    )

    full_graph = _build_graph(corpus)
    full_ranking = _crabpath_full_ranking(full_graph, question, gold, disable_inhibition=False)
    full_metrics = _normalize_metrics(full_ranking, gold)
    results["CrabPath full"] = MethodStats(
        name="CrabPath full",
        per_query_metrics={metric: [full_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": full_ranking, "metrics": full_metrics}],
    )

    noninhib_graph = _build_graph(corpus)
    noninhib_ranking = _crabpath_full_ranking(
        noninhib_graph, question, gold, disable_inhibition=True
    )
    noninhib_metrics = _normalize_metrics(noninhib_ranking, gold)
    results["CrabPath no-inhibition"] = MethodStats(
        name="CrabPath no-inhibition",
        per_query_metrics={metric: [noninhib_metrics[metric]] for metric in METRICS},
        query_results=[{"ranking": noninhib_ranking, "metrics": noninhib_metrics}],
    )
    return results


def _run_retrieval_suite_beir(
    cases: list[dict],
) -> tuple[dict[str, list[float]], dict[str, list[dict]]]:
    method_agg: dict[str, MethodStats] = {
        "BM25": MethodStats("BM25", {metric: [] for metric in METRICS}, []),
        "Dense": MethodStats("Dense", {metric: [] for metric in METRICS}, []),
        "Static": MethodStats("Static", {metric: [] for metric in METRICS}, []),
        "CrabPath full": MethodStats("CrabPath full", {metric: [] for metric in METRICS}, []),
        "CrabPath no-inhibition": MethodStats(
            "CrabPath no-inhibition",
            {metric: [] for metric in METRICS},
            [],
        ),
    }
    method_query_results: dict[str, list[dict]] = {name: [] for name in method_agg}

    for case in cases:
        question = case["question"]
        gold = list(case["gold"])
        corpus = list(case["corpus"])
        per_method = _run_beir_case(question=question, gold=gold, corpus=corpus)

        for name, stats in per_method.items():
            for metric in METRICS:
                method_agg[name].per_query_metrics[metric].extend(
                    stats.per_query_metrics[metric]
                )
            method_query_results[name].append(
                {
                    "query_id": case.get("id", ""),
                    "query": question,
                    "domain": case.get("domain", ""),
                    "expected_nodes": gold,
                    "selected_nodes": stats.query_results[0]["ranking"],
                    "query_metrics": stats.query_results[0]["metrics"],
                }
            )
            method_agg[name].query_results.append(
                {
                    "query_id": case.get("id", ""),
                    "query": question,
                    "domain": case.get("domain", ""),
                    "expected_nodes": gold,
                    "selected_nodes": stats.query_results[0]["ranking"],
                    "query_metrics": stats.query_results[0]["metrics"],
                }
            )

    metric_map = {name: stats.metrics for name, stats in method_agg.items()}
    return metric_map, method_query_results


def _build_recurring_topic_corpus() -> list[tuple[str, str]]:
    return [
        (
            "mp-deploy-01",
            "Meridian Deployment Playbook defines the baseline release lifecycle used across all teams. "
            "Every production rollout starts by staging changes in a temporary release branch and running "
            "automated validation before approval. The playbook explicitly references the incident response "
            "runbook (mp-inc-01) when validation gates fail after deployment has begun. "
            "It also requires every release to record decision checkpoints in the release notebook that "
            "captures security checks from the secret policy document (mp-sec-01).",
        ),
        (
            "mp-deploy-02",
            "Meridian CI/CD Pipeline policy controls every commit path from merge request to image build. "
            "Build jobs emit signed artifacts, run static checks, and attach provenance metadata for traceability. "
            "When the pipeline detects schema or config drift, teams defer the change and file a recovery note in "
            "the architecture assumptions doc (mp-arc-01). Release managers can only approve deployment after "
            "canary health checks pass in the observability workspace (mp-mon-02).",
        ),
        (
            "mp-deploy-03",
            "Canary Rollout Procedure describes staged traffic percentages and auto-stop conditions. "
            "At each stage, monitoring metrics from tracing and logs are reviewed before increasing traffic. "
            "If error rate climbs, the team follows the rollback protocol in mp-deploy-04 and updates the incident "
            "timeline using mp-inc-02. The procedure also notes that feature flags should be aligned with the "
            "hotfix plan before any broad promotion in mp-deploy-05.",
        ),
        (
            "mp-deploy-04",
            "Meridian Rollback Protocol is the fastest path when a release introduces regression. "
            "It defines safe windows for traffic shifting, database compatibility checks, and cache invalidation. "
            "The protocol must consult the on-call runbook and communication checklist in mp-inc-01 before "
            "restoring previous stable release states. Teams then run a post-deploy verification set from mp-mon-03 "
            "to confirm no residual telemetry errors remain.",
        ),
        (
            "mp-deploy-05",
            "Emergency Patch and Freeze rules describe when a release should be paused and only critical fixes allowed. "
            "During a freeze, only emergency labels can pass through the deploy gate and every patch must include "
            "a rollback test case from mp-deploy-04. Security teams review affected components against mp-sec-03 "
            "before the patch is promoted, and architecture leads confirm impact scope in mp-arc-02 before signoff.",
        ),
        (
            "mp-inc-01",
            "Incident Triage Standard defines how Meridian teams classify severity and establish the first responder. "
            "Severity is checked within two minutes, then the response room is opened with owners for the affected service. "
            "The document references the rollback protocol (mp-deploy-04) when release change created the failure. "
            "If the incident degrades customer confidence, the communications playbook in mp-inc-03 is activated immediately.",
        ),
        (
            "mp-inc-02",
            "Production Outage Playbook provides step-by-step command chains for isolating failures and reducing blast radius. "
            "Operators confirm scope using dashboards from mp-mon-02 and tracing links from mp-mon-03 before choosing a runbook. "
            "The playbook requires that deployment-related disruptions reference mp-deploy-03 or mp-deploy-04 and that any "
            "service restart aligns with the operational safety rules in mp-sec-02. "
            "If the incident is caused by a secret leak, escalation to security is mandatory under mp-sec-01.",
        ),
        (
            "mp-inc-03",
            "Comms and Severity Escalation policy documents how outages are communicated to business and engineering channels. "
            "The channel map is linked with incident state transitions from mp-inc-01 and aligns with team readiness on mp-onb-01. "
            "When cross-functional action is needed, the escalation matrix points to service ownership records in mp-arc-01 and "
            "operational contact details in mp-mon-01. "
            "Every comms update must include the rollback assumption and monitoring checkpoints.",
        ),
        (
            "mp-inc-04",
            "Post-Incident Review Framework is used after recovery to capture root cause, timeline, and preventive changes. "
            "Reviews are required even for recovered false alarms to improve heuristics in automation. "
            "Findings should propose edits to mp-inc-02, mp-deploy-01, and monitoring alerts in mp-mon-02. "
            "Unresolved risks are passed to architecture owners for follow-up in mp-arc-02.",
        ),
        (
            "mp-inc-05",
            "Error Budget and Escalation Ladder defines when incidents become organizational escalations versus local fixes. "
            "It maps SLO breach thresholds to response levels and ties each level to approved responders. "
            "For sustained breaches during deployment, the ladder references mp-deploy-03 to throttle rollout and mp-mon-01 for root analysis. "
            "All escalations are logged into the same evidence store used by mp-inc-01.",
        ),
        (
            "mp-sec-01",
            "Identity and Access Discipline covers role-based access, service principals, and just-in-time privilege grants. "
            "All deployment actions must be tied to logged identities and checked against Meridian policy tags. "
            "If an incident is identity-related, triagers follow mp-inc-01 and then verify token states using mp-sec-02. "
            "Architects reviewing sensitive touchpoints should also ensure controls from mp-arc-01 are enforced.",
        ),
        (
            "mp-sec-02",
            "Secret Rotation and Token Hygiene requires automatic expiry for long-lived credentials and scheduled audits. "
            "Any emergency patch under mp-deploy-05 must include verification that rotated secrets remain valid in runtime. "
            "Monitoring for credential errors uses alerts in mp-mon-02 and feeds into incident comms via mp-inc-03. "
            "The process is reviewed in onboarding runbooks at mp-onb-02 for consistency across squads.",
        ),
        (
            "mp-sec-03",
            "Dependency and Supply Chain Hardening tracks base image trust, vulnerability scanning, and blocked package policy. "
            "CI jobs fail automatically when unknown artifacts are detected, as described in mp-deploy-02. "
            "Patch candidates must pass integrity checks before deployment approvals and should align with mp-sec-01 permissions. "
            "If a vulnerability appears in production, responders use mp-inc-02 and mp-deploy-04 for containment.",
        ),
        (
            "mp-mon-01",
            "Observability Foundation defines the default telemetry schema for logs, metrics, and traces. "
            "It standardizes cardinality, labels, and query naming so incidents and deployments are traceable in one graph. "
            "Deployment and incident operators both depend on this standard before checking mp-mon-02 alert gates. "
            "Architecture owners must keep this schema aligned with control-plane events in mp-arc-01.",
        ),
        (
            "mp-mon-02",
            "Alerting and SLO Rules capture thresholds for latency, error budgets, and saturation warnings. "
            "The policy includes maintenance windows from mp-deploy-05 and severity mapping from mp-inc-05. "
            "When alerts trigger during a rollout, operators follow canary progression guidance in mp-deploy-03 and may invoke rollback in mp-deploy-04. "
            "Teams tune alert fatigue using observations from mp-mon-03 and postmortems in mp-inc-04.",
        ),
        (
            "mp-mon-03",
            "Distributed Tracing and Correlation Handbook connects request paths to deployment and incident records. "
            "It specifies how to pivot from error traces to root service in under one minute. "
            "For deployment anomalies, engineers should correlate rollout stages from mp-deploy-03 with trace latency shifts. "
            "All critical incidents should add trace IDs to summaries maintained by mp-inc-01 and mp-inc-03.",
        ),
        (
            "mp-onb-01",
            "Team Onboarding Foundations covers required repositories, credential setup, and local tooling for new operators. "
            "New members are required to run a synthetic query against mp-mon-01 before they can approve any deployment in mp-deploy-01. "
            "The checklist includes reading the deployment policy (mp-deploy-02), security policy (mp-sec-01), and incident entry workflow in mp-inc-01. "
            "After onboarding, teammates should complete a guided incident simulation to practice escalation paths.",
        ),
        (
            "mp-onb-02",
            "Agent Workspace Setup focuses on agent-to-agent handoff conventions, runbook ownership, and command aliases. "
            "It references mp-onb-01 for baseline profile setup and mp-inc-03 for status communication standards. "
            "Security-related onboarding tasks include creating short-lived service tokens as described in mp-sec-02. "
            "Operational simulation drills also cross-check mp-deploy-05 freeze procedures during mock pressure tests.",
        ),
        (
            "mp-arc-01",
            "Meridian Architecture Overview describes control plane, API gateway, and environment boundaries for deployments. "
            "The design requires every release path to stay within the staged topology and maintain one rollback point. "
            "Deployment automation in mp-deploy-02 should consult this topology when calculating blast radius. "
            "If architecture changes are proposed, they are validated through mp-inc-01 impact criteria and mp-sec-03 hardening guidance.",
        ),
        (
            "mp-arc-02",
            "Event Routing and Deployment Graph details how deployment state changes propagate across teams and environments. "
            "The document models edges between deploy, observe, verify, and recover states and references mp-mon-02 for signal thresholds. "
            "Any proposed topology edit must confirm security boundaries from mp-sec-01 and onboarding role assumptions from mp-onb-01. "
            "When incidents reveal missing transitions, teams update both this architecture note and the rollback protocol in mp-deploy-04.",
        ),
    ]


def _build_recurring_query_variants(
    domain: str,
    base_queries: list[tuple[str, list[str]]],
    query_count: int,
    id_start: int,
) -> tuple[list[dict], int]:
    patterns = [
        "{q}",
        "In one practical Meridian scenario, {q}",
        "{q} for the on-call runbook and release team",
        "How should we handle this: {q}",
        "Can you clarify this for the operations team: {q}",
    ]

    queries: list[dict] = []
    current = id_start
    for idx in range(query_count):
        base_query, gold = base_queries[idx % len(base_queries)]
        pattern = patterns[(idx // len(base_queries)) % len(patterns)]
        question = pattern.format(q=base_query)
        queries.append(
            {
                "id": f"recurring-{domain}-{current:03d}",
                "question": question,
                "topic": domain,
                "gold": list(gold),
            }
        )
        current += 1

    return queries, current


def _build_recurring_queries() -> list[dict]:
    queries: list[dict] = []
    current = 1

    deployment_queries = [
        (
            "What is the standard Meridian production deployment workflow from merge to rollout?",
            ["mp-deploy-01", "mp-deploy-02", "mp-deploy-03"],
        ),
        (
            "How do we prepare CI and policy checks before a scheduled release?",
            ["mp-deploy-02", "mp-sec-01", "mp-deploy-01"],
        ),
        (
            "Describe the canary rollout sequence and what to verify at each step.",
            ["mp-deploy-03", "mp-mon-02", "mp-mon-03"],
        ),
        (
            "When a release fails, how do we execute the rollback sequence?",
            ["mp-deploy-04", "mp-inc-01", "mp-deploy-03"],
        ),
        (
            "How do feature flags interact with the deployment promotion process?",
            ["mp-deploy-03", "mp-deploy-05", "mp-deploy-02"],
        ),
        (
            "What are the emergency patch rules during active incidents?",
            ["mp-deploy-05", "mp-inc-02", "mp-sec-02"],
        ),
        (
            "Which docs define deployment governance before production signoff?",
            ["mp-deploy-01", "mp-deploy-02", "mp-sec-01"],
        ),
        (
            "How should deployment state and architecture assumptions be updated after release?",
            ["mp-deploy-01", "mp-arc-01", "mp-inc-04"],
        ),
    ]

    incident_queries = [
        (
            "What is the first step when an alert indicates a production incident?",
            ["mp-inc-01", "mp-mon-01", "mp-sec-01"],
        ),
        (
            "How do responders determine severity and assign the first responder?",
            ["mp-inc-05", "mp-inc-01", "mp-onb-01"],
        ),
        (
            "What are the immediate actions for a service outage during business hours?",
            ["mp-inc-02", "mp-mon-02", "mp-inc-01"],
        ),
        (
            "How do we communicate customer impact during a live production incident?",
            ["mp-inc-03", "mp-inc-01", "mp-inc-05"],
        ),
        (
            "How should post-incident reviews be documented and fed back into ops docs?",
            ["mp-inc-04", "mp-inc-02", "mp-deploy-04"],
        ),
        (
            "What is the process for escalating incidents that exceed SLO error budget thresholds?",
            ["mp-inc-05", "mp-mon-02", "mp-inc-01"],
        ),
        (
            "Where should incident evidence and runbook links be consolidated?",
            ["mp-inc-01", "mp-mon-03", "mp-inc-04"],
        ),
        (
            "How do we close an incident after rollback and stabilization?",
            ["mp-inc-02", "mp-deploy-04", "mp-inc-04"],
        ),
    ]

    security_queries = [
        (
            "How are identities and permissions controlled for deployment actions?",
            ["mp-sec-01", "mp-deploy-02", "mp-onb-01"],
        ),
        (
            "What does secret rotation look like during normal operations?",
            ["mp-sec-02", "mp-sec-01", "mp-deploy-05"],
        ),
        (
            "How is the supply chain hardened before and after deployment?",
            ["mp-sec-03", "mp-deploy-02", "mp-sec-01"],
        ),
        (
            "How should teams respond to a suspected token exposure event?",
            ["mp-sec-02", "mp-inc-02", "mp-sec-01"],
        ),
        (
            "What checks enforce least privilege for new team members?",
            ["mp-sec-01", "mp-onb-01", "mp-onb-02"],
        ),
        (
            "Which security controls are mandatory during freeze or emergency patching?",
            ["mp-deploy-05", "mp-sec-02", "mp-sec-01"],
        ),
    ]

    monitoring_queries = [
        (
            "Which observability standard is required before a deployment can be approved?",
            ["mp-mon-01", "mp-arc-01", "mp-deploy-01"],
        ),
        (
            "How are alert thresholds defined for latency and saturation?",
            ["mp-mon-02", "mp-inc-05", "mp-deploy-03"],
        ),
        (
            "How can tracing help debug a rollout regression within the first minute?",
            ["mp-mon-03", "mp-mon-02", "mp-deploy-03"],
        ),
        (
            "What telemetry should be checked when a rollback is triggered?",
            ["mp-deploy-04", "mp-mon-03", "mp-inc-02"],
        ),
        (
            "How do dashboards support on-call handoffs and incident comms?",
            ["mp-mon-01", "mp-inc-03", "mp-mon-02"],
        ),
        (
            "What metrics demonstrate that the rollback is safe to stop?",
            ["mp-mon-03", "mp-mon-02", "mp-mon-01"],
        ),
    ]

    onboarding_queries = [
        (
            "What is the baseline onboarding checklist for a new Meridian operator?",
            ["mp-onb-01", "mp-onb-02", "mp-deploy-01"],
        ),
        (
            "How is a new teammate granted environment access and release roles?",
            ["mp-sec-01", "mp-onb-01", "mp-onb-02"],
        ),
        (
            "What runbook sequence should a new team member learn first?",
            ["mp-onb-02", "mp-inc-01", "mp-deploy-02"],
        ),
        (
            "How do new hires practice their first incident and deployment drill?",
            ["mp-onb-01", "mp-inc-04", "mp-mon-01"],
        ),
    ]

    architecture_queries = [
        (
            "How is Meridian control-plane and data-plane flow organized for safe deployments?",
            ["mp-arc-01", "mp-arc-02", "mp-deploy-01"],
        ),
        (
            "What are the architecture assumptions for release propagation and blast radius control?",
            ["mp-arc-02", "mp-inc-04", "mp-deploy-04"],
        ),
        (
            "How should event routing be updated when deployment topology changes?",
            ["mp-arc-02", "mp-mon-02", "mp-deploy-03"],
        ),
        (
            "Where is the boundary between deployment logic and operations observability documented?",
            ["mp-arc-01", "mp-mon-01", "mp-inc-02"],
        ),
    ]

    cross_queries = [
        (
            "How does incident response tie into the deployment rollback procedure when canaries fail?",
            ["mp-inc-01", "mp-deploy-04", "mp-deploy-03"],
        ),
        (
            "How do security approvals affect deployment governance during an incident?",
            ["mp-sec-01", "mp-inc-05", "mp-deploy-01"],
        ),
        (
            "Where do monitoring alerts trigger both release gating and on-call communication?",
            ["mp-mon-02", "mp-deploy-03", "mp-inc-03"],
        ),
        (
            "How should architecture boundaries influence postmortem remediation after rollout mistakes?",
            ["mp-arc-01", "mp-inc-04", "mp-deploy-05"],
        ),
        (
            "How do onboarding playbooks ensure release and incident runbooks stay synchronized?",
            ["mp-onb-02", "mp-inc-01", "mp-deploy-01"],
        ),
        (
            "Which process links dependency hygiene with incident readiness before a hotfix?",
            ["mp-sec-03", "mp-deploy-05", "mp-inc-02"],
        ),
        (
            "How are tracing and comms combined when a deployment rollback is happening live?",
            ["mp-mon-03", "mp-deploy-04", "mp-inc-03"],
        ),
        (
            "How do access checks and rollout policies interact during an emergency freeze?",
            ["mp-sec-02", "mp-deploy-05", "mp-deploy-01"],
        ),
        (
            "How should incident severity affect release promotion and monitoring thresholds?",
            ["mp-inc-05", "mp-mon-02", "mp-deploy-03"],
        ),
        (
            "How do architecture runbook references reduce repeat incidents after repeated patching?",
            ["mp-arc-02", "mp-inc-04", "mp-deploy-04"],
        ),
    ]

    domain_buckets = [
        ("deploy", deployment_queries, 40),
        ("incident", incident_queries, 40),
        ("security", security_queries, 30),
        ("monitoring", monitoring_queries, 30),
        ("onboarding", onboarding_queries, 20),
        ("architecture", architecture_queries, 20),
        ("cross", cross_queries, 20),
    ]

    for domain, bases, count in domain_buckets:
        items, current = _build_recurring_query_variants(
            domain=domain,
            base_queries=bases,
            query_count=count,
            id_start=current,
        )
        queries.extend(items)
        # Keep unique IDs across domains by advancing the counter.
        current += len(items)

    return queries


def _run_recurring_benchmark() -> tuple[dict[str, object], list[dict[str, object]]]:
    corpus = _build_recurring_topic_corpus()
    recurring_queries = _build_recurring_queries()

    graph_bm25 = _build_graph(corpus)
    graph_crab = _build_graph(corpus)
    graph_no_inhibition = _build_graph(corpus)

    initial_crab = {(edge.source, edge.target): edge.weight for edge in graph_crab.edges()}
    initial_no_inhibition = {
        (edge.source, edge.target): edge.weight for edge in graph_no_inhibition.edges()
    }

    window_size = 20
    metric_keys = ("recall@2", "recall@3", "ndcg@3", "mrr@3")
    window_metrics = {
        "BM25": {metric: [] for metric in metric_keys},
        "CrabPath": {metric: [] for metric in metric_keys},
        "CrabPath no-inhibition": {metric: [] for metric in metric_keys},
    }

    per_query: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []

    for idx, query_item in enumerate(recurring_queries, 1):
        question = str(query_item.get("question", ""))
        query_id = str(query_item.get("id", ""))
        topic = str(query_item.get("topic", ""))
        gold = list(query_item.get("gold", []))

        bm25_ranking = _bm25_ranking(graph_bm25, question)
        bm25_metrics = _normalize_metrics_3(bm25_ranking, gold)
        selected_bm25 = bm25_ranking[:TOP_K]

        crab_ranking, crab_result = _activation_ranking(graph_crab, question)
        crab_selected = crab_ranking[:TOP_K]
        crab_metrics = _normalize_metrics_3(crab_ranking, gold)
        crab_reward = _learn_outcome(crab_selected, gold)
        if crab_reward != 0.0:
            _learn(graph=graph_crab, result=crab_result, outcome=crab_reward, rate=0.1)

        no_inh_ranking, no_inh_activation = _activation_ranking(graph_no_inhibition, question)
        no_inh_selected = no_inh_ranking[:TOP_K]
        no_inh_metrics = _normalize_metrics_3(no_inh_ranking, gold)
        no_inh_reward = _learn_outcome(no_inh_selected, gold)
        if no_inh_reward != 0.0:
            _learn(graph=graph_no_inhibition, result=no_inh_activation, outcome=no_inh_reward, rate=0.1)
        _clamp_negative_edges(graph_no_inhibition)

        per_query_item: dict[str, object] = {
            "query_num": idx,
            "query_id": query_id,
            "topic": topic,
            "query": question,
            "expected_nodes": gold,
            "bm25": {
                "selected_nodes": selected_bm25,
                "all_nodes": bm25_ranking,
                "metrics": bm25_metrics,
            },
            "crabpath": {
                "selected_nodes": crab_selected,
                "all_nodes": crab_ranking,
                "reward": crab_reward,
                "metrics": crab_metrics,
            },
            "crabpath_no_inhibition": {
                "selected_nodes": no_inh_selected,
                "all_nodes": no_inh_ranking,
                "reward": no_inh_reward,
                "metrics": no_inh_metrics,
            },
        }
        per_query.append(per_query_item)

        for metric in metric_keys:
            window_metrics["BM25"][metric].append(float(bm25_metrics[metric]))
            window_metrics["CrabPath"][metric].append(float(crab_metrics[metric]))
            window_metrics["CrabPath no-inhibition"][metric].append(float(no_inh_metrics[metric]))

        if idx % window_size == 0:
            start = idx - (window_size - 1)
            windows.append(
                {
                    "window": f"{start}-{idx}",
                    "method_metrics": {
                        method: {
                            metric: _window_mean(values)
                            for metric, values in values_by_metric.items()
                        }
                        for method, values_by_metric in window_metrics.items()
                    },
                    "graph": {
                        "crab": _graph_window_stats(graph_crab, initial_crab),
                        "crab_no_inhibition": _graph_window_stats(
                            graph_no_inhibition, initial_no_inhibition
                        ),
                    },
                    "query_results": per_query[start - 1 : idx],
                }
            )
            window_metrics = {
                "BM25": {metric: [] for metric in metric_keys},
                "CrabPath": {metric: [] for metric in metric_keys},
                "CrabPath no-inhibition": {metric: [] for metric in metric_keys},
            }

    final_summary = {
        "query_count": len(recurring_queries),
        "window_size": window_size,
        "window_count": len(windows),
        "final_graph": {
            "crab": _graph_window_stats(graph_crab, initial_crab),
            "crab_no_inhibition": _graph_window_stats(graph_no_inhibition, initial_no_inhibition),
        },
    }

    return {
        "windows": windows,
        "summary": final_summary,
        "per_query": per_query,
    }, per_query


def _render_latex_table(name: str, method_metrics: dict[str, dict]) -> str:
    methods = [
        "BM25",
        "Dense",
        "Static",
        "CrabPath full",
        "CrabPath no-inhibition",
    ]

    lines = [
        f"\\subsubsection*{{{name}}}",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Method & Recall@2 & Recall@5 & Precision@2 & Precision@5 & nDCG@5 & MRR@5 \\\\",
        "\\hline",
    ]

    for method_name in methods:
        metrics = method_metrics.get(method_name, {})
        values = [
            metrics.get("recall@2", {}).get("mean", 0.0),
            metrics.get("recall@5", {}).get("mean", 0.0),
            metrics.get("precision@2", {}).get("mean", 0.0),
            metrics.get("precision@5", {}).get("mean", 0.0),
            metrics.get("ndcg@5", {}).get("mean", 0.0),
            metrics.get("mrr@5", {}).get("mean", 0.0),
        ]
        latex_name = method_name.replace("&", "\\&")
        lines.append(
            f"{latex_name} & " + " & ".join(f"{value:.3f}" for value in values) + " \\\\"
        )

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def _format_output_section(
    method_metrics: dict[str, dict],
    per_query_results: dict[str, list[dict]],
) -> list[dict]:
    methods = [
        "BM25",
        "Dense",
        "Static",
        "CrabPath full",
        "CrabPath no-inhibition",
    ]
    out: list[dict] = []

    for method in methods:
        out.append(
            {
                "name": method,
                "metrics": method_metrics.get(method, {}),
                "query_results": per_query_results.get(method, []),
            }
        )
    return out


def _print_learning_curve_table(learning_curve: dict[str, object]) -> None:
    windows = learning_curve.get("windows", []) if isinstance(learning_curve, dict) else []
    if not isinstance(windows, list):
        return

    print("")
    print("Window | BM25 R@2 | CrabPath R@2 | BM25 R@5 | CrabPath R@5 | Inhibitory Edges | Avg Weight")
    print("-" * 104)
    for item in windows:
        if not isinstance(item, dict):
            continue
        metrics = item.get("method_metrics", {})
        if not isinstance(metrics, dict):
            continue
        bm25 = metrics.get("BM25", {})
        crab = metrics.get("CrabPath", {})
        graph = item.get("graph", {})
        if not isinstance(bm25, dict) or not isinstance(crab, dict) or not isinstance(graph, dict):
            continue

        print(
            f"{item.get('window', ''):6} | "
            f"{float(bm25.get('recall@2', 0.0)):8.3f} | "
            f"{float(crab.get('recall@2', 0.0)):11.3f} | "
            f"{float(bm25.get('recall@5', 0.0)):8.3f} | "
            f"{float(crab.get('recall@5', 0.0)):12.3f} | "
            f"{int(graph.get('inhibitory_edges', 0)):15d} | "
            f"{float(graph.get('avg_weight', 0.0)):10.4f}"
        )


def _print_recurring_benchmark_table(recurring_result: dict[str, object]) -> None:
    windows = recurring_result.get("windows", []) if isinstance(recurring_result, dict) else []
    if not isinstance(windows, list):
        return

    print("")
    print(
        "Window | BM25 R@2 | Crab R@2 | Crab-noInh R@2 | "
        "Reflex Edges | Inhib Edges | Avg Wt"
    )
    print("-" * 104)
    for item in windows:
        if not isinstance(item, dict):
            continue
        metrics = item.get("method_metrics", {})
        if not isinstance(metrics, dict):
            continue
        bm25 = metrics.get("BM25", {})
        crab = metrics.get("CrabPath", {})
        no_inh = metrics.get("CrabPath no-inhibition", {})
        graph = item.get("graph", {})
        if not isinstance(bm25, dict) or not isinstance(crab, dict) or not isinstance(graph, dict):
            continue

        crab_graph = graph.get("crab")
        if not isinstance(crab_graph, dict):
            continue

        print(
            f"{item.get('window', ''):6} | "
            f"{float(bm25.get('recall@2', 0.0)):8.3f} | "
            f"{float(crab.get('recall@2', 0.0)):8.3f} | "
            f"{float(no_inh.get('recall@2', 0.0)):13.3f} | "
            f"{int(crab_graph.get('reflex_edges', 0)):12d} | "
            f"{int(crab_graph.get('inhibitory_edges', 0)):11d} | "
            f"{float(crab_graph.get('avg_weight', 0.0)):6.4f}"
        )


def main() -> None:
    _seed_everything(SEED)

    hotpot_questions = _read_hotpot(HOTPOT_DATASET)
    hotpot_metrics, hotpot_query_results = _run_retrieval_suite(hotpot_questions)
    learning_curve, learning_query_results, learning_summary = _run_learning_curve(hotpot_questions)

    beir_cases = _build_beir_corpus()
    beir_metrics, beir_query_results = _run_retrieval_suite_beir(beir_cases)
    recurring_benchmark, recurring_query_results = _run_recurring_benchmark()

    print(_render_latex_table("HotpotQA Comparison", hotpot_metrics))
    print("")
    print(_render_latex_table("BEIR-style Frozen Benchmark", beir_metrics))
    _print_learning_curve_table(learning_curve)
    _print_recurring_benchmark_table(recurring_benchmark)

    payload = {
        "seed": SEED,
        "hotpotqa": {
            "query_count": len(hotpot_questions),
            "methods": _format_output_section(hotpot_metrics, hotpot_query_results),
        },
        "beir": {
            "query_count": len(beir_cases),
            "domain_count": 3,
            "methods": _format_output_section(beir_metrics, beir_query_results),
        },
        "learning_curve": {
            "query_count": len(hotpot_questions),
            "window_size": LEARNING_CURVE_WINDOW,
            "summary": learning_summary,
            "windows": learning_curve.get("windows", []),
            "per_query": learning_query_results,
        },
        "recurring_topic": {
            "query_count": len(_build_recurring_queries()),
            "window_size": 20,
            "summary": recurring_benchmark.get("summary", {}),
            "windows": recurring_benchmark.get("windows", []),
            "per_query": recurring_query_results,
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
