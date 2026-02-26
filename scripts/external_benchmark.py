#!/usr/bin/env python3
"""External benchmark script for HotpotQA subset and frozen BEIR-style suites."""

from __future__ import annotations

import json
import math
import random
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


def main() -> None:
    _seed_everything(SEED)

    hotpot_questions = _read_hotpot(HOTPOT_DATASET)
    hotpot_metrics, hotpot_query_results = _run_retrieval_suite(hotpot_questions)

    beir_cases = _build_beir_corpus()
    beir_metrics, beir_query_results = _run_retrieval_suite_beir(beir_cases)

    print(_render_latex_table("HotpotQA Comparison", hotpot_metrics))
    print("")
    print(_render_latex_table("BEIR-style Frozen Benchmark", beir_metrics))

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
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
