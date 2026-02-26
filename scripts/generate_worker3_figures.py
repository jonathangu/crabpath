#!/usr/bin/env python3
"""Generate worker 3 paper figures from precomputed benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"

CONTEXT_NOISE_DRIFT_RESULTS = ROOT / "scripts" / "context_noise_drift_results.json"
DOWNSTREAM_ACCURACY_RESULTS = ROOT / "scripts" / "downstream_accuracy_results.json"
RAG_COLLAPSE_RESULTS = ROOT / "scripts" / "rag_collapse_results.json"
DOWNSTREAM_QA_RESULTS = ROOT / "scripts" / "downstream_qa_benchmark_results.json"

COLORS = {
    "bm25": "#ff7f0e",
    "crabpath": "#1f77b4",
}

METHOD_COLORS = {
    "bm25": "#ff7f0e",
    "crabpath": "#1f77b4",
    "dense cosine": "#9467bd",
    "static": "#8c8c8c",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", rc={
        "figure.dpi": 140,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#4d4d4d",
    })


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _method_color(name: str) -> str:
    lower = name.lower()
    if "crab" in lower:
        return COLORS["crabpath"]
    if "bm25" in lower:
        return COLORS["bm25"]
    return METHOD_COLORS.get(lower, "#7f7f7f")


def plot_context_utilization(data: dict) -> Path:
    context = data["context_utilization"]
    bm = context["bm25"]
    cp = context["crabpath"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: average token budget loaded.
    ax0 = axes[0]
    x0 = np.arange(1)
    ax0.bar(
        x0 - 0.14,
        [bm["avg_tokens_loaded"]],
        width=0.28,
        color=COLORS["bm25"],
        label="BM25",
    )
    ax0.bar(
        x0 + 0.14,
        [cp["avg_tokens_loaded"]],
        width=0.28,
        color=COLORS["crabpath"],
        label="CrabPath",
    )
    ax0.set_xticks(x0)
    ax0.set_xticklabels(["Tokens"])
    ax0.set_ylabel("Average tokens loaded")
    ax0.set_title("Avg tokens loaded")
    ax0.legend(frameon=False)

    # Right: precision and waste.
    ax1 = axes[1]
    metrics = ["avg_precision", "avg_waste"]
    x1 = np.arange(len(metrics))
    ax1.bar(
        x1 - 0.14,
        [bm["avg_precision"], bm["avg_waste"]],
        width=0.28,
        color=COLORS["bm25"],
        label="BM25",
    )
    ax1.bar(
        x1 + 0.14,
        [cp["avg_precision"], cp["avg_waste"]],
        width=0.28,
        color=COLORS["crabpath"],
        label="CrabPath",
    )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(["Precision", "Waste"])
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Precision vs waste")
    ax1.legend(frameon=False)

    fig.suptitle("Context Utilization: BM25 vs CrabPath", y=1.02)

    return _save(fig, "context_utilization.png")


def plot_noise_sensitivity(data: dict) -> Path:
    levels = sorted(data["noise_sensitivity"]["noise_levels"], key=lambda item: item["distractor_count"])
    distractors = [entry["distractor_count"] for entry in levels]
    bm_recall = [entry["bm25"]["recall_eval"] for entry in levels]
    cp_recall = [entry["crabpath"]["recall_eval"] for entry in levels]
    bm_fpr = [entry["bm25"]["fpr_eval"] for entry in levels]
    cp_fpr = [entry["crabpath"]["fpr_eval"] for entry in levels]

    fig, ax_recall = plt.subplots(figsize=(8, 4.8))
    ax_fpr = ax_recall.twinx()

    ax_recall.plot(distractors, bm_recall, marker="o", color=COLORS["bm25"], linewidth=2.1, label="BM25 Recall@3")
    ax_recall.plot(distractors, cp_recall, marker="o", color=COLORS["crabpath"], linewidth=2.1, label="CrabPath Recall@3")
    ax_recall.set_xlabel("Number of distractors")
    ax_recall.set_ylabel("Recall@3")
    ax_recall.set_ylim(0.0, 1.0)

    ax_fpr.plot(distractors, bm_fpr, marker="s", linestyle="--", color=COLORS["bm25"], linewidth=1.6, label="BM25 FPR")
    ax_fpr.plot(distractors, cp_fpr, marker="s", linestyle="--", color=COLORS["crabpath"], linewidth=1.6, label="CrabPath FPR")
    ax_fpr.set_ylabel("False positive rate")

    ax_recall.set_title("Noise Sensitivity: Distractor Injection")
    ax_recall.set_xticks(distractors)

    lines_recall, labels_recall = ax_recall.get_legend_handles_labels()
    lines_fpr, labels_fpr = ax_fpr.get_legend_handles_labels()
    ax_recall.legend(lines_recall + lines_fpr, labels_recall + labels_fpr, frameon=False, loc="best", ncol=2)

    return _save(fig, "noise_sensitivity.png")


def plot_temporal_drift(data: dict) -> Path:
    drift = data["temporal_drift"]

    phases = [
        ("Phase 1", drift["phase1"]["recall"]["bm25"], drift["phase1"]["recall"]["crabpath"]),
        ("Phase 2", drift["phase2"]["recall"]["bm25"], drift["phase2"]["recall"]["crabpath"]),
        ("Phase 3", drift["phase3"]["recall"]["bm25"], drift["phase3"]["recall"]["crabpath"]),
    ]

    labels = [item[0] for item in phases]
    bm = [item[1] for item in phases]
    cp = [item[2] for item in phases]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - w / 2, bm, width=w, color=COLORS["bm25"], label="BM25")
    ax.bar(x + w / 2, cp, width=w, color=COLORS["crabpath"], label="CrabPath")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Recall@3")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Temporal Drift: Recall@3 by Phase")
    ax.legend(frameon=False, loc="upper right")

    return _save(fig, "temporal_drift.png")


def plot_downstream_task_quality(data: dict) -> Path:
    payload = data["downstream_accuracy"]
    bm = payload["bm25"]
    cp = payload["crabpath"]

    metrics = ["f1", "coverage", "noise_ratio", "answer_length"]
    values_bm = [bm[metric] for metric in metrics]
    values_cp = [cp[metric] for metric in metrics]
    y = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    height = 0.36
    ax.barh(y - height / 2, values_bm, height=height, color=COLORS["bm25"], label="BM25")
    ax.barh(y + height / 2, values_cp, height=height, color=COLORS["crabpath"], label="CrabPath")

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Metric value")
    ax.set_title("Downstream Task Quality")
    ax.axvline(0, color="#4d4d4d", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")

    return _save(fig, "downstream_accuracy.png")


def plot_ruler_multi_fact(data: dict) -> Path:
    ruler = data["ruler_multi_fact"]
    # JSON may serialize integer keys as strings.
    n_values = sorted(ruler.keys(), key=lambda k: int(k))

    all_found_bm = [ruler[str(n)]["bm25"]["all_found"] for n in n_values]
    all_found_cp = [ruler[str(n)]["crabpath"]["all_found"] for n in n_values]
    partial_bm = [ruler[str(n)]["bm25"]["partial_recall"] for n in n_values]
    partial_cp = [ruler[str(n)]["crabpath"]["partial_recall"] for n in n_values]

    x = np.arange(len(n_values))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    axes[0].bar(
        x - width / 2,
        all_found_bm,
        width=width,
        color=COLORS["bm25"],
        label="BM25",
    )
    axes[0].bar(
        x + width / 2,
        all_found_cp,
        width=width,
        color=COLORS["crabpath"],
        label="CrabPath",
    )
    axes[0].set_title("All-found")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(n_values)
    axes[0].set_xlabel("N facts")

    axes[1].bar(
        x - width / 2,
        partial_bm,
        width=width,
        color=COLORS["bm25"],
        label="BM25",
    )
    axes[1].bar(
        x + width / 2,
        partial_cp,
        width=width,
        color=COLORS["crabpath"],
        label="CrabPath",
    )
    axes[1].set_title("Partial recall")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(n_values)
    axes[1].set_xlabel("N facts")

    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("RULER Multi-Fact Retrieval", y=1.02)

    return _save(fig, "ruler_multi_fact.png")


def plot_traversal_comparison() -> Path:
    # Values from quick benchmark (size, config -> (Recall@3, avg_hops)).
    values = {
        50: {
            "A": (0.100, 3.0),
            "B": (0.120, 28.6),
            "C": (0.140, 30.0),
            "D": (0.100, 28.8),
        },
        200: {
            "A": (0.000, 3.0),
            "B": (0.040, 30.0),
            "C": (0.020, 30.0),
            "D": (0.040, 30.0),
        },
    }

    configs = ["A", "B", "C", "D"]
    sizes = [50, 200]
    x = np.arange(len(configs))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, size in enumerate(sizes):
        recalls = [values[size][cfg][0] for cfg in configs]
        hops = [values[size][cfg][1] for cfg in configs]
        offset = -width / 2 if i == 0 else width / 2
        bars = ax.bar(
            x + offset,
            recalls,
            width=width,
            color=(COLORS["crabpath"] if i == 1 else "#8da0cb"),
            label=f"Size={size}",
            alpha=1.0,
        )
        for bar, hop in zip(bars, hops):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{hop:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Recall@3")
    ax.set_ylim(0.0, 0.17)
    ax.set_xlabel("Traversal configuration")
    ax.set_title("Traversal Mechanisms: Recall@3 Comparison")
    ax.legend(frameon=False, loc="upper right", ncol=2)

    return _save(fig, "traversal_comparison.png")


def plot_rag_collapse(data: dict) -> Path:
    comp = data["comparison"]
    systems = ["static", "rag", "crab"]
    labels = ["Static", "RAG", "CrabPath"]
    values = [
        comp["static_chars_per_query"],
        comp["rag_chars_per_query"],
        comp["crab_chars_per_query"],
    ]
    colors = [COLORS["bm25"], "#8da0cb", COLORS["crabpath"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.56)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tokens per turn")
    ax.set_title("Where RAG Fails: Semantic Similarity Trap")

    for idx, val in enumerate(values):
        ax.text(idx, val + 20, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    return _save(fig, "rag_collapse.png")


def plot_downstream_qa(data: dict) -> Path:
    part1 = data["part1"]["metrics"]
    part2a = data["part2a_distractor_injection"]["false_positive_rate"]
    part2c = data["part2c_temporal_drift"]
    methods = [
        "BM25",
        "Dense cosine",
        "Static",
        "CrabPath full",
        "CrabPath no-inhibition",
    ]
    methods_existing = [m for m in methods if m in {row["method"] for row in part1}]
    method_map = {row["method"]: row for row in part1}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=False)

    # Part 1: base recall comparison.
    recall_vals = [method_map[m]["avg_recall"] for m in methods_existing]
    colors1 = [_method_color(m) for m in methods_existing]
    axes[0].bar(np.arange(len(methods_existing)), recall_vals, color=colors1)
    axes[0].set_xticks(np.arange(len(methods_existing)))
    axes[0].set_xticklabels(methods_existing, rotation=22, ha="right")
    axes[0].set_ylabel("Recall (supporting docs)")
    axes[0].set_title("Part 1 — Base comparison")
    axes[0].set_ylim(0.0, 1.0)

    # Part 2a: distractor injection FPR.
    fpr_vals = [part2a[m] for m in methods_existing]
    colors2 = [_method_color(m) for m in methods_existing]
    axes[1].bar(np.arange(len(methods_existing)), fpr_vals, color=colors2)
    axes[1].set_xticks(np.arange(len(methods_existing)))
    axes[1].set_xticklabels(methods_existing, rotation=22, ha="right")
    axes[1].set_ylabel("False positive rate")
    axes[1].set_title("Part 2a — Distractor injection")
    axes[1].set_ylim(0.0, 1.0)

    # Part 2c: temporal drift (pre/post).
    drift_methods = [m for m in methods_existing if m in part2c]
    pre = [part2c[m]["pre_avg_recall"] for m in drift_methods]
    post = [part2c[m]["post_avg_recall"] for m in drift_methods]
    x = np.arange(len(drift_methods))
    w = 0.34
    axes[2].bar(x - w / 2, pre, width=w, color="#8da0cb", label="Pre-update recall")
    axes[2].bar(x + w / 2, post, width=w, color=COLORS["crabpath"], label="Post-update recall")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(drift_methods, rotation=22, ha="right")
    axes[2].set_title("Part 2c — Temporal drift")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Recall")
    axes[2].legend(frameon=False, loc="lower right")

    return _save(fig, "downstream_qa.png")


def main() -> None:
    _style()

    context_data = _load_json(CONTEXT_NOISE_DRIFT_RESULTS)
    downstream_acc_data = _load_json(DOWNSTREAM_ACCURACY_RESULTS)
    rag_collapse_data = _load_json(RAG_COLLAPSE_RESULTS)
    downstream_qa_data = _load_json(DOWNSTREAM_QA_RESULTS)

    plot_context_utilization(context_data)
    plot_noise_sensitivity(context_data)
    plot_temporal_drift(context_data)
    plot_downstream_task_quality(downstream_acc_data)
    plot_ruler_multi_fact(downstream_acc_data)
    plot_traversal_comparison()
    plot_rag_collapse(rag_collapse_data)
    plot_downstream_qa(downstream_qa_data)


if __name__ == "__main__":
    main()
