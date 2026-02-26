#!/usr/bin/env python3
"""Generate worker 2 paper figures from benchmark JSON outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"

CRAB_COLOR = "#1f77b4"
BM25_COLOR = "#ff7f0e"
GRAY_COLORS = ["#6e6e6e", "#a0a0a0", "#7b7b7b"]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
        }
    )


def save_png(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _read_json(path: str) -> Dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _ci_lohi(rec: Dict, metric: str, method: str):
    if "ci" not in rec["metrics"][metric]:
        return None
    ci = rec["metrics"][metric]["ci"]
    if isinstance(ci, dict):
        return ci.get("mean", 0.0), ci.get("lower", 0.0), ci.get("upper", 0.0)
    if isinstance(ci, (list, tuple)) and len(ci) == 3:
        return ci[0], ci[1], ci[2]
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return ci[0], ci[1], ci[1]
    return None


def plot_hotpotqa_cold_start() -> Path:
    set_style()
    data = _read_json("scripts/external_benchmark_results.json")
    methods = [
        ("BM25", BM25_COLOR),
        ("Dense", GRAY_COLORS[0]),
        ("Static", GRAY_COLORS[1]),
        ("CrabPath full", CRAB_COLOR),
        ("CrabPath no-inhibition", GRAY_COLORS[2]),
    ]
    metrics = ["recall@2", "recall@5", "ndcg@5"]
    metric_labels = ["Recall@2", "Recall@5", "NDCG@5"]
    method_map = {entry["name"]: entry for entry in data["hotpotqa"]["methods"]}

    x = np.arange(len(metric_labels))
    width = 0.14

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for i, (method_name, color) in enumerate(methods):
        rec = method_map[method_name]
        values = [float(rec["metrics"][m]["mean"]) for m in metrics]
        ci_values = [_ci_lohi(rec, m, None) for m in metrics]
        errs = [0.0, 0.0, 0.0]
        for j, ci in enumerate(ci_values):
            if ci is None:
                continue
            _, lo, hi = ci
            errs[j] = (values[j] - lo, hi - values[j])
        pos = x + (i - (len(methods) - 1) / 2) * width
        if any(e != 0.0 for e in errs):
            ax.bar(
                pos,
                values,
                width=width,
                color=color,
                label=method_name,
                yerr=np.array(errs).T,
                capsize=2.0,
                edgecolor="#111111",
                linewidth=0.4,
            )
        else:
            ax.bar(
                pos,
                values,
                width=width,
                color=color,
                label=method_name,
                edgecolor="#111111",
                linewidth=0.4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Metric value")
    ax.set_title("HotpotQA (100 questions, cold start)")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return save_png(fig, "hotpotqa_cold_start.png")


def plot_learning_curve() -> Path:
    set_style()
    data = _read_json("scripts/external_benchmark_results.json")["learning_curve"]["windows"]

    windows = [w["window"] for w in data]
    x = np.arange(len(windows))
    bm25 = [w["method_metrics"]["BM25"]["recall@2"] for w in data]
    crab = [w["method_metrics"]["CrabPath"]["recall@2"] for w in data]
    inhibitory = [w["graph"]["inhibitory_edges"] for w in data]
    avg_weight = [w["graph"]["avg_weight"] for w in data]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.0))
    ax2 = ax1.twinx()

    ax1.plot(x, bm25, marker="o", linewidth=2.1, color=BM25_COLOR, label="BM25")
    ax1.plot(x, crab, marker="o", linewidth=2.1, color=CRAB_COLOR, label="CrabPath")
    ax2.plot(x, inhibitory, marker="s", linewidth=1.8, linestyle="--", color="#2ca02c", label="Inhibitory edges")
    ax2.plot(x, avg_weight, marker="^", linewidth=1.8, linestyle=":", color="#9467bd", label="Avg weight")

    ax1.set_xticks(x)
    ax1.set_xticklabels(windows)
    ax1.set_xlabel("Query window")
    ax1.set_ylabel("Recall@2")
    ax2.set_ylabel("Inhibitory edges / Avg weight")
    ax1.set_title("Persistent Graph Learning")
    ax1.set_ylim(0.0, 1.0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, ncol=2)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    return save_png(fig, "learning_curve.png")


def plot_recurring_topic() -> Path:
    set_style()
    data = _read_json("scripts/external_benchmark_results.json")["recurring_topic"]["windows"]

    windows = [w["window"] for w in data]
    x = np.arange(len(windows))
    bm25 = [w["method_metrics"]["BM25"]["recall@2"] for w in data]
    crab = [w["method_metrics"]["CrabPath"]["recall@2"] for w in data]
    crab_no_inh = [w["method_metrics"]["CrabPath no-inhibition"]["recall@2"] for w in data]
    reflex = [w["graph"]["crab"]["reflex_edges"] for w in data]
    inhibitory = [w["graph"]["crab"]["inhibitory_edges"] for w in data]
    dormant = [w["graph"]["crab"]["dormant_edges"] for w in data]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.0))
    ax2 = ax1.twinx()

    ax1.plot(x, bm25, marker="o", linewidth=2.0, color=BM25_COLOR, label="BM25")
    ax1.plot(x, crab, marker="o", linewidth=2.0, color=CRAB_COLOR, label="CrabPath")
    ax1.plot(
        x,
        crab_no_inh,
        marker="o",
        linewidth=2.0,
        color=GRAY_COLORS[2],
        label="CrabPath no-inhibition",
    )

    ax2.plot(x, reflex, linewidth=1.7, linestyle="--", color="#2ca02c", label="Reflex edges")
    ax2.plot(x, inhibitory, linewidth=1.7, linestyle=":", color="#d62728", label="Inhibitory edges")
    ax2.plot(x, dormant, linewidth=1.7, linestyle="-.", color="#9467bd", label="Dormant edges")

    ax1.set_xticks(x)
    ax1.set_xticklabels(windows)
    ax1.set_xlabel("Query window")
    ax1.set_ylabel("Recall@2")
    ax2.set_ylabel("Edges")
    ax1.set_title("Recurring Topic (200 queries, 20 docs)")
    ax1.set_ylim(0.0, 1.0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, ncol=2)
    return save_png(fig, "recurring_topic.png")


def _parse_margin(value: str):
    m = re.match(r"^(CRAB|BM25)\+?(-?\d*\.?\d+)$", value or "", re.IGNORECASE)
    if not m:
        return None, np.nan
    return m.group(1).upper(), float(m.group(2))


def plot_sparsity_crossover() -> Path:
    set_style()
    data = _read_json("scripts/sparsity_scale_results.json")
    sizes = list(data["sizes"])
    row_defs = data["sparsity_levels"]
    rows = [r["name"] for r in row_defs]
    table = data["crossover_table"]

    score = np.zeros((len(rows), len(sizes)))
    annot = [["" for _ in sizes] for _ in rows]
    for i, row in enumerate(rows):
        for j, size in enumerate(sizes):
            raw = table[row][str(size)]
            winner, margin = _parse_margin(raw)
            annot[i][j] = f"{margin:+.3f}"
            score[i, j] = 1.0 if winner == "CRAB" else -1.0

    cmap = sns.color_palette(["#d73027", "#1a9850"], as_cmap=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.heatmap(
        score,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=sizes,
        yticklabels=rows,
        center=0.0,
        ax=ax,
    )
    ax.set_xlabel("Graph size")
    ax.set_ylabel("Sparsity level")
    ax.set_title("Phase2 Recall@3: CrabPath vs BM25 Crossover")
    return save_png(fig, "sparsity_crossover.png")


def _find_scaling_record_scaling(data: List[Dict], size: int):
    for rec in data:
        if rec["size"] == size:
            return rec
    return None


def plot_scaling_curves() -> Path:
    set_style()
    data = _read_json("scripts/niah_scaling_results.json")["scaling"]["results"]
    sizes = sorted({rec["size"] for rec in data})

    bm25 = []
    crab = []
    bm_tokens = []
    cp_tokens = []
    for size in sizes:
        rec = _find_scaling_record_scaling(data, size)
        if rec is None:
            continue
        bm25.append(rec["bm25"]["metrics"]["recall@3"]["mean"])
        crab.append(rec["crabpath"]["metrics"]["recall@3"]["mean"])
        bm_tokens.append(rec["bm25"]["avg_tokens_loaded"]["bm25"])
        cp_tokens.append(rec["crabpath"]["avg_tokens_loaded"]["crabpath"])

    fig, ax1 = plt.subplots(figsize=(7.4, 4.1))
    ax2 = ax1.twinx()
    ax1.plot(sizes, bm25, marker="o", linewidth=2.0, color=BM25_COLOR, label="BM25 Recall@3")
    ax1.plot(sizes, crab, marker="o", linewidth=2.0, color=CRAB_COLOR, label="CrabPath Recall@3")
    ax2.plot(sizes, bm_tokens, marker="s", linewidth=1.5, linestyle="--", color="#2ca02c", label="BM25 tokens/query")
    ax2.plot(sizes, cp_tokens, marker="^", linewidth=1.5, linestyle=":", color="#9467bd", label="CrabPath tokens/query")

    ax1.set_xscale("log")
    ax1.set_xlabel("Graph size")
    ax1.set_ylabel("Recall@3")
    ax2.set_ylabel("Tokens loaded per query")
    ax1.set_title("Scaling: Recall@3 vs Graph Size (sparse, edge_ratio=0.1)")
    ax1.set_ylim(0.0, 1.0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, ncol=1)
    return save_png(fig, "scaling_curves.png")


def plot_niah_multi_needle() -> Path:
    set_style()
    records = [r for r in _read_json("scripts/niah_scaling_results.json")["niah"]["results"] if r["size"] == 1000]
    records = sorted(records, key=lambda r: r["k"])

    ks = [r["k"] for r in records]
    bm = [r["bm25"]["metrics"]["partial_recall"]["mean"] for r in records]
    crab = [r["crabpath"]["metrics"]["partial_recall"]["mean"] for r in records]

    x = np.arange(len(ks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - width / 2, bm, width=width, color=BM25_COLOR, label="BM25")
    ax.bar(x + width / 2, crab, width=width, color=CRAB_COLOR, label="CrabPath")

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K")
    ax.set_ylabel("Partial Recall")
    ax.set_title("Multi-Needle Retrieval at 1000 Nodes")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    return save_png(fig, "niah_multi_needle.png")


def main() -> int:
    plot_hotpotqa_cold_start()
    plot_learning_curve()
    plot_recurring_topic()
    plot_sparsity_crossover()
    plot_scaling_curves()
    plot_niah_multi_needle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
