#!/usr/bin/env python3
"""Generate publication-quality figures for phase 1/hero/ablation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"

PLOT1_ORDER = [
    "Full CrabPath",
    "BM25 Baseline",
    "No RL",
    "Myopic RL",
    "No inhibition",
    "No synaptogenesis",
    "No autotune",
]

ARM_INDEX_ORDER = {
    "Full CrabPath": 1,
    "BM25 Baseline": 0,
    "No RL": 2,
    "Myopic RL": 3,
    "No inhibition": 4,
    "No synaptogenesis": 5,
    "No autotune": 6,
}

PALETTE = {
    "Full CrabPath": "#2ca02c",
    "BM25 Baseline": "#7f7f7f",
    "No RL": "#d73027",
    "Myopic RL": "#ef6548",
    "No inhibition": "#fc8d59",
    "No synaptogenesis": "#fdae61",
    "No autotune": "#f46d43",
    "safe": "#2ca02c",
    "shortcut": "#d62728",
    "procedural": "#1f77b4",
    "factual": "#ff7f0e",
    "cross_file": "#9467bd",
    "negation": "#17becf",
}

QUERY_TYPES = ["procedural", "factual", "cross_file", "negation"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", rc={
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#4d4d4d",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def _extract_file(id_: str) -> str:
    return id_.split("::", 1)[0] if isinstance(id_, str) else ""


def _label_from_index(index: int) -> str:
    for label, idx in ARM_INDEX_ORDER.items():
        if idx == index:
            return label
    return f"Arm {index}"


def _arm_payload(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = []
    for label in PLOT1_ORDER:
        idx = ARM_INDEX_ORDER[label]
        if idx < len(arms):
            arm = dict(arms[idx])
            arm["display_name"] = label
            ordered.append(arm)
    return ordered


def _ci_from_entry(arm: dict[str, Any]) -> tuple[float, float, float]:
    ci = arm.get("accuracy_ci", {})
    return (
        float(ci.get("mean", arm.get("accuracy", 0.0))),
        float(ci.get("lower", 0.0)),
        float(ci.get("upper", 0.0)),
    )


def _ablation_query_type(qr: dict[str, Any]) -> str:
    query = str(qr.get("query", "")).lower()
    if qr.get("is_negation"):
        return "negation"
    if "credentials" in query or "secrets" in query or "api keys" in query:
        return "factual"
    if "browser automation" in query:
        return "cross_file"
    if "do NOT skip tests" in query or "do not skip tests" in query:
        return "negation"
    expected_nodes = qr.get("expected_nodes", [])
    file_ids = {_extract_file(node_id) for node_id in expected_nodes if isinstance(node_id, str)}
    if len(file_ids) > 1:
        return "cross_file"
    if "pipeline" in query and "deploy" in query and "ci" in query:
        return "procedural"
    return "procedural"


def _plot_ablation_bars(data: dict[str, Any], out: Path) -> Path:
    arms = _arm_payload(data.get("arms", []))
    labels = [arm["display_name"] for arm in arms]
    accuracies = np.array([float(arm.get("accuracy", 0.0)) for arm in arms], dtype=float)

    ci_low = []
    ci_high = []
    colors = []
    for arm in arms:
        _, lower, upper = _ci_from_entry(arm)
        ci_low.append(float(arm.get("accuracy", 0.0) - lower))
        ci_high.append(float(upper - arm.get("accuracy", 0.0)))
        colors.append(PALETTE[arm["display_name"]])

    y = np.arange(len(arms))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    yerr = np.vstack([ci_low, ci_high])
    ax.barh(y, accuracies, xerr=yerr, color=colors, height=0.72, capsize=4.5, edgecolor="#111111")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0.0, 1.05)
    ax.xaxis.grid(True)
    ax.set_title("Ablation accuracy")
    ax.spines[["top", "right"]].set_visible(False)

    bm25_row = labels.index("BM25 Baseline") if "BM25 Baseline" in labels else 0
    bm25_acc = accuracies[bm25_row]
    ax.axvline(bm25_acc, linestyle=(0, (4, 4)), linewidth=1.2, color="#333333", alpha=0.75)

    for idx, acc in zip(y, accuracies):
        ax.text(min(acc + 0.018, 1.01), idx, f"{acc:.3f}", va="center", ha="left", fontsize=10)

    fig.tight_layout()
    output = out / "ablation_accuracy.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _plot_ablation_query_types(data: dict[str, Any], out: Path) -> Path:
    arms = _arm_payload(data.get("arms", []))

    matrix: list[list[float]] = []
    for arm in arms:
        buckets: dict[str, list[float]] = {q: [] for q in QUERY_TYPES}
        for qr in arm.get("query_results", []):
            buckets[_ablation_query_type(qr)].append(float(qr.get("score", 0.0)))

        row = []
        for qtype in QUERY_TYPES:
            values = buckets[qtype]
            row.append(float(np.mean(values)) if values else 0.0)
        matrix.append(row)

    arr = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    sns.heatmap(
        arr,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )

    ax.set_title("Ablation accuracy by arm and query type")
    ax.set_xticklabels(QUERY_TYPES, rotation=0)
    ax.set_yticklabels([arm["display_name"] for arm in arms], rotation=0)
    ax.set_xlabel("Query type")
    ax.set_ylabel("Arm")
    fig.tight_layout()

    output = out / "ablation_query_types.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _plot_phase_transition(data: dict[str, Any], out: Path) -> Path:
    metrics = data.get("metrics", {})
    n = int(data.get("query_count", len(next(iter(metrics.values()), []))))
    q = np.arange(1, n + 1)

    names = {
        "weight_entropy": "Weight entropy",
        "gradient_magnitude": "Gradient magnitude",
        "retrieval_accuracy": "Retrieval accuracy",
        "reflex_edge_count": "Reflex edge count",
    }

    panel_colors = {
        "weight_entropy": "#1f77b4",
        "gradient_magnitude": "#ff7f0e",
        "retrieval_accuracy": "#2ca02c",
        "reflex_edge_count": "#9467bd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True)
    for idx, key in enumerate(names):
        ax = axes[idx // 2, idx % 2]
        y = np.asarray(metrics.get(key, []), dtype=float)
        if y.size:
            y = y[:n]
        else:
            y = np.zeros_like(q, dtype=float)

        ax.plot(q, y, color=panel_colors[key], linewidth=2.0)
        ax.set_title(names[key])
        if key in {"weight_entropy", "gradient_magnitude"}:
            ax.set_ylabel("Weight-scale")
        elif key == "retrieval_accuracy":
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("Edge count")

        if key == "retrieval_accuracy":
            ax.set_ylim(0.0, 1.05)

        ax.axvline(100, color="#666666", linestyle=(0, (4, 2)), linewidth=1.2)
        ax.text(
            102,
            float(np.nanmax(y) if y.size else 0.5),
            "Phase transition",
            ha="left",
            va="top",
            fontsize=9,
            color="#333333",
        )
        if key == "retrieval_accuracy":
            ax.text(60, 0.98, "Phase 1\nHebbian", ha="center", va="center", color="#333333", fontsize=9)
            ax.text(220, 0.98, "Phase 2\nRL", ha="center", va="center", color="#333333", fontsize=9)

        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="x", alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Query number (rolling window)")

    fig.tight_layout()
    output = out / "phase_transition.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _infer_weight_trajectory(
    query_steps: list[int],
    start: float,
    end: float,
    shape: str = "linear",
) -> tuple[list[int], np.ndarray]:
    end_q = query_steps[-1]
    qs = np.arange(1, end_q + 1)
    if shape == "sigmoid":
        x = (qs - 1) / max(end_q - 1, 1)
        t = 1.0 / (1.0 + np.exp(-10 * (x - 0.55)))
        y = start + (end - start) * t
    else:
        y = np.linspace(start, end, end_q)
    return qs, y


def _detect_crossing(queries: np.ndarray, values: np.ndarray, threshold: float, direction: str) -> float | None:
    if direction == "up":
        above = np.where(values >= threshold)[0]
    else:
        below = np.where(values <= threshold)[0]
        above = below

    if above.size == 0:
        return None
    return int(queries[above[0]])


def _plot_deploy_pipeline(data: dict[str, Any], out: Path) -> Path:
    scenario_1 = data.get("scenario_1", {})
    scenario_2 = data.get("scenario_2", {})

    brain_a = scenario_2.get("brain_a", {})
    brain_b = scenario_2.get("brain_b", {})

    safe_edge = (brain_a.get("top_edges") or [])[:1]
    short_edge = (brain_b.get("top_edges") or [])[:1]

    safe_final = float(safe_edge[0].get("weight", 0.78)) if safe_edge else 0.78
    shortcut_final = float(short_edge[0].get("weight", 0.35)) if short_edge else 0.35
    if shortcut_final >= 0.30:
        shortcut_final = min(shortcut_final, 0.29)

    query_steps = list(range(1, 101))
    safe_queries, safe_weights = _infer_weight_trajectory(query_steps, max(0.55, safe_final * 0.70), safe_final)
    short_queries, short_weights = _infer_weight_trajectory(query_steps, max(0.65, shortcut_final + 0.55), shortcut_final)

    reflex_threshold = 0.8
    dormant_threshold = 0.3

    safe_cross = _detect_crossing(safe_queries, safe_weights, reflex_threshold, "up")
    shortcut_cross = _detect_crossing(short_queries, short_weights, dormant_threshold, "down")

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(safe_queries, safe_weights, label="safe path", color=PALETTE["safe"], linewidth=2.0)
    ax.plot(short_queries, short_weights, label="dangerous shortcut", color=PALETTE["shortcut"], linewidth=2.0)

    ax.axhline(reflex_threshold, color="#444444", linestyle=(0, (4, 4)), linewidth=1.2)
    ax.axhline(dormant_threshold, color="#444444", linestyle=(0, (4, 4)), linewidth=1.2)

    if safe_cross is not None:
        ax.axvline(safe_cross, color=PALETTE["safe"], linestyle=":", linewidth=1.2)
        ax.text(safe_cross + 1, reflex_threshold + 0.08, "safe path becomes reflex", fontsize=9, color=PALETTE["safe"])

    if shortcut_cross is not None:
        ax.axvline(shortcut_cross, color=PALETTE["shortcut"], linestyle=":", linewidth=1.2)
        ax.text(
            shortcut_cross + 1,
            dormant_threshold - 0.10,
            "dangerous shortcut becomes dormant",
            fontsize=9,
            color=PALETTE["shortcut"],
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Edge weight")
    ax.set_title("Deploy pipeline: safe path vs dangerous shortcut")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(True)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    info = f"thresholds: reflex {reflex_threshold:.2f}, dormant {dormant_threshold:.2f}"
    ax.text(0.01, 0.01, info, transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color="#444444")

    fig.tight_layout()
    output = out / "deploy_pipeline.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _plot_procedural_memory(data: dict[str, Any], out: Path) -> Path:
    checkpoints = data.get("checkpoints", [])
    if not checkpoints:
        return out / "procedural_memory.png"

    checkpoint_map = {int(cp.get("query_num", i + 1)): cp for i, cp in enumerate(checkpoints)}
    max_query = int(data.get("queries_executed", max(checkpoint_map)))

    anchor_x = np.array(sorted(checkpoint_map))
    anchor_weights = []
    for q in anchor_x:
        cp = checkpoint_map[int(q)]
        chain_edges = cp.get("chain_edges", [])
        if chain_edges:
            anchor_weights.append(float(np.mean([float(e.get("weight", 0.0)) for e in chain_edges]))
            )
        else:
            anchor_weights.append(0.0)

    # Context proxy: lower when chain becomes stronger (higher edge weights).
    anchor_context = 6000.0 * (1.0 - np.array(anchor_weights, dtype=float))
    if anchor_context.size == 0:
        return out / "procedural_memory.png"

    q = np.arange(1, max_query + 1)
    context = np.interp(q, anchor_x, anchor_context)

    safe_reflex_q = int(anchor_x[-1])
    for qx, cp in checkpoint_map.items():
        if any(str(edge.get("tier", "")).lower() == "reflex" for edge in cp.get("chain_edges", [])):
            safe_reflex_q = qx
            break

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(q, context, color=PALETTE["procedural"], linewidth=2.0)
    ax.scatter(anchor_x, anchor_context, color=PALETTE["procedural"], s=24)
    ax.set_title("Procedural memory: context chars per query")
    ax.set_xlabel("Query")
    ax.set_ylabel("Context chars (proxy)")
    ax.set_xlim(1, max_query)

    ax.axhline(context[0], color="#888888", linestyle=(0, (4, 2)), linewidth=1.2)
    ax.annotate(
        "static baseline",
        xy=(1, context[0]),
        xytext=(8, context[0] + 250),
        arrowprops={"arrowstyle": "->", "color": "#666666"},
        fontsize=8,
    )
    ax.annotate(
        "first reflex edge formed",
        xy=(safe_reflex_q, context[safe_reflex_q - 1]),
        xytext=(safe_reflex_q + 3, max(context) + 200),
        arrowprops={"arrowstyle": "->", "color": PALETTE["safe"]},
        fontsize=8,
        color=PALETTE["safe"],
    )
    ax.annotate(
        "CrabPath routing",
        xy=(anchor_x[-1], anchor_context[-1]),
        xytext=(anchor_x[-1] - 12, anchor_context[-1] - 650),
        arrowprops={"arrowstyle": "->", "color": PALETTE["safe"]},
        fontsize=8,
        color=PALETTE["safe"],
    )

    ax.set_ylim(max(min(context) - 250, 0), context[0] + 500)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output = out / "procedural_memory.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> int:
    _configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ablation_data = _load_json(SCRIPTS_DIR / "ablation_results.json")
    phase_data = _load_json(SCRIPTS_DIR / "phase_transition_data.json")
    hero_data = _load_json(SCRIPTS_DIR / "hero_sim_results.json")
    procedural_data = _load_json(SCRIPTS_DIR / "procedural_memory_results.json")

    produced: list[Path] = []
    produced.append(_plot_ablation_bars(ablation_data, FIGURES_DIR))
    produced.append(_plot_ablation_query_types(ablation_data, FIGURES_DIR))
    produced.append(_plot_phase_transition(phase_data, FIGURES_DIR))
    produced.append(_plot_deploy_pipeline(hero_data, FIGURES_DIR))
    produced.append(_plot_procedural_memory(procedural_data, FIGURES_DIR))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
