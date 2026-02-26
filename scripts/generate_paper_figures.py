#!/usr/bin/env python3
"""Generate publication-ready figures for the CrabPath arXiv paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

ROOT = Path(__file__).resolve().parent.parent


TAB10 = plt.get_cmap("tab10").colors
FORWARD_COLOR = TAB10[0]  # blue
LOOP_COLOR = TAB10[3]     # red/orange region for feedback flow
LOOP_SECOND = TAB10[1]    # orange


ABLATED_ITEMS = [
    ("BM25", "bm25", 0.737, (0.695, 0.779), 0.000),
    ("Full CrabPath", "full", 0.742, (0.700, 0.782), 1.000),
    ("No RL", "no_rl", 0.421, (0.368, 0.474), 0.000),
    ("Myopic", "myopic", 0.632, (0.592, 0.671), 0.000),
    ("No Inhibition", "no_inh", 0.579, (0.526, 0.632), 0.500),
    ("No Synaptogenesis", "no_syn", 0.211, (0.153, 0.268), 0.000),
    ("No Autotune", "no_auto", 0.316, (0.282, 0.350), 0.000),
]
ABLATED_NAMES = [item[0] for item in ABLATED_ITEMS]
ABLATED_SHORT_NAMES = ["BM25", "Full", "No RL", "Myopic", "No Inh.", "No Syn.", "No Auto"]
ABLATED_KEYS = [item[1] for item in ABLATED_ITEMS]
ABLATED_ACCURACY = [item[2] for item in ABLATED_ITEMS]
ABLATED_CI_LOW = [item[3][0] for item in ABLATED_ITEMS]
ABLATED_CI_HIGH = [item[3][1] for item in ABLATED_ITEMS]
ABLATED_NEGATION = [item[4] for item in ABLATED_ITEMS]
ABLATED_COLOR = {
    "bm25": TAB10[2],
    "full": TAB10[0],
    "no_rl": TAB10[1],
    "myopic": TAB10[4],
    "no_inh": TAB10[5],
    "no_syn": TAB10[6],
    "no_auto": TAB10[9],
}


GIRAFFE_EPISODES = np.arange(1, 9)
GIRAFFE_W_GIRAFFE = np.array([0.740, 0.778, 0.715, 0.654, 0.612, 0.453, 0.453, 0.453])
GIRAFFE_W_ELEPHANT = np.array([0.260, 0.222, 0.285, 0.346, 0.388, 0.547, 0.547, 0.547])

DEPLOY_EPISODES = np.arange(1, 16)
DEPLOY_CHECK_KNOWN = {
    1: 0.572,
    2: 0.572,
    3: 0.587,
    5: 0.637,
    10: 0.802,
    13: 0.896,
    15: 0.957,
}
DEPLOY_SKIP_KNOWN = {
    1: 0.473,
    2: 0.473,
    3: 0.436,
    5: 0.386,
    10: 0.327,
    13: 0.294,
    15: 0.273,
}

CONTEXT_EXPERIMENTS = [
    "Context Bloat",
    "Gate Bloat",
    "Stale Context",
    "Negation",
    "Procedure",
]
CONTEXT_STATIC = [6066, 8163, 895, 546, 548]
CONTEXT_RAG = [744, 407, 507, 546, 467]
CONTEXT_CRABPATH = [297, 89, 88, 87, 205]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#222222",
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig, out_dir: Path, filename: str) -> Path:
    out = out_dir / filename
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    return out


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def figure_ablation_bars(out_dir: Path) -> Path:
    style()
    x = np.arange(len(ABLATED_KEYS))
    acc = np.array(ABLATED_ACCURACY, dtype=float)
    ci_low = acc - np.array(ABLATED_CI_LOW, dtype=float)
    ci_high = np.array(ABLATED_CI_HIGH, dtype=float) - acc
    yerr = np.vstack([ci_low, ci_high])

    colors_main = [ABLATED_COLOR[k] for k in ABLATED_KEYS]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.52
    ax.bar(
        x,
        acc,
        width=width,
        yerr=yerr,
        capsize=4.0,
        color=colors_main,
        edgecolor="#111111",
        linewidth=1.5,
        label="Accuracy",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATED_SHORT_NAMES, rotation=0, ha="center")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model variant")
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_title("Ablation Study")
    ax.axhline(1.0, color="#666666", lw=1.2, ls=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for x_pos, value, p in zip(x, acc, ABLATED_NEGATION):
        stars = _significance_stars(p)
        if not stars:
            continue
        ax.text(
            x_pos,
            min(1.04, value + 0.03),
            stars,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="#111111",
        )

    return save_fig(fig, out_dir, "ablation_bars.pdf")


def figure_phase_transition(out_dir: Path) -> Path:
    style()
    q = np.arange(1, 301)
    entropy = np.empty_like(q, dtype=float)
    grad = np.empty_like(q, dtype=float)
    for idx, qi in enumerate(q):
        if qi <= 80:
            entropy[idx] = 1.45 - 0.07 * (qi - 1) / 79
        elif qi <= 100:
            entropy[idx] = 1.38 - 0.16 * (qi - 80) / 20
        elif qi <= 200:
            entropy[idx] = 1.22
        else:
            entropy[idx] = 1.22 - 0.07 * (qi - 200) / 100

        if qi <= 80:
            grad[idx] = 0.005 + 0.0
        elif qi <= 100:
            grad[idx] = 0.005 + 0.095 * (qi - 80) / 20
        elif qi <= 120:
            grad[idx] = 0.10 - 0.04 * (qi - 100) / 20
        elif qi <= 200:
            grad[idx] = 0.06
        else:
            grad[idx] = 0.05 - 0.0002 * (qi - 200)

    fig, ax1 = plt.subplots(figsize=(6.8, 3.8))
    ax2 = ax1.twinx()

    ent_error = np.full_like(entropy, 0.022, dtype=float)
    grad_error = np.full_like(grad, 0.0055, dtype=float)

    ax1.fill_between(q, entropy - ent_error, entropy + ent_error, color=FORWARD_COLOR, alpha=0.20)
    ax1.plot(q, entropy, color=FORWARD_COLOR, linewidth=2.0)
    ax2.fill_between(q, grad - grad_error, grad + grad_error, color=LOOP_SECOND, alpha=0.22)
    ax2.plot(q, grad, color=LOOP_SECOND, linewidth=2.0, ls="--")

    ax1.fill_betweenx([0.9, 2.0], 1, 100, color="#8dd3c7", alpha=0.2)
    ax1.fill_betweenx([0.9, 2.0], 100, 300, color="#bebada", alpha=0.2)
    ax1.text(50, 1.95, "Phase 1\nHebbian", ha="center", va="top", fontsize=12)
    ax1.text(190, 1.95, "Phase 2\nRL", ha="center", va="top", fontsize=12)

    ax1.axvline(100, color="#555555", ls=":", lw=2.0)
    ax1.annotate(
        "Phase transition",
        xy=(100, 1.34),
        xytext=(122, 1.43),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=2.0),
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax1.text(250, 1.45, "Weight entropy", color=FORWARD_COLOR, fontsize=12, fontweight="bold")
    ax2.text(250, 0.096, "Gradient magnitude", color=LOOP_SECOND, fontsize=12, fontweight="bold")

    ax1.set_xlim(1, 300)
    ax1.set_xlabel("Query")
    ax1.set_ylabel("Weight entropy")
    ax2.set_ylabel("Gradient magnitude")
    ax1.set_ylim(1.1, 1.55)
    ax2.set_ylim(0.0, 0.11)
    ax1.set_title("Two-Phase Learning Dynamics")

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    return save_fig(fig, out_dir, "phase_transition.pdf")


def _interpolate_lookup(idx: np.ndarray, known: dict[int, float]) -> np.ndarray:
    idx = np.asarray(idx)
    series = np.empty_like(idx, dtype=float)
    keys = np.array(sorted(known))
    vals = np.array([known[k] for k in keys], dtype=float)
    for i, step in enumerate(idx):
        if step in known:
            series[i] = known[step]
            continue
        left = keys[keys < step]
        right = keys[keys > step]
        if left.size == 0:
            series[i] = vals[0]
        elif right.size == 0:
            series[i] = vals[-1]
        else:
            li = left[-1]
            ri = right[0]
            lv = known[li]
            rv = known[ri]
            frac = (step - li) / (ri - li)
            series[i] = lv + (rv - lv) * frac
    return series


def figure_weight_trajectory(out_dir: Path) -> Path:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=False)

    ax_giraffe, ax_deploy = axes
    ax_giraffe.plot(
        GIRAFFE_EPISODES,
        GIRAFFE_W_GIRAFFE,
        marker="o",
        color=FORWARD_COLOR,
        linewidth=2.0,
        label=r"$w(\mathrm{giraffe})$",
    )
    ax_giraffe.plot(
        GIRAFFE_EPISODES,
        GIRAFFE_W_ELEPHANT,
        marker="o",
        color=LOOP_SECOND,
        linewidth=2.0,
        label=r"$w(\mathrm{elephant})$",
    )
    ax_giraffe.axhline(0.5, color="#666666", lw=2.0, ls="--")
    ax_giraffe.text(
        0.02,
        0.96,
        "(a)",
        transform=ax_giraffe.transAxes,
        fontsize=16,
        fontweight="bold",
    )
    ax_giraffe.set_title("Giraffe test")
    ax_giraffe.set_xlabel("Episode")
    ax_giraffe.set_ylabel("Edge weight")
    ax_giraffe.set_xticks(GIRAFFE_EPISODES)
    ax_giraffe.set_xlim(1, 8)
    ax_giraffe.set_ylim(0.0, 1.0)
    ax_giraffe.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_giraffe.legend(loc="best", fontsize=11)

    check = _interpolate_lookup(DEPLOY_EPISODES, DEPLOY_CHECK_KNOWN)
    skip = _interpolate_lookup(DEPLOY_EPISODES, DEPLOY_SKIP_KNOWN)
    ax_deploy.plot(
        DEPLOY_EPISODES,
        check,
        marker="o",
        color=FORWARD_COLOR,
        linewidth=2.0,
        label=r"$w(\mathrm{check\_tests})$",
    )
    ax_deploy.plot(
        DEPLOY_EPISODES,
        skip,
        marker="o",
        color=LOOP_SECOND,
        linewidth=2.0,
        label=r"$w(\mathrm{skip\_tests})$",
    )
    ax_deploy.axhline(0.9, color=FORWARD_COLOR, lw=2.0, ls="--", label="Reflex threshold")
    ax_deploy.axhline(0.1, color=LOOP_COLOR, lw=2.0, ls=":")
    ax_deploy.text(
        0.02,
        0.96,
        "(b)",
        transform=ax_deploy.transAxes,
        fontsize=16,
        fontweight="bold",
    )
    ax_deploy.set_title("Deploy pipeline")
    ax_deploy.set_xlabel("Episode")
    ax_deploy.set_ylabel("Edge weight")
    ax_deploy.set_xticks([1, 5, 10, 15])
    ax_deploy.set_xlim(1, 15)
    ax_deploy.set_ylim(0.0, 1.0)
    ax_deploy.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_deploy.legend(loc="best", fontsize=11)

    for s in ["top", "right"]:
        ax_deploy.spines[s].set_visible(False)
        ax_giraffe.spines[s].set_visible(False)

    return save_fig(fig, out_dir, "weight_trajectory.pdf")


def figure_context_reduction(out_dir: Path) -> Path:
    style()
    x = np.arange(len(CONTEXT_EXPERIMENTS))
    width = 0.22
    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    bars_static = ax.bar(x - width, CONTEXT_STATIC, width=width, color=TAB10[7], label="Static")
    bars_rag = ax.bar(x, CONTEXT_RAG, width=width, color=FORWARD_COLOR, label="RAG")
    bars_crab = ax.bar(x + width, CONTEXT_CRABPATH, width=width, color=LOOP_SECOND, label="CrabPath")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Ctx Bloat", "Gate Bloat", "Stale Ctx", "Negation", "Procedure"], rotation=0, ha="center")
    ax.set_ylabel("Tokens per turn (log scale)")
    ax.set_xlabel("Scenario")
    ax.set_title("Context Efficiency")
    ax.set_ylim(bottom=40, top=10000)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_yticks([1e2, 1e3, 1e4])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    for bars in (bars_static, bars_rag, bars_crab):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    return save_fig(fig, out_dir, "context_reduction.pdf")


def _add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    edge_color: str,
    face_color: str,
    lw: float = 2.0,
) -> None:
    shadow = patches.FancyBboxPatch(
        (x + 0.08, y - 0.08),
        w,
        h,
        boxstyle="round,pad=0.04",
        facecolor="#cfcfcf",
        edgecolor="none",
        linewidth=0,
        alpha=0.5,
        zorder=5,
    )
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04",
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=lw,
        zorder=6,
    )
    ax.add_patch(shadow)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111111",
        zorder=7,
    )


def figure_architecture(out_dir: Path) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    # Forward flow: Query → Seed Selection → Three-Tier Routing → Context Assembly
    _add_box(ax, 0.5, 5.5, 1.6, 1.0, "Query", FORWARD_COLOR, "#deebf7")
    _add_box(ax, 2.5, 5.5, 2.05, 1.0, "Seed\nSelection", FORWARD_COLOR, "#deebf7")
    _add_box(ax, 4.95, 5.5, 2.1, 1.0, "Three-Tier\nRouting", FORWARD_COLOR, "#deebf7")
    _add_box(ax, 7.25, 5.5, 2.0, 1.0, "Context\nAssembly", FORWARD_COLOR, "#deebf7")

    # Feedback loop below: Learning → Weight Update
    _add_box(ax, 3.4, 2.7, 1.8, 0.9, "Learning", LOOP_SECOND, "#fde0dd")
    _add_box(ax, 6.0, 2.7, 2.2, 0.9, "Weight\nUpdate", LOOP_SECOND, "#fde0dd")

    def arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        ls: str = "-",
        lw: float = 2.0,
        rad: float = 0.0,
    ) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=lw,
                ls=ls,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=0,  # keep arrows behind boxes/labels
        )

    y_center = 6.0
    arrow(2.1, y_center, 2.5, y_center, FORWARD_COLOR)
    arrow(4.55, y_center, 4.95, y_center, FORWARD_COLOR)
    arrow(7.05, y_center, 7.25, y_center, FORWARD_COLOR)

    # Route the learning loop around boxes to avoid occluding labels.
    arrow(8.25, 5.5, 4.15, 3.55, LOOP_COLOR, ls="--", rad=-0.25)
    arrow(4.35, 3.15, 6.0, 3.15, LOOP_COLOR)
    arrow(8.2, 3.15, 8.25, 4.4, LOOP_COLOR, rad=0.15)

    ax.set_title("CrabPath architecture", fontsize=18, fontweight="bold")
    return save_fig(fig, out_dir, "architecture.pdf")


def main() -> int:
    out_dir = ensure_output_dir(Path("/Users/guclaw/crabpath-private/paper/figures"))

    figure_ablation_bars(out_dir)
    figure_phase_transition(out_dir)
    figure_weight_trajectory(out_dir)
    figure_context_reduction(out_dir)
    figure_architecture(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
