"""pg_vs_heuristic_v2.py — Realistic PG vs heuristic comparison.

Key insight: measure EDGE WEIGHTS at the critical branch point, not just
softmax probabilities. PG redistributes mass (correct edge up, others down);
heuristic only inflates the correct edge.

The sim forces the traversal through a known path and applies mixed feedback.
This is closer to how OpenClawBrain actually works in production: the agent
queries, gets context, responds, then learns from the outcome on the fired
trajectory.

Run:
  cd ~/openclawbrain && python3 sims/pg_vs_heuristic_v2.py
"""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.learn import LearningConfig, apply_outcome, apply_outcome_pg


RESULT_PATH = Path(__file__).with_name("pg_vs_heuristic_v2_results.json")


# ── Graph construction ──────────────────────────────────────────

def build_graph(rng: random.Random) -> tuple[Graph, dict]:
    """Build a focused graph for testing branch disambiguation.

    Structure (50 nodes):
      hub ─┬─ gate_A ─ a0 ─ a1 ─ a2 ─ a3
           ├─ gate_B ─ b0 ─ b1 ─ b2 ─ b3
           ├─ gate_C ─ c0 ─ c1 ─ c2 ─ c3
           └─ distractor edges (4-6 from hub, 2-3 per chain node)

    All correct-path edges start at 0.35.
    Distractor edges start at 0.15-0.25.
    """
    g = Graph()
    g.add_node(Node("hub", "Central routing hub"))

    info = {}
    for name in ["A", "B", "C"]:
        gate = f"gate_{name}"
        chain = [f"{name}_{i}" for i in range(4)]
        distractors = [f"{name}_d{i}" for i in range(5)]

        g.add_node(Node(gate, f"Gate for {name}"))
        for n in chain + distractors:
            g.add_node(Node(n, f"Node {n}"))

        # Correct path
        g.add_edge(Edge("hub", gate, weight=0.35, kind="sibling"))
        g.add_edge(Edge(gate, chain[0], weight=0.35, kind="sibling"))
        for i in range(len(chain) - 1):
            g.add_edge(Edge(chain[i], chain[i + 1], weight=0.35, kind="sibling"))

        # Distractor edges from each chain node
        for src in [gate] + chain:
            for d in rng.sample(distractors, k=2):
                g.add_edge(Edge(src, d, weight=rng.uniform(0.15, 0.25), kind="sibling"))

        # Some distractor cross-links
        for d in distractors:
            tgt = rng.choice(distractors + chain)
            if tgt != d:
                g.add_edge(Edge(d, tgt, weight=rng.uniform(0.08, 0.18), kind="sibling"))

        info[name] = {
            "gate": gate,
            "chain": chain,
            "distractors": distractors,
            "path": ["hub", gate] + chain,
        }

    # Hub distractor edges (fewer = cleaner branch signal)
    all_distractors = [n for cl in info.values() for n in cl["distractors"]]
    for d in rng.sample(all_distractors, k=4):
        g.add_edge(Edge("hub", d, weight=rng.uniform(0.10, 0.20), kind="sibling"))

    return g, info


# ── Simulation ──────────────────────────────────────────────────

@dataclass
class StepMetrics:
    t: int
    phase: str
    query: str
    outcome: float
    # Edge weights at hub for all 3 gates
    w_gate_A: float
    w_gate_B: float
    w_gate_C: float
    # Weight of correct gate for this query
    w_correct_gate: float
    # Sum of weights of WRONG gates
    w_wrong_gates: float
    # Total graph weight
    total_weight: float
    # Separation = correct_gate_weight - max(wrong_gate_weights)
    separation: float


def edge_weight(graph: Graph, src: str, tgt: str) -> float:
    e = graph._edges.get(src, {}).get(tgt)
    return float(e.weight) if e else 0.0


def run_condition(
    graph: Graph,
    info: dict,
    method: str,
    *,
    steps: int = 400,
    drift_at: int = 200,
    seed: int = 42,
    lr: float = 0.08,
    discount: float = 0.95,
    tau: float = 1.0,
) -> dict:
    """Run simulation. Instead of traversing, we directly apply learning on
    known-correct and known-incorrect paths. This isolates the learning rule
    behavior from traversal randomness.

    Schedule:
    - Pre-drift (0..drift_at): correct path for each query is its own cluster
    - Post-drift (drift_at..steps): correct path rotates (A→B, B→C, C→A)

    Feedback mix: 65% positive on correct path, 20% negative on wrong path,
    15% partial (first 2 edges positive, rest negative on a wrong path)
    """
    rng = random.Random(seed)
    cfg = LearningConfig(learning_rate=lr, discount=discount)

    clusters = ["A", "B", "C"]
    # Pre-drift: query X → cluster X correct
    # Post-drift: query X → cluster (X+1) correct
    drift_map = {"A": "B", "B": "C", "C": "A"}

    records: list[StepMetrics] = []

    for t in range(steps):
        phase = "pre" if t < drift_at else "post"
        query_cluster = rng.choice(clusters)

        if phase == "pre":
            correct_cluster = query_cluster
        else:
            correct_cluster = drift_map[query_cluster]

        correct_path = info[correct_cluster]["path"]

        # Choose wrong clusters
        wrong_clusters = [c for c in clusters if c != correct_cluster]

        # Feedback
        r = rng.random()
        if r < 0.65:
            # Positive: learn on the correct path
            path = correct_path
            outcome = 1.0
            per_node = None
        elif r < 0.85:
            # Negative: agent followed a wrong path
            wrong_cl = rng.choice(wrong_clusters)
            path = info[wrong_cl]["path"]
            outcome = -1.0
            per_node = None
        else:
            # Partial: started right (hub + correct gate) then went wrong
            wrong_cl = rng.choice(wrong_clusters)
            # First 2 steps of correct path, then wrong chain
            path = correct_path[:2] + info[wrong_cl]["chain"]
            outcome = -1.0
            per_node = {}
            for idx in range(len(path) - 1):
                per_node[path[idx]] = 0.6 if idx < 2 else -1.0

        # Apply update
        if method == "heuristic":
            apply_outcome(graph, path, outcome, config=cfg,
                          per_node_outcomes=per_node)
        else:
            apply_outcome_pg(graph, path, outcome, config=cfg,
                             per_node_outcomes=per_node,
                             baseline=0.0, temperature=tau)

        # Record metrics
        w_A = edge_weight(graph, "hub", "gate_A")
        w_B = edge_weight(graph, "hub", "gate_B")
        w_C = edge_weight(graph, "hub", "gate_C")

        gate_weights = {"A": w_A, "B": w_B, "C": w_C}
        w_correct = gate_weights[correct_cluster]
        w_wrong = [gate_weights[c] for c in clusters if c != correct_cluster]

        tw = sum(e.weight for d in graph._edges.values() for e in d.values())

        records.append(StepMetrics(
            t=t, phase=phase, query=query_cluster, outcome=outcome,
            w_gate_A=round(w_A, 4), w_gate_B=round(w_B, 4), w_gate_C=round(w_C, 4),
            w_correct_gate=round(w_correct, 4),
            w_wrong_gates=round(sum(w_wrong), 4),
            total_weight=round(tw, 2),
            separation=round(w_correct - max(w_wrong), 4),
        ))

    # ── Aggregate ──

    def avg_window(records, start, end, key):
        vals = [getattr(r, key) for r in records[start:end]]
        return round(statistics.mean(vals), 4) if vals else 0.0

    # Separation = how much higher correct gate is than best wrong gate
    sep_pre_late = avg_window(records, drift_at - 40, drift_at, "separation")
    sep_post_early = avg_window(records, drift_at, drift_at + 40, "separation")
    sep_post_late = avg_window(records, steps - 40, steps, "separation")

    # Recovery: first step after drift where separation > 0
    recovery = None
    for r in records[drift_at:]:
        if r.separation > 0:
            recovery = r.t - drift_at
            break

    # Weight stats
    tw_start = records[0].total_weight
    tw_end = records[-1].total_weight

    # Correct gate weight pre vs post
    cw_pre_late = avg_window(records, drift_at - 40, drift_at, "w_correct_gate")
    cw_post_late = avg_window(records, steps - 40, steps, "w_correct_gate")

    # Wrong gate suppression: average wrong gate weight
    ww_pre_late = avg_window(records, drift_at - 40, drift_at, "w_wrong_gates") / 2
    ww_post_late = avg_window(records, steps - 40, steps, "w_wrong_gates") / 2

    return {
        "method": method,
        "separation_pre_late": sep_pre_late,
        "separation_post_early": sep_post_early,
        "separation_post_late": sep_post_late,
        "recovery_steps": recovery,
        "correct_gate_pre": cw_pre_late,
        "correct_gate_post": cw_post_late,
        "wrong_gate_avg_pre": round(ww_pre_late, 4),
        "wrong_gate_avg_post": round(ww_post_late, 4),
        "total_weight_start": tw_start,
        "total_weight_end": tw_end,
        "total_weight_delta": round(tw_end - tw_start, 2),
        "timeseries": {
            "gate_A": [r.w_gate_A for r in records],
            "gate_B": [r.w_gate_B for r in records],
            "gate_C": [r.w_gate_C for r in records],
            "separation": [r.separation for r in records],
            "total_weight": [r.total_weight for r in records],
        },
    }


# ── Main ────────────────────────────────────────────────────────

def main() -> None:
    all_results = []

    for seed in range(10):
        rng = random.Random(seed)
        g0, info = build_graph(rng)

        g_h = copy.deepcopy(g0)
        g_pg = copy.deepcopy(g0)

        res_h = run_condition(g_h, info, "heuristic", seed=seed + 100,
                              steps=400, drift_at=200)
        res_pg = run_condition(g_pg, info, "pg", seed=seed + 100,
                               steps=400, drift_at=200)

        all_results.append({"seed": seed, "heuristic": res_h, "pg": res_pg})

    # Aggregate
    def avg_metric(results, method, key):
        vals = [r[method][key] for r in results if r[method].get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    metrics = [
        "separation_pre_late", "separation_post_early", "separation_post_late",
        "recovery_steps", "correct_gate_pre", "correct_gate_post",
        "wrong_gate_avg_pre", "wrong_gate_avg_post", "total_weight_delta",
    ]

    summary = {
        "seeds": len(all_results),
        "heuristic": {k: avg_metric(all_results, "heuristic", k) for k in metrics},
        "pg": {k: avg_metric(all_results, "pg", k) for k in metrics},
    }

    output = {"summary": summary, "runs": all_results}
    RESULT_PATH.write_text(json.dumps(output, indent=2))

    # Print
    print("\n=== PG vs Heuristic Simulation (v2) ===")
    print(f"  10 seeds, 400 steps each, drift at 200\n")
    print(f"{'Metric':<30} {'Heuristic':>12} {'PG':>12} {'Winner':>10}")
    print("-" * 66)
    for key in metrics:
        h = summary["heuristic"].get(key)
        p = summary["pg"].get(key)
        h_s = f"{h}" if h is not None else "N/A"
        p_s = f"{p}" if p is not None else "N/A"

        winner = ""
        if h is not None and p is not None:
            if key in ("wrong_gate_avg_pre", "wrong_gate_avg_post", "total_weight_delta", "recovery_steps"):
                winner = "PG" if p < h else ("Heur" if h < p else "Tie")
            else:
                winner = "PG" if p > h else ("Heur" if h > p else "Tie")

        print(f"{key:<30} {h_s:>12} {p_s:>12} {winner:>10}")

    print(f"\nResults written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
