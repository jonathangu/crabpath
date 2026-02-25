"""Calibrate warm-start defaults for CrabPath via deterministic grid search."""

from __future__ import annotations

import re
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.autotune import HEALTH_TARGETS, GraphHealth, measure_health, self_tune
from crabpath.decay import DecayConfig, apply_decay
from crabpath.graph import Graph
from crabpath.lifecycle_sim import SimConfig, make_mock_llm_all, make_mock_router, workspace_scenario
from crabpath.mitosis import MitosisConfig, MitosisState, bootstrap_workspace, mitosis_maintenance
from crabpath.synaptogenesis import SynaptogenesisConfig, SynaptogenesisState, decay_proto_edges, record_cofiring, record_skips


SIBLING_WEIGHTS: list[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
PROMOTION_THRESHOLDS: list[int] = [2, 3, 4]
DECAY_HALF_LIVES: list[int] = [40, 60, 80, 100, 120]
HEBBIAN_INCREMENTS: list[float] = [0.04, 0.06, 0.08, 0.10]


def _query_stats_from_run(
    *,
    fired_counts: list[int],
    context_chars: list[int],
    promotions: int,
    proto_created: int,
    reconverged_families: int,
    queries_seen: int,
) -> dict[str, object]:
    return {
        "fired_counts": list(fired_counts),
        "context_chars": list(context_chars),
        "chars": list(context_chars),
        "avg_nodes_fired_per_query": (sum(fired_counts) / queries_seen) if queries_seen else 0.0,
        "promotions": promotions,
        "proto_created": proto_created,
        "reconverged_families": reconverged_families,
        "queries": queries_seen,
    }


def _run_workspace_simulation(
    *,
    sibling_weight: float,
    promotion_threshold: int,
    decay_half_life: int,
    hebbian_increment: float,
    enable_self_tune: bool,
) -> dict[str, object]:
    """Run the deterministic workspace scenario with explicit core knobs."""

    workspace_files, queries = workspace_scenario()
    sim_config = SimConfig(decay_half_life=decay_half_life)

    graph = Graph()
    mitosis_state = MitosisState()
    synapse_state = SynaptogenesisState()
    mitosis_config = MitosisConfig(sibling_weight=sibling_weight)
    synapse_config = SynaptogenesisConfig(
        promotion_threshold=promotion_threshold,
        hebbian_increment=hebbian_increment,
    )
    decay_config = DecayConfig(half_life_turns=decay_half_life)

    llm = make_mock_llm_all()
    router = make_mock_router()
    bootstrap_workspace(graph, workspace_files, llm, mitosis_state, mitosis_config)

    fired_counts: list[int] = []
    context_chars: list[int] = []
    promotions = 0
    proto_created = 0

    last_adjusted: dict[str, int] = {}

    for qi, query in enumerate(queries, 1):
        candidates = []
        query_words = set(query.text.lower().split())

        for node in graph.nodes():
            node_words = set(node.content.lower().split())
            overlap = len(query_words & node_words)
            score = min(overlap / max(len(query_words), 1), 1.0)
            if score > 0.1:
                candidates.append((node.id, score, node.summary or node.content[:80]))

        candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = candidates[:10]

        selected = router(query.text, candidates)
        cofire_result = record_cofiring(graph, selected, synapse_state, synapse_config)
        promotions += cofire_result["promoted"]
        proto_created += cofire_result["proto_created"]

        skips = 0
        if selected:
            candidate_ids = [c[0] for c in candidates]
            skips = record_skips(graph, selected[0], candidate_ids, selected, synapse_config)

        fired_count = len(selected)
        fired_counts.append(fired_count)
        selected_chars = sum(
            len(graph.get_node(node_id).content if graph.get_node(node_id) else "")
            for node_id in selected
        )
        context_chars.append(selected_chars)

        if qi % sim_config.decay_interval == 0:
            apply_decay(graph, turns_elapsed=sim_config.decay_interval, config=decay_config)
            decay_proto_edges(synapse_state, synapse_config)

        if qi % sim_config.maintenance_interval == 0:
            mitosis_maintenance(graph, llm, mitosis_state, mitosis_config)

            if enable_self_tune:
                self_tune(
                    graph=graph,
                    state=mitosis_state,
                    query_stats=_query_stats_from_run(
                        fired_counts=fired_counts,
                        context_chars=context_chars,
                        promotions=promotions,
                        proto_created=proto_created,
                        reconverged_families=0,
                        queries_seen=qi,
                    ),
                    syn_config=synapse_config,
                    decay_config=decay_config,
                    mitosis_config=mitosis_config,
                    cycle_number=qi,
                    last_adjusted=last_adjusted,
                )

        del skips

    query_stats = _query_stats_from_run(
        fired_counts=fired_counts,
        context_chars=context_chars,
        promotions=promotions,
        proto_created=proto_created,
        reconverged_families=0,
        queries_seen=len(queries),
    )

    health = measure_health(graph, mitosis_state, query_stats)

    return {
        "health": health,
        "graph": graph,
        "state": mitosis_state,
        "synapse_config": synapse_config,
        "decay_config": decay_config,
        "mitosis_config": mitosis_config,
        "query_stats": query_stats,
    }


def _metric_in_range(value: float, target: tuple[float | None, float | None]) -> bool:
    min_v, max_v = target
    if min_v is not None and value < min_v:
        return False
    if max_v is not None and value > max_v:
        return False
    return True


def _metric_distance(value: float, target: tuple[float | None, float | None]) -> float:
    min_v, max_v = target

    if min_v is not None and max_v is not None:
        if max_v == min_v:
            if value == min_v:
                return 0.0
            return abs(value - min_v)
        center = (min_v + max_v) / 2.0
        width = max_v - min_v
        return abs(value - center) / width if width else 0.0

    if min_v is not None:
        if value >= min_v:
            return 0.0
        scale = abs(min_v) if abs(min_v) > 0 else 1.0
        return abs(min_v - value) / scale

    if max_v is not None:
        if value <= max_v:
            return 0.0
        scale = abs(max_v) if abs(max_v) > 0 else 1.0
        return abs(value - max_v) / scale

    return 0.0


def _score_health(health: GraphHealth) -> dict[str, object]:
    in_range = 0
    distance = 0.0
    per_metric: dict[str, dict[str, object]] = {}

    for key, target in HEALTH_TARGETS.items():
        value = float(getattr(health, key))
        in_range_flag = _metric_in_range(value, target)
        if in_range_flag:
            in_range += 1
        metric_distance = _metric_distance(value, target)
        distance += metric_distance
        per_metric[key] = {
            "value": value,
            "target": target,
            "in_range": in_range_flag,
            "distance": metric_distance,
        }

    return {
        "in_range_count": in_range,
        "distance": distance,
        "per_metric": per_metric,
    }


def _format_health_line(health: GraphHealth) -> str:
    parts = []
    for key in HEALTH_TARGETS:
        value = getattr(health, key)
        parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts)


def _update_medium_defaults(winner: dict[str, float | int]) -> None:
    path = ROOT / "crabpath/autotune.py"
    lines = path.read_text(encoding="utf-8").splitlines()

    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == '"medium": {':
            start_idx = idx
            continue
        if start_idx is not None and line.strip() == "},":
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        raise RuntimeError("Could not locate DEFAULTS['medium'] block in crabpath/autotune.py")

    for idx in range(start_idx + 1, end_idx):
        stripped = lines[idx].strip()
        if not (stripped.startswith('"') and ":" in stripped):
            continue
        key = stripped.split(":", 1)[0].strip().strip('"')
        if key in winner:
            lines[idx] = f'        "{key}": {winner[key]!r},'

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_grid_search() -> list[dict[str, object]]:
    combos = product(
        SIBLING_WEIGHTS,
        PROMOTION_THRESHOLDS,
        DECAY_HALF_LIVES,
        HEBBIAN_INCREMENTS,
    )

    results: list[dict[str, object]] = []
    for sibling_weight, promotion_threshold, decay_half_life, hebbian_increment in combos:
        outcome = _run_workspace_simulation(
            sibling_weight=sibling_weight,
            promotion_threshold=promotion_threshold,
            decay_half_life=decay_half_life,
            hebbian_increment=hebbian_increment,
            enable_self_tune=False,
        )
        health = outcome["health"]
        if not isinstance(health, GraphHealth):
            raise TypeError("Expected GraphHealth from simulation result")
        score = _score_health(health)
        results.append(
            {
                "config": {
                    "sibling_weight": sibling_weight,
                    "promotion_threshold": promotion_threshold,
                    "decay_half_life": decay_half_life,
                    "hebbian_increment": hebbian_increment,
                },
                "health": health,
                "score": score,
            }
        )

    results.sort(
        key=lambda item: (
            -item["score"]["in_range_count"],
            item["score"]["distance"],
        ),
    )
    return results


def main() -> None:
    print("Running warm-start grid search: 360 combinations...")
    results = _run_grid_search()

    print("Top 10 combinations (by in-range count then distance):")
    top_10 = results[:10]
    for rank, item in enumerate(top_10, 1):
        score = item["score"]
        cfg = item["config"]
        health = item["health"]
        print(
            f"{rank:>2}. cfg={cfg} | "
            f"in_range={score['in_range_count']}/8 | "
            f"distance={float(score['distance']):.4f}"
        )
        print(f"    health: {_format_health_line(health)}")

    best = top_10[0]
    best_config = best["config"]
    best_health = best["health"]
    best_score = best["score"]

    print("\nBest config without self_tune:")
    print(f"  {best_config}")
    print(f"  in_range={best_score['in_range_count']}/8")
    print(f"  distance={float(best_score['distance']):.4f}")
    print(f"  health: {_format_health_line(best_health)}")

    without_tune = best_health
    with_tune = _run_workspace_simulation(
        sibling_weight=float(best_config["sibling_weight"]),
        promotion_threshold=int(best_config["promotion_threshold"]),
        decay_half_life=int(best_config["decay_half_life"]),
        hebbian_increment=float(best_config["hebbian_increment"]),
        enable_self_tune=True,
    )["health"]
    without_tune_result = _score_health(without_tune)
    with_tune_result = _score_health(with_tune)  # type: ignore[arg-type]

    print("\nSelf_tune@maintenance comparison:")
    print(f"  without self_tune -> in_range={without_tune_result['in_range_count']}/8, distance={float(without_tune_result['distance']):.4f}")
    print(f"  with self_tune    -> in_range={with_tune_result['in_range_count']}/8, distance={float(with_tune_result['distance']):.4f}")
    delta_in_range = int(with_tune_result["in_range_count"]) - int(without_tune_result["in_range_count"])
    delta_distance = float(with_tune_result["distance"]) - float(without_tune_result["distance"])
    print(f"  delta -> in_range={delta_in_range:+d}, distance={delta_distance:+.4f}")

    _update_medium_defaults({
        "sibling_weight": best_config["sibling_weight"],
        "promotion_threshold": best_config["promotion_threshold"],
        "decay_half_life": best_config["decay_half_life"],
        "hebbian_increment": best_config["hebbian_increment"],
    })
    print("\nUpdated crabpath/autotune.py DEFAULTS['medium'] with winner.")


if __name__ == "__main__":
    main()
