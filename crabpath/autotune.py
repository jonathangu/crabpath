"""Adaptive configuration helpers for warm-start and runtime tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph import Graph
from .decay import DecayConfig
from .mitosis import MitosisConfig
from .synaptogenesis import SynaptogenesisConfig
from .mitosis import MitosisState
from .synaptogenesis import edge_tier_stats


# ---------------------------------------------------------------------------
# Tuned defaults by workspace regime.
# ---------------------------------------------------------------------------

DEFAULTS = {
    "small": {
        "sibling_weight": 0.60,
        "promotion_threshold": 2,
        "decay_half_life": 60,
        "decay_interval": 6,
        "min_content_chars": 160,
        "hebbian_increment": 0.06,
        "skip_factor": 0.92,
        "max_outgoing": 16,
    },
    "medium": {
        "sibling_weight": 0.65,
        "promotion_threshold": 3,
        "decay_half_life": 80,
        "decay_interval": 10,
        "min_content_chars": 200,
        "hebbian_increment": 0.05,
        "skip_factor": 0.90,
        "max_outgoing": 20,
    },
    "large": {
        "sibling_weight": 0.70,
        "promotion_threshold": 4,
        "decay_half_life": 120,
        "decay_interval": 12,
        "min_content_chars": 260,
        "hebbian_increment": 0.04,
        "skip_factor": 0.88,
        "max_outgoing": 28,
    },
}


# ---------------------------------------------------------------------------
# Outcome-driven health targets and measurement schema.
# ---------------------------------------------------------------------------

MetricRange = tuple[float | None, float | None]


@dataclass(frozen=True)
class GraphHealth:
    avg_nodes_fired_per_query: float
    cross_file_edge_pct: float
    dormant_pct: float
    reflex_pct: float
    context_compression: float
    proto_promotion_rate: float
    reconvergence_rate: float
    orphan_nodes: int


@dataclass(frozen=True)
class Adjustment:
    metric: str
    current: float | int
    target_range: MetricRange
    suggested_change: dict[str, Any]
    reason: str


HEALTH_TARGETS: dict[str, MetricRange] = {
    "avg_nodes_fired_per_query": (3.0, 8.0),
    "cross_file_edge_pct": (5.0, 20.0),
    "dormant_pct": (60.0, 90.0),
    "reflex_pct": (1.0, 5.0),
    "context_compression": (None, 20.0),
    "proto_promotion_rate": (5.0, 15.0),
    "reconvergence_rate": (None, 10.0),
    "orphan_nodes": (0.0, 0.0),
}

ADJUSTMENT_COOLDOWN_CYCLES = 2


# ---------------------------------------------------------------------------
# Warm-start defaults
# ---------------------------------------------------------------------------


def _workspace_size(workspace_files: dict[str, str]) -> str:
    file_count = len(workspace_files)
    total_chars = sum(len(v or "") for v in workspace_files.values())

    if file_count < 10 and total_chars < 50_000:
        return "small"
    if file_count >= 50 and total_chars >= 500_000:
        return "large"
    return "medium"


def suggest_config(workspace_files: dict[str, str]) -> dict[str, int | float]:
    """Return tuned configuration defaults for the given workspace size."""

    size = _workspace_size(workspace_files)
    return dict(DEFAULTS[size])


# ---------------------------------------------------------------------------
# Runtime tuning
# ---------------------------------------------------------------------------


def _extract_proto_edges(graph: Graph) -> int:
    for attr in ("proto_edges", "_proto_edges"):
        value = getattr(graph, attr, None)
        if isinstance(value, dict):
            return len(value)

    value = getattr(graph, "synapse_state", None)
    if value is not None and isinstance(getattr(value, "proto_edges", None), dict):
        return len(value.proto_edges)

    return 0


def _node_file_id(node_id: str) -> str:
    return str(node_id).split("::", 1)[0]


def _cross_file_edges(graph: Graph) -> int:
    if graph.node_count <= 1:
        return 0

    cross = 0
    for edge in graph.edges():
        if _node_file_id(edge.source) != _node_file_id(edge.target):
            cross += 1
    return cross


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metric_from_query_stats(
    query_stats: dict[str, Any],
    keys: list[str],
    default: float | int | None = None,
) -> float | int | list[float] | list[int] | None:
    for key in keys:
        if key not in query_stats:
            continue
        return query_stats[key]
    return default


def _avg_query_value(query_stats: dict[str, Any], keys: list[str], *, default: float = 0.0) -> float:
    value = _metric_from_query_stats(query_stats, keys, default=None)
    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        values: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple, set)):
                values.append(len(item))
            else:
                num = _coerce_float(item, default=None)
                if num is not None:
                    values.append(num)
        if not values:
            return default
        return sum(values) / len(values)

    if isinstance(value, (int, float)):
        return float(value)

    return default


def _sum_query_value(query_stats: dict[str, Any], keys: list[str], *, default: int = 0) -> int:
    value = _metric_from_query_stats(query_stats, keys, default=None)
    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        return sum(_coerce_int(item, default=0) for item in value)

    return _coerce_int(value, default)


def _average_context_chars(query_stats: dict[str, Any]) -> float:
    total = query_stats.get("context_chars")
    if total is None:
        total = query_stats.get("chars")
    if total is None:
        total = query_stats.get("context_chars_loaded")
    if total is None:
        return 0.0

    if isinstance(total, (list, tuple)):
        if not total:
            return 0.0
        return sum(_coerce_float(v) for v in total) / len(total)

    total_f = _coerce_float(total)
    queries = _coerce_int(query_stats.get("queries"), 0) or _coerce_int(
        query_stats.get("queries_replayed"),
        0,
    )
    if queries > 0:
        return total_f / queries

    return total_f


def _orphan_nodes(graph: Graph) -> int:
    total = 0
    for node in graph.nodes():
        if not graph.incoming(node.id) and not graph.outgoing(node.id):
            total += 1
    return total


def _within_range(value: float, target: MetricRange) -> bool:
    min_v, max_v = target
    if min_v is not None and value < min_v:
        return False
    if max_v is not None and value > max_v:
        return False
    return True


def _range_desc(target: MetricRange) -> str:
    min_v, max_v = target
    if min_v is None:
        return f"<={max_v}%".replace("=%", "")
    if max_v is None:
        return f">={min_v}"
    return f"{min_v}-{max_v}"


def measure_health(graph: Graph, state: MitosisState, query_stats: dict[str, Any]) -> GraphHealth:
    """Compute post-query graph health metrics from observed activity."""
    tiers = edge_tier_stats(graph)
    total_edges = sum(tiers.values())

    cross_file_edges = _cross_file_edges(graph)
    cross_file_edge_pct = (cross_file_edges / total_edges * 100.0) if total_edges else 0.0

    dormant = tiers.get("dormant", 0)
    reflex = tiers.get("reflex", 0)
    dormant_pct = (dormant / total_edges * 100.0) if total_edges else 0.0
    reflex_pct = (reflex / total_edges * 100.0) if total_edges else 0.0

    avg_nodes_fired_per_query = _avg_query_value(
        query_stats,
        ["avg_nodes_fired_per_query", "avg_nodes_fired", "avg_fired", "fired_counts"],
    )

    total_chars = sum(len(node.content) for node in graph.nodes())
    loaded_chars = _average_context_chars(query_stats)
    context_compression = (loaded_chars / total_chars * 100.0) if total_chars else 0.0

    promoted = _sum_query_value(
        query_stats,
        ["promotions", "total_promotions", "promoted", "promoted_count"],
        default=0,
    )
    proto_created = _sum_query_value(
        query_stats,
        ["proto_created", "total_protos_created", "proto_creations", "created"],
        default=0,
    )
    proto_promotion_rate = (promoted / proto_created * 100.0) if proto_created else 0.0

    reconvergence_count = _coerce_float(
        _metric_from_query_stats(
            query_stats,
            ["reconverged_families", "reconvergence_events", "reconverged_count"],
        ),
        default=0.0,
    )
    total_families = len(getattr(state, "families", {}) or {})
    reconvergence_rate = (
        reconvergence_count / total_families * 100.0
        if total_families > 0
        else 0.0
    )

    return GraphHealth(
        avg_nodes_fired_per_query=_coerce_float(avg_nodes_fired_per_query, default=0.0),
        cross_file_edge_pct=_coerce_float(cross_file_edge_pct, default=0.0),
        dormant_pct=_coerce_float(dormant_pct, default=0.0),
        reflex_pct=_coerce_float(reflex_pct, default=0.0),
        context_compression=_coerce_float(context_compression, default=0.0),
        proto_promotion_rate=_coerce_float(proto_promotion_rate, default=0.0),
        reconvergence_rate=_coerce_float(reconvergence_rate, default=0.0),
        orphan_nodes=_orphan_nodes(graph),
    )


def autotune(graph: Graph, health: GraphHealth) -> list[Adjustment]:
    """Suggest configuration changes based on measured graph health."""
    del graph

    adjustments: list[Adjustment] = []

    # avg_nodes_fired_per_query
    min_fired, max_fired = HEALTH_TARGETS["avg_nodes_fired_per_query"]
    if health.avg_nodes_fired_per_query > (max_fired or 0.0):
        adjustments.append(
            Adjustment(
                metric="avg_nodes_fired_per_query",
                current=health.avg_nodes_fired_per_query,
                target_range=(min_fired or 0.0, max_fired or 0.0),
                suggested_change={"decay_half_life": "decrease"},
                reason="Too many nodes fired per query; stronger decay (lower half_life) should narrow spread.",
            )
        )
    elif health.avg_nodes_fired_per_query < (min_fired or 0.0):
        adjustments.append(
            Adjustment(
                metric="avg_nodes_fired_per_query",
                current=health.avg_nodes_fired_per_query,
                target_range=(min_fired or 0.0, max_fired or 0.0),
                suggested_change={
                    "decay_half_life": "increase",
                    "promotion_threshold": "decrease",
                },
                reason="Too few nodes fired per query; slower decay and lower promotion threshold should increase spread.",
            )
        )

    # cross_file_edge_pct
    min_cross, max_cross = HEALTH_TARGETS["cross_file_edge_pct"]
    cross_file_low = health.cross_file_edge_pct < (min_cross or 0.0)
    if health.cross_file_edge_pct < (min_cross or 0.0):
        adjustments.append(
            Adjustment(
                metric="cross_file_edge_pct",
                current=health.cross_file_edge_pct,
                target_range=(min_cross or 0.0, max_cross or 0.0),
                suggested_change={"promotion_threshold": "decrease"},
                reason="Cross-file evidence is sparse; lowering promotion threshold can speed cross-domain edge discovery.",
            )
        )
    elif max_cross is not None and health.cross_file_edge_pct > max_cross:
        adjustments.append(
            Adjustment(
                metric="cross_file_edge_pct",
                current=health.cross_file_edge_pct,
                target_range=(min_cross or 0.0, max_cross or 0.0),
                suggested_change={"promotion_threshold": "increase"},
                reason="Cross-file edges are over-represented; raising promotion threshold can curb noisy shortcuts.",
            )
        )

    # dormant_pct
    min_dormant, max_dormant = HEALTH_TARGETS["dormant_pct"]
    if (health.dormant_pct < (min_dormant or 0.0)) and not cross_file_low:
        adjustments.append(
            Adjustment(
                metric="dormant_pct",
                current=health.dormant_pct,
                target_range=(min_dormant or 0.0, max_dormant or 0.0),
                suggested_change={"decay_half_life": "decrease"},
                reason="Too few dormant links; increasing decay helps remove weakly reinforced candidates faster.",
            )
        )
    elif max_dormant is not None and health.dormant_pct > max_dormant:
        adjustments.append(
            Adjustment(
                metric="dormant_pct",
                current=health.dormant_pct,
                target_range=(min_dormant or 0.0, max_dormant or 0.0),
                suggested_change={"decay_half_life": "increase"},
                reason="Dormant links are over-abundant; relaxing decay may help useful links stabilize.",
            )
        )

    # reflex_pct
    min_reflex, max_reflex = HEALTH_TARGETS["reflex_pct"]
    if health.reflex_pct < (min_reflex or 0.0):
        adjustments.append(
            Adjustment(
                metric="reflex_pct",
                current=health.reflex_pct,
                target_range=(min_reflex or 0.0, max_reflex or 0.0),
                suggested_change={"hebbian_increment": "increase"},
                reason="Reflex is too low; stronger Hebbian updates help edges move into reflex tier faster.",
            )
        )
    if max_reflex is not None and health.reflex_pct > max_reflex:
        adjustments.append(
            Adjustment(
                metric="reflex_pct",
                current=health.reflex_pct,
                target_range=(min_reflex or 0.0, max_reflex or 0.0),
                suggested_change={
                    "decay_half_life": "decrease",
                    "reflex_threshold": "increase",
                },
                reason="Reflex edges are too common; increase decay and/or require stronger evidence for reflex edges.",
            )
        )

    # context_compression
    min_context, max_context = HEALTH_TARGETS["context_compression"]
    if max_context is not None and health.context_compression > max_context:
        adjustments.append(
            Adjustment(
                metric="context_compression",
                current=health.context_compression,
                target_range=(min_context or 0.0, max_context or 0.0),
                suggested_change={"decay_half_life": "decrease"},
                reason="Context compression is high; lowering decay half-life reduces loaded context churn.",
            )
        )

    # proto_promotion_rate
    min_promo, max_promo = HEALTH_TARGETS["proto_promotion_rate"]
    if health.proto_promotion_rate < (min_promo or 0.0):
        adjustments.append(
            Adjustment(
                metric="proto_promotion_rate",
                current=health.proto_promotion_rate,
                target_range=(min_promo or 0.0, max_promo or 0.0),
                suggested_change={"promotion_threshold": "decrease"},
                reason="Promotion is too slow; lowering threshold should convert proto-links sooner.",
            )
        )
    elif max_promo is not None and health.proto_promotion_rate > max_promo:
        adjustments.append(
            Adjustment(
                metric="proto_promotion_rate",
                current=health.proto_promotion_rate,
                target_range=(min_promo or 0.0, max_promo or 0.0),
                suggested_change={"promotion_threshold": "increase"},
                reason="Promotion is too aggressive; raising threshold can reduce weak proto-link conversion.",
            )
        )

    # orphan_nodes
    if health.orphan_nodes > 0:
        adjustments.append(
            Adjustment(
                metric="orphan_nodes",
                current=health.orphan_nodes,
                target_range=HEALTH_TARGETS["orphan_nodes"],
                suggested_change={"investigation": "trace why nodes have no edges"},
                reason="Orphan nodes indicate disconnected storage; investigate routing, promotion, and split/merge side-effects.",
            )
        )

    if not adjustments:
        return []

    return adjustments


def _record_change(
    changes: dict[str, dict[str, float | int]],
    key: str,
    before: float | int,
    after: float | int,
) -> None:
    if after == before:
        return
    changes[key] = {"before": before, "after": after, "delta": after - before}


def apply_adjustments(
    adjustments: list[Adjustment],
    syn_config: SynaptogenesisConfig,
    decay_config: DecayConfig,
    mitosis_config: MitosisConfig,
    last_adjusted: dict[str, int] | None = None,
    cycle_number: int = 0,
) -> dict[str, dict[str, float | int]]:
    """Apply suggestions from autotune() directly to live config objects.

    Returns:
        Mapping of changed public keys to `{before, after, delta}`.
    """
    del mitosis_config
    if last_adjusted is None:
        last_adjusted = {}
    changes: dict[str, dict[str, float | int]] = {}

    for adjustment in adjustments:
        for key, direction in adjustment.suggested_change.items():
            last_seen = last_adjusted.get(key, -10**9)
            if cycle_number - last_seen <= ADJUSTMENT_COOLDOWN_CYCLES:
                continue
            if key == "decay_half_life":
                before = decay_config.half_life_turns
                if direction == "decrease":
                    after = before * 0.75
                elif direction == "increase":
                    after = before * 1.33
                else:
                    continue
                after = int(round(max(20, min(300, after))))
                decay_config.half_life_turns = after
                _record_change(changes, "decay_half_life", before, after)
                last_adjusted[key] = cycle_number

            elif key == "promotion_threshold":
                before = int(syn_config.promotion_threshold)
                if direction == "decrease":
                    after = before - 1
                elif direction == "increase":
                    after = before + 1
                else:
                    continue
                after = max(2, min(10, after))
                syn_config.promotion_threshold = after
                _record_change(changes, "promotion_threshold", before, after)
                last_adjusted[key] = cycle_number

            elif key == "reflex_threshold":
                before = float(syn_config.reflex_threshold)
                if direction == "increase":
                    after = before + 0.05
                else:
                    continue
                after = min(0.95, after)
                syn_config.reflex_threshold = after
                _record_change(changes, "reflex_threshold", before, after)
                last_adjusted[key] = cycle_number

            elif key == "skip_factor":
                before = float(syn_config.skip_factor)
                if direction == "increase":
                    after = before + 0.02
                elif direction == "decrease":
                    after = before - 0.02
                else:
                    continue
                after = max(0.0, min(1.0, after))
                syn_config.skip_factor = after
                _record_change(changes, "skip_factor", before, after)
                last_adjusted[key] = cycle_number

            elif key == "hebbian_increment":
                before = float(syn_config.hebbian_increment)
                if direction == "increase":
                    after = before + 0.01
                elif direction == "decrease":
                    after = before - 0.01
                else:
                    continue
                after = max(0.02, min(0.15, after))
                syn_config.hebbian_increment = after
                _record_change(changes, "hebbian_increment", before, after)
                last_adjusted[key] = cycle_number

    return changes


def self_tune(
    graph: Graph,
    state: MitosisState,
    query_stats: dict[str, Any],
    syn_config: SynaptogenesisConfig,
    decay_config: DecayConfig,
    mitosis_config: MitosisConfig,
    cycle_number: int = 0,
    last_adjusted: dict[str, int] | None = None,
) -> tuple[GraphHealth, list[Adjustment], dict[str, dict[str, float | int]]]:
    """Run one full health-tuning cycle."""
    health = measure_health(graph, state, query_stats)
    adjustments = autotune(graph, health)
    changes = apply_adjustments(
        adjustments,
        syn_config,
        decay_config,
        mitosis_config,
        last_adjusted=last_adjusted,
        cycle_number=cycle_number,
    )
    return health, adjustments, changes
