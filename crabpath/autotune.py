"""Adaptive configuration helpers for warm-start and runtime tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .graph import Graph
from ._structural_utils import count_cross_file_edges
from ._structural_utils import JSONStateMixin
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
        "sibling_weight": 0.45,
        "promotion_threshold": 2,
        "decay_half_life": 50,
        "decay_interval": 4,
        "min_content_chars": 160,
        "hebbian_increment": 0.08,
        "skip_factor": 0.9,
        "max_outgoing": 14,
    },
    "medium": {
        "sibling_weight": 0.5,
        "promotion_threshold": 2,
        "decay_half_life": 80,
        "decay_interval": 10,
        "min_content_chars": 200,
        "hebbian_increment": 0.06,
        "skip_factor": 0.9,
        "max_outgoing": 20,
    },
    "large": {
        "sibling_weight": 0.70,
        "promotion_threshold": 4,
        "decay_half_life": 120,
        "decay_interval": 12,
        "min_content_chars": 260,
        "hebbian_increment": 0.04,
        "skip_factor": 0.9,
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


@dataclass
class TuneHistoryRecord:
    cycle: int
    metric: str
    knob: str
    direction: str
    before_value: float | int
    after_value: float | int
    before_health: GraphHealth
    after_health: GraphHealth | None = None
    delta_toward_target: int = 0
    score: int = 0


@dataclass
class TuneHistory:
    records: list[TuneHistoryRecord] = field(default_factory=list)
    pending: list[TuneHistoryRecord] = field(default_factory=list)
    previous_config: dict[str, float | int] | None = None
    query_stats_snapshot: dict[str, Any] = field(default_factory=dict)

    def record_adjustments(
        self,
        cycle: int,
        pre_health: GraphHealth,
        adjustments: list[Adjustment],
        changes: dict[str, dict[str, float | int]],
    ) -> list[TuneHistoryRecord]:
        """Store applied adjustments for deferred evaluation on a later cycle."""
        new_records: list[TuneHistoryRecord] = []
        seen: set[tuple[str, str, str]] = set()

        for adjustment in adjustments:
            for knob, direction in adjustment.suggested_change.items():
                if knob not in changes:
                    continue
                key = (adjustment.metric, knob, direction)
                if key in seen:
                    continue
                seen.add(key)

                change = changes[knob]
                record = TuneHistoryRecord(
                    cycle=cycle,
                    metric=adjustment.metric,
                    knob=knob,
                    direction=direction,
                    before_value=change["before"],
                    after_value=change["after"],
                    before_health=pre_health,
                )
                self.pending.append(record)
                new_records.append(record)

        return new_records

    def evaluate_pending(self, current_health: GraphHealth) -> list[TuneHistoryRecord]:
        """Evaluate previously recorded adjustments against current health."""
        if not self.pending:
            return []

        evaluated: list[TuneHistoryRecord] = []
        for record in self.pending:
            metric = record.metric
            pre_metric = _coerce_float(getattr(record.before_health, metric), default=0.0)
            post_metric = _coerce_float(getattr(current_health, metric), default=0.0)
            target = HEALTH_TARGETS[metric]

            score = _score_metric_toward_target(pre_metric, post_metric, target)
            record.after_health = current_health
            record.delta_toward_target = score
            record.score = score
            self.records.append(record)
            evaluated.append(record)

        self.pending = []
        return evaluated


@dataclass
class TuneMemory(JSONStateMixin):
    scores: dict[tuple[str, str, str], int] = field(default_factory=dict)

    @staticmethod
    def _key(metric: str, knob: str, direction: str) -> tuple[str, str, str]:
        return metric, knob, direction

    @staticmethod
    def _triple_to_key(metric: str, knob: str, direction: str) -> str:
        return f"{metric}|{knob}|{direction}"

    @staticmethod
    def _key_to_triple(key: str) -> tuple[str, str, str] | None:
        parts = key.split("|", 2)
        if len(parts) != 3:
            return None
        return parts[0], parts[1], parts[2]

    def get_score(self, metric: str, knob: str, direction: str) -> int:
        return self.scores.get(self._key(metric, knob, direction), 0)

    def is_blocked(self, metric: str, knob: str, direction: str) -> bool:
        return self.get_score(metric, knob, direction) < -2

    def is_preferred(self, metric: str, knob: str, direction: str) -> bool:
        return self.get_score(metric, knob, direction) > 2

    def update(self, history: TuneHistory) -> None:
        for record in history.records:
            if record.score == 0:
                continue
            key = self._key(record.metric, record.knob, record.direction)
            self.scores[key] = self.scores.get(key, 0) + record.score

    def report(self) -> str:
        works: list[str] = []
        blocked: list[str] = []
        preferred: list[str] = []

        for (metric, knob, direction), score in sorted(self.scores.items()):
            triple = f"{metric}:{knob}:{direction}"
            if score > 2:
                preferred.append(f"{triple} => {score}")
            elif score < -2:
                blocked.append(f"{triple} => {score}")
            elif score > 0:
                works.append(f"{triple} => {score}")

        if not works:
            works.append("<none>")
        if not blocked:
            blocked.append("<none>")
        if not preferred:
            preferred.append("<none>")

        return "\n".join(
            [
                "TuneMemory report",
                "Working triples:",
                *[f"- {item}" for item in works],
                "Blocked triples:",
                *[f"- {item}" for item in blocked],
                "Preferred triples:",
                *[f"- {item}" for item in preferred],
            ]
        )

    def save(self, path: str) -> None:
        data = {
            self._triple_to_key(metric, knob, direction): score
            for (metric, knob, direction), score in self.scores.items()
        }
        self._write_json_file(path, data, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "TuneMemory":
        data = cls._load_json_file(path, default=None)
        if not isinstance(data, dict):
            return cls()

        scores: dict[tuple[str, str, str], int] = {}
        for key, score in data.items():
            triple = cls._key_to_triple(str(key))
            if triple is None:
                continue
            try:
                scores[triple] = int(score)
            except (TypeError, ValueError):
                continue
        return cls(scores=scores)


HEALTH_TARGETS: dict[str, MetricRange] = {
    # These targets are EMPIRICALLY CALIBRATED from 5 playback runs.
    # They represent what a healthy ~300-node graph looks like after ~100 queries.
    # They should be recalibrated for significantly different graph sizes.
    # Wider targets keep the autotuner from generating false positives too often.
    "avg_nodes_fired_per_query": (3.0, 10.0),
    "cross_file_edge_pct": (3.0, 25.0),
    "dormant_pct": (60.0, 97.0),
    "reflex_pct": (0.5, 5.0),
    "context_compression": (None, 25.0),
    "proto_promotion_rate": (5.0, 25.0),
    "reconvergence_rate": (None, 15.0),
    "orphan_nodes": (0.0, 0.0),
}

ADJUSTMENT_COOLDOWN_CYCLES = 2


@dataclass
class SafetyBounds:
    # Decay: too fast = brain death, too slow = no differentiation
    min_decay_half_life: int = 30
    max_decay_half_life: int = 200

    # Promotion: too low = noise, too high = no cross-file edges ever
    min_promotion_threshold: int = 2
    max_promotion_threshold: int = 6

    # Hebbian: too high = reflex runaway, too low = no learning
    min_hebbian_increment: float = 0.02
    max_hebbian_increment: float = 0.12

    # Retrieval scoring: filter weak/helpful and harmful signals in RL
    min_helpfulness_gate: float = 0.05
    max_helpfulness_gate: float = 0.50
    min_harmful_reward_threshold: float = -1.0
    max_harmful_reward_threshold: float = -0.2

    # Max adjustments per cycle: prevent thrashing
    max_adjustments_per_cycle: int = 3

    # Emergency brake: if >X% of metrics got WORSE after an adjustment, revert ALL
    revert_threshold: float = 0.5


ADJUSTMENT_PRIORITIES = {
    "decay_half_life": 0,
    "promotion_threshold": 1,
    "hebbian_increment": 2,
    "helpfulness_gate": 3,
    "harmful_reward_threshold": 4,
}


def validate_config(
    syn_config: SynaptogenesisConfig,
    decay_config: DecayConfig,
    safety_bounds: SafetyBounds | None = None,
) -> bool:
    """Return whether the current tunable config is within hard safety bounds."""
    if safety_bounds is None:
        safety_bounds = SafetyBounds()

    checks: list[tuple[float | int, float | int, float | int]] = [
        (
            decay_config.half_life_turns,
            safety_bounds.min_decay_half_life,
            safety_bounds.max_decay_half_life,
        ),
        (
            syn_config.promotion_threshold,
            safety_bounds.min_promotion_threshold,
            safety_bounds.max_promotion_threshold,
        ),
        (
            syn_config.hebbian_increment,
            safety_bounds.min_hebbian_increment,
            safety_bounds.max_hebbian_increment,
        ),
        (
            syn_config.helpfulness_gate,
            safety_bounds.min_helpfulness_gate,
            safety_bounds.max_helpfulness_gate,
        ),
        (
            syn_config.harmful_reward_threshold,
            safety_bounds.min_harmful_reward_threshold,
            safety_bounds.max_harmful_reward_threshold,
        ),
    ]

    for value, min_value, max_value in checks:
        try:
            if value < min_value or value > max_value:
                return False
        except TypeError:
            return False

    return True


def _snapshot_tune_config(
    syn_config: SynaptogenesisConfig,
    decay_config: DecayConfig,
) -> dict[str, float | int]:
    return {
        "decay_half_life": decay_config.half_life_turns,
        "promotion_threshold": syn_config.promotion_threshold,
        "hebbian_increment": syn_config.hebbian_increment,
        "helpfulness_gate": syn_config.helpfulness_gate,
        "harmful_reward_threshold": syn_config.harmful_reward_threshold,
    }


def _restore_tune_config(
    snapshot: dict[str, float | int],
    syn_config: SynaptogenesisConfig,
    decay_config: DecayConfig,
) -> None:
    decay_config.half_life_turns = int(snapshot["decay_half_life"])
    syn_config.promotion_threshold = int(snapshot["promotion_threshold"])
    syn_config.hebbian_increment = float(snapshot["hebbian_increment"])
    syn_config.helpfulness_gate = float(snapshot["helpfulness_gate"])
    syn_config.harmful_reward_threshold = float(snapshot["harmful_reward_threshold"])


def _should_revert(
    records: list[TuneHistoryRecord],
    safety_bounds: SafetyBounds,
) -> bool:
    if not records:
        return False
    worse_count = sum(1 for record in records if record.score < 0)
    return (worse_count / len(records)) >= safety_bounds.revert_threshold


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


def _avg_query_value(
    query_stats: dict[str, Any], keys: list[str], *, default: float = 0.0
) -> float:
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


def compute_window_stats(
    current_stats: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Build a delta stats dict representing query activity since the snapshot."""
    snapshot = snapshot or {}

    windowed: dict[str, Any] = {}
    for key, value in current_stats.items():
        if isinstance(value, list):
            prior = snapshot.get(key)
            if isinstance(prior, (list, tuple)):
                if len(value) >= len(prior):
                    windowed[key] = list(value[len(prior):])
                else:
                    windowed[key] = list(value)
            else:
                windowed[key] = list(value)
        elif isinstance(value, tuple):
            prior = snapshot.get(key)
            if isinstance(prior, (list, tuple)):
                if len(value) >= len(prior):
                    windowed[key] = list(value[len(prior):])
                else:
                    windowed[key] = list(value)
            else:
                windowed[key] = list(value)
        elif isinstance(value, dict):
            windowed[key] = dict(value)
        else:
            windowed[key] = value

    current_promotions = _sum_query_value(
        current_stats,
        ["promotions", "total_promotions", "promoted", "promoted_count"],
        default=0,
    )
    snapshot_promotions = _sum_query_value(
        snapshot,
        ["promotions", "total_promotions", "promoted", "promoted_count"],
        default=0,
    )
    total_promotions = current_promotions - snapshot_promotions
    if any(key in current_stats or key in snapshot for key in ["promotions", "total_promotions", "promoted", "promoted_count"]):
        windowed["promotions"] = max(0, total_promotions)
        windowed["total_promotions"] = max(0, total_promotions)

    current_protos_created = _sum_query_value(
        current_stats,
        ["proto_created", "total_protos_created", "proto_creations", "created"],
        default=0,
    )
    snapshot_protos_created = _sum_query_value(
        snapshot,
        ["proto_created", "total_protos_created", "proto_creations", "created"],
        default=0,
    )
    total_protos_created = current_protos_created - snapshot_protos_created
    if any(
        key in current_stats or key in snapshot
        for key in ["proto_created", "total_protos_created", "proto_creations", "created"]
    ):
        windowed["proto_created"] = max(0, total_protos_created)
        windowed["total_protos_created"] = max(0, total_protos_created)

    return windowed


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


def _clone_query_stats(query_stats: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in query_stats.items():
        if isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = tuple(value)
        elif isinstance(value, dict):
            cloned[key] = dict(value)
        else:
            cloned[key] = value
    return cloned


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


def _metric_distance_to_target(value: float, target: MetricRange) -> float:
    min_v, max_v = target
    if min_v is not None and value < min_v:
        return min_v - value
    if max_v is not None and value > max_v:
        return value - max_v
    return 0.0


def _score_metric_toward_target(before: float, after: float, target: MetricRange) -> int:
    before_distance = _metric_distance_to_target(before, target)
    after_distance = _metric_distance_to_target(after, target)
    if after_distance < before_distance:
        return 1
    if after_distance > before_distance:
        return -1
    return 0


def _filter_adjustments(
    adjustments: list[Adjustment],
    tune_memory: TuneMemory | None,
) -> list[Adjustment]:
    if tune_memory is None:
        return adjustments

    preferred_adjustments: list[Adjustment] = []
    regular_adjustments: list[Adjustment] = []

    for adjustment in adjustments:
        filtered: dict[str, str] = {}
        has_preferred = False

        for knob, direction in adjustment.suggested_change.items():
            if tune_memory.is_blocked(adjustment.metric, knob, direction):
                continue
            if tune_memory.is_preferred(adjustment.metric, knob, direction):
                has_preferred = True
            filtered[knob] = direction

        if not filtered:
            continue

        filtered_adjustment = Adjustment(
            metric=adjustment.metric,
            current=adjustment.current,
            target_range=adjustment.target_range,
            suggested_change=filtered,
            reason=adjustment.reason,
        )
        if has_preferred:
            preferred_adjustments.append(filtered_adjustment)
        else:
            regular_adjustments.append(filtered_adjustment)

    return preferred_adjustments + regular_adjustments


def measure_health(graph: Graph, state: MitosisState, query_stats: dict[str, Any]) -> GraphHealth:
    """Compute post-query graph health metrics from observed activity."""
    tiers = edge_tier_stats(graph)
    total_edges = sum(tiers.values())

    cross_file_edges = count_cross_file_edges(graph)
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
    reconvergence_rate = reconvergence_count / total_families * 100.0 if total_families > 0 else 0.0

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
                reason=(
                    "Too many nodes fired per query; stronger decay "
                    "(lower half_life) should narrow spread."
                ),
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
                reason=(
                    "Too few nodes fired per query; slower decay and lower "
                    "promotion threshold should increase spread."
                ),
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
                reason=(
                    "Cross-file evidence is sparse; lowering promotion "
                    "threshold can speed cross-domain edge discovery."
                ),
            )
        )
    elif max_cross is not None and health.cross_file_edge_pct > max_cross:
        adjustments.append(
            Adjustment(
                metric="cross_file_edge_pct",
                current=health.cross_file_edge_pct,
                target_range=(min_cross or 0.0, max_cross or 0.0),
                suggested_change={"promotion_threshold": "increase"},
                reason=(
                    "Cross-file edges are over-represented; raising promotion "
                    "threshold can curb noisy shortcuts."
                ),
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
                reason=(
                    "Too few dormant links; increasing decay helps remove "
                    "weakly reinforced candidates faster."
                ),
            )
        )
    elif max_dormant is not None and health.dormant_pct > max_dormant:
        adjustments.append(
            Adjustment(
                metric="dormant_pct",
                current=health.dormant_pct,
                target_range=(min_dormant or 0.0, max_dormant or 0.0),
                suggested_change={"decay_half_life": "increase"},
                reason=(
                    "Dormant links are over-abundant; relaxing decay may help "
                    "useful links stabilize."
                ),
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
                reason=(
                    "Reflex is too low; stronger Hebbian updates help edges "
                    "move into reflex tier faster."
                ),
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
                    "hebbian_increment": "decrease",
                },
                reason=(
                    "Reflex edges are too common; increase decay and/or require "
                    "weaker Hebbian growth for edge promotion into reflex."
                ),
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
                reason=(
                    "Context compression is high; lowering decay half-life "
                    "reduces loaded context churn."
                ),
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
                suggested_change={
                    "promotion_threshold": "decrease",
                    "helpfulness_gate": "decrease",
                },
                reason=(
                    "Promotion is too slow; lowering threshold and helpfulness "
                    "gate should increase RL throughput."
                ),
            )
        )
    elif max_promo is not None and health.proto_promotion_rate > max_promo:
        adjustments.append(
            Adjustment(
                metric="proto_promotion_rate",
                current=health.proto_promotion_rate,
                target_range=(min_promo or 0.0, max_promo or 0.0),
                suggested_change={
                    "promotion_threshold": "increase",
                    "helpfulness_gate": "increase",
                    "harmful_reward_threshold": "decrease",
                },
                reason=(
                    "Promotion is too aggressive; raising threshold can "
                    "reduce weak proto-link conversion and tighten RL filtering."
                ),
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
                reason=(
                    "Orphan nodes indicate disconnected storage; investigate "
                    "routing, promotion, and split/merge side-effects."
                ),
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
    bounded: bool = False,
) -> None:
    if after == before and not bounded:
        return
    change: dict[str, float | int | bool] = {
        "before": before,
        "after": after,
        "delta": after - before,
    }
    if bounded:
        change["bounded"] = True
    changes[key] = change


def apply_adjustments(
    adjustments: list[Adjustment],
    syn_config: SynaptogenesisConfig,
    decay_config: DecayConfig,
    mitosis_config: MitosisConfig,
    last_adjusted: dict[str, int] | None = None,
    cycle_number: int = 0,
    safety_bounds: SafetyBounds | None = None,
) -> dict[str, dict[str, float | int]]:
    """Apply suggestions from autotune() directly to live config objects.

    Returns:
        Mapping of changed public keys to `{before, after, delta}`.
    """
    if safety_bounds is None:
        safety_bounds = SafetyBounds()

    del mitosis_config
    if last_adjusted is None:
        last_adjusted = {}
    changes: dict[str, dict[str, float | int]] = {}

    requested_adjustments: list[tuple[str, str]] = []
    for adjustment in adjustments:
        for key, direction in adjustment.suggested_change.items():
            if key not in ADJUSTMENT_PRIORITIES:
                continue
            last_seen = last_adjusted.get(key, -(10**9))
            if cycle_number - last_seen <= ADJUSTMENT_COOLDOWN_CYCLES:
                continue
            requested_adjustments.append((key, direction))

    def _priority(item: tuple[str, str]) -> int:
        return ADJUSTMENT_PRIORITIES.get(item[0], len(ADJUSTMENT_PRIORITIES))

    requested_adjustments = list(dict.fromkeys(requested_adjustments))
    requested_adjustments.sort(key=_priority)
    requested_adjustments = requested_adjustments[: safety_bounds.max_adjustments_per_cycle]

    for key, direction in requested_adjustments:
        if key == "decay_half_life":
            before = decay_config.half_life_turns
            if direction == "decrease":
                candidate = before * 0.75
            elif direction == "increase":
                candidate = before * 1.33
            else:
                continue
            clamped = int(
                round(
                    max(
                        safety_bounds.min_decay_half_life,
                        min(safety_bounds.max_decay_half_life, candidate),
                    )
                )
            )
            bounded = int(round(candidate)) != clamped
            decay_config.half_life_turns = clamped
            _record_change(changes, "decay_half_life", before, clamped, bounded=bounded)
            last_adjusted[key] = cycle_number

        elif key == "promotion_threshold":
            before = int(syn_config.promotion_threshold)
            if direction == "decrease":
                candidate = before - 1
            elif direction == "increase":
                candidate = before + 1
            else:
                continue
            clamped = int(
                max(
                    safety_bounds.min_promotion_threshold,
                    min(safety_bounds.max_promotion_threshold, candidate),
                )
            )
            bounded = candidate != clamped
            syn_config.promotion_threshold = clamped
            _record_change(changes, "promotion_threshold", before, clamped, bounded=bounded)
            last_adjusted[key] = cycle_number

        elif key == "helpfulness_gate":
            before = float(syn_config.helpfulness_gate)
            if direction == "increase":
                candidate = before + 0.05
            elif direction == "decrease":
                candidate = before - 0.05
            else:
                continue
            clamped = float(
                max(
                    safety_bounds.min_helpfulness_gate,
                    min(safety_bounds.max_helpfulness_gate, candidate),
                )
            )
            bounded = candidate != clamped
            syn_config.helpfulness_gate = clamped
            _record_change(changes, "helpfulness_gate", before, clamped, bounded=bounded)
            last_adjusted[key] = cycle_number

        elif key == "harmful_reward_threshold":
            before = float(syn_config.harmful_reward_threshold)
            if direction == "increase":
                candidate = before + 0.05
            elif direction == "decrease":
                candidate = before - 0.05
            else:
                continue
            clamped = float(
                max(
                    safety_bounds.min_harmful_reward_threshold,
                    min(safety_bounds.max_harmful_reward_threshold, candidate),
                )
            )
            bounded = candidate != clamped
            syn_config.harmful_reward_threshold = clamped
            _record_change(
                changes,
                "harmful_reward_threshold",
                before,
                clamped,
                bounded=bounded,
            )
            last_adjusted[key] = cycle_number

        elif key == "hebbian_increment":
            before = float(syn_config.hebbian_increment)
            if direction == "increase":
                candidate = before + 0.01
            elif direction == "decrease":
                candidate = before - 0.01
            else:
                continue
            clamped = float(
                max(
                    safety_bounds.min_hebbian_increment,
                    min(safety_bounds.max_hebbian_increment, candidate),
                )
            )
            bounded = candidate != clamped
            syn_config.hebbian_increment = clamped
            _record_change(changes, "hebbian_increment", before, clamped, bounded=bounded)
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
    tune_history: TuneHistory | None = None,
    tune_memory: TuneMemory | None = None,
    safety_bounds: SafetyBounds | None = None,
) -> tuple[GraphHealth, list[Adjustment], dict[str, dict[str, float | int]]]:
    """Run one full health-tuning cycle."""
    if safety_bounds is None:
        safety_bounds = SafetyBounds()

    if tune_history is None:
        tune_history = TuneHistory()

    if tune_history.query_stats_snapshot:
        window_query_stats = compute_window_stats(
            query_stats,
            tune_history.query_stats_snapshot,
        )
    else:
        window_query_stats = query_stats

    health = measure_health(graph, state, window_query_stats)

    previous_config = _snapshot_tune_config(syn_config, decay_config)
    if tune_history.pending:
        completed_records = tune_history.evaluate_pending(health)
        if (
            _should_revert(completed_records, safety_bounds)
            and tune_history.previous_config is not None
        ):
            _restore_tune_config(tune_history.previous_config, syn_config, decay_config)
        if tune_memory is not None:
            tune_memory.update(TuneHistory(records=completed_records))

    adjustments = autotune(graph, health)
    adjustments = _filter_adjustments(adjustments, tune_memory)
    effective_max = 1 if tune_memory is not None else safety_bounds.max_adjustments_per_cycle
    changes = apply_adjustments(
        adjustments,
        syn_config,
        decay_config,
        mitosis_config,
        last_adjusted=last_adjusted,
        cycle_number=cycle_number,
        safety_bounds=SafetyBounds(
            min_decay_half_life=safety_bounds.min_decay_half_life,
            max_decay_half_life=safety_bounds.max_decay_half_life,
            min_promotion_threshold=safety_bounds.min_promotion_threshold,
            max_promotion_threshold=safety_bounds.max_promotion_threshold,
            min_hebbian_increment=safety_bounds.min_hebbian_increment,
            max_hebbian_increment=safety_bounds.max_hebbian_increment,
            min_helpfulness_gate=safety_bounds.min_helpfulness_gate,
            max_helpfulness_gate=safety_bounds.max_helpfulness_gate,
            min_harmful_reward_threshold=safety_bounds.min_harmful_reward_threshold,
            max_harmful_reward_threshold=safety_bounds.max_harmful_reward_threshold,
            max_adjustments_per_cycle=effective_max,
            revert_threshold=safety_bounds.revert_threshold,
        ),
    )
    tune_history.previous_config = previous_config
    tune_history.query_stats_snapshot = _clone_query_stats(query_stats)
    tune_history.record_adjustments(cycle_number, health, adjustments, changes)

    return health, adjustments, changes
