"""
🦀 CrabPath Synaptogenesis — How edges are born, change, and die.

Edges form mechanically from usage patterns. No LLM needed.
The LLM routes. The mechanics wire.

Proto-edges: tentative connections that must prove themselves.
  - First co-selection stores credit in a side table
  - Repeated co-firing promotes to real edge (dormant, weight 0.15)
  - If not reinforced, proto-edges decay and die

Real edges:
  - Dormant (< 0.3): invisible to router, must prove themselves
  - Habitual (0.3 - 0.8): visible, LLM decides whether to follow
  - Reflex (> 0.8): auto-follow, no LLM call

Edge changes:
  - Co-fire: Hebbian reinforcement (+)
  - Skip: candidate not chosen, mild penalty (×0.9)
  - Reward: policy gradient along trajectory
  - Decay: mechanical, time-based
  - Death: below min threshold, pruned
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ._structural_utils import ConfigBase, JSONStateMixin, classify_edge_tier
from .graph import Edge, Graph
from .inhibition import InhibitionConfig
from .inhibition import apply_correction as apply_inhibition_correction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SynaptogenesisConfig(ConfigBase):
    """All the knobs for edge formation."""

    # Proto-edge promotion
    promotion_threshold: int = 2  # Co-fires needed to promote (2 = responsive)
    proto_decay_rate: float = 0.85  # Proto-credit multiplier per maintenance cycle
    proto_min_credit: float = 0.4  # Below this, proto-edge dies

    # New edge weights
    cofire_initial_weight: float = 0.18  # Symmetric co-fire edge (just above dormant floor)
    causal_initial_weight: float = 0.28  # Directed A→B (A fired before B)

    # Reinforcement
    hebbian_increment: float = 0.06  # Per co-fire reinforcement
    skip_factor: float = 0.9  # Multiply weight when skipped as candidate

    # Limits
    max_outgoing: int = 20  # Max outgoing edges per node
    min_edge_weight: float = 0.05  # Below this, edge dies

    # Correction
    negative_cap: float = -1.0  # Never let inhibitory edges drop below this
    correction_decay: float = 0.5  # How much to halve positive edges on correction

    # Tiers
    helpfulness_gate: float = 0.1  # Minimum retrieval score needed for RL reward signal
    harmful_reward_threshold: float = -0.5  # Min node score treated as actively harmful
    dormant_threshold: float = 0.1  # Fixed — decay controls visibility
    # Below = dormant (invisible to router)
    reflex_threshold: float = 0.9  # Fixed — hebbian controls compilation
    # Above = reflex (auto-follow)


# ---------------------------------------------------------------------------
# Proto-edge state
# ---------------------------------------------------------------------------


@dataclass
class ProtoEdge:
    """A tentative connection. Not yet a real graph edge."""

    source: str
    target: str
    credit: float = 1.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    causal_count: int = 0  # Times source fired before target
    reverse_count: int = 0  # Times target fired before source


@dataclass
class SynaptogenesisState(JSONStateMixin):
    """Tracks proto-edges and co-firing history."""

    # (source, target) -> ProtoEdge
    proto_edges: dict[tuple[str, str], ProtoEdge] = field(default_factory=dict)

    def save(self, path: str) -> None:
        data: dict[str, Any] = {}
        for (s, t), pe in self.proto_edges.items():
            data[f"{s}|{t}"] = {
                "source": pe.source,
                "target": pe.target,
                "credit": pe.credit,
                "first_seen": pe.first_seen,
                "last_seen": pe.last_seen,
                "causal_count": pe.causal_count,
                "reverse_count": pe.reverse_count,
            }
        self._write_json_file(path, data)

    @classmethod
    def load(cls, path: str) -> SynaptogenesisState:
        data = cls._load_json_file(path, default=None)
        if not isinstance(data, dict):
            return cls()

        state = cls()
        for val in data.values():
            if not isinstance(val, dict):
                continue
            s = val.get("source")
            t = val.get("target")
            if not isinstance(s, str) or not isinstance(t, str):
                continue
            state.proto_edges[(s, t)] = ProtoEdge(
                source=s,
                target=t,
                credit=val.get("credit", 1.0),
                first_seen=val.get("first_seen", 0.0),
                last_seen=val.get("last_seen", 0.0),
                causal_count=val.get("causal_count", 0),
                reverse_count=val.get("reverse_count", 0),
            )
        return state


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def record_cofiring(
    graph: Graph,
    fired_nodes: list[str],
    state: SynaptogenesisState,
    config: SynaptogenesisConfig | None = None,
) -> dict[str, Any]:
    """Record that these nodes co-fired. Update proto-edges or reinforce real edges.

    Args:
        graph: The graph.
        fired_nodes: Ordered list of node IDs that fired (order = causal sequence).
        state: Proto-edge state.
        config: Config.

    Returns:
        Dict with counts of actions taken.
    """
    config = config or SynaptogenesisConfig()
    now = time.time()
    promoted = 0
    reinforced = 0
    proto_created = 0

    # For every pair of co-fired nodes
    for i, node_a in enumerate(fired_nodes):
        for j, node_b in enumerate(fired_nodes):
            if i == j:
                continue

            edge = graph.get_edge(node_a, node_b)

            if edge is not None:
                # Real edge exists — reinforce
                edge.weight = min(edge.weight + config.hebbian_increment, 5.0)
                edge.provenance = "hebbian"
                edge.last_followed_ts = now
                edge.follow_count = (edge.follow_count or 0) + 1
                reinforced += 1
            else:
                # No real edge — update or create proto-edge
                key = (node_a, node_b)
                proto = state.proto_edges.get(key)

                if proto is None:
                    # First time seeing this pair
                    proto = ProtoEdge(
                        source=node_a,
                        target=node_b,
                        credit=1.0,
                        first_seen=now,
                        last_seen=now,
                        causal_count=1 if i < j else 0,
                        reverse_count=1 if i > j else 0,
                    )
                    state.proto_edges[key] = proto
                    proto_created += 1
                else:
                    # Existing proto — accumulate
                    proto.credit += 1.0
                    proto.last_seen = now
                    if i < j:
                        proto.causal_count += 1
                    else:
                        proto.reverse_count += 1

                # Check promotion
                if proto.credit >= config.promotion_threshold:
                    # Promote! Determine weight based on causal evidence
                    if proto.causal_count > proto.reverse_count * 2:
                        # Strong causal direction
                        weight = config.causal_initial_weight
                    else:
                        weight = config.cofire_initial_weight

                    new_edge = Edge(
                        source=node_a,
                        target=node_b,
                        weight=weight,
                        created_by="auto",
                    )
                    new_edge.last_followed_ts = now
                    if _add_edge_with_competition(graph, new_edge, config):
                        # Remove proto only when the edge was actually added.
                        del state.proto_edges[key]
                        promoted += 1

    return {
        "reinforced": reinforced,
        "proto_created": proto_created,
        "promoted": promoted,
    }


def record_skips(
    graph: Graph,
    current_node: str,
    candidates: list[str],
    selected: list[str],
    config: SynaptogenesisConfig | None = None,
) -> int:
    """Apply skip penalty to candidates that were NOT selected.

    Returns count of edges penalized.
    """
    config = config or SynaptogenesisConfig()
    skipped = set(candidates) - set(selected)
    penalized = 0

    for node_id in skipped:
        edge = graph.get_edge(current_node, node_id)
        if edge is not None:
            edge.weight *= config.skip_factor
            edge.skip_count = (edge.skip_count or 0) + 1
            penalized += 1

    return penalized


def record_correction(
    graph: Graph,
    trajectory: list[str],
    reward: float = -1.0,
    config: SynaptogenesisConfig | None = None,
) -> list[dict[str, float | str | None]]:
    """Compatibility wrapper around inhibition.apply_correction."""
    config = config or SynaptogenesisConfig()
    return apply_inhibition_correction(
        graph=graph,
        trajectory=trajectory,
        reward=reward,
        config=InhibitionConfig(
            correction_decay=config.correction_decay,
            negative_cap=config.negative_cap,
        ),
    )


def decay_proto_edges(
    state: SynaptogenesisState,
    config: SynaptogenesisConfig | None = None,
) -> int:
    """Decay proto-edges. Remove ones below threshold. Returns count removed."""
    config = config or SynaptogenesisConfig()
    removed = 0
    to_remove = []

    for key, proto in state.proto_edges.items():
        proto.credit *= config.proto_decay_rate
        if proto.credit < config.proto_min_credit:
            to_remove.append(key)

    for key in to_remove:
        del state.proto_edges[key]
        removed += 1

    return removed


def _add_edge_with_competition(
    graph: Graph,
    new_edge: Edge,
    config: SynaptogenesisConfig,
) -> bool:
    """Add edge, enforcing max outgoing limit. Weakest edge dies if full.

    Returns True when an edge was added/replaced.
    """
    outgoing = graph.outgoing(new_edge.source)

    if len(outgoing) >= config.max_outgoing:
        # Find weakest non-protected outgoing edge
        weakest = None
        weakest_weight = float("inf")
        for target_node, edge in outgoing:
            if graph.is_node_protected(edge.target):
                continue
            if abs(edge.weight) < weakest_weight:
                weakest = edge
                weakest_weight = abs(edge.weight)

        if weakest is not None and abs(new_edge.weight) > weakest_weight:
            # Replace weakest
            graph._remove_edge(weakest.source, weakest.target)
        else:
            # New edge is weaker than all existing — don't add
            return False

    graph.add_edge(new_edge)
    return True


def classify_tier(weight: float, config: SynaptogenesisConfig | None = None) -> str:
    """Classify an edge weight into a tier."""
    config = config or SynaptogenesisConfig()
    if weight < 0:
        return "inhibitory"
    return classify_edge_tier(
        weight,
        dormant_threshold=config.dormant_threshold,
        reflex_threshold=config.reflex_threshold,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def edge_tier_stats(graph: Graph, config: SynaptogenesisConfig | None = None) -> dict[str, int]:
    """Count edges in each tier."""
    config = config or SynaptogenesisConfig()
    stats = {"reflex": 0, "habitual": 0, "dormant": 0, "inhibitory": 0}
    for edge in graph.edges():
        tier = classify_tier(edge.weight, config)
        stats[tier] += 1
    return stats
