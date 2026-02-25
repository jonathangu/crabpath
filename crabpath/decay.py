from __future__ import annotations

from dataclasses import dataclass

from .graph import Graph


@dataclass
class DecayConfig:
    half_life_turns: int = 80  # 80 turns to halve (sim-validated baseline)
    min_weight: float = -5.0
    max_weight: float = 5.0


def decay_factor(half_life_turns: int, elapsed_turns: int | float) -> float:
    """Return decay multiplier for a given half-life and elapsed turns."""
    if half_life_turns <= 0:
        return 0.0
    if elapsed_turns <= 0:
        return 1.0
    return 2 ** (-_as_float(elapsed_turns) / _as_float(half_life_turns))


def decay_weight(weight: float | int, elapsed_turns: int | float, config: DecayConfig) -> float:
    """Return a clamped decayed weight after the configured number of turns."""
    decayed = _as_float(weight) * decay_factor(config.half_life_turns, elapsed_turns)
    if decayed < config.min_weight:
        decayed = config.min_weight
    if decayed > config.max_weight:
        decayed = config.max_weight
    return decayed


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def apply_decay(
    graph: Graph, turns_elapsed: int | float, config: DecayConfig | None = None
) -> dict[str, float]:
    """Apply exponential decay to all non-protected edges.

    Args:
        graph: Graph to mutate in place.
        turns_elapsed: Number of turns since last decay.
        config: Optional override for decay config.

    Returns:
        Map of edge keys to new values for edges whose weight changed.
    """
    if config is None:
        config = DecayConfig()

    factor = decay_factor(config.half_life_turns, turns_elapsed)
    if factor == 1.0:
        return {}

    changed = {}
    for edge in graph.edges():
        if graph.is_node_protected(edge.source) or graph.is_node_protected(edge.target):
            continue

        decayed = decay_weight(edge.weight, turns_elapsed, config)
        if decayed != edge.weight:
            edge.weight = decayed
            changed[f"{edge.source}->{edge.target}"] = decayed

    return changed
