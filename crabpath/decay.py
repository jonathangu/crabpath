from __future__ import annotations

from dataclasses import dataclass

from .graph import Graph


@dataclass
class DecayConfig:
    half_life_turns: int = 120
    min_weight: float = -5.0
    max_weight: float = 5.0


def decay_factor(half_life_turns, elapsed_turns) -> float:
    if half_life_turns <= 0:
        return 0.0
    if elapsed_turns <= 0:
        return 1.0
    return 2 ** (-_as_float(elapsed_turns) / _as_float(half_life_turns))


def decay_weight(weight, elapsed_turns, config) -> float:
    decayed = _as_float(weight) * decay_factor(config.half_life_turns, elapsed_turns)
    if decayed < config.min_weight:
        decayed = config.min_weight
    if decayed > config.max_weight:
        decayed = config.max_weight
    return decayed


def _as_float(value: float | int) -> float:
    return float(value)


def apply_decay(graph: Graph, turns_elapsed, config: DecayConfig | None = None) -> dict[str, float]:
    if config is None:
        config = DecayConfig()

    factor = decay_factor(config.half_life_turns, turns_elapsed)
    if factor == 1.0:
        return {}

    changed = {}
    for edge in graph.edges():
        decayed = decay_weight(edge.weight, turns_elapsed, config)
        if decayed != edge.weight:
            edge.weight = decayed
            changed[f"{edge.source}->{edge.target}"] = decayed

    return changed
