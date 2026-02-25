from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Sequence

from .graph import Edge, Graph


@dataclass
class RewardSignal:
    episode_id: str
    final_reward: float
    step_rewards: list[float] | None = None
    outcome: str | None = None
    feedback: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class LearningConfig:
    learning_rate: float = 0.05
    discount: float = 1.0
    baseline_decay: float = 0.95
    clip_min: float = -5
    clip_max: float = 5


@dataclass
class EdgeUpdate:
    source: str
    target: str
    delta: float
    new_weight: float
    rationale: str


@dataclass
class LearningResult:
    updates: list[EdgeUpdate]
    baseline: float
    avg_reward: float


_BASELINE_STATE: dict[str, float] = {}


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _step_get(step: object, key: str) -> Any:
    if isinstance(step, dict):
        if key not in step:
            raise KeyError(key)
        return step[key]
    return getattr(step, key)


def _extract_reward(reward: RewardSignal | float | int) -> float:
    if isinstance(reward, RewardSignal):
        return _as_float(reward.final_reward)
    return _as_float(reward)


def _as_candidates(candidates: Any) -> list[tuple[str, float]]:
    if isinstance(candidates, dict):
        return [(str(k), _as_float(v)) for k, v in candidates.items()]
    if candidates is None:
        return []

    pairs: list[tuple[str, float]] = []
    for item in candidates:
        if isinstance(item, tuple) and len(item) == 2:
            pairs.append((str(item[0]), _as_float(item[1])))
            continue
        if isinstance(item, Edge):
            pairs.append((item.target, _as_float(item.weight)))
            continue
        if hasattr(item, "target") and hasattr(item, "weight"):
            pairs.append((str(item.target), _as_float(getattr(item, "weight"))))
            continue
        if hasattr(item, "to") and hasattr(item, "score"):
            pairs.append((str(item.to), _as_float(getattr(item, "score"))))
            continue
        raise TypeError("unsupported candidate entry type")
    return pairs


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values) if values else 0.0
    exps = [exp(v - max_value) for v in values]
    total = sum(exps)
    if total == 0.0:
        return [0.0 for _ in values]
    return [v / total for v in exps]


def gu_corrected_advantage(
    trajectory_steps: Sequence[Any],
    reward: RewardSignal | float | int,
    baseline: float,
    discount: float | int,
) -> list[float]:
    """Compute baseline-shifted advantages for one trajectory.

    Discounting, when used, is applied by remaining steps to terminal
    (e.g. step at index 0 in a length-3 trajectory uses discount^2).
    A discount of 1.0 means no discounting and is preferred for CrabPath's
    short terminal-reward trajectories.
    """
    reward_value = _extract_reward(reward)
    reward_baselined = reward_value - _as_float(baseline)
    remaining_steps = len(trajectory_steps)
    return [
        reward_baselined * (_as_float(discount) ** (remaining_steps - index - 1))
        for index, _ in enumerate(trajectory_steps)
    ]


def policy_gradient_update(
    trajectory_steps: Sequence[Any],
    reward: RewardSignal | float | int,
    config: LearningConfig,
    baseline: float = 0.0,
) -> tuple[float, list[float]]:
    """Compute REINFORCE-style policy-loss proxy and per-step advantages.

    For short trajectories with terminal-only reward, this trajectory-summed
    REINFORCE objective is equivalent to Gu's decomposition when discount=1.0.
    """
    advantages = gu_corrected_advantage(
        trajectory_steps,
        reward,
        baseline=baseline,
        discount=config.discount,
    )
    total_loss = 0.0

    for index, step in enumerate(trajectory_steps):
        candidates = _as_candidates(_step_get(step, "candidates"))
        if not candidates:
            continue

        chosen = str(_step_get(step, "to_node"))
        probs = _softmax([weight for _, weight in candidates])
        candidate_targets = [target for target, _ in candidates]
        if chosen not in candidate_targets:
            continue
        chosen_prob = probs[candidate_targets.index(chosen)]
        if chosen_prob > 0.0:
            total_loss -= advantages[index] * log(chosen_prob)

    return total_loss, advantages


def weight_delta(
    trajectory_steps: Sequence[Any],
    advantages: Sequence[float],
    config: LearningConfig,
) -> list[tuple[str, str, float]]:
    """Turn trajectory advantage values into edge-weight deltas."""
    deltas: dict[tuple[str, str], float] = {}
    for index, step in enumerate(trajectory_steps):
        candidates = _as_candidates(_step_get(step, "candidates"))
        if not candidates:
            continue

        source = str(_step_get(step, "from_node"))
        chosen = str(_step_get(step, "to_node"))
        weights = [w for _, w in candidates]
        probs = _softmax(weights)

        candidate_map = [
            (target, probs[target_index]) for target_index, (target, _) in enumerate(candidates)
        ]

        for target, probability in candidate_map:
            baseline_grad = 1.0 if target == chosen else 0.0
            grad = baseline_grad - probability
            delta = config.learning_rate * advantages[index] * grad
            # Per-step clipping is kept for numerical safety, while deltas are still
            # accumulated per-edge to preserve trajectory-summed signal.
            if delta > config.clip_max:
                delta = config.clip_max
            if delta < config.clip_min:
                delta = config.clip_min
            deltas[(source, target)] = deltas.get((source, target), 0.0) + delta

    return [(source, target, delta) for (source, target), delta in deltas.items()]


def _set_count(edge: Edge, field: str, delta: int) -> None:
    current = getattr(edge, field, 0)
    setattr(edge, field, int(current) + delta)


def apply_weight_updates(
    graph: Graph, deltas: Sequence[tuple[object, object, object]], config: LearningConfig
) -> list[EdgeUpdate]:
    """Apply policy gradients to graph edges and score skipped candidates.

    The policy-gradient terms are reward-conditional (from REINFORCE) and update
    action propensities based on outcomes.
    Skip penalties are reward-independent structural updates that act as an
    auxiliary signal: they encourage deterministic routing (entropy reduction) by
    reinforcing co-occurrence patterns separately from reward-driven updates.
    """
    updates: list[EdgeUpdate] = []
    deltas_by_source: dict[str, set[str]] = {}
    raw_by_source: dict[tuple[str, str], float] = {}

    for source, target, delta in deltas:
        raw_by_source[(str(source), str(target))] = _as_float(delta)
        deltas_by_source.setdefault(str(source), set()).add(str(target))

    for (source, target), delta in raw_by_source.items():
        edge = graph.get_edge(source, target)
        if edge is None:
            continue

        old_weight = edge.weight
        new_weight = old_weight + delta
        if new_weight > config.clip_max:
            new_weight = config.clip_max
        if new_weight < config.clip_min:
            new_weight = config.clip_min
        edge.weight = new_weight
        _set_count(edge, "follow_count", 1)

        updates.append(
            EdgeUpdate(
                source=source,
                target=target,
                delta=new_weight - old_weight,
                new_weight=new_weight,
                rationale="policy-gradient update",
            )
        )

    for source, updated_targets in deltas_by_source.items():
        for target, edge in graph.outgoing(source):
            if target.id in updated_targets:
                continue
            _set_count(edge, "skip_count", 1)

    return updates


def make_learning_step(
    graph: Graph,
    trajectory_steps: Sequence[Any],
    reward: RewardSignal,
    config: LearningConfig,
) -> LearningResult:
    """Run a learning step end-to-end and return all weight updates."""
    # episode_id is intended to represent a recurring query family/group so that
    # baselines are shared across similar trajectories; per-query unique IDs
    # reduce baseline effectiveness.
    prev_baseline = _BASELINE_STATE.get(reward.episode_id, 0.0)
    loss, advantages = policy_gradient_update(
        trajectory_steps,
        reward,
        config,
        baseline=prev_baseline,
    )
    deltas = weight_delta(trajectory_steps, advantages, config)
    updates = apply_weight_updates(graph, deltas, config)
    final_reward = _extract_reward(reward)
    updated_baseline = (
        config.baseline_decay * prev_baseline + (1.0 - config.baseline_decay) * final_reward
    )
    _BASELINE_STATE[reward.episode_id] = updated_baseline

    return LearningResult(updates=updates, baseline=updated_baseline, avg_reward=final_reward)
