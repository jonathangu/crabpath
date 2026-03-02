from __future__ import annotations

from openclawbrain.reward import RewardSource, RewardWeights, scale_reward


def test_reward_weights_defaults_and_scale() -> None:
    weights = RewardWeights()
    assert weights.human == 1.0
    assert weights.self == 0.6
    assert weights.harvester == 0.3
    assert weights.teacher == 0.1
    assert scale_reward(2.0, RewardSource.TEACHER, weights) == 0.2
    assert scale_reward(-1.0, RewardSource.HUMAN, weights) == -1.0


def test_reward_weights_parse_string() -> None:
    weights = RewardWeights.from_string("human=0.9,self=0.5,harvester=0.2,teacher=0.05")
    assert weights == RewardWeights(human=0.9, self=0.5, harvester=0.2, teacher=0.05)
