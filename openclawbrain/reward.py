"""Reward source taxonomy and weighting helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

DEFAULT_REWARD_WEIGHTS_ENV = "OPENCLAWBRAIN_REWARD_WEIGHTS"


class RewardSource(str, Enum):
    """Origin of a supervision reward signal."""

    HUMAN = "human"
    SELF = "self"
    HARVESTER = "harvester"
    TEACHER = "teacher"

    @classmethod
    def parse(cls, value: object, *, default: "RewardSource" = TEACHER) -> "RewardSource":
        """Parse a reward source from string/enum with a stable default."""
        if value is None:
            return default
        if isinstance(value, RewardSource):
            return value
        if not isinstance(value, str):
            raise ValueError("reward_source must be one of: human, self, harvester, teacher")
        normalized = value.strip().lower()
        for source in cls:
            if source.value == normalized:
                return source
        raise ValueError("reward_source must be one of: human, self, harvester, teacher")


@dataclass(frozen=True)
class RewardWeights:
    """Scalar multipliers per reward source."""

    human: float = 1.0
    self: float = 0.6
    harvester: float = 0.3
    teacher: float = 0.1

    def for_source(self, source: RewardSource) -> float:
        """Return the configured multiplier for one source."""
        if source == RewardSource.HUMAN:
            return float(self.human)
        if source == RewardSource.SELF:
            return float(self.self)
        if source == RewardSource.HARVESTER:
            return float(self.harvester)
        return float(self.teacher)

    @classmethod
    def from_string(cls, value: str) -> "RewardWeights":
        """Parse a comma-separated key=value string.

        Example: ``human=1.0,self=0.7,harvester=0.4,teacher=0.2``
        """
        raw = (value or "").strip()
        if not raw:
            return cls()

        values: dict[str, float] = {
            "human": cls().human,
            "self": cls().self,
            "harvester": cls().harvester,
            "teacher": cls().teacher,
        }
        for chunk in raw.split(","):
            item = chunk.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"invalid reward weights entry: {item!r}")
            key_raw, val_raw = item.split("=", 1)
            key = key_raw.strip().lower()
            if key not in values:
                raise ValueError(f"unknown reward weight key: {key!r}")
            try:
                parsed = float(val_raw.strip())
            except ValueError as exc:
                raise ValueError(f"invalid reward weight value for {key!r}: {val_raw!r}") from exc
            values[key] = parsed
        return cls(
            human=values["human"],
            self=values["self"],
            harvester=values["harvester"],
            teacher=values["teacher"],
        )

    @classmethod
    def from_env(cls, env_var: str = DEFAULT_REWARD_WEIGHTS_ENV) -> "RewardWeights":
        """Parse weights from env var; defaults when unset/blank."""
        raw = os.environ.get(env_var, "")
        if not raw.strip():
            return cls()
        return cls.from_string(raw)


def scale_reward(outcome: float, source: RewardSource, weights: RewardWeights) -> float:
    """Scale one reward value by source weight."""
    return float(outcome) * weights.for_source(source)
