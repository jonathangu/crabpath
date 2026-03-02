"""Brain profile config loader with strict validation and env overrides."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .protocol import parse_bool, parse_float, parse_int, parse_route_mode


class BrainProfileError(ValueError):
    """Raised for invalid BrainProfile config content."""


@dataclass(frozen=True)
class ProfilePaths:
    state_path: str | None = None
    journal_path: str | None = None


@dataclass(frozen=True)
class ProfilePolicy:
    max_prompt_context_chars: int = 30000
    max_fired_nodes: int = 30
    route_mode: str = "off"
    route_top_k: int = 5
    route_alpha_sim: float = 0.5
    route_use_relevance: bool = True


@dataclass(frozen=True)
class ProfileReward:
    source: str = "explicit"
    weight_correction: float = -1.0
    weight_teaching: float = 0.5
    weight_directive: float = 0.75
    weight_reinforcement: float = 1.0


@dataclass(frozen=True)
class ProfileEmbedder:
    embed_model: str = "auto"


@dataclass(frozen=True)
class BrainProfile:
    """Config profile for daemon/socket defaults and policy scaffolding."""

    paths: ProfilePaths = field(default_factory=ProfilePaths)
    policy: ProfilePolicy = field(default_factory=ProfilePolicy)
    reward: ProfileReward = field(default_factory=ProfileReward)
    embedder: ProfileEmbedder = field(default_factory=ProfileEmbedder)

    @classmethod
    def load(cls, path: str | None, *, env_prefix: str = "OPENCLAWBRAIN_") -> "BrainProfile":
        """Load profile from JSON file plus optional env overrides."""
        payload: dict[str, object] = {}
        if path is not None:
            profile_path = Path(path).expanduser()
            try:
                raw = json.loads(profile_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise BrainProfileError(f"profile not found: {profile_path}") from exc
            except json.JSONDecodeError as exc:
                raise BrainProfileError(f"invalid profile JSON at {profile_path}: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise BrainProfileError("profile root must be a JSON object")
            payload = dict(raw)

        env_patch = _env_overrides(env_prefix=env_prefix)
        merged = _deep_merge(payload, env_patch)
        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BrainProfile":
        """Build validated profile from a nested dict."""
        try:
            _assert_allowed_keys(payload, {"paths", "policy", "reward", "embedder"}, "profile")
            raw_paths = _require_obj(payload.get("paths"), "paths")
            raw_policy = _require_obj(payload.get("policy"), "policy")
            raw_reward = _require_obj(payload.get("reward"), "reward")
            raw_embedder = _require_obj(payload.get("embedder"), "embedder")

            _assert_allowed_keys(raw_paths, {"state_path", "journal_path"}, "paths")
            _assert_allowed_keys(
                raw_policy,
                {
                    "max_prompt_context_chars",
                    "max_fired_nodes",
                    "route_mode",
                    "route_top_k",
                    "route_alpha_sim",
                    "route_use_relevance",
                },
                "policy",
            )
            _assert_allowed_keys(
                raw_reward,
                {
                    "source",
                    "weight_correction",
                    "weight_teaching",
                    "weight_directive",
                    "weight_reinforcement",
                },
                "reward",
            )
            _assert_allowed_keys(raw_embedder, {"embed_model"}, "embedder")

            state_path = _parse_optional_non_empty_str(raw_paths.get("state_path"), "paths.state_path")
            journal_path = _parse_optional_non_empty_str(raw_paths.get("journal_path"), "paths.journal_path")

            max_prompt_context_chars = parse_int(
                raw_policy.get("max_prompt_context_chars"),
                "policy.max_prompt_context_chars",
                default=30000,
            )
            max_fired_nodes = parse_int(
                raw_policy.get("max_fired_nodes"),
                "policy.max_fired_nodes",
                default=30,
            )
            route_mode = parse_route_mode(raw_policy.get("route_mode"))
            route_top_k = parse_int(
                raw_policy.get("route_top_k"),
                "policy.route_top_k",
                default=5,
            )
            route_alpha_sim = parse_float(
                raw_policy.get("route_alpha_sim"),
                "policy.route_alpha_sim",
                default=0.5,
            )
            route_use_relevance = parse_bool(
                raw_policy.get("route_use_relevance"),
                "policy.route_use_relevance",
                default=True,
            )

            reward_source = _parse_non_empty_str(raw_reward.get("source"), "reward.source", default="explicit")
            reward_weight_correction = parse_float(
                raw_reward.get("weight_correction"),
                "reward.weight_correction",
                default=-1.0,
            )
            reward_weight_teaching = parse_float(
                raw_reward.get("weight_teaching"),
                "reward.weight_teaching",
                default=0.5,
            )
            reward_weight_directive = parse_float(
                raw_reward.get("weight_directive"),
                "reward.weight_directive",
                default=0.75,
            )
            reward_weight_reinforcement = parse_float(
                raw_reward.get("weight_reinforcement"),
                "reward.weight_reinforcement",
                default=1.0,
            )

            embed_model = _parse_non_empty_str(raw_embedder.get("embed_model"), "embedder.embed_model", default="auto")

            return cls(
                paths=ProfilePaths(
                    state_path=state_path,
                    journal_path=journal_path,
                ),
                policy=ProfilePolicy(
                    max_prompt_context_chars=max_prompt_context_chars,
                    max_fired_nodes=max_fired_nodes,
                    route_mode=route_mode,
                    route_top_k=route_top_k,
                    route_alpha_sim=route_alpha_sim,
                    route_use_relevance=route_use_relevance,
                ),
                reward=ProfileReward(
                    source=reward_source,
                    weight_correction=reward_weight_correction,
                    weight_teaching=reward_weight_teaching,
                    weight_directive=reward_weight_directive,
                    weight_reinforcement=reward_weight_reinforcement,
                ),
                embedder=ProfileEmbedder(embed_model=embed_model),
            )
        except ValueError as exc:
            raise BrainProfileError(str(exc)) from exc

    def to_dict(self) -> dict[str, object]:
        """Convert profile to a stable, JSON-serializable dict."""
        return {
            "paths": {
                "state_path": self.paths.state_path,
                "journal_path": self.paths.journal_path,
            },
            "policy": {
                "max_prompt_context_chars": self.policy.max_prompt_context_chars,
                "max_fired_nodes": self.policy.max_fired_nodes,
                "route_mode": self.policy.route_mode,
                "route_top_k": self.policy.route_top_k,
                "route_alpha_sim": self.policy.route_alpha_sim,
                "route_use_relevance": self.policy.route_use_relevance,
            },
            "reward": {
                "source": self.reward.source,
                "weight_correction": self.reward.weight_correction,
                "weight_teaching": self.reward.weight_teaching,
                "weight_directive": self.reward.weight_directive,
                "weight_reinforcement": self.reward.weight_reinforcement,
            },
            "embedder": {
                "embed_model": self.embedder.embed_model,
            },
        }

    def to_json(self) -> str:
        """Convert profile to deterministic JSON for storage/diffing."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _env_overrides(*, env_prefix: str) -> dict[str, object]:
    mapping = {
        "STATE_PATH": (("paths", "state_path"), str),
        "JOURNAL_PATH": (("paths", "journal_path"), str),
        "MAX_PROMPT_CONTEXT_CHARS": (("policy", "max_prompt_context_chars"), int),
        "MAX_FIRED_NODES": (("policy", "max_fired_nodes"), int),
        "ROUTE_MODE": (("policy", "route_mode"), str),
        "ROUTE_TOP_K": (("policy", "route_top_k"), int),
        "ROUTE_ALPHA_SIM": (("policy", "route_alpha_sim"), float),
        "ROUTE_USE_RELEVANCE": (("policy", "route_use_relevance"), _parse_env_bool),
        "REWARD_SOURCE": (("reward", "source"), str),
        "REWARD_WEIGHT_CORRECTION": (("reward", "weight_correction"), float),
        "REWARD_WEIGHT_TEACHING": (("reward", "weight_teaching"), float),
        "REWARD_WEIGHT_DIRECTIVE": (("reward", "weight_directive"), float),
        "REWARD_WEIGHT_REINFORCEMENT": (("reward", "weight_reinforcement"), float),
        "EMBED_MODEL": (("embedder", "embed_model"), str),
    }
    patch: dict[str, object] = {}
    for suffix, (path, caster) in mapping.items():
        key = f"{env_prefix}{suffix}"
        if key not in os.environ:
            continue
        raw = os.environ[key]
        try:
            value = caster(raw)
        except ValueError as exc:
            raise BrainProfileError(f"{key} is invalid: {exc}") from exc
        _set_nested_value(patch, path, value)
    return patch


def _set_nested_value(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    cursor = payload
    for segment in path[:-1]:
        existing = cursor.get(segment)
        if not isinstance(existing, dict):
            cursor[segment] = {}
        cursor = cursor[segment]  # type: ignore[assignment]
    cursor[path[-1]] = value


def _deep_merge(base: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in patch.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _require_obj(value: object, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BrainProfileError(f"{label} must be an object")
    return dict(value)


def _assert_allowed_keys(payload: dict[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise BrainProfileError(f"{label} has unknown field(s): {joined}")


def _parse_optional_non_empty_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BrainProfileError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise BrainProfileError(f"{label} must be a non-empty string when set")
    return cleaned


def _parse_non_empty_str(value: object, label: str, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise BrainProfileError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise BrainProfileError(f"{label} must be a non-empty string")
    return cleaned


def _parse_env_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError("expected true or false")
