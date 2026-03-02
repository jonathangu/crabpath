from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclawbrain.profile import BrainProfile, BrainProfileError


def test_brain_profile_loads_from_json(tmp_path: Path) -> None:
    profile_path = tmp_path / "brainprofile.json"
    profile_path.write_text(
        json.dumps(
            {
                "paths": {
                    "state_path": str(tmp_path / "state.json"),
                    "journal_path": str(tmp_path / "journal.jsonl"),
                },
                "policy": {
                    "max_prompt_context_chars": 22222,
                    "max_fired_nodes": 44,
                    "route_mode": "edge+sim",
                    "route_top_k": 9,
                    "route_alpha_sim": 0.7,
                    "route_use_relevance": False,
                },
                "reward": {
                    "source": "explicit",
                    "weight_correction": -0.9,
                    "weight_teaching": 0.3,
                    "weight_directive": 0.6,
                    "weight_reinforcement": 1.3,
                },
                "embedder": {"embed_model": "text-embedding-3-small"},
            }
        ),
        encoding="utf-8",
    )

    profile = BrainProfile.load(str(profile_path))
    assert profile.paths.state_path == str(tmp_path / "state.json")
    assert profile.paths.journal_path == str(tmp_path / "journal.jsonl")
    assert profile.policy.max_prompt_context_chars == 22222
    assert profile.policy.max_fired_nodes == 44
    assert profile.policy.route_mode == "edge+sim"
    assert profile.policy.route_top_k == 9
    assert profile.policy.route_alpha_sim == 0.7
    assert profile.policy.route_use_relevance is False
    assert profile.reward.source == "explicit"
    assert profile.reward.weight_correction == -0.9
    assert profile.reward.weight_teaching == 0.3
    assert profile.reward.weight_directive == 0.6
    assert profile.reward.weight_reinforcement == 1.3
    assert profile.embedder.embed_model == "text-embedding-3-small"
    assert json.loads(profile.to_json()) == profile.to_dict()


def test_brain_profile_env_overrides(monkeypatch, tmp_path: Path) -> None:
    profile_path = tmp_path / "brainprofile.json"
    profile_path.write_text(
        json.dumps(
            {
                "policy": {"route_mode": "off", "max_fired_nodes": 30},
                "embedder": {"embed_model": "auto"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAWBRAIN_ROUTE_MODE", "edge")
    monkeypatch.setenv("OPENCLAWBRAIN_MAX_FIRED_NODES", "88")
    monkeypatch.setenv("OPENCLAWBRAIN_EMBED_MODEL", "hash")

    profile = BrainProfile.load(str(profile_path))
    assert profile.policy.route_mode == "edge"
    assert profile.policy.max_fired_nodes == 88
    assert profile.embedder.embed_model == "hash"


def test_brain_profile_validation_fails_on_unknown_fields(tmp_path: Path) -> None:
    profile_path = tmp_path / "brainprofile.json"
    profile_path.write_text(json.dumps({"policy": {"unexpected": 1}}), encoding="utf-8")

    with pytest.raises(BrainProfileError, match="unknown field"):
        BrainProfile.load(str(profile_path))


def test_brain_profile_validation_fails_on_bad_values(tmp_path: Path) -> None:
    profile_path = tmp_path / "brainprofile.json"
    profile_path.write_text(
        json.dumps(
            {
                "policy": {"route_mode": "invalid-mode"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BrainProfileError, match="route_mode must be one of"):
        BrainProfile.load(str(profile_path))
