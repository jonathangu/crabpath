from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_shadow_mode_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "crabpath-private"
        / "scripts"
        / "shadow_mode_v2.py"
    )
    spec = spec_from_file_location("shadow_mode_v2", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow_mode = _load_shadow_mode_module()


def test_scoring_enabled_by_default() -> None:
    parser = shadow_mode._build_parser()
    parsed = parser.parse_args(["query"])
    assert parsed.score is True


def test_no_score_flag_disables_scoring() -> None:
    parser = shadow_mode._build_parser()
    parsed = parser.parse_args(["query", "--no-score"])
    assert parsed.score is False


def test_reward_from_scoring_honor_gate() -> None:
    reward, avg = shadow_mode._derive_score_and_gate(
        learn_reward=None,
        score_enabled=True,
        retrieval_scores={"scores": {"a": 1.0}, "overall": 0.22},
    )
    assert reward == 1.0
    assert avg == 0.22


def test_reward_from_scoring_honors_helpfulness_threshold() -> None:
    reward, avg = shadow_mode._derive_score_and_gate(
        learn_reward=None,
        score_enabled=True,
        retrieval_scores={"scores": {"a": 1.0}, "overall": 0.31},
    )
    assert reward == 1.0
    assert avg == 0.31


def test_reward_from_learn_overrides_scoring_gate() -> None:
    reward, avg = shadow_mode._derive_score_and_gate(
        learn_reward=1.0,
        score_enabled=False,
        retrieval_scores=None,
    )
    assert reward == 1.0
    assert avg == 1.0
