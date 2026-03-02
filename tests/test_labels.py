from __future__ import annotations

from pathlib import Path

from openclawbrain.labels import (
    LabelRecord,
    from_human_correction,
    from_self_learning_event,
    from_teacher_output,
    read_labels_jsonl,
    write_labels_jsonl,
)
from openclawbrain.reward import RewardSource


def test_label_record_serialization_roundtrip(tmp_path: Path) -> None:
    record = LabelRecord(
        query_id="q1",
        decision_point_idx=2,
        candidate_scores={"a": 1.0, "b": -0.5},
        reward_source=RewardSource.HUMAN,
        weight=0.8,
        ts=123.0,
        metadata={"note": "ok"},
    )
    path = tmp_path / "labels.jsonl"
    write_labels_jsonl(path, [record])

    loaded = read_labels_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0] == record


def test_label_conversions_human_self_teacher() -> None:
    human = from_human_correction({"query_id": "q", "decision_point_idx": 1, "target_id": "n1", "ts": 1.0})
    assert human is not None
    assert human.reward_source == RewardSource.HUMAN
    assert human.candidate_scores == {"n1": 1.0}

    self_label = from_self_learning_event(
        {"query_id": "q", "decision_point_idx": 0, "fired_ids": ["n2"], "outcome": -1.0, "ts": 2.0}
    )
    assert self_label is not None
    assert self_label.reward_source == RewardSource.SELF
    assert self_label.candidate_scores == {"n2": -1.0}

    teacher = from_teacher_output(query_id="q", decision_point_idx=3, teacher_scores={"n3": 0.5}, ts=3.0)
    assert teacher.reward_source == RewardSource.TEACHER
    assert teacher.candidate_scores == {"n3": 0.5}
