"""Unified label records across human/self/teacher supervision sources."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .reward import RewardSource


@dataclass(frozen=True)
class LabelRecord:
    query_id: str
    decision_point_idx: int
    candidate_scores: dict[str, float]
    reward_source: RewardSource
    weight: float
    ts: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "decision_point_idx": int(self.decision_point_idx),
            "candidate_scores": {str(k): float(v) for k, v in sorted(self.candidate_scores.items())},
            "reward_source": self.reward_source.value,
            "weight": float(self.weight),
            "ts": float(self.ts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelRecord":
        scores_raw = payload.get("candidate_scores")
        scores: dict[str, float] = {}
        if isinstance(scores_raw, dict):
            for key, value in scores_raw.items():
                try:
                    scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        return cls(
            query_id=str(payload.get("query_id", "")),
            decision_point_idx=int(payload.get("decision_point_idx", 0)),
            candidate_scores=scores,
            reward_source=RewardSource.parse(payload.get("reward_source"), default=RewardSource.TEACHER),
            weight=float(payload.get("weight", 1.0)),
            ts=float(payload.get("ts", 0.0)),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )


def from_human_correction(event: dict[str, Any]) -> LabelRecord | None:
    target_id = event.get("target_id") or event.get("correct_target_id")
    if not isinstance(target_id, str) or not target_id:
        return None
    query_id = str(event.get("query_id", event.get("chat_id", "")))
    point_idx = int(event.get("decision_point_idx", 0))
    score = float(event.get("score", 1.0))
    return LabelRecord(
        query_id=query_id,
        decision_point_idx=point_idx,
        candidate_scores={target_id: score},
        reward_source=RewardSource.HUMAN,
        weight=float(event.get("weight", 1.0)),
        ts=float(event.get("ts", time.time())),
        metadata={"kind": "human_correction", **(event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {})},
    )


def from_self_learning_event(event: dict[str, Any]) -> LabelRecord | None:
    fired = event.get("fired_ids") or event.get("fired")
    if not isinstance(fired, list) or not fired:
        return None
    candidate_scores = {
        str(node_id): float(event.get("outcome", -1.0))
        for node_id in fired
        if isinstance(node_id, str) and node_id
    }
    if not candidate_scores:
        return None
    return LabelRecord(
        query_id=str(event.get("query_id", event.get("chat_id", ""))),
        decision_point_idx=int(event.get("decision_point_idx", 0)),
        candidate_scores=candidate_scores,
        reward_source=RewardSource.SELF,
        weight=float(event.get("weight", 1.0)),
        ts=float(event.get("ts", time.time())),
        metadata={"kind": "self_learning", **(event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {})},
    )


def from_teacher_output(
    *,
    query_id: str,
    decision_point_idx: int,
    teacher_scores: dict[str, float],
    ts: float | None = None,
    weight: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> LabelRecord:
    return LabelRecord(
        query_id=query_id,
        decision_point_idx=decision_point_idx,
        candidate_scores={str(k): float(v) for k, v in teacher_scores.items()},
        reward_source=RewardSource.TEACHER,
        weight=float(weight),
        ts=float(ts if ts is not None else time.time()),
        metadata=dict(metadata or {}),
    )


def write_labels_jsonl(path: str | Path, records: list[LabelRecord]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")


def read_labels_jsonl(path: str | Path) -> list[LabelRecord]:
    source = Path(path).expanduser()
    if not source.exists():
        return []
    out: list[LabelRecord] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict):
            out.append(LabelRecord.from_dict(payload))
    return out
