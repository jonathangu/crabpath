"""First-class route trace and decision-point schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .reward import RewardSource


@dataclass(frozen=True)
class RouteCandidate:
    """One possible route action at a decision point."""

    target_id: str
    edge_weight: float
    edge_relevance: float
    similarity: float | None = None
    target_preview: str = ""
    target_file: str | None = None
    target_authority: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "edge_weight": float(self.edge_weight),
            "edge_relevance": float(self.edge_relevance),
            "similarity": None if self.similarity is None else float(self.similarity),
            "target_preview": self.target_preview,
            "target_file": self.target_file,
            "target_authority": self.target_authority,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteCandidate":
        return cls(
            target_id=str(payload.get("target_id", "")),
            edge_weight=float(payload.get("edge_weight", 0.0)),
            edge_relevance=float(payload.get("edge_relevance", 0.0)),
            similarity=float(payload["similarity"]) if payload.get("similarity") is not None else None,
            target_preview=str(payload.get("target_preview", "")),
            target_file=str(payload["target_file"]) if payload.get("target_file") is not None else None,
            target_authority=str(payload["target_authority"]) if payload.get("target_authority") is not None else None,
        )


def _candidate_sort_key(candidate: RouteCandidate) -> tuple[float, str]:
    return (-float(candidate.edge_weight), str(candidate.target_id))


@dataclass(frozen=True)
class RouteDecisionPoint:
    """One routed step with candidate actions and optional teacher labels."""

    query_text: str
    source_id: str
    source_preview: str
    chosen_target_id: str = ""
    candidates: list[RouteCandidate] = field(default_factory=list)
    teacher_choose: list[str] = field(default_factory=list)
    teacher_scores: dict[str, float] = field(default_factory=dict)
    ts: float = 0.0
    reward_source: RewardSource = RewardSource.TEACHER

    def sorted_candidates(self) -> list[RouteCandidate]:
        """Deterministic candidate order: edge_weight desc, target_id asc."""
        return sorted(self.candidates, key=_candidate_sort_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "source_id": self.source_id,
            "source_preview": self.source_preview,
            "chosen_target_id": self.chosen_target_id,
            "candidates": [item.to_dict() for item in self.sorted_candidates()],
            "teacher_choose": sorted({str(item) for item in self.teacher_choose}),
            "teacher_scores": {k: float(v) for k, v in sorted(self.teacher_scores.items())},
            "ts": float(self.ts),
            "reward_source": self.reward_source.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteDecisionPoint":
        raw_source = payload.get("reward_source")
        reward_source = RewardSource.parse(raw_source, default=RewardSource.TEACHER)
        raw_candidates = payload.get("candidates")
        candidates: list[RouteCandidate] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if isinstance(item, dict):
                    candidates.append(RouteCandidate.from_dict(item))
        raw_scores = payload.get("teacher_scores")
        teacher_scores: dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                try:
                    teacher_scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        raw_choose = payload.get("teacher_choose")
        teacher_choose = [str(item) for item in raw_choose] if isinstance(raw_choose, list) else []
        return cls(
            query_text=str(payload.get("query_text", "")),
            source_id=str(payload.get("source_id", "")),
            source_preview=str(payload.get("source_preview", "")),
            chosen_target_id=str(payload.get("chosen_target_id", "")),
            candidates=candidates,
            teacher_choose=teacher_choose,
            teacher_scores=teacher_scores,
            ts=float(payload.get("ts", 0.0)),
            reward_source=reward_source,
        )


@dataclass(frozen=True)
class RouteTrace:
    """Replayable route trace for one query event."""

    query_id: str
    ts: float
    query_text: str
    seeds: list[list[Any]] = field(default_factory=list)
    fired_nodes: list[str] = field(default_factory=list)
    traversal_config: dict[str, Any] = field(default_factory=dict)
    route_policy: dict[str, Any] = field(default_factory=dict)
    chat_id: str | None = None
    query_vector: list[float] | None = None
    decision_points: list[RouteDecisionPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "ts": float(self.ts),
            "chat_id": self.chat_id,
            "query_text": self.query_text,
            "seeds": list(self.seeds),
            "fired_nodes": list(self.fired_nodes),
            "traversal_config": dict(self.traversal_config),
            "route_policy": dict(self.route_policy),
            "query_vector": list(self.query_vector) if self.query_vector is not None else None,
            "decision_points": [item.to_dict() for item in self.decision_points],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteTrace":
        raw_decision_points = payload.get("decision_points")
        decision_points: list[RouteDecisionPoint] = []
        if isinstance(raw_decision_points, list):
            for item in raw_decision_points:
                if isinstance(item, dict):
                    decision_points.append(RouteDecisionPoint.from_dict(item))
        raw_seeds = payload.get("seeds")
        seeds = raw_seeds if isinstance(raw_seeds, list) else []
        return cls(
            query_id=str(payload.get("query_id", "")),
            ts=float(payload.get("ts", 0.0)),
            chat_id=str(payload["chat_id"]) if payload.get("chat_id") is not None else None,
            query_text=str(payload.get("query_text", "")),
            seeds=list(seeds),
            fired_nodes=[str(item) for item in payload.get("fired_nodes", [])] if isinstance(payload.get("fired_nodes"), list) else [],
            traversal_config=dict(payload.get("traversal_config", {})) if isinstance(payload.get("traversal_config"), dict) else {},
            route_policy=dict(payload.get("route_policy", {})) if isinstance(payload.get("route_policy"), dict) else {},
            query_vector=[float(item) for item in payload.get("query_vector", [])]
            if isinstance(payload.get("query_vector"), list)
            else None,
            decision_points=decision_points,
        )


def route_trace_to_json(trace: RouteTrace) -> str:
    """Serialize one trace deterministically."""
    return json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def route_trace_from_json(raw: str) -> RouteTrace:
    """Parse one trace line from JSON."""
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("trace JSON payload must be an object")
    return RouteTrace.from_dict(payload)
