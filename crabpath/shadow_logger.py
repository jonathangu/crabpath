from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _get_node_id(node: Any) -> str:
    if isinstance(node, str):
        return node
    return _coerce_to_str(getattr(node, "id", node))


def _get_node_content(node: Any) -> str:
    if isinstance(node, str):
        return ""
    return _coerce_to_str(getattr(node, "content", ""))


def _node_snippet(node: Any, max_len: int = 50) -> str:
    return _coerce_to_str(_get_node_content(node))[:max_len]


def _coerce_trajectory_edges(trajectory: Any) -> list[tuple[str, str]]:
    if trajectory is None:
        return []

    if hasattr(trajectory, "steps"):
        candidate_steps = list(getattr(trajectory, "steps"))
    elif isinstance(trajectory, Iterable) and not isinstance(trajectory, (str, bytes)):
        candidate_steps = list(trajectory)
    else:
        return []

    edges: list[list[str]] = []
    for step in candidate_steps:
        if hasattr(step, "from_node") and hasattr(step, "to_node"):
            edges.append([str(step.from_node), str(step.to_node)])
            continue
        if isinstance(step, dict):
            source = step.get("from_node")
            target = step.get("to_node")
        elif isinstance(step, (tuple, list)) and len(step) >= 2:
            source, target = step[0], step[1]
        else:
            continue
        if source is not None and target is not None:
            edges.append([str(source), str(target)])
    return edges


def _coerce_tier_snapshot(tiers: Any) -> dict[str, int]:
    if not isinstance(tiers, dict):
            return {"reflex": 0, "habitual": 0, "dormant": 0, "inhibitory": 0}
    return {
        "reflex": _safe_int(tiers.get("reflex"), 0),
        "habitual": _safe_int(tiers.get("habitual"), 0),
        "dormant": _safe_int(tiers.get("dormant"), 0),
        "inhibitory": _safe_int(tiers.get("inhibitory"), 0),
    }


def _coerce_tiers_for_reward_source(scores: Any, reward: Any) -> str:
    if reward is None:
        return "none"
    if isinstance(scores, dict) and (
        isinstance(scores.get("scores"), dict) or isinstance(scores.get("overall"), (int, float))
    ):
        return "scoring"
    return "correction"


@dataclass(frozen=True)
class _ShadowLogRecord:
    event: str
    timestamp: float


class ShadowLog:
    """Write per-query and system telemetry into JSONL for shadow-mode analysis."""

    def __init__(self, path: str | Path = "~/.crabpath/shadow.jsonl") -> None:
        self.path = Path(path).expanduser()

    def log_query(
        self,
        query: str,
        selected_nodes: list[Any],
        scores: dict[str, Any] | None,
        reward: float | None,
        trajectory: Any,
        tiers: dict[str, Any] | None,
    ) -> dict[str, Any]:
        selected_node_ids = [_get_node_id(node) for node in (selected_nodes or [])]
        selected_nodes_with_snippets = [
            {"id": node_id, "snippet": _node_snippet(node)}
            for node_id, node in ((_get_node_id(node), node) for node in (selected_nodes or []))
        ]

        tier_snapshot = _coerce_tier_snapshot(tiers)
        proto_edge_count = 0
        if isinstance(tiers, dict):
            proto_edge_count = _safe_int(
                tiers.get("proto_edge_count", tiers.get("proto_edges", 0)),
                0,
            )

        reward_source = _coerce_tiers_for_reward_source(scores, reward)
        record: dict[str, Any] = {
            "event": "query",
            "timestamp": time.time(),
            "query": _coerce_to_str(query)[:100],
            "selected_node_ids": selected_node_ids,
            "selected_node_snippets": selected_nodes_with_snippets,
            "retrieval_scores": scores or {},
            "reward": _safe_float(reward, None),
            "reward_source": reward_source,
            "trajectory_edges": _coerce_trajectory_edges(trajectory),
            "tier_snapshot": tier_snapshot,
            "proto_edge_count": proto_edge_count,
        }
        self._append(record)
        return record

    def log_health(self, health_metrics: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "health",
            "timestamp": time.time(),
            "health_metrics": dict(health_metrics),
        }
        self._append(record)
        return record

    def log_tune(self, adjustments: Any, changes: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "tune",
            "timestamp": time.time(),
            "adjustments": adjustments,
            "changes": changes,
        }
        self._append(record)
        return record

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        records: list[dict[str, Any]] = []
        for raw in lines[-n:]:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records

    def summary(self, last_n: int = 50) -> dict[str, Any]:
        query_records = [entry for entry in self.tail(last_n) if entry.get("event") == "query"]
        if not query_records:
            return {
                "queries": 0,
                "avg_selected": 0.0,
                "avg_reward": 0.0,
                "tier_trends": {"reflex": 0.0, "habitual": 0.0, "dormant": 0.0, "inhibitory": 0.0},
            }

        total_selected = sum(len(entry.get("selected_node_ids", [])) for entry in query_records)
        rewards = [
            _safe_float(entry.get("reward"), None)
            for entry in query_records
            if isinstance(entry.get("reward"), (int, float))
        ]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        tier_totals = {"reflex": 0, "habitual": 0, "dormant": 0, "inhibitory": 0}
        for entry in query_records:
            snapshot = _coerce_tier_snapshot(entry.get("tier_snapshot"))
            for key, value in snapshot.items():
                tier_totals[key] += _safe_int(value)

        count = len(query_records)
        tier_trends = {
            key: tier_totals[key] / count for key in tier_totals
        }
        return {
            "queries": count,
            "avg_selected": total_selected / count,
            "avg_reward": avg_reward,
            "tier_trends": tier_trends,
        }

    def _append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload))
            stream.write("\n")
