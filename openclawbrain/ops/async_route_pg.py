"""Background teacher-routing PG updates over recent query journal events."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

from ..graph import Edge, Graph
from ..journal import read_journal
from ..learn import apply_outcome_pg
from ..replay import default_keyword_seed_fn
from ..store import load_state, save_state
from ..traverse import TraversalConfig, traverse
from .._util import _extract_json


TEACHER_SYSTEM_PROMPT = (
    "You are a routing teacher for a memory graph traversal policy.\n"
    "Given the user query, source node context, and candidate targets, return JSON only.\n"
    'Allowed JSON forms: {"choose": ["target_id", ...]} and/or {"scores": {"target_id": -1.0..1.0}}.\n'
    "Prefer sparse useful supervision. Use IDs exactly as given."
)


@dataclass
class DecisionPoint:
    """One supervised routing decision from traversal replay."""

    query: str
    source_id: str
    chosen_target_id: str
    candidates: list[dict[str, object]]


@dataclass
class AsyncRoutePgSummary:
    """Structured summary for CLI output."""

    teacher_requested: str
    teacher_available: bool
    teacher_model: str
    sampled_queries: int
    decision_points_total: int
    decision_points_labeled: int
    labeled_edges: int
    updates_applied: int
    total_abs_weight_delta: float
    max_abs_weight_delta: float
    dry_run: bool
    max_decision_points_hit: bool
    score_scale: float
    state_path: str
    journal_path: str
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "teacher_requested": self.teacher_requested,
            "teacher_available": self.teacher_available,
            "teacher_model": self.teacher_model,
            "sampled_queries": self.sampled_queries,
            "decision_points_total": self.decision_points_total,
            "decision_points_labeled": self.decision_points_labeled,
            "labeled_edges": self.labeled_edges,
            "updates_applied": self.updates_applied,
            "total_abs_weight_delta": self.total_abs_weight_delta,
            "max_abs_weight_delta": self.max_abs_weight_delta,
            "dry_run": self.dry_run,
            "max_decision_points_hit": self.max_decision_points_hit,
            "score_scale": self.score_scale,
            "state_path": self.state_path,
            "journal_path": self.journal_path,
            "errors": list(self.errors),
        }


def parse_teacher_route_labels(raw: str, valid_target_ids: set[str]) -> dict[str, float]:
    """Parse model output supporting choose-only, scores-only, or both."""
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return {}

    labels: dict[str, float] = {}
    raw_scores = parsed.get("scores")
    has_scores = isinstance(raw_scores, dict)
    if has_scores:
        for target_id, value in raw_scores.items():
            target = str(target_id)
            if target not in valid_target_ids:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            labels[target] = max(-1.0, min(1.0, score))

    raw_choose = parsed.get("choose")
    if isinstance(raw_choose, list):
        choose_ids = [str(item) for item in raw_choose if str(item) in valid_target_ids]
        if has_scores:
            for target in choose_ids:
                labels.setdefault(target, 1.0)
        else:
            for target in choose_ids:
                labels[target] = 1.0
    return labels


def _preview(text: str, limit: int = 180) -> str:
    value = " ".join((text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _teacher_user_prompt(point: DecisionPoint) -> str:
    payload = {
        "query": point.query,
        "source_id": point.source_id,
        "source_preview": _preview(point.candidates[0].get("source_preview", "")) if point.candidates else "",
        "chosen_target_id_runtime": point.chosen_target_id,
        "candidates": point.candidates,
        "response_schema": {
            "choose": ["target_id"],
            "scores": {"target_id": "number in [-1,1]"},
        },
    }
    return json.dumps(payload, ensure_ascii=True)


def _candidate_sort_key(edge: Edge) -> tuple[float, float, str]:
    return (abs(edge.weight), edge.weight, edge.target)


def _decision_points_for_query(
    graph: Graph,
    query: str,
    *,
    max_candidates_per_node: int,
) -> list[DecisionPoint]:
    seeds = default_keyword_seed_fn(graph, query)
    if not seeds:
        return []
    result = traverse(
        graph=graph,
        seeds=seeds,
        config=TraversalConfig(),
        query_text=query,
    )
    if not result.steps:
        return []

    points: list[DecisionPoint] = []
    for step in result.steps:
        source = graph.get_node(step.from_node)
        if source is None:
            continue
        outgoing = graph._edges.get(step.from_node, {})
        habitual_edges = [edge for edge in outgoing.values() if 0.15 <= edge.weight < 0.6]
        reflex_edges = [edge for edge in outgoing.values() if edge.weight >= 0.6]

        candidates = sorted(habitual_edges, key=_candidate_sort_key, reverse=True)
        if len(candidates) < max_candidates_per_node:
            for edge in sorted(reflex_edges, key=_candidate_sort_key, reverse=True):
                if edge.target not in {item.target for item in candidates}:
                    candidates.append(edge)
                if len(candidates) >= max_candidates_per_node:
                    break
        candidates = candidates[:max_candidates_per_node]
        if not candidates:
            continue

        candidate_payload: list[dict[str, object]] = []
        for edge in candidates:
            target_node = graph.get_node(edge.target)
            if target_node is None:
                continue
            metadata = target_node.metadata if isinstance(target_node.metadata, dict) else {}
            candidate_payload.append(
                {
                    "source_preview": _preview(source.content),
                    "target_id": edge.target,
                    "edge_weight": edge.weight,
                    "target_preview": _preview(target_node.content),
                    "file": metadata.get("file"),
                    "authority": metadata.get("authority"),
                }
            )
        if not candidate_payload:
            continue

        points.append(
            DecisionPoint(
                query=query,
                source_id=step.from_node,
                chosen_target_id=step.to_node,
                candidates=candidate_payload,
            )
        )
    return points


def _sample_queries(
    journal_path: str,
    *,
    since_hours: float,
    max_queries: int,
    sample_rate: float,
) -> list[str]:
    entries = read_journal(journal_path=journal_path)
    cutoff = time.time() - max(0.0, since_hours) * 3600.0
    raw_queries: list[str] = []
    for entry in entries:
        if entry.get("type") != "query":
            continue
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and float(ts) < cutoff:
            continue
        raw_queries.append(query.strip())

    rng = random.Random(0)
    sampled: list[str] = []
    for query in raw_queries:
        if len(sampled) >= max_queries:
            break
        if rng.random() <= max(0.0, min(1.0, sample_rate)):
            sampled.append(query)
    return sampled


def _teacher_labels_openai(
    decision_points: list[DecisionPoint],
    *,
    teacher_model: str,
) -> tuple[list[dict[str, float]], list[str]]:
    errors: list[str] = []
    if not decision_points:
        return [], errors

    from ..openai_llm import openai_llm_batch_fn

    requests = [
        {
            "id": idx,
            "model": teacher_model,
            "system": TEACHER_SYSTEM_PROMPT,
            "user": _teacher_user_prompt(point),
        }
        for idx, point in enumerate(decision_points)
    ]
    responses = openai_llm_batch_fn(requests)
    by_id: dict[int, dict] = {}
    for row in responses:
        if not isinstance(row, dict):
            continue
        request_id = row.get("id")
        if isinstance(request_id, int):
            by_id[request_id] = row

    labels_per_point: list[dict[str, float]] = []
    for idx, point in enumerate(decision_points):
        response = by_id.get(idx, {})
        raw = response.get("response")
        if not isinstance(raw, str):
            labels_per_point.append({})
            err = response.get("error")
            if isinstance(err, str) and err:
                errors.append(err)
            continue
        valid_ids = {str(item["target_id"]) for item in point.candidates if "target_id" in item}
        labels_per_point.append(parse_teacher_route_labels(raw, valid_ids))
        err = response.get("error")
        if isinstance(err, str) and err:
            errors.append(err)
    return labels_per_point, errors


def run_async_route_pg(
    *,
    state_path: str,
    journal_path: str,
    since_hours: float = 24.0,
    max_queries: int = 200,
    sample_rate: float = 0.1,
    max_candidates_per_node: int = 12,
    max_decision_points: int = 500,
    teacher: str = "openai",
    teacher_model: str = "gpt-5-mini",
    apply: bool = False,
    write_relevance_metadata: bool = True,
    score_scale: float = 0.3,
    teacher_labeler: Callable[[list[DecisionPoint]], tuple[list[dict[str, float]], list[str]]] | None = None,
) -> AsyncRoutePgSummary:
    """Run teacher-shadow routing updates over recent query events."""
    graph, index, meta = load_state(state_path)
    sampled_queries = _sample_queries(
        journal_path=journal_path,
        since_hours=since_hours,
        max_queries=max_queries,
        sample_rate=sample_rate,
    )

    decision_points: list[DecisionPoint] = []
    cap_hit = False
    for query in sampled_queries:
        points = _decision_points_for_query(
            graph=graph,
            query=query,
            max_candidates_per_node=max_candidates_per_node,
        )
        for point in points:
            decision_points.append(point)
            if len(decision_points) >= max_decision_points:
                cap_hit = True
                break
        if cap_hit:
            break

    teacher_available = False
    errors: list[str] = []
    labels_by_point: list[dict[str, float]] = [{} for _ in decision_points]
    if teacher_labeler is not None:
        teacher_available = True
        labels_by_point, custom_errors = teacher_labeler(decision_points)
        errors.extend(custom_errors)
    elif teacher == "openai" and os.environ.get("OPENAI_API_KEY"):
        teacher_available = True
        try:
            labels_by_point, model_errors = _teacher_labels_openai(
                decision_points=decision_points,
                teacher_model=teacher_model,
            )
            errors.extend(model_errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            labels_by_point = [{} for _ in decision_points]
            teacher_available = False
    elif teacher == "openai":
        errors.append("OPENAI_API_KEY not set; teacher unavailable")
    else:
        errors.append("teacher disabled")

    working_graph = graph if apply else copy.deepcopy(graph)
    decision_points_labeled = 0
    labeled_edges = 0
    updates_applied = 0
    total_abs_weight_delta = 0.0
    max_abs_weight_delta = 0.0
    for point, labels in zip(decision_points, labels_by_point):
        if not labels:
            continue
        decision_points_labeled += 1
        for target_id, score in labels.items():
            if score == 0:
                continue
            source_node = working_graph.get_node(point.source_id)
            target_node = working_graph.get_node(target_id)
            if source_node is None or target_node is None:
                continue
            updates = apply_outcome_pg(
                graph=working_graph,
                fired_nodes=[point.source_id, target_id],
                outcome=score_scale * score,
            )
            update_key = f"{point.source_id}->{target_id}"
            delta = float(updates.get(update_key, 0.0))
            abs_delta = abs(delta)
            total_abs_weight_delta += abs_delta
            if abs_delta > max_abs_weight_delta:
                max_abs_weight_delta = abs_delta
            updates_applied += 1
            labeled_edges += 1
            if write_relevance_metadata:
                edge = working_graph._edges.get(point.source_id, {}).get(target_id)
                if edge is not None:
                    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
                    metadata["relevance"] = max(-1.0, min(1.0, float(score)))
                    edge.metadata = metadata
                    working_graph._edges[point.source_id][target_id] = edge

    if apply and teacher_available and updates_applied > 0:
        save_state(graph=working_graph, index=index, path=state_path, meta=meta)

    return AsyncRoutePgSummary(
        teacher_requested=teacher,
        teacher_available=teacher_available,
        teacher_model=teacher_model,
        sampled_queries=len(sampled_queries),
        decision_points_total=len(decision_points),
        decision_points_labeled=decision_points_labeled,
        labeled_edges=labeled_edges,
        updates_applied=updates_applied,
        total_abs_weight_delta=total_abs_weight_delta,
        max_abs_weight_delta=max_abs_weight_delta,
        dry_run=not apply,
        max_decision_points_hit=cap_hit,
        score_scale=score_scale,
        state_path=state_path,
        journal_path=journal_path,
        errors=errors,
    )
