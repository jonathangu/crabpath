"""Background teacher-routing PG updates over recent query journal events."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..graph import Edge, Graph
from ..hasher import HashEmbedder
from ..learn import apply_outcome_pg
from ..replay import default_keyword_seed_fn
from ..reward import RewardSource, RewardWeights, scale_reward
from ..storage import EventStore, JsonStateStore, JsonlEventStore, StateStore
from ..trace import RouteCandidate, RouteDecisionPoint, RouteTrace, route_trace_from_json, route_trace_to_json
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
    """Backward-compatible teacher labeling payload."""

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
    reward_source: str
    reward_weights: dict[str, float]
    traces_in: str | None
    traces_out: str | None
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
            "reward_source": self.reward_source,
            "reward_weights": dict(self.reward_weights),
            "traces_in": self.traces_in,
            "traces_out": self.traces_out,
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


def _candidate_sort_key(edge: Edge) -> tuple[float, str]:
    return (-float(edge.weight), str(edge.target))


def _decisionpoint_candidate_sort_key(item: dict[str, object]) -> tuple[float, str]:
    weight_raw = item.get("edge_weight", 0.0)
    try:
        weight = float(weight_raw)
    except (TypeError, ValueError):
        weight = 0.0
    return (-weight, str(item.get("target_id", "")))


def _teacher_user_prompt(point: DecisionPoint) -> str:
    payload = {
        "query": point.query,
        "source_id": point.source_id,
        "source_preview": _preview(point.candidates[0].get("source_preview", "")) if point.candidates else "",
        "candidates": sorted(point.candidates, key=_decisionpoint_candidate_sort_key),
        "response_schema": {
            "choose": ["target_id"],
            "scores": {"target_id": "number in [-1,1]"},
        },
    }
    if point.chosen_target_id:
        payload["chosen_target_id_runtime"] = point.chosen_target_id
    return json.dumps(payload, ensure_ascii=True)


def _read_chat_id(entry: dict[str, object]) -> str | None:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw_chat_id = metadata.get("chat_id")
    if not isinstance(raw_chat_id, str):
        return None
    value = raw_chat_id.strip()
    return value or None


def _select_query_entries(
    journal_path: str,
    *,
    since_hours: float,
    max_queries: int,
    sample_rate: float,
    event_store: EventStore | None = None,
) -> list[tuple[int, dict[str, object]]]:
    resolved_event_store = event_store or JsonlEventStore(journal_path)
    entries = resolved_event_store.iter_since(None)
    cutoff = time.time() - max(0.0, since_hours) * 3600.0
    filtered: list[tuple[int, dict[str, object]]] = []
    for idx, entry in enumerate(entries):
        if entry.get("type") != "query":
            continue
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and float(ts) < cutoff:
            continue
        filtered.append((idx, entry))

    rng = random.Random(0)
    sampled: list[tuple[int, dict[str, object]]] = []
    clamped = max(0.0, min(1.0, sample_rate))
    for idx, entry in filtered:
        if len(sampled) >= max_queries:
            break
        if rng.random() <= clamped:
            sampled.append((idx, entry))
    return sampled


def _build_traces_from_journal(
    graph: Graph,
    index,
    *,
    journal_path: str,
    since_hours: float,
    max_queries: int,
    sample_rate: float,
    max_candidates_per_node: int,
    max_decision_points: int,
    reward_source: RewardSource,
    include_query_vector: bool,
    event_store: EventStore | None = None,
) -> tuple[list[RouteTrace], bool]:
    sampled = _select_query_entries(
        journal_path=journal_path,
        since_hours=since_hours,
        max_queries=max_queries,
        sample_rate=sample_rate,
        event_store=event_store,
    )
    config = TraversalConfig()
    policy = {"route_mode": "off"}

    traces: list[RouteTrace] = []
    cap_hit = False
    total_points = 0
    hash_embedder = HashEmbedder()

    def _resolve_query_vector(seeds: list[tuple[str, float]], query: str) -> list[float] | None:
        if not include_query_vector:
            return None
        weighted: list[float] | None = None
        total_weight = 0.0
        for node_id, score in seeds:
            vector = index._vectors.get(node_id)
            if vector is None:
                continue
            weight = max(0.001, float(score))
            if weighted is None:
                weighted = [0.0] * len(vector)
            if len(weighted) != len(vector):
                continue
            for idx, value in enumerate(vector):
                weighted[idx] += float(value) * weight
            total_weight += weight
        if weighted is not None and total_weight > 0:
            return [value / total_weight for value in weighted]
        if index._vectors:
            first = next(iter(index._vectors.values()))
            guess = hash_embedder.embed(query)
            if len(first) == len(guess):
                return [float(value) for value in guess]
        return None

    for journal_idx, entry in sampled:
        query = str(entry.get("query", "")).strip()
        if not query:
            continue

        seeds = default_keyword_seed_fn(graph, query)
        if not seeds:
            trace = RouteTrace(
                query_id=f"journal:{journal_idx}",
                ts=float(entry.get("ts", 0.0) or 0.0),
                chat_id=_read_chat_id(entry),
                query_text=query,
                seeds=[],
                fired_nodes=[],
                traversal_config=asdict(config),
                route_policy=policy,
                query_vector=None,
                decision_points=[],
            )
            traces.append(trace)
            continue

        traversal = traverse(graph=graph, seeds=seeds, config=config, query_text=query)
        query_vector = _resolve_query_vector(seeds, query)
        points: list[RouteDecisionPoint] = []
        for step in traversal.steps:
            if total_points >= max_decision_points:
                cap_hit = True
                break

            source = graph.get_node(step.from_node)
            if source is None:
                continue

            outgoing = graph._edges.get(step.from_node, {})
            habitual_edges = [edge for edge in outgoing.values() if config.habitual_range[0] <= edge.weight < config.habitual_range[1]]
            reflex_edges = [edge for edge in outgoing.values() if edge.weight >= config.reflex_threshold]

            ordered: list[Edge] = sorted(habitual_edges, key=_candidate_sort_key)
            seen_targets = {edge.target for edge in ordered}
            if len(ordered) < max_candidates_per_node:
                for edge in sorted(reflex_edges, key=_candidate_sort_key):
                    if edge.target in seen_targets:
                        continue
                    ordered.append(edge)
                    seen_targets.add(edge.target)
                    if len(ordered) >= max_candidates_per_node:
                        break
            ordered = ordered[:max_candidates_per_node]
            if not ordered:
                continue

            candidates: list[RouteCandidate] = []
            chosen_edge_meta: dict[str, object] | None = None
            for edge in ordered:
                target_node = graph.get_node(edge.target)
                if target_node is None:
                    continue
                target_meta = target_node.metadata if isinstance(target_node.metadata, dict) else {}
                edge_meta = edge.metadata if isinstance(edge.metadata, dict) else {}
                if edge.target == step.to_node:
                    chosen_edge_meta = edge_meta
                raw_relevance = edge_meta.get("relevance", 0.0)
                edge_relevance = float(raw_relevance) if isinstance(raw_relevance, (int, float)) else 0.0
                candidates.append(
                    RouteCandidate(
                        target_id=str(edge.target),
                        edge_weight=float(edge.weight),
                        edge_relevance=edge_relevance,
                        similarity=None,
                        target_preview=_preview(target_node.content),
                        target_file=str(target_meta["file"]) if target_meta.get("file") is not None else None,
                        target_authority=str(target_meta["authority"]) if target_meta.get("authority") is not None else None,
                        graph_prior_score=float(edge_meta["graph_prior_score"])
                        if isinstance(edge_meta.get("graph_prior_score"), (int, float))
                        else None,
                        router_score_raw=float(edge_meta["router_score_raw"])
                        if isinstance(edge_meta.get("router_score_raw"), (int, float))
                        else None,
                        final_score=float(edge_meta["final_score"]) if isinstance(edge_meta.get("final_score"), (int, float)) else None,
                    )
                )
            if not candidates:
                continue

            points.append(
                RouteDecisionPoint(
                    query_text=query,
                    source_id=step.from_node,
                    source_preview=_preview(source.content),
                    chosen_target_id=step.to_node,
                    candidates=candidates,
                    teacher_choose=[],
                    teacher_scores={},
                    router_entropy=float(chosen_edge_meta["router_entropy"])
                    if isinstance((chosen_edge_meta or {}).get("router_entropy"), (int, float))
                    else None,
                    router_conf=float(chosen_edge_meta["router_conf"])
                    if isinstance((chosen_edge_meta or {}).get("router_conf"), (int, float))
                    else None,
                    router_margin=float(chosen_edge_meta["router_margin"])
                    if isinstance((chosen_edge_meta or {}).get("router_margin"), (int, float))
                    else None,
                    relevance_entropy=float(chosen_edge_meta["relevance_entropy"])
                    if isinstance((chosen_edge_meta or {}).get("relevance_entropy"), (int, float))
                    else None,
                    relevance_conf=float(chosen_edge_meta["relevance_conf"])
                    if isinstance((chosen_edge_meta or {}).get("relevance_conf"), (int, float))
                    else None,
                    policy_disagreement=float(chosen_edge_meta["policy_disagreement"])
                    if isinstance((chosen_edge_meta or {}).get("policy_disagreement"), (int, float))
                    else None,
                    ts=float(entry.get("ts", 0.0) or 0.0),
                    reward_source=reward_source,
                )
            )
            total_points += 1

        trace = RouteTrace(
            query_id=f"journal:{journal_idx}",
            ts=float(entry.get("ts", 0.0) or 0.0),
            chat_id=_read_chat_id(entry),
            query_text=query,
            seeds=[[str(node_id), float(score)] for node_id, score in seeds],
            fired_nodes=[str(node_id) for node_id in traversal.fired],
            traversal_config=asdict(config),
            route_policy=policy,
            query_vector=query_vector,
            decision_points=points,
        )
        traces.append(trace)
        if cap_hit:
            break

    return traces, cap_hit


def _write_traces_jsonl(path: str, traces: list[RouteTrace]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(route_trace_to_json(trace) + "\n")


def _read_traces_jsonl(path: str) -> list[RouteTrace]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"missing traces file: {source}")

    traces: list[RouteTrace] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        traces.append(route_trace_from_json(raw))
    return traces


def _flatten_traces(
    traces: list[RouteTrace],
    *,
    max_decision_points: int,
) -> tuple[list[DecisionPoint], list[RouteDecisionPoint], bool]:
    compat_points: list[DecisionPoint] = []
    route_points: list[RouteDecisionPoint] = []
    cap_hit = False

    for trace in traces:
        for idx, point in enumerate(trace.decision_points):
            if len(route_points) >= max_decision_points:
                cap_hit = True
                break
            candidate_payload = [
                {
                    "source_preview": point.source_preview,
                    "target_id": candidate.target_id,
                    "edge_weight": candidate.edge_weight,
                    "edge_relevance": candidate.edge_relevance,
                    "similarity": candidate.similarity,
                    "target_preview": candidate.target_preview,
                    "target_file": candidate.target_file,
                    "target_authority": candidate.target_authority,
                    "graph_prior_score": candidate.graph_prior_score,
                    "router_score_raw": candidate.router_score_raw,
                    "final_score": candidate.final_score,
                }
                for candidate in point.sorted_candidates()
            ]
            compat_points.append(
                DecisionPoint(
                    query=point.query_text,
                    source_id=point.source_id,
                    chosen_target_id=point.chosen_target_id,
                    candidates=candidate_payload,
                )
            )
            route_points.append(point)
        if cap_hit:
            break

    return compat_points, route_points, cap_hit


def _normalize_labels(
    labels_by_point: list[dict[str, float]],
    *,
    expected_len: int,
) -> list[dict[str, float]]:
    if len(labels_by_point) >= expected_len:
        return labels_by_point[:expected_len]
    padded = list(labels_by_point)
    padded.extend({} for _ in range(expected_len - len(padded)))
    return padded


def _router_conf_multiplier(router_conf: float | None) -> float:
    """Modulate reinforcement by router confidence when available."""
    if router_conf is None:
        return 1.0
    clamped = max(0.0, min(1.0, float(router_conf)))
    return 0.5 + (0.5 * clamped)


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
    traces_out: str | None = None,
    traces_in: str | None = None,
    include_query_vector: bool = False,
    reward_source: RewardSource | str = RewardSource.TEACHER,
    reward_weights: RewardWeights | None = None,
    state_store: StateStore | None = None,
    event_store: EventStore | None = None,
    teacher_labeler: Callable[[list[DecisionPoint]], tuple[list[dict[str, float]], list[str]]] | None = None,
) -> AsyncRoutePgSummary:
    """Run teacher-shadow routing updates over recent query events."""
    resolved_state_store = state_store or JsonStateStore()
    resolved_event_store = event_store or JsonlEventStore(journal_path)
    graph, index, meta = resolved_state_store.load(state_path)

    effective_source = RewardSource.parse(reward_source, default=RewardSource.TEACHER)
    effective_weights = reward_weights or RewardWeights.from_env()

    cap_hit = False
    if traces_in:
        traces = _read_traces_jsonl(traces_in)
    else:
        traces, cap_hit = _build_traces_from_journal(
            graph=graph,
            index=index,
            journal_path=journal_path,
            since_hours=since_hours,
            max_queries=max_queries,
            sample_rate=sample_rate,
            max_candidates_per_node=max_candidates_per_node,
            max_decision_points=max_decision_points,
            reward_source=effective_source,
            include_query_vector=include_query_vector,
            event_store=resolved_event_store,
        )

    if traces_out:
        _write_traces_jsonl(traces_out, traces)

    decision_points, route_points, flatten_cap_hit = _flatten_traces(traces, max_decision_points=max_decision_points)
    cap_hit = cap_hit or flatten_cap_hit

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

    labels_by_point = _normalize_labels(labels_by_point, expected_len=len(route_points))

    working_graph = graph if apply else copy.deepcopy(graph)
    decision_points_labeled = 0
    labeled_edges = 0
    updates_applied = 0
    total_abs_weight_delta = 0.0
    max_abs_weight_delta = 0.0
    for point, labels in zip(route_points, labels_by_point):
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
            conf_multiplier = _router_conf_multiplier(point.router_conf)
            scaled_outcome = scale_reward(
                outcome=score_scale * float(score) * conf_multiplier,
                source=effective_source,
                weights=effective_weights,
            )
            updates = apply_outcome_pg(
                graph=working_graph,
                fired_nodes=[point.source_id, target_id],
                outcome=scaled_outcome,
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
                    if point.relevance_conf is not None:
                        metadata["relevance_conf"] = max(0.0, min(1.0, float(point.relevance_conf)))
                    if point.router_conf is not None:
                        metadata["router_conf"] = max(0.0, min(1.0, float(point.router_conf)))
                    if point.policy_disagreement is not None:
                        metadata["policy_disagreement"] = abs(float(point.policy_disagreement))
                    score_count_raw = metadata.get("teacher_score_count", 0)
                    score_count = int(score_count_raw) if isinstance(score_count_raw, (int, float)) else 0
                    score_sum_raw = metadata.get("teacher_score_sum", 0.0)
                    score_sum = float(score_sum_raw) if isinstance(score_sum_raw, (int, float)) else 0.0
                    score_sum += float(score)
                    score_count += 1
                    metadata["teacher_score_sum"] = score_sum
                    metadata["teacher_score_count"] = score_count
                    metadata["teacher_score_mean"] = score_sum / max(1, score_count)
                    edge.metadata = metadata
                    working_graph._edges[point.source_id][target_id] = edge

    if apply and teacher_available and updates_applied > 0:
        resolved_state_store.save(state_path, graph=working_graph, index=index, meta=meta)

    return AsyncRoutePgSummary(
        teacher_requested=teacher,
        teacher_available=teacher_available,
        teacher_model=teacher_model,
        sampled_queries=len(traces),
        decision_points_total=len(route_points),
        decision_points_labeled=decision_points_labeled,
        labeled_edges=labeled_edges,
        updates_applied=updates_applied,
        total_abs_weight_delta=total_abs_weight_delta,
        max_abs_weight_delta=max_abs_weight_delta,
        dry_run=not apply,
        max_decision_points_hit=cap_hit,
        score_scale=score_scale,
        reward_source=effective_source.value,
        reward_weights={
            "human": effective_weights.human,
            "self": effective_weights.self,
            "harvester": effective_weights.harvester,
            "teacher": effective_weights.teacher,
        },
        traces_in=traces_in,
        traces_out=traces_out,
        state_path=state_path,
        journal_path=journal_path,
        errors=errors,
    )
