"""Train low-rank learned routing model from traces + label records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import LabelRecord, read_labels_jsonl
from .reward import RewardWeights
from .route_model import RouteModel
from .trace import RouteDecisionPoint, RouteTrace, route_trace_from_json
from .store import load_state


@dataclass(frozen=True)
class TrainRouteModelSummary:
    traces_path: str
    labels_path: str | None
    out_path: str
    rank: int
    epochs: int
    lr: float
    label_temp: float
    points_total: int
    points_used: int
    initial_ce_loss: float
    final_ce_loss: float
    epoch_losses: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "traces_path": self.traces_path,
            "labels_path": self.labels_path,
            "out_path": self.out_path,
            "rank": self.rank,
            "epochs": self.epochs,
            "lr": self.lr,
            "label_temp": self.label_temp,
            "points_total": self.points_total,
            "points_used": self.points_used,
            "initial_ce_loss": self.initial_ce_loss,
            "final_ce_loss": self.final_ce_loss,
            "epoch_losses": list(self.epoch_losses),
        }


def _read_traces(path: str) -> list[RouteTrace]:
    source = Path(path).expanduser()
    traces: list[RouteTrace] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        traces.append(route_trace_from_json(raw))
    return traces


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - np.max(logits)
    exp = np.exp(stable)
    denom = np.sum(exp)
    if denom <= 0:
        return np.ones_like(logits) / max(1, logits.shape[0])
    return exp / denom


def _labels_index(labels: list[LabelRecord]) -> dict[tuple[str, int], list[LabelRecord]]:
    indexed: dict[tuple[str, int], list[LabelRecord]] = {}
    for record in labels:
        key = (record.query_id, int(record.decision_point_idx))
        indexed.setdefault(key, []).append(record)
    return indexed


def _teacher_distribution(
    trace: RouteTrace,
    point_idx: int,
    point: RouteDecisionPoint,
    candidate_ids: list[str],
    labels_by_key: dict[tuple[str, int], list[LabelRecord]],
    *,
    label_temp: float,
    reward_weights: RewardWeights,
) -> np.ndarray:
    scores = {candidate_id: 0.0 for candidate_id in candidate_ids}

    if point.teacher_scores:
        for target_id, value in point.teacher_scores.items():
            if target_id in scores:
                scores[target_id] += float(value)
    elif point.teacher_choose:
        for target_id in point.teacher_choose:
            if target_id in scores:
                scores[target_id] += 1.0
    elif point.chosen_target_id in scores:
        scores[point.chosen_target_id] += 1.0

    for record in labels_by_key.get((trace.query_id, point_idx), []):
        source_weight = reward_weights.for_source(record.reward_source)
        total_weight = float(record.weight) * float(source_weight)
        for target_id, value in record.candidate_scores.items():
            if target_id in scores:
                scores[target_id] += total_weight * float(value)

    score_vec = np.asarray([scores[candidate_id] for candidate_id in candidate_ids], dtype=float)
    if np.allclose(score_vec, 0.0):
        return np.ones(len(candidate_ids), dtype=float) / max(1, len(candidate_ids))
    denom = max(1e-6, float(label_temp))
    return _softmax(score_vec / denom)


def _point_logits(
    model: RouteModel,
    query_vector: np.ndarray,
    targets: list[np.ndarray],
    features: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    q_proj = model.project_query(query_vector)
    target_projs = [model.project_target(t_vec) for t_vec in targets]
    logits = np.asarray([model.score_projected(q_proj, t_proj, feat_vec) for t_proj, feat_vec in zip(target_projs, features)], dtype=float)
    return logits, q_proj, target_projs


def _collect_points(
    traces: list[RouteTrace],
    index_vectors: dict[str, list[float]],
) -> tuple[int, list[tuple[RouteTrace, int, RouteDecisionPoint, np.ndarray, list[np.ndarray], list[np.ndarray], list[str]]]]:
    points_total = sum(len(trace.decision_points) for trace in traces)
    points: list[tuple[RouteTrace, int, RouteDecisionPoint, np.ndarray, list[np.ndarray], list[np.ndarray], list[str]]] = []
    for trace in traces:
        if trace.query_vector is None:
            continue
        query_vector = np.asarray(trace.query_vector, dtype=float)
        for point_idx, point in enumerate(trace.decision_points):
            candidate_ids: list[str] = []
            targets: list[np.ndarray] = []
            features: list[np.ndarray] = []
            for candidate in point.sorted_candidates():
                target_vector = index_vectors.get(candidate.target_id)
                if target_vector is None:
                    continue
                target_arr = np.asarray(target_vector, dtype=float)
                if target_arr.ndim != 1:
                    continue
                candidate_ids.append(candidate.target_id)
                targets.append(target_arr)
                features.append(
                    np.asarray(
                        [float(candidate.edge_weight), float(candidate.edge_relevance), 1.0],
                        dtype=float,
                    )
                )
            if len(candidate_ids) < 2:
                continue
            points.append((trace, point_idx, point, query_vector, targets, features, candidate_ids))
    return points_total, points


def evaluate_ce_loss(
    model: RouteModel,
    traces: list[RouteTrace],
    index_vectors: dict[str, list[float]],
    labels: list[LabelRecord],
    *,
    label_temp: float,
    reward_weights: RewardWeights,
) -> tuple[float, int, int]:
    points_total, points = _collect_points(traces, index_vectors)
    labels_by_key = _labels_index(labels)
    losses: list[float] = []
    for trace, point_idx, point, query_vector, targets, features, candidate_ids in points:
        logits, _q_proj, _target_projs = _point_logits(model, query_vector, targets, features)
        probs = _softmax(logits)
        labels_dist = _teacher_distribution(
            trace,
            point_idx,
            point,
            candidate_ids,
            labels_by_key,
            label_temp=label_temp,
            reward_weights=reward_weights,
        )
        loss = -float(np.sum(labels_dist * np.log(np.clip(probs, 1e-12, 1.0))))
        losses.append(loss)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, points_total, len(points)


def train_route_model(
    *,
    state_path: str,
    traces_in: str,
    labels_in: str | None,
    out_path: str,
    rank: int = 16,
    epochs: int = 3,
    lr: float = 0.01,
    label_temp: float = 0.5,
    reward_weights: RewardWeights | None = None,
) -> TrainRouteModelSummary:
    traces = _read_traces(traces_in)
    labels = read_labels_jsonl(labels_in) if labels_in else []
    graph, index, _meta = load_state(state_path)
    _ = graph

    parsed_weights = reward_weights or RewardWeights.from_env()
    points_total, points = _collect_points(traces, index._vectors)
    if not points:
        raise ValueError("no trainable decision points found (need query_vector + >=2 candidates with target vectors)")

    dq = int(points[0][3].shape[0])
    dt = int(points[0][4][0].shape[0])
    model = RouteModel.init_random(dq=dq, dt=dt, df=3, rank=max(1, int(rank)))

    labels_by_key = _labels_index(labels)
    epoch_losses: list[float] = []

    initial_loss, _, points_used = evaluate_ce_loss(
        model,
        traces,
        index._vectors,
        labels,
        label_temp=label_temp,
        reward_weights=parsed_weights,
    )

    lr_value = float(lr)
    inv_t = 1.0 / max(1e-6, float(model.T))
    for _ in range(max(1, int(epochs))):
        losses: list[float] = []
        for trace, point_idx, point, query_vector, targets, features, candidate_ids in points:
            logits, q_proj, target_projs = _point_logits(model, query_vector, targets, features)
            probs = _softmax(logits)
            labels_dist = _teacher_distribution(
                trace,
                point_idx,
                point,
                candidate_ids,
                labels_by_key,
                label_temp=label_temp,
                reward_weights=parsed_weights,
            )

            grad = probs - labels_dist
            grad_A = np.zeros_like(model.A)
            grad_B = np.zeros_like(model.B)
            grad_w = np.zeros_like(model.w_feat)
            grad_b = 0.0

            for idx, g_i in enumerate(grad.tolist()):
                g_scaled = float(g_i) * inv_t
                grad_A += g_scaled * np.outer(query_vector, target_projs[idx])
                grad_B += g_scaled * np.outer(targets[idx], q_proj)
                grad_w += g_scaled * features[idx]
                grad_b += g_scaled

            model.A -= lr_value * grad_A
            model.B -= lr_value * grad_B
            model.w_feat -= lr_value * grad_w
            model.b -= lr_value * grad_b

            loss = -float(np.sum(labels_dist * np.log(np.clip(probs, 1e-12, 1.0))))
            losses.append(loss)

        epoch_losses.append(float(np.mean(losses)) if losses else 0.0)

    final_loss, _, _ = evaluate_ce_loss(
        model,
        traces,
        index._vectors,
        labels,
        label_temp=label_temp,
        reward_weights=parsed_weights,
    )

    model.save_npz(out_path)
    return TrainRouteModelSummary(
        traces_path=str(traces_in),
        labels_path=str(labels_in) if labels_in else None,
        out_path=str(out_path),
        rank=int(model.r),
        epochs=max(1, int(epochs)),
        lr=lr_value,
        label_temp=float(label_temp),
        points_total=points_total,
        points_used=points_used,
        initial_ce_loss=float(initial_loss),
        final_ce_loss=float(final_loss),
        epoch_losses=epoch_losses,
    )


def write_summary_json(summary: TrainRouteModelSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2)
