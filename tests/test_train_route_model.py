from __future__ import annotations

import json
from pathlib import Path

from openclawbrain.route_model import RouteModel
from openclawbrain import Edge, Graph, Node, VectorIndex, load_state, save_state
from openclawbrain.reward import RewardSource, RewardWeights
from openclawbrain.trace import RouteCandidate, RouteDecisionPoint, RouteTrace, route_trace_to_json
from openclawbrain.train_route_model import _read_traces, evaluate_ce_loss, train_route_model


def _write_state(path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("seed", "seed"))
    graph.add_node(Node("target_a", "target a"))
    graph.add_node(Node("target_b", "target b"))
    graph.add_edge(Edge("seed", "target_a", weight=0.4, metadata={"relevance": 0.0}))
    graph.add_edge(Edge("seed", "target_b", weight=0.4, metadata={"relevance": 0.0}))

    index = VectorIndex()
    index.upsert("seed", [1.0, 0.0])
    index.upsert("target_a", [1.0, 0.0])
    index.upsert("target_b", [0.0, 1.0])
    save_state(graph=graph, index=index, path=str(path), meta={"embedder_name": "hash-v1", "embedder_dim": 2})


def _write_traces(path: Path) -> None:
    traces: list[RouteTrace] = []
    for idx in range(8):
        traces.append(
            RouteTrace(
                query_id=f"q{idx}",
                ts=1000.0 + idx,
                query_text="choose a",
                seeds=[["seed", 1.0]],
                fired_nodes=["seed", "target_a"],
                traversal_config={"max_hops": 15},
                route_policy={"route_mode": "off"},
                query_vector=[1.0, 0.0],
                decision_points=[
                    RouteDecisionPoint(
                        query_text="choose a",
                        source_id="seed",
                        source_preview="seed",
                        chosen_target_id="target_a",
                        candidates=[
                            RouteCandidate(target_id="target_a", edge_weight=0.4, edge_relevance=0.0),
                            RouteCandidate(target_id="target_b", edge_weight=0.4, edge_relevance=0.0),
                        ],
                        teacher_choose=["target_a"],
                        teacher_scores={"target_a": 1.0, "target_b": -1.0},
                        ts=1000.0 + idx,
                        reward_source=RewardSource.TEACHER,
                    )
                ],
            )
        )

    path.write_text("\n".join(route_trace_to_json(trace) for trace in traces) + "\n", encoding="utf-8")


def test_train_route_model_ce_loss_decreases(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    traces_path = tmp_path / "traces.jsonl"
    out_path = tmp_path / "route_model.npz"

    _write_state(state_path)
    _write_traces(traces_path)

    summary = train_route_model(
        state_path=str(state_path),
        traces_in=str(traces_path),
        labels_in=None,
        out_path=str(out_path),
        rank=2,
        epochs=1,
        lr=0.1,
        label_temp=0.5,
    )
    assert out_path.exists()
    model = RouteModel.load_npz(out_path)
    assert model.df == 1
    assert summary.points_used > 0
    assert summary.final_ce_loss < summary.initial_ce_loss


def test_evaluate_ce_loss_supports_legacy_df3_features(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    traces_path = tmp_path / "traces.jsonl"
    _write_state(state_path)
    _write_traces(traces_path)
    traces = _read_traces(str(traces_path))
    model = RouteModel.init_identity(d=2, df=3)
    _, index, _ = load_state(str(state_path))

    loss, points_total, points_used = evaluate_ce_loss(
        model=model,
        traces=traces,
        index_vectors=index._vectors,
        labels=[],
        label_temp=0.5,
        reward_weights=RewardWeights(),
    )
    assert loss >= 0.0
    assert points_total > 0
    assert points_used > 0
