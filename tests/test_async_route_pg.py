from __future__ import annotations

import json
from pathlib import Path

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.index import VectorIndex
from openclawbrain.ops.async_route_pg import parse_teacher_route_labels, run_async_route_pg
from openclawbrain.reward import RewardSource
from openclawbrain.store import load_state, save_state
from openclawbrain.trace import RouteCandidate, RouteDecisionPoint, RouteTrace, route_trace_to_json


def _write_state(path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("seed", "alpha routing seed"))
    graph.add_node(Node("target_a", "alpha preferred target"))
    graph.add_node(Node("target_b", "alpha alternative target"))
    graph.add_edge(Edge("seed", "target_a", 0.35))
    graph.add_edge(Edge("seed", "target_b", 0.30))
    save_state(graph=graph, index=VectorIndex(), path=str(path), meta={"embedder_name": "hash-v1", "embedder_dim": 1024})


def _write_journal(path: Path) -> None:
    entry = {"type": "query", "query": "alpha", "ts": 9999999999.0, "iso": "2286-11-20T17:46:39+0000"}
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def test_parse_teacher_route_labels_choose_only() -> None:
    labels = parse_teacher_route_labels('{"choose":["target_a"]}', {"target_a", "target_b"})
    assert labels == {"target_a": 1.0}


def test_parse_teacher_route_labels_scores() -> None:
    labels = parse_teacher_route_labels(
        '{"scores":{"target_a":0.4,"target_b":-2.0,"unknown":1.0}}',
        {"target_a", "target_b"},
    )
    assert labels == {"target_a": 0.4, "target_b": -1.0}


def test_run_async_route_pg_positive_label_increases_chosen_edge_relative_to_others(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.jsonl"
    _write_state(state_path)
    _write_journal(journal_path)

    graph_before, _, _ = load_state(str(state_path))
    before_gap = graph_before._edges["seed"]["target_a"].weight - graph_before._edges["seed"]["target_b"].weight

    def _teacher(points):
        labels = []
        for point in points:
            if point.source_id == "seed" and point.chosen_target_id == "target_a":
                labels.append({"target_a": 1.0})
            else:
                labels.append({})
        return labels, []

    summary = run_async_route_pg(
        state_path=str(state_path),
        journal_path=str(journal_path),
        since_hours=24.0,
        max_queries=10,
        sample_rate=1.0,
        max_candidates_per_node=12,
        max_decision_points=500,
        teacher="none",
        apply=True,
        teacher_labeler=_teacher,
    )
    assert summary.updates_applied >= 1

    graph_after, _, _ = load_state(str(state_path))
    after_gap = graph_after._edges["seed"]["target_a"].weight - graph_after._edges["seed"]["target_b"].weight
    assert after_gap > before_gap


def test_run_async_route_pg_dry_run_does_not_modify_state_json(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.jsonl"
    _write_state(state_path)
    _write_journal(journal_path)
    before = state_path.read_bytes()

    def _teacher(points):
        labels = []
        for point in points:
            if point.source_id == "seed":
                labels.append({"target_a": 1.0})
            else:
                labels.append({})
        return labels, []

    summary = run_async_route_pg(
        state_path=str(state_path),
        journal_path=str(journal_path),
        since_hours=24.0,
        max_queries=10,
        sample_rate=1.0,
        max_candidates_per_node=12,
        max_decision_points=500,
        teacher="none",
        apply=False,
        teacher_labeler=_teacher,
    )
    after = state_path.read_bytes()
    assert summary.updates_applied >= 1
    assert before == after


def test_run_async_route_pg_dry_run_traces_out_emits_expected_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.jsonl"
    traces_path = tmp_path / "route_traces.jsonl"
    _write_state(state_path)
    _write_journal(journal_path)

    summary = run_async_route_pg(
        state_path=str(state_path),
        journal_path=str(journal_path),
        since_hours=24.0,
        max_queries=10,
        sample_rate=1.0,
        max_candidates_per_node=12,
        max_decision_points=500,
        teacher="none",
        apply=False,
        traces_out=str(traces_path),
        include_query_vector=True,
    )

    assert summary.updates_applied == 0
    lines = [line for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    trace = json.loads(lines[0])
    assert "query_id" in trace
    assert "query_text" in trace
    assert "decision_points" in trace
    assert "traversal_config" in trace
    assert "route_policy" in trace
    assert isinstance(trace["decision_points"], list)
    assert "query_vector" in trace
    if trace["decision_points"]:
        point = trace["decision_points"][0]
        assert "source_id" in point
        assert "candidates" in point
        assert "reward_source" in point


def test_run_async_route_pg_modulates_updates_by_router_confidence(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.jsonl"
    traces_path = tmp_path / "route_traces.jsonl"
    _write_state(state_path)
    _write_journal(journal_path)

    trace = RouteTrace(
        query_id="q-1",
        ts=1.0,
        query_text="alpha",
        decision_points=[
            RouteDecisionPoint(
                query_text="alpha",
                source_id="seed",
                source_preview="seed",
                chosen_target_id="target_a",
                candidates=[RouteCandidate(target_id="target_a", edge_weight=0.35, edge_relevance=0.0)],
                router_conf=1.0,
                reward_source=RewardSource.TEACHER,
            ),
            RouteDecisionPoint(
                query_text="alpha",
                source_id="seed",
                source_preview="seed",
                chosen_target_id="target_b",
                candidates=[RouteCandidate(target_id="target_b", edge_weight=0.30, edge_relevance=0.0)],
                router_conf=0.0,
                reward_source=RewardSource.TEACHER,
            ),
        ],
    )
    traces_path.write_text(route_trace_to_json(trace) + "\n", encoding="utf-8")

    def _teacher(points):
        return [{"target_a": 1.0}, {"target_b": 1.0}], []

    run_async_route_pg(
        state_path=str(state_path),
        journal_path=str(journal_path),
        traces_in=str(traces_path),
        teacher="none",
        apply=True,
        teacher_labeler=_teacher,
    )

    graph_after, _, _ = load_state(str(state_path))
    weight_a = graph_after._edges["seed"]["target_a"].weight
    weight_b = graph_after._edges["seed"]["target_b"].weight
    assert (weight_a - 0.35) > (weight_b - 0.30)
