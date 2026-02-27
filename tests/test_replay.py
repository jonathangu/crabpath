from __future__ import annotations

import json

from pathlib import Path

import pytest
from crabpath.cli import main
from crabpath.graph import Edge, Graph, Node
from crabpath.replay import extract_queries, extract_queries_from_dir, replay_queries
from crabpath.traverse import TraversalConfig




def _write_graph_payload(path: Path) -> None:
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "alpha chunk", "summary": "", "metadata": {"file": "a.md"}},
                {"id": "b", "content": "beta chunk", "summary": "", "metadata": {"file": "b.md"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.5, "kind": "sibling", "metadata": {}},
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_queries_openclaw_format(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "how do i deploy?"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "ignored"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {"role": "user", "content": [{"type": "text", "text": "roll back now"}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert extract_queries(path) == ["how do i deploy?", "roll back now"]


def test_extract_queries_flat_format(tmp_path: Path) -> None:
    path = tmp_path / "session_flat.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "assistant", "content": "ignore"}),
                json.dumps({"role": "user", "content": "restart service"}),
                json.dumps({"role": "user", "content": "check logs"}),
            ]
        ),
        encoding="utf-8",
    )

    assert extract_queries(path) == ["restart service", "check logs"]


def test_extract_queries_from_directory(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "a.jsonl").write_text(json.dumps({"role": "user", "content": "one"}), encoding="utf-8")
    (sessions / "b.jsonl").write_text(
        "\n".join([json.dumps({"role": "user", "content": "two"}), json.dumps({"role": "user", "content": "three"})]),
        encoding="utf-8",
    )
    (sessions / "ignore.txt").write_text("not jsonl", encoding="utf-8")

    assert extract_queries_from_dir(sessions) == ["one", "two", "three"]


def test_extract_queries_filtering_since_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "one", "ts": 100}),
                json.dumps({"role": "user", "content": "two", "ts": 200}),
                json.dumps({"role": "user", "content": "three", "ts": 300}),
            ]
        ),
        encoding="utf-8",
    )

    assert extract_queries(path, since_ts=150.0) == ["two", "three"]


def test_replay_queries_filters_by_since_ts() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "b", 0.5))

    stats = replay_queries(
        graph=graph,
        queries=[("alpha", 1.0), ("alpha", 2.0), ("alpha", 3.0)],
        config=TraversalConfig(max_hops=1),
        since_ts=2.0,
    )

    assert stats["queries_replayed"] == 1
    assert stats["last_replayed_ts"] == 3.0


def test_replay_strengthens_edges() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "b", 0.5))

    stats = replay_queries(graph=graph, queries=["alpha"] * 10, config=TraversalConfig(max_hops=1))

    assert stats["queries_replayed"] == 10
    assert stats["edges_reinforced"] > 0
    assert graph.get_node("a") is not None
    assert graph.get_node("b") is not None
    assert graph._edges["a"]["b"].weight > 0.5


def test_replay_creates_cross_file_edges() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "b.md"}))

    stats = replay_queries(graph=graph, queries=["alpha beta"], config=TraversalConfig(max_hops=1))

    assert stats["queries_replayed"] == 1
    assert stats["cross_file_edges_created"] == 1

    assert graph._edges["b"]["a"].source == "b"
    assert graph._edges["b"]["a"].target == "a"


def test_replay_queries_supports_outcome_fn_negative_learning() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    graph.add_node(Node("bad", "bad", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "bad", 0.8))

    stats = replay_queries(
        graph=graph,
        queries=["bad example", "normal"],
        config=TraversalConfig(max_hops=1),
        outcome_fn=lambda query: -1.0 if "bad" in query else 1.0,
    )

    assert stats["queries_replayed"] == 2
    assert graph._edges["a"]["bad"].kind == "inhibitory"
    assert graph._edges["a"]["bad"].weight < 0.8
