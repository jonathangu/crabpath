from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from crabpath import Edge, EmbeddingIndex, Graph, Node
from crabpath.mitosis import MitosisState


def _run_cli(
    args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    proc_env = os.environ.copy()
    proc_env["PYTHONPATH"] = os.getcwd()
    if env:
        proc_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "crabpath.cli", *args],
        text=True,
        capture_output=True,
        env=proc_env,
    )


def _load_json_output(raw: str) -> dict:
    return json.loads(raw.strip())


def test_query_outputs_json(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"

    graph = Graph()
    graph.add_node(Node(id="deploy-config", content="deploy broke after config change"))
    graph.add_node(Node(id="ignore", content="nothing relevant"))
    graph.save(str(graph_path))
    index = EmbeddingIndex()
    index.vectors = {"deploy-config": [1.0], "ignore": [0.2]}
    index.save(str(index_path))

    result = _run_cli(
        [
            "query",
            "deploy broke after config change",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "12",
        ],
        env={"OPENAI_API_KEY": ""},
    )
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert "fired" in payload
    assert payload["fired"][0]["id"] == "deploy-config"
    assert "inhibited" in payload
    assert "guardrails" in payload


def test_learn_updates_graph_and_outputs_edges_updated(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    graph = Graph()
    graph.add_node(Node(id="a", content="start"))
    graph.add_node(Node(id="b", content="next"))
    graph.add_edge(Edge(source="a", target="b", weight=1.0))
    graph.save(str(graph_path))

    result = _run_cli(
        [
            "learn",
            "--graph",
            str(graph_path),
            "--outcome",
            "1.0",
            "--fired-ids",
            "a,b",
        ]
    )
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["ok"] is True
    assert payload["edges_updated"] >= 1

    updated = Graph.load(str(graph_path))
    assert updated.get_edge("a", "b").weight > 1.0


def test_snapshot_and_feedback_roundtrip(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph = Graph()
    graph.add_node(Node(id="a", content="seed"))
    graph.save(str(graph_path))

    snapshot_path = tmp_path / "session.events.db"
    snapshot_env = {
        "CRABPATH_SNAPSHOT_PATH": str(snapshot_path),
        "OPENAI_API_KEY": "",
    }

    snapshot = _run_cli(
        [
            "snapshot",
            "--graph",
            str(graph_path),
            "--session",
            "sess-123",
            "--turn",
            "42",
            "--fired-ids",
            "a",
        ],
        env=snapshot_env,
    )
    assert snapshot.returncode == 0
    snapshot_payload = _load_json_output(snapshot.stdout)
    assert snapshot_payload["ok"] is True
    assert snapshot_payload["snapshot_path"] == str(snapshot_path)

    feedback = _run_cli(
        [
            "feedback",
            "--session",
            "sess-123",
            "--turn-window",
            "5",
        ],
        env={**snapshot_env, "CRABPATH_CORRECTION_TURN": "45"},
    )
    assert feedback.returncode == 0
    feedback_payload = _load_json_output(feedback.stdout)

    assert feedback_payload["turn_id"] == 42
    assert feedback_payload["fired_ids"] == ["a"]
    assert feedback_payload["turns_since_fire"] == 3
    assert feedback_payload["suggested_outcome"] == -1.0


def test_stats_outputs_graph_summary(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    graph = Graph()
    graph.add_node(Node(id="a", content="start"))
    graph.add_node(Node(id="b", content="bridge"))
    graph.add_node(Node(id="c", content="end"))
    graph.add_edge(Edge(source="a", target="b", weight=0.4))
    graph.add_edge(Edge(source="b", target="c", weight=0.7))
    graph.add_edge(Edge(source="a", target="c", weight=0.2))
    graph.save(str(graph_path))

    result = _run_cli(["stats", "--graph", str(graph_path)])
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["nodes"] == 3
    assert payload["edges"] == 3
    assert abs(payload["avg_weight"] - 0.43333333333333335) < 1e-12
    assert payload["top_hubs"] == ["a", "b", "c"]


def test_health_cli_text_without_query_stats(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    graph = Graph()
    graph.add_node(Node(id="file_a::a", content="alpha block"))
    graph.add_node(Node(id="file_a::b", content="beta block"))
    graph.add_node(Node(id="file_b::c", content="gamma block"))
    graph.add_edge(Edge(source="file_a::a", target="file_a::b", weight=0.2))
    graph.add_edge(Edge(source="file_a::b", target="file_b::c", weight=0.5))
    graph.add_edge(Edge(source="file_b::c", target="file_a::a", weight=0.95))
    graph.save(str(graph_path))

    result = _run_cli(["health", "--graph", str(graph_path)])
    assert result.returncode == 0
    assert "Graph Health:" in result.stdout
    assert "avg_nodes_fired_per_query" in result.stdout
    assert "target" in result.stdout
    assert "⚠️" in result.stdout
    assert "n/a (collect query stats)" in result.stdout


def test_health_cli_json_with_query_stats_and_state(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    query_stats_path = tmp_path / "query_stats.json"
    state_path = tmp_path / "state.json"

    graph = Graph()
    graph.add_node(Node(id="file_a::a", content="alpha " * 20))
    graph.add_node(Node(id="file_a::b", content="beta " * 20))
    graph.add_node(Node(id="file_b::c", content="gamma " * 20))
    graph.add_edge(Edge(source="file_a::a", target="file_a::b", weight=0.2))
    graph.add_edge(Edge(source="file_a::b", target="file_b::c", weight=0.5))
    graph.add_edge(Edge(source="file_b::c", target="file_a::a", weight=0.95))
    graph.save(str(graph_path))

    query_stats = {
        "fired_counts": [1, 2, 3],
        "chars": [20, 30],
        "promotions": 1,
        "proto_created": 10,
        "reconverged_families": 1,
    }
    query_stats_path.write_text(json.dumps(query_stats), encoding="utf-8")
    MitosisState(families={"x": ["file_a::a"], "y": ["file_a::b", "file_b::c"]}).save(
        str(state_path)
    )

    result = _run_cli(
        [
            "health",
            "--graph",
            str(graph_path),
            "--query-stats",
            str(query_stats_path),
            "--mitosis-state",
            str(state_path),
            "--json",
        ]
    )
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["query_stats_provided"] is True
    assert payload["mitosis_state"] == str(state_path)
    assert payload["metrics"]
    metrics = {row["metric"]: row for row in payload["metrics"]}
    assert metrics["avg_nodes_fired_per_query"]["value"] == 2.0
    assert metrics["reconvergence_rate"]["value"] == 50.0
    assert metrics["context_compression"]["target_range"] == [None, 20.0]


def test_evolve_cli_appends_snapshot_and_report(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    snapshot_path = tmp_path / "evolve.jsonl"

    graph = Graph()
    graph.add_node(Node(id="file_a::a", content="alpha"))
    graph.add_node(Node(id="file_b::b", content="bravo"))
    graph.add_edge(Edge(source="file_a::a", target="file_b::b", weight=0.2))
    graph.save(str(graph_path))

    first = _run_cli(["evolve", "--graph", str(graph_path), "--snapshots", str(snapshot_path)])
    assert first.returncode == 0
    first_payload = _load_json_output(first.stdout)
    assert first_payload["ok"] is True
    assert first_payload["snapshot"]["nodes"] == 2
    assert first_payload["snapshot"]["edges"] == 1
    assert first_payload["snapshot"]["cross_file_edges"] == 1
    assert snapshot_path.exists()
    assert len(snapshot_path.read_text().splitlines()) == 1

    graph.add_node(Node(id="file_c::c", content="charlie"))
    graph.add_edge(Edge(source="file_b::b", target="file_c::c", weight=0.8))
    graph.save(str(graph_path))

    report = _run_cli(
        ["evolve", "--graph", str(graph_path), "--snapshots", str(snapshot_path), "--report"]
    )
    assert report.returncode == 0
    assert report.stdout.count("Evolution timeline") == 1
    assert "nodes 3 (+1)" in report.stdout
    assert "# 2" in report.stdout


def test_consolidate_outputs_prune_counts(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    graph = Graph()
    graph.add_node(Node(id="a", content="a"))
    graph.add_node(Node(id="b", content="b"))
    graph.add_node(Node(id="c", content="c"))
    graph.add_node(Node(id="d", content="d"))
    graph.add_edge(Edge(source="a", target="b", weight=0.02))
    graph.add_edge(Edge(source="b", target="c", weight=0.6))
    graph.add_edge(Edge(source="c", target="a", weight=0.03))
    graph.add_edge(Edge(source="a", target="d", weight=0.09))
    graph.save(str(graph_path))

    result = _run_cli(["consolidate", "--graph", str(graph_path), "--min-weight", "0.05"])
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["ok"] is True
    assert payload["pruned_edges"] == 2

    consolidated = Graph.load(str(graph_path))
    assert consolidated.edge_count == 2
    assert consolidated.get_edge("b", "c") is not None


def test_query_error_when_graph_missing(tmp_path: Path) -> None:
    missing_graph = tmp_path / "missing_graph.json"
    result = _run_cli(
        [
            "query",
            "deploy broke",
            "--graph",
            str(missing_graph),
            "--index",
            str(tmp_path / "missing_index.json"),
        ],
        env={"OPENAI_API_KEY": ""},
    )

    assert result.returncode == 1
    error = _load_json_output(result.stderr)
    assert "error" in error
    assert "graph file not found" in error["error"]


def test_learn_bad_args(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph = Graph()
    graph.add_node(Node(id="a", content="start"))
    graph.save(str(graph_path))

    result = _run_cli(
        [
            "learn",
            "--graph",
            str(graph_path),
            "--outcome",
            "not-a-number",
            "--fired-ids",
            "a",
        ]
    )
    assert result.returncode == 1
    error = _load_json_output(result.stderr)
    assert "error" in error


def test_migrate_cli_outputs_graph_and_info(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text(
        "## Rules\nUse concise, safe, and direct instructions.\n\n"
        "## Tools\nUse CLI and browser for debugging."
    )
    (workspace / "SOUL.md").write_text("## Identity\nI am a memory graph test harness.")

    graph_path = tmp_path / "migrated_graph.json"
    embeddings_path = tmp_path / "migrated_embeddings.json"

    result = _run_cli(
        [
            "migrate",
            "--workspace",
            str(workspace),
            "--no-include-memory",
            "--output-graph",
            str(graph_path),
            "--output-embeddings",
            str(embeddings_path),
            "--verbose",
            "--include-docs",
        ]
    )
    assert result.returncode == 0

    payload = _load_json_output(result.stdout)
    assert payload["ok"] is True
    assert payload["graph_path"] == str(graph_path)
    assert payload["embeddings_path"] == str(embeddings_path)
    assert payload["info"]["bootstrap"]["nodes"] > 0

    assert graph_path.exists()
    assert embeddings_path.exists()
    migrated = Graph.load(str(graph_path))
    assert migrated.node_count >= 2


def test_split_cli_saves_chunks(tmp_path: Path) -> None:
    graph_path = tmp_path / "split_graph.json"
    graph = Graph()
    graph.add_node(
        Node(
            id="soul",
            content=(
                "## Identity\nI am the main identity file. This file captures the "
                "core persona and operating guidelines for this agent.\n\n"
                "## Tools\nUse codex for coding and browser for web tasks.\n\n"
                "## Safety\nNever expose credentials.\n\n"
                "## Memory\nKeep daily notes and migration logs. This text should "
                "be sufficiently long and realistic "
                "so that CLI split operations can test chunking behavior across markdown sections, "
                "ensuring output contains multiple coherent chunks from the source content.\n\n"
                "## Notes\nContinue extending these notes with project decisions and "
                "rationale so downstream systems "
                "can find useful overlap signals and stable retrieval anchors."
            ),
        )
    )
    graph.save(str(graph_path))

    result = _run_cli(
        [
            "split",
            "--graph",
            str(graph_path),
            "--node-id",
            "soul",
            "--save",
        ]
    )
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["ok"] is True
    assert payload["action"] == "split"
    assert payload["chunk_count"] >= 2
    assert payload["node_id"] == "soul"

    split_graph = Graph.load(str(graph_path))
    assert split_graph.get_node("soul") is None
    assert len(payload["chunk_ids"]) == payload["chunk_count"]


def test_sim_cli_runs_and_outputs_snapshots(tmp_path: Path) -> None:
    output = tmp_path / "sim.json"
    result = _run_cli(
        [
            "sim",
            "--queries",
            "5",
            "--decay-interval",
            "2",
            "--decay-half-life",
            "40",
            "--output",
            str(output),
        ]
    )
    assert result.returncode == 0

    payload = _load_json_output(result.stdout)
    assert payload["ok"] is True
    assert payload["queries"] == 5
    assert payload["result"]["final"]["nodes"] >= 1
    assert len(payload["result"]["snapshots"]) == 5
    assert output.exists()
