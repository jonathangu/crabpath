from __future__ import annotations

import json
from pathlib import Path

import pytest

from crabpath.cli import main


def _write_graph_payload(path: Path) -> None:
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "alpha", "summary": "", "metadata": {"file": "a.md"}},
                {"id": "b", "content": "beta", "summary": "", "metadata": {"file": "b.md"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.7, "kind": "sibling", "metadata": {}},
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_index(path: Path) -> None:
    path.write_text(json.dumps({"a": [1, 0], "b": [0, 1]}), encoding="utf-8")


def test_init_command_creates_workspace_graph(tmp_path, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path / "graph.json"

    code = main(["init", "--workspace", str(workspace), "--output", str(output)])
    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "graph" in data
    assert "node_texts" in data
    assert len(data["graph"]["nodes"]) == 1


def test_init_command_with_empty_workspace(tmp_path) -> None:
    workspace = tmp_path / "empty"
    workspace.mkdir()
    output = tmp_path / "graph.json"

    code = main(["init", "--workspace", str(workspace), "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["graph"]["nodes"] == []
    assert payload["graph"]["edges"] == []


def test_query_command_returns_json_with_fired_nodes(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(index_path)

    code = main(
        [
            "query",
            "seed",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
            "--json",
            "--query-vector",
            "1,0",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert "a" in out["fired"]


def test_query_command_error_on_missing_graph(tmp_path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    with pytest.raises(SystemExit):
        main(["query", "seed", "--graph", str(tmp_path / "missing.json"), "--index", str(index_path), "--query-vector", "1,0"])


def test_query_command_non_vector_error() -> None:
    with pytest.raises(SystemExit):
        main(["query", "seed", "--graph", "dummy.json", "--index", "dummy_idx.json"])


def test_learn_command_updates_graph_weights(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["learn", "--graph", str(graph_path), "--outcome", "1.0", "--fired-ids", "a,b"])
    assert code == 0

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["graph"]["edges"][0]["weight"] > 0.7


def test_learn_command_supports_json_output(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["learn", "--graph", str(graph_path), "--outcome", "-1.0", "--fired-ids", "a,b", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out.strip())
    assert data["edges"][0]["source"] == "a"


def test_health_command_outputs_all_metrics(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)
    code = main(["health", "--graph", str(graph_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert set(payload.keys()) == {"dormant_pct", "habitual_pct", "reflex_pct", "cross_file_edge_pct", "orphan_nodes"}


def test_init_output_flag_writes_path(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("single paragraph", encoding="utf-8")
    output = tmp_path / "custom.json"

    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--json"])
    assert code == 0
    assert output.exists()


def test_init_command_invalid_arguments(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["init", "--workspace", str(tmp_path)])


def test_health_invalid_graph_path_error_is_clear(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["health", "--graph", str(tmp_path / "nope.json")])


def test_cli_help_text_for_all_commands() -> None:
    for command in ["init", "query", "learn", "health"]:
        with pytest.raises(SystemExit):
            main([command, "--help"])
