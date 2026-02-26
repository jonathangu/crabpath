from __future__ import annotations

import json
from io import StringIO
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


def test_init_command_creates_workspace_graph(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path

    code = main(["init", "--workspace", str(workspace), "--output", str(output)])
    assert code == 0

    graph_path = output / "graph.json"
    texts_path = output / "texts.json"
    assert graph_path.exists()
    assert texts_path.exists()
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    assert len(graph_data["nodes"]) == 1
    texts_data = json.loads(texts_path.read_text(encoding="utf-8"))
    assert len(texts_data) == 1


def test_init_command_with_empty_workspace(tmp_path) -> None:
    workspace = tmp_path / "empty"
    workspace.mkdir()
    output = tmp_path / "out"
    output.mkdir()

    code = main(["init", "--workspace", str(workspace), "--output", str(output)])
    assert code == 0
    graph_data = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert graph_data["nodes"] == []
    assert graph_data["edges"] == []


def test_query_command_returns_json_with_fired_nodes(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(index_path)

    code = main(
        [
            "query",
            "alpha",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert "a" in out["fired"]


def test_query_uses_vector_from_stdin(tmp_path, capsys, monkeypatch) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(index_path)

    monkeypatch.setattr("sys.stdin", StringIO(json.dumps([1.0, 0.0])))

    code = main(
        [
            "query",
            "seed",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "1",
            "--query-vector-stdin",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"][0] == "a"


def test_query_command_error_on_missing_graph(tmp_path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    with pytest.raises(SystemExit):
        main(["query", "seed", "--graph", str(tmp_path / "missing.json"), "--index", str(index_path)])


def test_query_command_keywords_without_index(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["query", "alpha", "--graph", str(graph_path), "--top", "2", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert out["fired"][0] == "a"


def test_learn_command_updates_graph_weights(tmp_path) -> None:
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
    assert data["graph"]["edges"][0]["source"] == "a"


def test_merge_command_suggests_and_applies(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "alpha", "summary": "", "metadata": {}},
                {"id": "b", "content": "beta", "summary": "", "metadata": {}},
                {"id": "c", "content": "gamma", "summary": "", "metadata": {}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.95, "kind": "sibling", "metadata": {}},
                {"source": "b", "target": "a", "weight": 0.95, "kind": "sibling", "metadata": {}},
                {"source": "a", "target": "c", "weight": 0.2, "kind": "sibling", "metadata": {}},
                {"source": "c", "target": "a", "weight": 0.1, "kind": "sibling", "metadata": {}},
            ],
        }
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    code = main(["merge", "--graph", str(graph_path), "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert "suggestions" in out
    assert out["suggestions"]

    updated = json.loads(graph_path.read_text(encoding="utf-8"))
    if "graph" in updated:
        graph_payload = updated["graph"]
    else:
        graph_payload = updated
    assert "nodes" in graph_payload
    assert len(graph_payload["nodes"]) <= 3


def test_health_command_outputs_all_metrics(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)
    code = main(["health", "--graph", str(graph_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    expected = {
        "dormant_pct",
        "habitual_pct",
        "reflex_pct",
        "cross_file_edge_pct",
        "orphan_nodes",
        "nodes",
        "edges",
    }
    assert expected.issubset(set(payload.keys()))


def test_cli_help_text_for_commands() -> None:
    for command in ["init", "query", "learn", "merge", "health", "connect", "replay", "journal"]:
        with pytest.raises(SystemExit):
            main([command, "--help"])
