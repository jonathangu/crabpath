from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import crabpath.cli as cli
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


def _write_route_graph_payload(path: Path) -> None:
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "deploy start", "summary": "", "metadata": {}},
                {"id": "b", "content": "database setup", "summary": "", "metadata": {}},
                {"id": "c", "content": "deployment checklist", "summary": "", "metadata": {}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.5, "kind": "sibling", "metadata": {}},
                {"source": "a", "target": "c", "weight": 0.5, "kind": "sibling", "metadata": {}},
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_init_command_creates_workspace_graph(tmp_path, capsys) -> None:
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


def test_query_auto_embeds_query_vector_from_index(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(index_path)

    embed_script = tmp_path / "query_embed.py"
    embed_script.write_text(
        "import json\nimport sys\n\nfor line in sys.stdin:\n    payload = json.loads(line)\n    if payload.get('id') == 'query':\n        print(json.dumps({'id': 'query', 'embedding': [1, 0]}))\n",
        encoding="utf-8",
    )

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
            "--embed-command",
            f"{sys.executable} {embed_script}",
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
        main(["query", "seed", "--graph", str(tmp_path / "missing.json"), "--index", str(index_path), "--query-vector", "1,0"])


def test_query_command_keywords_without_index(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["query", "alpha", "--graph", str(graph_path), "--top", "2", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert out["fired"][0] == "a"


def test_query_route_command_selection(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_route_graph_payload(graph_path)

    route_script = tmp_path / "route.py"
    route_script.write_text(
        """import json
import sys

req = json.loads(sys.stdin.read())
print(json.dumps({'selected': ['b']}))
""",
        encoding="utf-8",
    )

    code = main(
        [
            "query",
            "deploy",
            "--graph",
            str(graph_path),
            "--top",
            "1",
            "--route-command",
            f"{sys.executable} {route_script}",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"][0] == "a"
    assert "b" in out["fired"]
    assert "c" not in out["fired"]


def test_query_route_command_failure_falls_back(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_route_graph_payload(graph_path)
    route_script = tmp_path / "route_fail.py"
    route_script.write_text("import sys\nprint('route failed', file=sys.stderr)\nsys.exit(1)\n", encoding="utf-8")

    code = main(
        [
            "query",
            "deploy",
            "--graph",
            str(graph_path),
            "--top",
            "2",
            "--route-command",
            f"{sys.executable} {route_script}",
            "--json",
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out.strip())
    assert "Warning: route command failed" in captured.err
    assert out["fired"]


def test_init_graph_texts_written_when_embedding_fails(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("single paragraph", encoding="utf-8")
    output = tmp_path / "out"
    embed_script = tmp_path / "embed_fail.py"
    embed_script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

    code = main(
        [
            "init",
            "--workspace",
            str(workspace),
            "--output",
            str(output),
            "--embed-command",
            f"{sys.executable} {embed_script}",
        ]
    )

    assert code == 0
    assert (output / "graph.json").exists()
    assert (output / "texts.json").exists()


def test_init_command_with_local_embeddings(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("local embedding test", encoding="utf-8")
    output = tmp_path / "out"

    class FakeSentenceTransformer:
        class _Vector(list):
            def tolist(self):  # type: ignore[override]
                return list(self)

        def __init__(self, model_name: str) -> None:
            assert model_name == "all-MiniLM-L6-v2"

        def encode(self, texts: str | list[str]) -> "FakeSentenceTransformer._Vector":
            if isinstance(texts, list):
                return FakeSentenceTransformer._Vector([self._vector() for _ in texts])
            return FakeSentenceTransformer._Vector(self._vector())

        @staticmethod
        def _vector() -> list[float]:
            return [7.0, 1.0]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    sys.modules.pop("crabpath.embeddings", None)

    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--embed-local"])
    assert code == 0
    index_path = output / "index.json"
    assert index_path.exists()

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload
    assert all(entry == [7.0, 1.0] for entry in index_payload.values())


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
    assert data["graph"]["edges"][0]["source"] == "a"


def test_learn_command_uses_scores(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(
        [
            "learn",
            "--graph",
            str(graph_path),
            "--fired-ids",
            "a,b",
            "--scores",
            json.dumps({"scores": {"a": 0.95, "b": 0.95}}),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["graph"]["edges"][0]["weight"] > 0.7


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
        },
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
    assert len(graph_payload["nodes"]) == 2


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
    for command in ["init", "query", "learn", "merge", "health", "connect", "replay", "journal", "embed"]:
        with pytest.raises(SystemExit):
            main([command, "--help"])
