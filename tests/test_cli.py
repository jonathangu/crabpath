from __future__ import annotations

import json
import os
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

import crabpath.cli as cli
from crabpath.cli import main


@pytest.fixture(autouse=True)
def _disable_auto_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRABPATH_NO_AUTO_DETECT", "1")


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


def _write_route_openai_stub(path: Path, selected_id: str) -> None:
    path.write_text(
        f"""import json


_SELECTED = {json.dumps([selected_id])}


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Resp:
    def __init__(self, selected):
        self.choices = [_Choice(json.dumps({{'selected': selected}}))]


class _Completions:
    def create(self, **kwargs):
        return _Resp(_SELECTED)


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class OpenAI:
    def __init__(self) -> None:
        self.chat = _Chat()
""",
        encoding="utf-8",
    )


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


def _write_route_command(path: Path, selected_id: str) -> None:
    path.write_text(
        f"""import json
import sys

req = json.loads(sys.stdin.read())
print(json.dumps({{'selected': [{json.dumps(selected_id)}]}}))
""",
        encoding="utf-8",
    )


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


def test_auto_detect_openai_from_env(monkeypatch) -> None:
    monkeypatch.delenv("CRABPATH_NO_AUTO_DETECT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert cli._auto_detect_provider() == "openai"


def test_auto_detect_openai_from_dotfile(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CRABPATH_NO_AUTO_DETECT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".zshrc").write_text(
        '  # shell comments ignored\n'
        'export OPENAI_API_KEY="  sk-zshrc  "\n'
        'UNRELATED=ignore\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    monkeypatch.setattr(
        cli.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError()),
    )

    assert cli._auto_detect_provider() == "openai"
    assert os.getenv("OPENAI_API_KEY") == "sk-zshrc"


def test_auto_detect_none(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CRABPATH_NO_AUTO_DETECT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError()))
    assert cli._auto_detect_provider() is None


def test_init_auto_embeds_when_key_available(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.md").write_text(
        "## Alpha\nalpha content\n\n## Beta\nbeta content",
        encoding="utf-8",
    )

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "openai.py").write_text(
        """class _Data:\n    def __init__(self, embedding):\n        self.embedding = embedding\n\n\nclass _Resp:\n    def __init__(self, embedding):\n        self.data = [_Data(embedding)]\n\n\nclass _Embeddings:\n    def create(self, model, input):\n        text = input[0]\n        return _Resp([float(len(text)), 1.0])\n\n\nclass OpenAI:\n    def __init__(self):\n        self.embeddings = _Embeddings()\n""",
        encoding="utf-8",
    )
    env_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), env_path]) if env_path else str(stub_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    output_dir = tmp_path / "crabpath-data"
    code = main(["init", "--workspace", str(workspace), "--output", str(output_dir), "--embed-provider", "openai"])
    assert code == 0
    assert (output_dir / "index.json").exists()


def test_init_command_with_route_provider_no_embed(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path / "out"

    code = main(
        [
            "init",
            "--workspace",
            str(workspace),
            "--output",
            str(output),
            "--route-provider",
            "openai",
            "--no-embed",
        ]
    )
    assert code == 0
    assert (output / "graph.json").exists()
    assert (output / "texts.json").exists()
    assert not (output / "index.json").exists()


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


def test_query_auto_embeds_query_vector_from_index(tmp_path, monkeypatch, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(index_path)

    embed_script = tmp_path / "query_embed.py"
    embed_script.write_text(
        "import json\nimport sys\n\nfor line in sys.stdin:\n    payload = json.loads(line)\n    if payload.get('id') == 'query':\n        print(json.dumps({'id': 'query', 'embedding': [1, 0]}))\n",
        encoding="utf-8",
    )

    def fake_build_embed_command(embed_command: str | None, embed_provider: str | None):
        return [sys.executable, str(embed_script)], None

    monkeypatch.setattr(cli, "_auto_detect_provider", lambda: "openai")
    monkeypatch.setattr(cli, "_build_embed_command", fake_build_embed_command)

    code = main(
        [
            "query",
            "seed",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--no-route",
            "--top",
            "1",
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


def test_route_provider_openai(tmp_path, monkeypatch, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_route_graph_payload(graph_path)

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    _write_route_openai_stub(stub_dir / "openai.py", selected_id="c")
    env_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), env_path]) if env_path else str(stub_dir))

    code = main(
        [
            "query",
            "deploy",
            "--graph",
            str(graph_path),
            "--top",
            "1",
            "--route-provider",
            "openai",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"][0] == "a"
    assert "c" in out["fired"]
    assert "b" not in out["fired"]


def test_query_with_route_command(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_route_graph_payload(graph_path)
    command_script = tmp_path / "route.py"
    _write_route_command(command_script, selected_id="b")

    code = main(
        [
            "query",
            "deploy",
            "--graph",
            str(graph_path),
            "--top",
            "1",
            "--route-command",
            f"{sys.executable} {command_script}",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"][0] == "a"
    assert "b" in out["fired"]
    assert "c" not in out["fired"]


def test_query_route_failure_falls_back(tmp_path, capsys) -> None:
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
    stderr = captured.err
    out = json.loads(captured.out.strip())
    assert "Warning: route command failed" in stderr
    assert out["fired"]


def test_query_creates_node_when_no_results(tmp_path, monkeypatch, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)
    monkeypatch.setattr(cli, "_keyword_seeds", lambda *args, **_kwargs: [])

    def fake_run_route_command(_command: list[str], _query_text: str, _candidates: list[dict[str, str | float]]) -> list[str]:
        return []

    def fake_run_llm_command(_command: list[str], system_prompt: str, _user_prompt: str) -> str:
        if "A query found no relevant documents." in system_prompt:
            return json.dumps(
                {
                    "should_create": True,
                    "content": "created for missing lookup",
                    "summary": "created summary",
                    "reason": "nothing matched",
                }
            )
        return json.dumps({"selected": []})

    monkeypatch.setattr(cli, "_run_route_command", fake_run_route_command)
    monkeypatch.setattr(cli, "_run_llm_command", fake_run_llm_command)

    code = main(
        [
            "query",
            "totally new query",
            "--graph",
            str(graph_path),
            "--route-provider",
            "openai",
            "--top",
            "2",
            "--json",
        ]
    )

    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["created_node_id"] is not None
    assert out["created_node_id"] in out["fired"]

    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert out["created_node_id"] in {node["id"] for node in graph_payload["nodes"]}


def test_query_with_route_provider(tmp_path, monkeypatch, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_route_graph_payload(graph_path)
    index_path.write_text(
        json.dumps({"a": [1, 0], "b": [0, 0], "c": [0, 0]}),
        encoding="utf-8",
    )

    stub_dir = tmp_path / "route_stubs"
    stub_dir.mkdir()
    _write_route_openai_stub(stub_dir / "openai.py", selected_id="b")
    env_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), env_path]) if env_path else str(stub_dir))

    code = main(
        [
            "query",
            "deploy",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--query-vector",
            "1,0",
            "--top",
            "1",
            "--route-provider",
            "openai",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"][0] == "a"
    assert "b" in out["fired"]
    assert "c" not in out["fired"]


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


def test_learn_auto_merge(tmp_path, monkeypatch, capsys) -> None:
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
            ],
        },
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli, "_auto_detect_provider", lambda: "openai")
    monkeypatch.setattr(cli, "_run_llm_command", lambda _command, _system, _user: json.dumps({"should_merge": True}))

    call_counter = {"build": 0}

    def fake_build_llm_command(provider: str) -> tuple[list[str], str | None]:
        call_counter["build"] += 1
        return [sys.executable], None

    monkeypatch.setattr(cli, "_build_llm_command", fake_build_llm_command)

    code = main(
        [
            "learn",
            "--graph",
            str(graph_path),
            "--outcome",
            "1.0",
            "--fired-ids",
            "a,b",
            "--auto-merge",
            "--json",
        ]
    )

    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["merges"]
    assert out["merges"][0]["from"] == ["a", "b"]

    updated = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_payload = updated["graph"] if "graph" in updated else updated
    assert len(graph_payload["nodes"]) == 2
    assert call_counter["build"] == 1


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


def test_query_with_llm_provider_returns_scores(tmp_path, monkeypatch, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_route_graph_payload(graph_path)
    index_path.write_text(json.dumps({"a": [1, 0], "b": [0, 0], "c": [0, 0]}), encoding="utf-8")

    stub_dir = tmp_path / "route_stubs"
    stub_dir.mkdir()
    _write_route_openai_stub(stub_dir / "openai.py", selected_id="c")
    env_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), env_path]) if env_path else str(stub_dir))

    code = main(
        [
            "query",
            "deploy",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--query-vector",
            "1,0",
            "--top",
            "1",
            "--route-provider",
            "openai",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert "scores" in out
    assert out["scores"].get("a") == 1.0


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


def test_init_with_auto_route_provider_message_when_detected(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path / "out"

    monkeypatch.delenv("CRABPATH_NO_AUTO_DETECT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "openai.py").write_text(
        """class _Message:\n    def __init__(self, content):\n        self.content = content\n\n\nclass _Choice:\n    def __init__(self, content):\n        self.message = _Message(content)\n\n\nclass _Resp:\n    def __init__(self, content):\n        self.choices = [_Choice(content)]\n\n\nclass _Chat:\n    def __init__(self):\n        self.completions = self\n\n    def create(self, **_kwargs):\n        return _Resp('{\"selected\": []}')\n\n\nclass OpenAI:\n    def __init__(self):\n        self.chat = _Chat()\n""",
        encoding="utf-8",
    )
    env_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), env_path]) if env_path else str(stub_dir))

    code = main(
        [
            "init",
            "--workspace",
            str(workspace),
            "--output",
            str(output),
            "--no-embed",
        ]
    )
    assert code == 0
    assert "Using LLM for: splitting, summaries" in capsys.readouterr().err


def test_init_bounded_llm_calls_are_limited(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for index in range(5):
        (workspace / f"file_{index}.md").write_text(
            f"token{index} token{index} token{index} token{index} token{index}",
            encoding="utf-8",
        )

    output = tmp_path / "out"

    call_count = {"split": 0, "summary": 0}

    def fake_run_llm(_command: list[str], system_prompt: str, _user_prompt: str) -> str:
        if "Split this document into coherent semantic sections." in system_prompt:
            call_count["split"] += 1
            return json.dumps({"sections": ["section"]})
        if "Write a one-line summary for this node." in system_prompt:
            call_count["summary"] += 1
            return json.dumps({"summary": "summary"})
        return json.dumps({})

    monkeypatch.setattr(cli, "_run_llm_command", fake_run_llm)

    code = main(
        [
            "init",
            "--workspace",
            str(workspace),
            "--output",
            str(output),
            "--route-provider",
            "openai",
            "--llm-split",
            "auto",
            "--llm-split-max-files",
            "2",
            "--llm-split-min-chars",
            "10",
            "--llm-summary",
            "auto",
            "--llm-summary-max-nodes",
            "3",
            "--no-embed",
        ]
    )

    assert code == 0
    assert call_count["split"] <= 2
    assert call_count["summary"] <= 3


def test_init_graph_texts_written_even_when_embedding_fails(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("single paragraph", encoding="utf-8")
    output = tmp_path / "out"
    embed_script = tmp_path / "embed_fail.py"
    embed_script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "init",
                "--workspace",
                str(workspace),
                "--output",
                str(output),
                "--no-route",
                "--embed-command",
                f"{sys.executable} {embed_script}",
            ]
        )

    assert (output / "graph.json").exists()
    assert (output / "texts.json").exists()
    assert not (output / "index.json").exists()


def test_init_output_flag_writes_path(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("single paragraph", encoding="utf-8")
    output = tmp_path / "custom_out"

    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--json"])
    assert code == 0
    assert (output / "graph.json").exists()
    assert (output / "texts.json").exists()


def test_init_command_invalid_arguments(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["init", "--workspace", str(tmp_path)])


def test_health_invalid_graph_path_error_is_clear(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["health", "--graph", str(tmp_path / "nope.json")])


def test_cli_help_text_for_all_commands() -> None:
    for command in ["init", "query", "learn", "merge", "health"]:
        with pytest.raises(SystemExit):
            main([command, "--help"])
