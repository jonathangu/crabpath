from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path

import pytest

from openclawbrain.cli import main
from openclawbrain.hasher import default_embed
from openclawbrain.journal import read_journal
from openclawbrain.state_lock import lock_path_for_state


def _write_graph_payload(path: Path) -> None:
    """ write graph payload."""
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


def _write_index(path: Path, payload: dict[str, list[float]] | None = None) -> None:
    """ write index."""
    if payload is None:
        payload = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_state(
    path: Path,
    graph_payload: dict | None = None,
    index_payload: dict[str, list[float]] | None = None,
    meta: dict[str, object] | None = None,
) -> None:
    """ write state."""
    if graph_payload is None:
        graph_payload = {
            "nodes": [
                {"id": "a", "content": "alpha", "summary": "", "metadata": {"file": "a.md"}},
                {"id": "b", "content": "beta", "summary": "", "metadata": {"file": "b.md"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.7, "kind": "sibling", "metadata": {}},
            ],
        }
    if index_payload is None:
        index_payload = {
            "a": default_embed("alpha"),
            "b": default_embed("beta"),
        }
    if meta is None:
        meta = {"embedder_name": "hash-v1", "embedder_dim": 1024}
    path.write_text(json.dumps({"graph": graph_payload, "index": index_payload, "meta": meta}), encoding="utf-8")


def test_init_command_creates_workspace_graph(tmp_path) -> None:
    """test init command creates workspace graph."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path

    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--embedder", "hash", "--llm", "none"])
    assert code == 0

    graph_path = output / "graph.json"
    texts_path = output / "texts.json"
    index_path = output / "index.json"
    assert graph_path.exists()
    assert texts_path.exists()
    assert index_path.exists()
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_payload = graph_data["graph"] if "graph" in graph_data else graph_data
    assert len(graph_payload["nodes"]) == 1
    assert graph_data.get("meta", {}).get("embedder_name") == "hash-v1"
    state_data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state_data["meta"]["embedder_name"] == "hash-v1"
    texts_data = json.loads(texts_path.read_text(encoding="utf-8"))
    assert len(texts_data) == 1


def test_init_command_with_empty_workspace(tmp_path) -> None:
    """test init command with empty workspace."""
    workspace = tmp_path / "empty"
    workspace.mkdir()
    output = tmp_path / "out"
    output.mkdir()

    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--embedder", "hash", "--llm", "none"])
    assert code == 0
    graph_file = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    graph_data = graph_file["graph"] if "graph" in graph_file else graph_file
    assert graph_data["nodes"] == []
    assert graph_data["edges"] == []
    assert (output / "index.json").exists()


def test_init_sets_authority_metadata_for_mapped_files(tmp_path, monkeypatch) -> None:
    """init assigns authority metadata for files in DEFAULT_AUTHORITY_MAP."""
    import openclawbrain.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "DEFAULT_AUTHORITY_MAP",
        {
            "SOUL.md": "constitutional",
            "AGENTS.md": "constitutional",
            "USER.md": "canonical",
        },
    )

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "SOUL.md").write_text("Constitutional guidance", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("Agent operating rules", encoding="utf-8")
    (workspace / "USER.md").write_text("User preferences", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--embedder", "hash", "--llm", "none"])
    assert code == 0

    state_data = json.loads((output / "state.json").read_text(encoding="utf-8"))
    nodes = state_data["graph"]["nodes"]

    def authorities_for(file_name: str) -> set[str]:
        return {
            str(node.get("metadata", {}).get("authority"))
            for node in nodes
            if node.get("metadata", {}).get("file") == file_name
        }

    assert "constitutional" in authorities_for("SOUL.md")
    assert "constitutional" in authorities_for("AGENTS.md")
    assert "canonical" in authorities_for("USER.md")


def test_query_command_returns_json_with_fired_nodes(tmp_path, capsys) -> None:
    """test query command returns json with fired nodes."""
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(
        index_path,
        {
            "a": default_embed("alpha"),
            "b": default_embed("beta"),
        },
    )

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
    assert out["tier_thresholds"] == {
        "reflex": ">= 0.6",
        "habitual": "0.15 - 0.6",
        "dormant": "< 0.15",
        "inhibitory": "< -0.01",
    }


def test_query_auto_embeds(tmp_path, capsys) -> None:
    """test query auto embeds."""
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(
        index_path,
        {
            "a": default_embed("alpha"),
            "b": default_embed("completely different text"),
        },
    )

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
    assert out["fired"][0] == "a"


def test_query_uses_vector_from_stdin(tmp_path, capsys, monkeypatch) -> None:
    """test query uses vector from stdin."""
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


def test_cli_state_flag_query(tmp_path, capsys) -> None:
    """test cli state flag query."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    code = main(
        [
            "query",
            "alpha",
            "--state",
            str(state_path),
            "--top",
            "2",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert "a" in out["fired"]


def test_query_max_context_chars_cap_in_query_command(tmp_path, capsys) -> None:
    """test query max context chars cap in query command."""
    state_path = tmp_path / "state.json"
    graph_payload = {
        "nodes": [
            {"id": "a", "content": "alpha " * 80, "summary": "", "metadata": {"file": "a.md"}},
            {"id": "b", "content": "zeta " * 80, "summary": "", "metadata": {"file": "b.md"}},
        ],
        "edges": [{"source": "a", "target": "b", "weight": 0.95, "kind": "sibling", "metadata": {}}],
    }
    _write_state(state_path, graph_payload=graph_payload)

    code = main(
        [
            "query",
            "alpha",
            "--state",
            str(state_path),
            "--top",
            "1",
            "--max-context-chars",
            "120",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert len(out["context"]) <= 120


def test_query_journal_written_to_state_directory(tmp_path) -> None:
    """test query journal written to state directory."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    main(
        [
            "query",
            "alpha",
            "--state",
            str(state_path),
            "--json",
        ]
    )
    journal_path = tmp_path / "journal.jsonl"
    assert journal_path.exists()

    entries = read_journal(journal_path=str(journal_path))
    assert entries
    assert entries[-1]["type"] == "query"


def test_cli_state_replay_uses_last_replayed_ts(tmp_path, capsys) -> None:
    """test cli state replay uses last replayed ts."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "user", "content": "alpha", "ts": 2.0}),
                json.dumps({"role": "user", "content": "alpha", "ts": 3.0}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    first = json.loads(capsys.readouterr().out.strip())
    assert first["queries_replayed"] == 3
    assert json.loads(state_path.read_text(encoding="utf-8"))["meta"]["last_replayed_ts"] == 3.0

    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 4.0}),
                json.dumps({"role": "user", "content": "alpha", "ts": 5.0}),
            ]
        ),
        encoding="utf-8",
    )
    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    second = json.loads(capsys.readouterr().out.strip())
    assert second["queries_replayed"] == 2
    assert json.loads(state_path.read_text(encoding="utf-8"))["meta"]["last_replayed_ts"] == 5.0


def test_cli_state_replay_uses_wall_clock_ts_when_session_ts_missing(tmp_path, capsys) -> None:
    """Replay persists wall-clock fallback timestamp and source when query ts is absent."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        json.dumps({"role": "user", "content": "alpha"}),
        encoding="utf-8",
    )

    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 1
    assert isinstance(payload["last_replayed_ts"], float)
    assert payload["last_replayed_ts_source"] == "wall_clock"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(state["meta"]["last_replayed_ts"], float)
    assert state["meta"]["last_replayed_ts_source"] == "wall_clock"

    code = main(["status", "--state", str(state_path), "--json"])
    assert code == 0
    status_payload = json.loads(capsys.readouterr().out.strip())
    assert status_payload["last_replayed"] != "never"


def test_cli_replay_discovers_reset_session_files(tmp_path, capsys) -> None:
    """test replay discovers and processes reset session files."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "current.jsonl").write_text(
        json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
        encoding="utf-8",
    )
    (sessions / "current.jsonl.reset.2026-02-27").write_text(
        json.dumps({"role": "user", "content": "alpha", "ts": 2.0}),
        encoding="utf-8",
    )

    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 2
    assert json.loads(state_path.read_text(encoding="utf-8"))["meta"]["last_replayed_ts"] == 2.0


def test_cli_replay_fails_cleanly_when_state_lock_held_unless_force(tmp_path, capsys, monkeypatch) -> None:
    """replay exits with a clean lock error unless --force is set."""
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.delenv("OPENCLAWBRAIN_STATE_LOCK_FORCE", raising=False)
    monkeypatch.delenv("OCB_STATE_LOCK_FORCE", raising=False)
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(json.dumps({"role": "user", "content": "alpha", "ts": 1.0}), encoding="utf-8")

    lock_path = lock_path_for_state(state_path)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="state write lock is already held"):
            main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])

        code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--force", "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["queries_replayed"] == 1
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_cli_replay_with_fast_learning_writes_learning_events_and_injects_nodes(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """test cli replay with fast learning writes learning events and injects nodes."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "do not do that", "ts": 1.0}),
                json.dumps({"role": "user", "content": "how to deploy", "ts": 2.0}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps(
            {
                "corrections": [
                    {
                        "content": "Never answer with sensitive data.",
                        "context": "user corrected behavior",
                        "severity": "high",
                    }
                ],
                "teachings": [],
                "reinforcements": [],
            }
        )

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--ignore-checkpoint",
            "--json",
        ]
    )
    assert code == 0
    first = json.loads(capsys.readouterr().out.strip())

    assert "fast_learning" in first
    assert first["fast_learning"]["events_injected"] >= 1
    assert first["fast_learning"]["events_appended"] >= 1

    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    learning_nodes = [node for node in first_state["graph"]["nodes"] if str(node["id"]).startswith("learning::")]
    assert learning_nodes

    events_path = Path(first["fast_learning"]["learning_events_path"])
    assert events_path.exists()
    initial_events = len(events_path.read_text(encoding="utf-8").splitlines())
    assert first["fast_learning"]["learning_events_path"] == str(state_path.parent / "learning_events.jsonl")

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--ignore-checkpoint",
            "--json",
        ]
    )
    assert code == 0
    second = json.loads(capsys.readouterr().out.strip())
    assert second["fast_learning"]["events_appended"] == 0
    assert second["fast_learning"]["events_injected"] == 0
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == initial_events

    second_state = json.loads(state_path.read_text(encoding="utf-8"))
    second_learning_nodes = [
        node for node in second_state["graph"]["nodes"] if str(node["id"]).startswith("learning::")
    ]
    assert len(second_learning_nodes) == len(learning_nodes)


def test_query_command_text_output_includes_node_ids(tmp_path, capsys) -> None:
    """test query text output includes node ids."""
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    graph_payload = {
        "graph": {
            "nodes": [
                {
                    "id": "deploy.md::0",
                    "content": "How to create a hotfix",
                    "summary": "",
                    "metadata": {"file": "deploy.md"},
                },
                {
                    "id": "deploy.md::1",
                    "content": "CI must pass for hotfixes",
                    "summary": "",
                    "metadata": {"file": "deploy.md"},
                },
            ],
            "edges": [
                {
                    "source": "deploy.md::0",
                    "target": "deploy.md::1",
                    "weight": 0.85,
                    "kind": "sibling",
                    "metadata": {},
                }
            ],
        }
    }
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
    _write_index(index_path, {"deploy.md::0": default_embed("deploy"), "deploy.md::1": default_embed("hotfix")})

    code = main(
        [
            "query",
            "deploy hotfix",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "deploy.md::0" in out
    assert "deploy.md::1" in out


def test_cli_dimension_mismatch_error(tmp_path) -> None:
    """test cli dimension mismatch error."""
    state_path = tmp_path / "state.json"
    _write_state(
        state_path,
        index_payload={"a": [1.0, 0.0] * 768, "b": [0.0, 1.0] * 768},
        meta={"embedder_name": "openai-text-embedding-3-small", "embedder_dim": 1536},
    )

    with pytest.raises(SystemExit, match=r"Index was built with openai-text-embedding-3-small \(dim=1536\). CLI hash embedder uses dim=1024. Dimension mismatch. Use --query-vector-stdin with matching embedder."):
        main(["query", "alpha", "--state", str(state_path), "--embedder", "hash", "--top", "2", "--json"])
        main(["query", "alpha", "--state", str(state_path), "--top", "2", "--json"])


def test_query_command_error_on_missing_graph(tmp_path) -> None:
    """test query command error on missing graph."""
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    with pytest.raises(SystemExit):
        main(["query", "seed", "--graph", str(tmp_path / "missing.json"), "--index", str(index_path)])


def test_query_command_keywords_without_index(tmp_path, capsys) -> None:
    """test query command keywords without index."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["query", "alpha", "--graph", str(graph_path), "--top", "2", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert out["fired"][0] == "a"


def test_learn_command_updates_graph_weights(tmp_path) -> None:
    """test learn command updates graph weights."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["learn", "--graph", str(graph_path), "--outcome", "1.0", "--fired-ids", "a,b"])
    assert code == 0

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["graph"]["edges"][0]["weight"] > 0.7


def test_cli_state_flag_learn(tmp_path) -> None:
    """test cli state flag learn."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    original = json.loads(state_path.read_text(encoding="utf-8"))
    original_weight = original["graph"]["edges"][0]["weight"]
    code = main(
        [
            "learn",
            "--state",
            str(state_path),
            "--outcome",
            "1.0",
            "--fired-ids",
            "a,b",
        ]
    )
    assert code == 0

    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["graph"]["edges"][0]["weight"] > original_weight


def test_cli_state_flag_health(tmp_path, capsys) -> None:
    """test cli state flag health."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    code = main(["health", "--state", str(state_path), "--json"])
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


def test_learn_command_supports_json_output(tmp_path, capsys) -> None:
    """test learn command supports json output."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["learn", "--graph", str(graph_path), "--outcome", "-1.0", "--fired-ids", "a,b", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out.strip())
    assert data["edges_updated"] >= 1
    assert data["max_weight_delta"] >= 0.0


def test_merge_command_suggests_and_applies(tmp_path, capsys) -> None:
    """test merge command suggests and applies."""
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
    """test health command outputs all metrics."""
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


def test_health_command_text_output_is_readable(tmp_path, capsys) -> None:
    """test health command has human-readable non-json output."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["health", "--graph", str(graph_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Brain health:" in out
    assert "Nodes:" in out
    assert "Edges:" in out
    assert "Cross-file edges:" in out


def test_cli_help_text_for_commands() -> None:
    """test cli help text for commands."""
    for command in [
        "init",
        "query",
        "learn",
        "merge",
        "health",
        "connect",
        "replay",
        "harvest",
        "journal",
        "sync",
        "serve",
    ]:
        with pytest.raises(SystemExit):
            main([command, "--help"])


def test_serve_command_prints_banner_and_calls_socket_server(monkeypatch, capsys) -> None:
    """serve should print a startup banner and delegate to socket_server.main."""
    called: list[list[str]] = []

    def fake_default_socket_path(state_path: str) -> str:
        return f"/tmp/{Path(state_path).parent.name}/daemon.sock"

    def fake_socket_server_main(argv: list[str] | None = None) -> int:
        called.append(list(argv or []))
        return 0

    from openclawbrain import socket_server

    monkeypatch.setattr(socket_server, "_default_socket_path", fake_default_socket_path)
    monkeypatch.setattr(socket_server, "main", fake_socket_server_main)

    code = main(["serve", "--state", "~/agent/state.json"])
    assert code == 0
    assert called == [["--state", str(Path("~/agent/state.json").expanduser())]]

    err = capsys.readouterr().err
    assert "OpenClawBrain socket service (foreground)" in err
    assert "socket path: /tmp/agent/daemon.sock" in err
    assert f"state path: {Path('~/agent/state.json').expanduser()}" in err
    assert "query status: openclawbrain status --state" in err
    assert "stop: Ctrl-C" in err


def test_serve_command_passes_explicit_socket_path(monkeypatch) -> None:
    """serve should pass --socket-path through to the socket server."""
    called: list[list[str]] = []

    def fake_socket_server_main(argv: list[str] | None = None) -> int:
        called.append(list(argv or []))
        return 0

    from openclawbrain import socket_server

    monkeypatch.setattr(socket_server, "main", fake_socket_server_main)

    code = main(["serve", "--state", "/tmp/state.json", "--socket-path", "/tmp/d.sock"])
    assert code == 0
    assert called == [["--state", "/tmp/state.json", "--socket-path", "/tmp/d.sock"]]
def test_inject_command_defaults_connect_min_sim_for_hash_embedder(tmp_path, capsys) -> None:
    """test inject uses a zero min similarity threshold by default for hash embeds."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    code = main(
        [
            "inject",
            "--state",
            str(state_path),
            "--id",
            "fix::ci",
            "--content",
            "alpha",
            "--type",
            "CORRECTION",
            "--json",
        ]
    )
    assert code == 0

    out = json.loads(capsys.readouterr().out.strip())
    assert out["connected_to"]
    assert out["inhibitory_edges_created"] > 0


def test_cli_replay_default_runs_full_learning(tmp_path, capsys, monkeypatch) -> None:
    """replay without flags defaults to full-learning (fast-learning + harvest)."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "do not do that", "ts": 1.0}),
                json.dumps({"role": "user", "content": "how to deploy", "ts": 2.0}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps(
            {
                "corrections": [
                    {
                        "content": "Default full-learning correction.",
                        "context": "test",
                        "severity": "high",
                    }
                ],
                "teachings": [],
                "reinforcements": [],
            }
        )

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--ignore-checkpoint",
            "--json",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out.strip())

    # Default should trigger fast_learning and harvest
    assert "fast_learning" in result
    assert result["fast_learning"]["events_appended"] >= 1
    assert "harvest" in result


def test_cli_replay_edges_only_skips_learning(tmp_path, capsys) -> None:
    """replay --edges-only does not run fast_learning or harvest."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--json",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out.strip())

    assert "fast_learning" not in result
    assert "harvest" not in result
    assert result["queries_replayed"] == 1


def test_cli_replay_tool_result_flags_forwarded_json_and_text(tmp_path, capsys, monkeypatch) -> None:
    """replay toolResult controls are forwarded to the replay loader."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text('{"role":"user","content":"alpha","ts":1.0}\n', encoding="utf-8")

    import openclawbrain.cli as cli_module

    seen: list[dict[str, object]] = []

    def fake_load_interactions_for_replay(session_paths, since_lines=None, **kwargs):
        seen.append(dict(kwargs))
        interactions = [
            {
                "query": "alpha",
                "response": None,
                "tool_calls": [],
                "ts": 1.0,
                "source": str(sessions),
                "line_no": 1,
            }
        ]
        return interactions, {str(sessions): 1}

    monkeypatch.setattr(cli_module, "load_interactions_for_replay", fake_load_interactions_for_replay)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--no-include-tool-results",
            "--tool-result-allowlist",
            "image,summarize",
            "--tool-result-max-chars",
            "123",
            "--json",
        ]
    )
    assert code == 0
    # JSON output should parse.
    json.loads(capsys.readouterr().out.strip())
    assert seen[-1]["include_tool_results"] is False
    assert seen[-1]["tool_result_allowlist"] == {"image", "summarize"}
    assert seen[-1]["tool_result_max_chars"] == 123

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--include-tool-results",
            "--tool-result-allowlist",
            "openai-whisper",
            "--tool-result-max-chars",
            "456",
        ]
    )
    assert code == 0
    assert seen[-1]["include_tool_results"] is True
    assert seen[-1]["tool_result_allowlist"] == {"openai-whisper"}
    assert seen[-1]["tool_result_max_chars"] == 456


def test_cli_replay_progress_events_jsonl(tmp_path, capsys) -> None:
    """replay emits JSONL progress events when --json and --progress-every are set."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--progress-every",
            "1",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert '{"type": "progress", "phase": "replay"' in out


def test_cli_replay_fast_learning_progress_events_jsonl(tmp_path, capsys, monkeypatch) -> None:
    """replay emits fast-learning JSONL progress events when enabled."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "that is wrong", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps(
            {
                "corrections": [{"content": "Use the runbook.", "context": "ctx", "severity": "high"}],
                "teachings": [],
                "reinforcements": [],
            }
        )

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--stop-after-fast-learning",
            "--progress-every",
            "1",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert '{"type": "progress", "phase": "fast_learning"' in out


def test_cli_replay_show_checkpoint_json_uses_new_schema_fixture(tmp_path, capsys) -> None:
    """replay --show-checkpoint emits stable JSON status for new schema checkpoints."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "new_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--checkpoint",
            str(checkpoint_path),
            "--show-checkpoint",
            "--resume",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["type"] == "checkpoint_status"
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["schema_version"] == 1
    assert payload["fast_learning"]["windows_processed"] == 4
    assert payload["fast_learning"]["windows_total"] == 10
    assert payload["fast_learning"]["status"] == "running"
    assert payload["replay"]["queries_processed"] == 8
    assert payload["replay"]["queries_total"] == 16
    assert payload["replay"]["merge_batches"] == 2
    assert payload["resume"]["would_take_effect"] is True


def test_cli_replay_show_checkpoint_text_legacy_fixture_warns(tmp_path, capsys) -> None:
    """replay --show-checkpoint warns when legacy top-level sessions are used."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "legacy_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.warns(UserWarning, match="legacy top-level 'sessions' offsets"):
        code = main(
            [
                "replay",
                "--state",
                str(state_path),
                "--checkpoint",
                str(checkpoint_path),
                "--show-checkpoint",
                "--resume",
            ]
        )
    assert code == 0
    out = capsys.readouterr().out
    assert f"Checkpoint: {checkpoint_path}" in out
    assert "Schema version: 0" in out
    assert "Resume would take effect: True (resume=True, ignore_checkpoint=False)" in out


def test_cli_replay_startup_banner_printed_for_text_mode(tmp_path, capsys) -> None:
    """replay prints startup banner in non-JSON mode."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "Replay startup:" in err
    assert "resume: False" in err
    assert "ignore_checkpoint: False" in err
    assert "phases: replay" in err


def test_cli_replay_quiet_suppresses_banners_and_progress(tmp_path, capsys) -> None:
    """quiet replay suppresses status banners and progress lines on stderr."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--progress-every",
            "1",
            "--quiet",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "Loaded " not in err
    assert "[replay]" not in err


def test_cli_replay_alias_extract_learning_events(tmp_path, capsys, monkeypatch) -> None:
    """--extract-learning-events is accepted as alias for --fast-learning."""
    from openclawbrain.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s",
        "--extract-learning-events", "--stop-after-fast-learning",
    ])
    assert args.fast_learning is True


def test_cli_replay_alias_full_pipeline(tmp_path, capsys, monkeypatch) -> None:
    """--full-pipeline is accepted as alias for --full-learning."""
    from openclawbrain.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s",
        "--full-pipeline",
    ])
    assert args.full_learning is True



def test_cli_replay_writes_checkpoint_and_resume_uses_it(tmp_path, capsys) -> None:
    """replay writes checkpoint and resume only replays new lines."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--checkpoint",
            str(checkpoint_path),
            "--checkpoint-every",
            "1",
            "--resume",
            "--json",
        ]
    )
    assert code == 0
    first_payload = json.loads(capsys.readouterr().out.strip())
    assert first_payload["queries_replayed"] == 1
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    replay_checkpoint = checkpoint.get("replay", {})
    assert replay_checkpoint.get("sessions")

    with sessions.open("a", encoding="utf-8") as handle:
        handle.write("\n" + json.dumps({"role": "user", "content": "beta", "ts": 2.0}))
        handle.write("\n" + json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}))

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--checkpoint",
            str(checkpoint_path),
            "--resume",
            "--json",
        ]
    )
    assert code == 0
    second_payload = json.loads(capsys.readouterr().out.strip())
    assert second_payload["queries_replayed"] == 1
    assert second_payload["last_replayed_ts"] == 2.1


def test_cli_replay_resume_legacy_checkpoint_warns_and_uses_offsets(tmp_path, capsys) -> None:
    """replay resume falls back to legacy top-level sessions offsets."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )
    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "legacy_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint = json.loads(fixture.read_text(encoding="utf-8"))
    checkpoint["sessions"] = {str(sessions): 2}
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.warns(UserWarning, match="replay checkpoint missing phase-scoped sessions"):
        code = main(
            [
                "replay",
                "--state",
                str(state_path),
                "--sessions",
                str(sessions),
                "--edges-only",
                "--checkpoint",
                str(checkpoint_path),
                "--resume",
                "--json",
            ]
        )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 1
    assert payload["last_replayed_ts"] == 2.1


def test_cli_replay_stop_after_fast_learning(tmp_path, capsys, monkeypatch) -> None:
    """stop-after-fast-learning exits before replay and harvest."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "do not do that", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps(
            {
                "corrections": [{"content": "Never do that.", "context": "ctx", "severity": "high"}],
                "teachings": [],
                "reinforcements": [],
            }
        )

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--stop-after-fast-learning",
            "--ignore-checkpoint",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["stopped_after_fast_learning"] is True
    assert "fast_learning" in payload
    assert "harvest" not in payload
    assert "queries_replayed" not in payload


def test_cli_replay_fast_learning_resume_legacy_checkpoint_warns(tmp_path, capsys, monkeypatch) -> None:
    """fast-learning resume falls back to legacy top-level sessions offsets."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "first", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "second", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps({"corrections": [], "teachings": [], "reinforcements": []})

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "legacy_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint = json.loads(fixture.read_text(encoding="utf-8"))
    checkpoint["sessions"] = {str(sessions): 2}
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.warns(UserWarning, match="fast_learning checkpoint missing phase-scoped sessions"):
        code = main(
            [
                "replay",
                "--state",
                str(state_path),
                "--sessions",
                str(sessions),
                "--fast-learning",
                "--stop-after-fast-learning",
                "--checkpoint",
                str(checkpoint_path),
                "--resume",
                "--json",
            ]
        )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["stopped_after_fast_learning"] is True


def test_cli_replay_parallel_mode_v0(tmp_path, capsys) -> None:
    """replay supports simple parallel reducer mode."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
                json.dumps({"role": "user", "content": "alpha beta", "ts": 3.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 3.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--replay-workers",
            "2",
            "--checkpoint-every",
            "1",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 3
    assert payload["replay_workers"] == 2
    assert payload["merge_batches"] >= 1


def test_cli_init_auto_embedder_falls_back_to_hash(tmp_path, monkeypatch) -> None:
    """init with auto embedder and no OPENAI_API_KEY uses hash embedder."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    code = main(["init", "--workspace", str(workspace), "--output", str(output)])
    assert code == 0

    state_data = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert state_data["meta"]["embedder_name"] == "hash-v1"
