from __future__ import annotations

import importlib

from examples import quickstart


def test_examples_import_cleanly() -> None:
    importlib.reload(quickstart)
    importlib.import_module("examples.hello_world")
    importlib.import_module("examples.openai_agent")
    importlib.import_module("examples.langchain_adapter")


def test_quickstart_main_runs_with_temporary_workspace(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "toy_workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("Agent identity", encoding="utf-8")
    (workspace / "SOUL.md").write_text("Core behavior", encoding="utf-8")
    (workspace / "TOOLS.md").write_text("Tools", encoding="utf-8")
    (workspace / "USER.md").write_text("Users", encoding="utf-8")
    (workspace / "MEMORY.md").write_text("Memory notes", encoding="utf-8")

    graph_path = tmp_path / "toy_workspace_graph.json"
    monkeypatch.setattr(quickstart, "WORKSPACE_PATH", workspace)
    monkeypatch.setattr(quickstart, "GRAPH_PATH", graph_path)

    quickstart.main()

    assert graph_path.exists()
