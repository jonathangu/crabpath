"""Tests for CrabPath migration — bootstrap + playback."""

import json

import pytest

from crabpath.migrate import (
    MigrateConfig,
    fallback_llm_split,
    gather_files,
    keyword_router,
    migrate,
    parse_session_logs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path):
    """Create a minimal workspace for testing."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "AGENTS.md").write_text(
        "## Rules\nFollow these rules carefully at all times. "
        "They define how the agent operates.\n\n"
        "## Tools\nUse these tools for coding and browsing. Codex for code, browser for web.\n\n"
        "## Safety\nStay safe. Never expose credentials. Always ask before destructive actions."
    )
    (ws / "SOUL.md").write_text(
        "## Identity\nI am a test agent built to help with development tasks and research.\n\n"
        "## Values\nBe helpful, direct, and honest. Own the outcome and verify facts."
    )
    (ws / "MEMORY.md").write_text(
        "## Key Facts\nThe project started in February 2026. "
        "Main stack is Python and TypeScript.\n\n"
        "## Decisions\nWe decided to use CrabPath for memory management in the agent system."
    )

    # Memory dir
    mem = ws / "memory"
    mem.mkdir()
    (mem / "2026-02-25.md").write_text(
        "## Morning\nWorked on the CrabPath bootstrap system and mitosis module for splitting.\n\n"
        "## Afternoon\nRan simulations showing graph self-organization over 200 queries with decay."
    )

    return ws


def _make_session_log(tmp_path, queries):
    """Create a JSONL session log."""
    log = tmp_path / "session.jsonl"
    with log.open("w") as f:
        for q in queries:
            f.write(json.dumps({"role": "user", "content": q}) + "\n")
    return log


# ---------------------------------------------------------------------------
# Tests: gather_files
# ---------------------------------------------------------------------------


def test_gather_injection_files(tmp_path):
    ws = _make_workspace(tmp_path)
    files = gather_files(ws)
    assert "agents" in files
    assert "soul" in files
    assert "memory" in files


def test_gather_with_memory(tmp_path):
    ws = _make_workspace(tmp_path)
    config = MigrateConfig(include_memory=True)
    files = gather_files(ws, config)
    # memory/*.md files get "memory-" prefix, injection MEMORY.md gets "memory"
    assert any("2026" in k for k in files)


def test_gather_without_memory(tmp_path):
    ws = _make_workspace(tmp_path)
    config = MigrateConfig(include_memory=False)
    files = gather_files(ws, config)
    assert not any(k.startswith("memory-2") for k in files)


# ---------------------------------------------------------------------------
# Tests: parse_session_logs
# ---------------------------------------------------------------------------


def test_parse_jsonl_logs(tmp_path):
    log = _make_session_log(
        tmp_path,
        [
            "how do I use codex for coding tasks",
            "what are the safety rules for credentials",
            "hi",  # short but > 5 chars with JSONL wrapping
        ],
    )
    queries = parse_session_logs([log])
    assert len(queries) >= 2  # At least the two substantive ones


def test_parse_openclaw_jsonl_message_field(tmp_path):
    openclaw_file = tmp_path / "openclaw.jsonl"
    payload = {
        "type": "message",
        "message": (
            "{'role': 'user', 'content': ["
            "{'type': 'text', 'text': 'openclaw query about safety checks'}]"
            "}"
        ),
    }
    openclaw_file.write_text(json.dumps(payload), encoding="utf-8")
    queries = parse_session_logs([openclaw_file])
    assert queries == ["openclaw query about safety checks"]


def test_parse_plain_text_logs(tmp_path):
    log = tmp_path / "plain.txt"
    log.write_text("how do I use codex\nwhat are the safety rules\n")
    queries = parse_session_logs([log])
    assert len(queries) == 2


def test_parse_max_queries(tmp_path):
    log = _make_session_log(tmp_path, [f"query number {i}" for i in range(100)])
    queries = parse_session_logs([log], max_queries=10)
    assert len(queries) == 10


def test_parse_missing_file():
    queries = parse_session_logs(["/nonexistent.jsonl"])
    assert queries == []


# ---------------------------------------------------------------------------
# Tests: keyword_router
# ---------------------------------------------------------------------------


def test_keyword_router_multi_select():
    candidates = [
        ("node-a", 0.5, "codex coding rules workflow"),
        ("node-b", 0.4, "codex worktree reset cleanup"),
        ("node-c", 0.1, "weather forecast"),
    ]
    selected = keyword_router("codex worktree cleanup", candidates)
    assert "node-a" in selected or "node-b" in selected
    assert "node-c" not in selected


def test_keyword_router_trivial():
    candidates = [("node-a", 0.5, "something")]
    selected = keyword_router("hello", candidates)
    assert selected == []


# ---------------------------------------------------------------------------
# Tests: migrate (full pipeline)
# ---------------------------------------------------------------------------


def test_migrate_basic(tmp_path):
    ws = _make_workspace(tmp_path)
    graph, info = migrate(
        workspace_dir=ws,
        config=MigrateConfig(include_memory=False),
        verbose=False,
    )
    assert graph.node_count > 0
    assert graph.edge_count > 0
    assert info["bootstrap"]["files"] >= 3
    assert info["replay"] is None


def test_migrate_with_replay(tmp_path):
    ws = _make_workspace(tmp_path)
    log = _make_session_log(
        tmp_path,
        [
            "what are the rules for agents",
            "how to stay safe with tools",
            "agent safety rules and tools",
            "rules and safety together",
            "what tools follow what rules",
        ]
        * 5,
    )  # 25 queries to generate some co-firing

    graph, info = migrate(
        workspace_dir=ws,
        session_logs=[log],
        config=MigrateConfig(include_memory=False),
        verbose=False,
    )
    assert info["replay"] is not None
    assert info["replay"]["queries_replayed"] == 25


def test_migrate_empty_workspace(tmp_path):
    ws = tmp_path / "empty"
    ws.mkdir()
    with pytest.raises(ValueError, match="No workspace files"):
        migrate(workspace_dir=ws)


# ---------------------------------------------------------------------------
# Tests: fallback_llm_split
# ---------------------------------------------------------------------------


def test_fallback_splits_by_headers():
    content = (
        "## Rules\nFollow these rules carefully at all times. "
        "They define agent behavior.\n\n"
        "## Tools\nUse these tools for coding and browsing. "
        "Codex for code, browser for web."
    )
    result = fallback_llm_split("system", f"---\n{content}\n---")
    parsed = json.loads(result)
    assert len(parsed["sections"]) == 2
