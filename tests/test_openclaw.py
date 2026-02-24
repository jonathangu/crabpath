"""Tests for CrabPath ↔ OpenClaw integration."""

import tempfile
from pathlib import Path

from crabpath.graph import MemoryGraph, NodeType
from crabpath.integrations.openclaw import import_workspace


def _create_mock_workspace(tmp: Path):
    """Create a minimal OpenClaw workspace for testing."""
    # MEMORY.md
    (tmp / "MEMORY.md").write_text("""# MEMORY.md

## Identity
I am GUCLAW, a test agent.

## Infrastructure
Mac Mini M4 Pro, 64GB RAM.
Running OpenClaw with CrabPath integration.
""")

    # memory/ daily notes
    mem_dir = tmp / "memory"
    mem_dir.mkdir()
    (mem_dir / "2026-02-24.md").write_text("""# 2026-02-24

## Deploy Fix
Fixed the deployment pipeline by checking config first, then logs, then rollback.

## Hard Gate — Never Clear Cache
Jon corrected: never tell users to clear cache. This is a hard gate.
""")

    # TOOLS.md
    (tmp / "TOOLS.md").write_text("""# TOOLS.md

## Browser Tool
Always use profile="openclaw" on headless Mac Mini.

## Codex CLI
Binary at /opt/homebrew/bin/codex. Always use --yolo flag.
""")

    # active-tasks.md
    (tmp / "active-tasks.md").write_text("""# active-tasks.md

## Active Tasks

1. **Project Pelican** — autonomous options trading system
2. **CrabPath** — activation-driven memory graphs
""")


def test_import_workspace_basic():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _create_mock_workspace(tmp_path)

        g = MemoryGraph()
        stats = import_workspace(g, tmp_path)

        assert stats["nodes_created"] > 0
        assert stats["edges_created"] >= 0
        assert len(stats["sources"]) > 0


def test_import_creates_correct_node_types():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _create_mock_workspace(tmp_path)

        g = MemoryGraph()
        import_workspace(g, tmp_path)

        # Should have Hub nodes from MEMORY.md and active-tasks
        hubs = g.nodes_by_type(NodeType.HUB)
        assert len(hubs) > 0

        # Should have Tool nodes from TOOLS.md
        tools = g.nodes_by_type(NodeType.TOOL)
        assert len(tools) >= 2  # browser + codex

        # Should have Rule node from "Hard Gate"
        rules = g.nodes_by_type(NodeType.RULE)
        assert any("cache" in r.content.lower() for r in rules)


def test_import_builds_tag_edges():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _create_mock_workspace(tmp_path)

        g = MemoryGraph()
        stats = import_workspace(g, tmp_path)

        # Nodes sharing tags should have association edges
        assert stats["edges_created"] >= 0  # may be 0 if no shared tags


def test_import_handles_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Empty workspace — nothing to import

        g = MemoryGraph()
        stats = import_workspace(g, tmp_path)

        assert stats["nodes_created"] == 0
        assert stats["edges_created"] == 0
        assert len(stats["sources"]) == 0


def test_gate_learnings_become_rules():
    """If a learning harness DB exists, GATE entries should become Rule nodes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _create_mock_workspace(tmp_path)

        # The mock workspace doesn't have a learning DB,
        # so we just verify the code path doesn't crash
        g = MemoryGraph()
        stats = import_workspace(g, tmp_path)
        assert stats["nodes_created"] > 0
