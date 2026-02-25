"""Tests for CrabPath mitosis — recursive cell division."""

import json
import pytest
from crabpath.graph import Graph, Node, Edge
from crabpath.mitosis import (
    MitosisConfig,
    MitosisState,
    split_node,
    split_with_llm,
    check_reconvergence,
    merge_and_resplit,
    bootstrap_workspace,
    mitosis_maintenance,
    _fallback_split,
    _make_chunk_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_llm_4way(system: str, user: str) -> str:
    """Mock LLM that splits content by paragraphs or evenly."""
    # Extract content from the prompt
    content = user.split("---\n", 1)[-1].rsplit("\n---", 1)[0] if "---" in user else user

    # Simple split: by double newlines, then merge to get 4
    parts = [p.strip() for p in content.split("\n\n") if p.strip()]

    if len(parts) >= 4:
        # Merge to exactly 4
        chunk_size = max(len(parts) // 4, 1)
        sections = []
        for i in range(4):
            start = i * chunk_size
            end = start + chunk_size if i < 3 else len(parts)
            sections.append("\n\n".join(parts[start:end]))
    elif len(parts) >= 2:
        sections = parts
    else:
        # Split by characters
        n = len(content) // 4
        sections = [content[i*n:(i+1)*n if i < 3 else len(content)] for i in range(4)]

    return json.dumps({"sections": sections})


def _mock_llm_different_split(system: str, user: str) -> str:
    """Mock LLM that splits differently (for re-split testing)."""
    content = user.split("---\n", 1)[-1].rsplit("\n---", 1)[0] if "---" in user else user
    n = len(content) // 4
    # Split at different offsets than the first time
    offset = n // 3
    sections = [
        content[:n + offset],
        content[n + offset:2*n],
        content[2*n:3*n - offset],
        content[3*n - offset:],
    ]
    return json.dumps({"sections": [s for s in sections if s.strip()]})


SAMPLE_CONTENT = """## Identity
I am GUCLAW. Jonathan's high-trust operator.

## Tools
Use Codex for coding. Use browser for web tasks.

## Safety Rules
Never expose credentials. Never delete without asking.

## Memory
MEMORY.md is long-term. Daily notes are raw logs."""


SMALL_CONTENT = "Too small."


# ---------------------------------------------------------------------------
# Tests: split_with_llm
# ---------------------------------------------------------------------------

def test_split_with_llm_basic():
    sections = split_with_llm(SAMPLE_CONTENT, 4, _mock_llm_4way)
    assert len(sections) >= 2
    # All content should be preserved (approximately)
    joined = "\n\n".join(sections)
    assert len(joined) > len(SAMPLE_CONTENT) * 0.5  # At least half preserved


def test_split_with_llm_fallback_on_bad_json():
    def bad_llm(s, u):
        return "not json at all"

    sections = split_with_llm(SAMPLE_CONTENT, 4, bad_llm)
    assert len(sections) >= 2  # Should fall back to header splitting


def test_split_with_llm_fallback_on_empty():
    def empty_llm(s, u):
        return json.dumps({"sections": []})

    sections = split_with_llm(SAMPLE_CONTENT, 4, empty_llm)
    assert len(sections) >= 2  # Should fall back


def test_split_with_llm_fallback_on_exception():
    def exploding_llm(s, u):
        raise RuntimeError("API error")

    sections = split_with_llm(SAMPLE_CONTENT, 4, exploding_llm)
    assert len(sections) >= 2  # Should fall back


# ---------------------------------------------------------------------------
# Tests: fallback_split
# ---------------------------------------------------------------------------

def test_fallback_split_by_headers():
    content = "## A\nFirst section\n\n## B\nSecond section\n\n## C\nThird\n\n## D\nFourth"
    sections = _fallback_split(content, 4)
    assert len(sections) == 4


def test_fallback_split_by_chars():
    content = "x" * 400  # No headers
    sections = _fallback_split(content, 4)
    assert len(sections) == 4
    assert all(len(s) > 0 for s in sections)


def test_fallback_split_merges_small():
    content = "## A\nA\n\n## B\nB\n\n## C\nC\n\n## D\nD\n\n## E\nE\n\n## F\nF"
    sections = _fallback_split(content, 4)
    assert len(sections) == 4


# ---------------------------------------------------------------------------
# Tests: split_node
# ---------------------------------------------------------------------------

def test_split_node_basic():
    g = Graph()
    g.add_node(Node(id="soul", content=SAMPLE_CONTENT, type="workspace_file"))
    state = MitosisState()

    result = split_node(g, "soul", _mock_llm_4way, state)

    assert result is not None
    assert result.parent_id == "soul"
    assert len(result.chunk_ids) >= 2
    assert result.generation == 1

    # Parent should be removed
    assert g.get_node("soul") is None

    # Chunks should exist
    for cid in result.chunk_ids:
        assert g.get_node(cid) is not None

    # Sibling edges should exist (all-to-all)
    for i, src in enumerate(result.chunk_ids):
        for j, tgt in enumerate(result.chunk_ids):
            if i != j:
                edge = g.get_edge(src, tgt)
                assert edge is not None
                assert edge.weight == 1.0

    # State should be updated
    assert "soul" in state.families
    assert state.generations["soul"] == 1


def test_split_node_too_small():
    g = Graph()
    g.add_node(Node(id="tiny", content=SMALL_CONTENT))
    state = MitosisState()

    result = split_node(g, "tiny", _mock_llm_4way, state)
    assert result is None
    assert g.get_node("tiny") is not None  # Node should still exist


def test_split_node_nonexistent():
    g = Graph()
    state = MitosisState()
    result = split_node(g, "nope", _mock_llm_4way, state)
    assert result is None


def test_split_node_preserves_edges():
    g = Graph()
    g.add_node(Node(id="soul", content=SAMPLE_CONTENT))
    g.add_node(Node(id="other", content="Other node"))
    g.add_edge(Edge(source="other", target="soul", weight=0.8))
    g.add_edge(Edge(source="soul", target="other", weight=0.5))
    state = MitosisState()

    result = split_node(g, "soul", _mock_llm_4way, state)
    assert result is not None

    # All chunks should have inherited edges from/to "other"
    for cid in result.chunk_ids:
        incoming = g.get_edge("other", cid)
        assert incoming is not None
        assert incoming.weight == 0.8

        outgoing = g.get_edge(cid, "other")
        assert outgoing is not None
        assert outgoing.weight == 0.5


def test_split_node_carries_protection():
    g = Graph()
    g.add_node(Node(id="safe", content=SAMPLE_CONTENT, metadata={"protected": True}))
    state = MitosisState()

    result = split_node(g, "safe", _mock_llm_4way, state)
    assert result is not None

    for cid in result.chunk_ids:
        node = g.get_node(cid)
        assert node.metadata.get("protected") is True


# ---------------------------------------------------------------------------
# Tests: check_reconvergence
# ---------------------------------------------------------------------------

def test_reconvergence_detected():
    g = Graph()
    state = MitosisState()

    # Create a family of 3 chunks with all edges at 1.0
    chunk_ids = ["parent::chunk-0-aaa", "parent::chunk-1-bbb", "parent::chunk-2-ccc"]
    for cid in chunk_ids:
        g.add_node(Node(id=cid, content="chunk content"))

    for i, src in enumerate(chunk_ids):
        for j, tgt in enumerate(chunk_ids):
            if i != j:
                g.add_edge(Edge(source=src, target=tgt, weight=1.0))

    state.families["parent"] = chunk_ids

    reconverged = check_reconvergence(g, state)
    assert "parent" in reconverged


def test_reconvergence_not_detected_when_decayed():
    g = Graph()
    state = MitosisState()

    chunk_ids = ["parent::chunk-0-aaa", "parent::chunk-1-bbb"]
    for cid in chunk_ids:
        g.add_node(Node(id=cid, content="chunk content"))

    # One edge decayed
    g.add_edge(Edge(source=chunk_ids[0], target=chunk_ids[1], weight=1.0))
    g.add_edge(Edge(source=chunk_ids[1], target=chunk_ids[0], weight=0.3))

    state.families["parent"] = chunk_ids

    reconverged = check_reconvergence(g, state)
    assert "parent" not in reconverged


# ---------------------------------------------------------------------------
# Tests: merge_and_resplit
# ---------------------------------------------------------------------------

def test_merge_and_resplit():
    g = Graph()
    state = MitosisState()

    # First split
    g.add_node(Node(id="tools", content=SAMPLE_CONTENT))
    result1 = split_node(g, "tools", _mock_llm_4way, state)
    assert result1 is not None

    gen1_chunks = list(result1.chunk_ids)

    # Force reconvergence (set all sibling edges to 1.0)
    for i, src in enumerate(gen1_chunks):
        for j, tgt in enumerate(gen1_chunks):
            if i != j:
                edge = g.get_edge(src, tgt)
                if edge:
                    edge.weight = 1.0

    # Re-split
    result2 = merge_and_resplit(g, "tools", _mock_llm_different_split, state)
    assert result2 is not None
    assert result2.generation == 2

    # Old chunks should be gone
    for cid in gen1_chunks:
        assert g.get_node(cid) is None

    # New chunks should exist
    for cid in result2.chunk_ids:
        assert g.get_node(cid) is not None


# ---------------------------------------------------------------------------
# Tests: bootstrap_workspace
# ---------------------------------------------------------------------------

def test_bootstrap_workspace():
    g = Graph()
    state = MitosisState()
    files = {
        "soul": SAMPLE_CONTENT,
        "tools": SAMPLE_CONTENT + "\n\n## Extra\nMore tools content here.",
    }

    results = bootstrap_workspace(g, files, _mock_llm_4way, state)

    assert len(results) == 2
    assert g.node_count >= 4  # At least 2 chunks per file
    assert g.edge_count > 0

    # Both families tracked
    assert "soul" in state.families
    assert "tools" in state.families


# ---------------------------------------------------------------------------
# Tests: mitosis_maintenance
# ---------------------------------------------------------------------------

def test_maintenance_detects_and_resplits():
    g = Graph()
    state = MitosisState()

    # Bootstrap
    files = {"test": SAMPLE_CONTENT}
    bootstrap_workspace(g, files, _mock_llm_4way, state)

    # Force all sibling edges to reconverge
    chunk_ids = state.families["test"]
    for i, src in enumerate(chunk_ids):
        for j, tgt in enumerate(chunk_ids):
            if i != j:
                edge = g.get_edge(src, tgt)
                if edge:
                    edge.weight = 1.0

    # Run maintenance
    result = mitosis_maintenance(g, _mock_llm_different_split, state)

    assert result["reconverged_families"] >= 1
    assert result["resplit_count"] >= 1


def test_maintenance_no_action_when_decayed():
    g = Graph()
    state = MitosisState()

    # Bootstrap
    files = {"test": SAMPLE_CONTENT}
    bootstrap_workspace(g, files, _mock_llm_4way, state)

    # Decay some edges
    chunk_ids = state.families["test"]
    if len(chunk_ids) >= 2:
        edge = g.get_edge(chunk_ids[0], chunk_ids[1])
        if edge:
            edge.weight = 0.3  # Decayed — chunks are separating

    result = mitosis_maintenance(g, _mock_llm_4way, state)
    assert result["reconverged_families"] == 0


# ---------------------------------------------------------------------------
# Tests: MitosisState persistence
# ---------------------------------------------------------------------------

def test_state_save_load(tmp_path):
    state = MitosisState()
    state.families["soul"] = ["chunk-0", "chunk-1", "chunk-2", "chunk-3"]
    state.generations["soul"] = 2
    state.chunk_to_parent["chunk-0"] = "soul"

    path = str(tmp_path / "state.json")
    state.save(path)

    loaded = MitosisState.load(path)
    assert loaded.families == state.families
    assert loaded.generations == state.generations
    assert loaded.chunk_to_parent == state.chunk_to_parent


def test_state_load_missing():
    state = MitosisState.load("/nonexistent/path.json")
    assert state.families == {}
    assert state.generations == {}


# ---------------------------------------------------------------------------
# Tests: chunk ID determinism
# ---------------------------------------------------------------------------

def test_chunk_id_deterministic():
    id1 = _make_chunk_id("soul", 0, "Some content here")
    id2 = _make_chunk_id("soul", 0, "Some content here")
    assert id1 == id2


def test_chunk_id_unique_per_content():
    id1 = _make_chunk_id("soul", 0, "Content A")
    id2 = _make_chunk_id("soul", 0, "Content B")
    assert id1 != id2


def test_chunk_id_unique_per_index():
    id1 = _make_chunk_id("soul", 0, "Same content")
    id2 = _make_chunk_id("soul", 1, "Same content")
    # Same content but different index — still same hash but different index label
    assert "chunk-0" in id1
    assert "chunk-1" in id2
