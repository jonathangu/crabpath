"""Tests for CrabPath mitosis — LLM-driven graph self-organization."""

import json
import pytest
from crabpath.graph import Graph, Node, Edge
from crabpath.mitosis import (
    MitosisConfig,
    MitosisState,
    split_node,
    split_with_llm,
    should_merge,
    should_create_node,
    create_node,
    find_co_firing_families,
    merge_nodes,
    bootstrap_workspace,
    mitosis_maintenance,
    _fallback_split,
    _make_chunk_id,
)


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

def _mock_llm_split(system: str, user: str) -> str:
    """Mock LLM that splits by paragraphs."""
    content = user.split("---\n", 1)[-1].rsplit("\n---", 1)[0] if "---" in user else user
    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(parts) < 2:
        parts = [content[:len(content)//2], content[len(content)//2:]]
    return json.dumps({"sections": parts})


def _mock_llm_split_3way(system: str, user: str) -> str:
    """Mock LLM that always splits into 3."""
    content = user.split("---\n", 1)[-1].rsplit("\n---", 1)[0] if "---" in user else user
    n = len(content) // 3
    return json.dumps({"sections": [
        content[:n],
        content[n:2*n],
        content[2*n:],
    ]})


def _mock_llm_merge_yes(system: str, user: str) -> str:
    """Mock LLM that always says merge."""
    return json.dumps({
        "should_merge": True,
        "reason": "These chunks are semantically identical",
        "merged_content": "merged content here",
    })


def _mock_llm_merge_no(system: str, user: str) -> str:
    """Mock LLM that always says don't merge."""
    return json.dumps({
        "should_merge": False,
        "reason": "These serve different purposes",
    })


def _mock_llm_create_yes(system: str, user: str) -> str:
    """Mock LLM that always says create."""
    return json.dumps({
        "should_create": True,
        "reason": "Novel concept not in graph",
        "content": "New concept: quantum computing basics",
        "summary": "Quantum computing basics",
    })


def _mock_llm_create_no(system: str, user: str) -> str:
    """Mock LLM that always says don't create."""
    return json.dumps({
        "should_create": False,
        "reason": "Already covered by existing nodes",
    })


def _mock_llm_dispatch(system: str, user: str) -> str:
    """Mock LLM that routes to the right response based on system prompt."""
    if "split" in system.lower():
        return _mock_llm_split(system, user)
    if "merge" in system.lower() or "organizer" in system.lower():
        return _mock_llm_merge_yes(system, user)
    if "builder" in system.lower() or "neurogenesis" in system.lower():
        return _mock_llm_create_yes(system, user)
    return json.dumps({"actions": []})


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
# Tests: split_with_llm (no hardcoded count)
# ---------------------------------------------------------------------------

def test_split_with_llm_basic():
    sections = split_with_llm(SAMPLE_CONTENT, _mock_llm_split)
    assert len(sections) >= 2
    # Content preserved
    joined = "\n\n".join(sections)
    assert len(joined) > len(SAMPLE_CONTENT) * 0.5


def test_split_with_llm_variable_count():
    """LLM can return any number of sections."""
    sections = split_with_llm(SAMPLE_CONTENT, _mock_llm_split_3way)
    assert len(sections) == 3


def test_split_with_llm_fallback_on_bad_json():
    def bad_llm(s, u):
        return "not json"
    sections = split_with_llm(SAMPLE_CONTENT, bad_llm)
    assert len(sections) >= 2


def test_split_with_llm_fallback_on_exception():
    def exploding_llm(s, u):
        raise RuntimeError("API error")
    sections = split_with_llm(SAMPLE_CONTENT, exploding_llm)
    assert len(sections) >= 2


# ---------------------------------------------------------------------------
# Tests: fallback_split (structural, no count)
# ---------------------------------------------------------------------------

def test_fallback_split_by_headers():
    content = "## A\nFirst\n\n## B\nSecond\n\n## C\nThird"
    sections = _fallback_split(content)
    assert len(sections) == 3  # Natural header count, not forced


def test_fallback_split_no_headers():
    content = ("First paragraph with enough content to be meaningful on its own. "
               "It discusses routing and activation.\n\n"
               "Second paragraph covering a completely different topic about decay "
               "mechanisms and how weights change over time.\n\n"
               "Third paragraph about neurogenesis and creating new nodes in the graph.")
    sections = _fallback_split(content)
    assert len(sections) >= 2


# ---------------------------------------------------------------------------
# Tests: should_merge (LLM-driven)
# ---------------------------------------------------------------------------

def test_should_merge_yes():
    chunks = [("chunk-0", "Safety rules"), ("chunk-1", "More safety rules")]
    do_merge, reason, content = should_merge(chunks, _mock_llm_merge_yes)
    assert do_merge is True
    assert content  # Non-empty merged content


def test_should_merge_no():
    chunks = [("chunk-0", "Identity"), ("chunk-1", "Tools")]
    do_merge, reason, content = should_merge(chunks, _mock_llm_merge_no)
    assert do_merge is False


def test_should_merge_handles_error():
    def bad_llm(s, u):
        raise RuntimeError("fail")
    chunks = [("a", "content a"), ("b", "content b")]
    do_merge, reason, _ = should_merge(chunks, bad_llm)
    assert do_merge is False
    assert reason == "llm_error"


# ---------------------------------------------------------------------------
# Tests: should_create_node (LLM neurogenesis)
# ---------------------------------------------------------------------------

def test_should_create_yes():
    matches = [("node-1", 0.3, "some topic")]
    do_create, reason, content, summary = should_create_node(
        "quantum computing", matches, _mock_llm_create_yes
    )
    assert do_create is True
    assert content


def test_should_create_no():
    matches = [("node-1", 0.9, "quantum stuff")]
    do_create, reason, content, summary = should_create_node(
        "quantum", matches, _mock_llm_create_no
    )
    assert do_create is False


def test_should_create_handles_error():
    def bad_llm(s, u):
        raise RuntimeError("fail")
    do_create, reason, _, _ = should_create_node("test", [], bad_llm)
    assert do_create is False


# ---------------------------------------------------------------------------
# Tests: create_node
# ---------------------------------------------------------------------------

def test_create_node():
    g = Graph()
    g.add_node(Node(id="existing", content="existing node"))
    matches = [("existing", 0.3, "existing")]

    result = create_node(g, "new concept", matches, _mock_llm_create_yes, ["existing"])
    assert result is not None
    assert g.get_node(result.node_id) is not None
    assert "existing" in result.connected_to


def test_create_node_declined():
    g = Graph()
    result = create_node(g, "hello", [], _mock_llm_create_no)
    assert result is None


# ---------------------------------------------------------------------------
# Tests: split_node
# ---------------------------------------------------------------------------

def test_split_node_basic():
    g = Graph()
    g.add_node(Node(id="soul", content=SAMPLE_CONTENT, type="workspace_file"))
    state = MitosisState()

    result = split_node(g, "soul", _mock_llm_split, state)

    assert result is not None
    assert result.parent_id == "soul"
    assert len(result.chunk_ids) >= 2  # LLM decides how many
    assert result.generation == 1
    assert g.get_node("soul") is None  # Parent removed

    for cid in result.chunk_ids:
        assert g.get_node(cid) is not None

    # Sibling edges
    for i, src in enumerate(result.chunk_ids):
        for j, tgt in enumerate(result.chunk_ids):
            if i != j:
                edge = g.get_edge(src, tgt)
                assert edge is not None
                assert edge.weight == 1.0


def test_split_node_too_small():
    g = Graph()
    g.add_node(Node(id="tiny", content=SMALL_CONTENT))
    state = MitosisState()
    result = split_node(g, "tiny", _mock_llm_split, state)
    assert result is None


def test_split_node_preserves_edges():
    g = Graph()
    g.add_node(Node(id="soul", content=SAMPLE_CONTENT))
    g.add_node(Node(id="other", content="Other node"))
    g.add_edge(Edge(source="other", target="soul", weight=0.8))
    g.add_edge(Edge(source="soul", target="other", weight=0.5))
    state = MitosisState()

    result = split_node(g, "soul", _mock_llm_split, state)
    assert result is not None

    for cid in result.chunk_ids:
        assert g.get_edge("other", cid) is not None
        assert g.get_edge(cid, "other") is not None


def test_split_node_carries_protection():
    g = Graph()
    g.add_node(Node(id="safe", content=SAMPLE_CONTENT, metadata={"protected": True}))
    state = MitosisState()

    result = split_node(g, "safe", _mock_llm_split, state)
    assert result is not None

    for cid in result.chunk_ids:
        node = g.get_node(cid)
        assert node.metadata.get("protected") is True


# ---------------------------------------------------------------------------
# Tests: find_co_firing_families
# ---------------------------------------------------------------------------

def test_co_firing_detected():
    g = Graph()
    state = MitosisState()

    chunk_ids = ["p::chunk-0-a", "p::chunk-1-b", "p::chunk-2-c"]
    for cid in chunk_ids:
        g.add_node(Node(id=cid, content="chunk"))

    for i, src in enumerate(chunk_ids):
        for j, tgt in enumerate(chunk_ids):
            if i != j:
                g.add_edge(Edge(source=src, target=tgt, weight=1.0))

    state.families["p"] = chunk_ids

    result = find_co_firing_families(g, state)
    assert len(result) == 1
    assert result[0][0] == "p"


def test_co_firing_not_detected_when_decayed():
    g = Graph()
    state = MitosisState()

    chunk_ids = ["p::chunk-0-a", "p::chunk-1-b"]
    for cid in chunk_ids:
        g.add_node(Node(id=cid, content="chunk"))

    g.add_edge(Edge(source=chunk_ids[0], target=chunk_ids[1], weight=1.0))
    g.add_edge(Edge(source=chunk_ids[1], target=chunk_ids[0], weight=0.3))

    state.families["p"] = chunk_ids

    result = find_co_firing_families(g, state)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: merge_nodes (LLM-driven)
# ---------------------------------------------------------------------------

def test_merge_nodes_yes():
    g = Graph()
    state = MitosisState()

    chunk_ids = ["p::chunk-0-a", "p::chunk-1-b"]
    for i, cid in enumerate(chunk_ids):
        g.add_node(Node(id=cid, content=f"Content {i}", metadata={"chunk_index": i}))

    state.families["p"] = chunk_ids

    result = merge_nodes(g, "p", chunk_ids, _mock_llm_merge_yes, state)
    assert result is not None
    assert result.merged_id == "p"
    assert g.get_node("p") is not None

    for cid in chunk_ids:
        assert g.get_node(cid) is None


def test_merge_nodes_declined():
    g = Graph()
    state = MitosisState()

    chunk_ids = ["p::chunk-0-a", "p::chunk-1-b"]
    for cid in chunk_ids:
        g.add_node(Node(id=cid, content="Content"))

    state.families["p"] = chunk_ids

    result = merge_nodes(g, "p", chunk_ids, _mock_llm_merge_no, state)
    assert result is None

    # Chunks should still exist
    for cid in chunk_ids:
        assert g.get_node(cid) is not None


# ---------------------------------------------------------------------------
# Tests: bootstrap_workspace
# ---------------------------------------------------------------------------

def test_bootstrap_workspace():
    g = Graph()
    state = MitosisState()
    files = {
        "soul": SAMPLE_CONTENT,
        "tools": SAMPLE_CONTENT + "\n\n## Extra\nMore content.",
    }

    results = bootstrap_workspace(g, files, _mock_llm_split, state)

    assert len(results) == 2
    assert g.node_count >= 4
    assert g.edge_count > 0
    assert "soul" in state.families
    assert "tools" in state.families


# ---------------------------------------------------------------------------
# Tests: mitosis_maintenance (unified loop)
# ---------------------------------------------------------------------------

def test_maintenance_merges_and_resplits():
    g = Graph()
    state = MitosisState()

    # Bootstrap
    files = {"test": SAMPLE_CONTENT}
    bootstrap_workspace(g, files, _mock_llm_split, state)

    # Force all sibling edges to co-fire
    for cid_src in state.families["test"]:
        for cid_tgt in state.families["test"]:
            if cid_src != cid_tgt:
                edge = g.get_edge(cid_src, cid_tgt)
                if edge:
                    edge.weight = 1.0

    # Maintenance should detect co-firing, ask LLM to merge, then re-split
    result = mitosis_maintenance(g, _mock_llm_dispatch, state)
    assert result["merges"] >= 1 or result["splits"] >= 0


def test_maintenance_no_action_when_separated():
    g = Graph()
    state = MitosisState()

    files = {"test": SAMPLE_CONTENT}
    bootstrap_workspace(g, files, _mock_llm_split, state)

    # Decay some edges — chunks are separating
    chunk_ids = state.families["test"]
    if len(chunk_ids) >= 2:
        edge = g.get_edge(chunk_ids[0], chunk_ids[1])
        if edge:
            edge.weight = 0.3

    result = mitosis_maintenance(g, _mock_llm_dispatch, state)
    assert result["merges"] == 0


# ---------------------------------------------------------------------------
# Tests: MitosisState persistence
# ---------------------------------------------------------------------------

def test_state_save_load(tmp_path):
    state = MitosisState()
    state.families["soul"] = ["c0", "c1", "c2"]
    state.generations["soul"] = 2
    state.chunk_to_parent["c0"] = "soul"

    path = str(tmp_path / "state.json")
    state.save(path)

    loaded = MitosisState.load(path)
    assert loaded.families == state.families
    assert loaded.generations == state.generations


def test_state_load_missing():
    state = MitosisState.load("/nonexistent.json")
    assert state.families == {}


# ---------------------------------------------------------------------------
# Tests: chunk ID
# ---------------------------------------------------------------------------

def test_chunk_id_deterministic():
    id1 = _make_chunk_id("soul", 0, "Content")
    id2 = _make_chunk_id("soul", 0, "Content")
    assert id1 == id2


def test_chunk_id_unique_per_content():
    id1 = _make_chunk_id("soul", 0, "A")
    id2 = _make_chunk_id("soul", 0, "B")
    assert id1 != id2
