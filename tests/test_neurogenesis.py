"""Tests for automatic neurogenesis behavior."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from crabpath import (
    Edge,
    Graph,
    Node,
    OpenClawCrabPathAdapter,
    deterministic_auto_id,
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def make_mock_embed_fn(texts: list[str]):
    vocabulary: list[str] = []
    for text in texts:
        for token in sorted(_tokenize(text)):
            if token not in vocabulary:
                vocabulary.append(token)

    dimension_lookup = {token: idx for idx, token in enumerate(vocabulary)}
    dim = len(vocabulary) if vocabulary else 1

    def embed_fn(batch: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in batch:
            present = _tokenize(text)
            vector = [0.0] * dim
            for token in present:
                idx = dimension_lookup.get(token)
                if idx is not None:
                    vector[idx] = 1.0
            norm = sum(v * v for v in vector) ** 0.5
            if norm:
                vectors.append([v / norm for v in vector])
            else:
                vectors.append([0.0] * dim)
        return vectors

    return embed_fn


def _build_adapter(tmpdir: str, nodes: list[tuple[str, str]], top_texts: list[str]) -> OpenClawCrabPathAdapter:
    graph_path = Path(tmpdir) / "graph.json"
    index_path = Path(tmpdir) / "index.json"

    adapter = OpenClawCrabPathAdapter(
        str(graph_path),
        str(index_path),
        embed_fn=make_mock_embed_fn(top_texts),
    )

    for node_id, content in nodes:
        adapter.graph.add_node(Node(id=node_id, content=content))

    adapter.index.build(adapter.graph, adapter.embed_fn)
    adapter.save()
    return adapter


def test_known_concept_no_creation(monkeypatch, tmp_path):
    adapter = _build_adapter(
        str(tmp_path),
        [("known", "worktree drift codex reset")],
        ["worktree drift codex reset"],
    )
    raw_scores = [("known", 0.76)]
    monkeypatch.setattr(adapter.index, "raw_scores", lambda *args, **kwargs: raw_scores)

    result = adapter.query("worktree drift codex reset", top_k=8)

    assert result["auto_node"] is not None
    assert result["auto_node"]["created"] is False
    assert result["auto_node"]["node_id"] is None
    assert result["auto_node"]["band"] == "known"


def test_novel_concept_creates_node(monkeypatch, tmp_path):
    adapter = _build_adapter(
        str(tmp_path),
        [("seed", "git diff and logs")],
        ["git diff and logs", "new concept tokens"],
    )

    monkeypatch.setattr(
        adapter.index,
        "raw_scores",
        lambda query, embed_fn, top_k=8: [("seed", 0.32)],
    )

    result = adapter.query("giraffe codeword sequence", memory_search_ids=["seed"], top_k=8)
    auto_id = result["auto_node"]["node_id"]

    assert auto_id is not None
    assert result["auto_node"]["created"] is True
    assert result["auto_node"]["band"] == "novel"
    assert auto_id.startswith("auto:")

    node = adapter.graph.get_node(auto_id)
    assert node is not None
    assert node.threshold == 0.8
    assert node.metadata["source"] == "auto"
    assert node.metadata["auto_probationary"] is True
    assert adapter.graph.get_edge("seed", auto_id) is not None
    assert adapter.graph.get_edge("seed", auto_id).weight == 0.15


def test_noise_no_creation(monkeypatch, tmp_path):
    adapter = _build_adapter(
        str(tmp_path),
        [("seed", "alpha beta gamma")],
        ["alpha beta gamma", "random nonsense token"],
    )
    monkeypatch.setattr(
        adapter.index,
        "raw_scores",
        lambda query, embed_fn, top_k=8: [("seed", 0.24)],
    )

    result = adapter.query("xyzzy plugh nonsense phrase", memory_search_ids=["seed"], top_k=8)

    assert result["auto_node"]["created"] is False
    assert result["auto_node"]["node_id"] is None
    assert result["auto_node"]["band"] == "noise"


def test_greeting_blocked(monkeypatch, tmp_path):
    adapter = _build_adapter(
        str(tmp_path),
        [("seed", "hello world"),
         ("seed2", "thanks for everything")],
        ["hello world", "thanks for everything"],
    )
    monkeypatch.setattr(
        adapter.index,
        "raw_scores",
        lambda query, embed_fn, top_k=8: [("seed", 0.95)],
    )

    for phrase in ("hello", "thanks"):
        result = adapter.query(phrase, memory_search_ids=["seed"], top_k=8)
        assert result["auto_node"]["created"] is False
        assert result["auto_node"]["band"] == "blocked"


def test_deterministic_id_stable():
    first = deterministic_auto_id("  Giraffe concept for codeword  ")
    second = deterministic_auto_id("giraffe concept for codeword")

    assert first == second
    assert first.startswith("auto:")
    assert len(first) > 5


def test_duplicate_query_updates_not_duplicates(monkeypatch, tmp_path):
    adapter = _build_adapter(
        str(tmp_path),
        [("seed", "memory seed node")],
        ["memory seed node", "repeat query phrase"],
    )
    raw_scores = [("seed", 0.35)]
    monkeypatch.setattr(
        adapter.index,
        "raw_scores",
        lambda query, embed_fn, top_k=8: raw_scores,
    )

    first = adapter.query("repeat query phrase now", memory_search_ids=["seed"])
    auto_id = first["auto_node"]["node_id"]
    assert auto_id is not None
    assert first["auto_node"]["created"] is True
    node_count = adapter.graph.node_count

    second = adapter.query("repeat query phrase now", memory_search_ids=["seed"])
    assert second["auto_node"]["node_id"] == auto_id
    assert second["auto_node"]["created"] is False
    assert adapter.graph.node_count == node_count
    assert adapter.graph.get_node(auto_id).metadata["auto_seed_count"] == 2


def test_auto_node_connects_to_seeds(monkeypatch, tmp_path):
    adapter = _build_adapter(
        str(tmp_path),
        [
            ("seed-a", "seed one"),
            ("seed-b", "seed two"),
        ],
        ["seed one", "seed two", "connect new concept"],
    )
    raw_scores = [("seed-a", 0.32), ("seed-b", 0.31)]
    monkeypatch.setattr(
        adapter.index,
        "raw_scores",
        lambda query, embed_fn, top_k=8: raw_scores,
    )

    result = adapter.query(
        "another novel phrase",
        memory_search_ids=["seed-a", "seed-b"],
        top_k=8,
    )
    auto_id = result["auto_node"]["node_id"]

    assert auto_id is not None
    assert adapter.graph.get_edge("seed-a", auto_id) is not None
    assert adapter.graph.get_edge("seed-b", auto_id) is not None
    assert adapter.graph.get_edge("seed-a", auto_id).weight == 0.15
    assert adapter.graph.get_edge("seed-b", auto_id).weight == 0.15


def test_upsert_makes_node_queryable(monkeypatch, tmp_path):
    adapter = _build_adapter(
        str(tmp_path),
        [("seed", "base memory node")],
        ["base memory node", "queryable concept phrase"],
    )
    original_raw_scores = adapter.index.raw_scores
    calls: dict[str, int] = {"count": 0}

    def fake_raw_scores(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return [("seed", 0.35)]
        return original_raw_scores(*args, **kwargs)

    monkeypatch.setattr(
        adapter.index,
        "raw_scores",
        fake_raw_scores,
    )

    first = adapter.query("queryable concept phrase", memory_search_ids=["seed"])
    auto_id = first["auto_node"]["node_id"]
    assert auto_id is not None
    assert first["auto_node"]["created"] is True

    raw_after = adapter.index.raw_scores(
        "queryable concept phrase",
        embed_fn=adapter.embed_fn,
        top_k=5,
    )
    assert raw_after[0][0] == auto_id

    second = adapter.query("queryable concept phrase", memory_search_ids=["seed"])
    assert second["auto_node"]["node_id"] is None
