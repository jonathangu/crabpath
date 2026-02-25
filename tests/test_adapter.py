"""Tests for OpenClaw adapter and feedback helpers."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

from crabpath import Edge, Graph, Node, OpenClawCrabPathAdapter
from crabpath.feedback import auto_outcome, map_correction_to_snapshot


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def make_mock_embed_fn(texts: Iterable[str]):
    """Deterministic bag-of-words embedder used in adapter tests."""
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


def _build_graph_and_index(tmpdir: str) -> OpenClawCrabPathAdapter:
    graph_path = Path(tmpdir) / "graph.json"
    index_path = Path(tmpdir) / "index.json"
    base_graph = Graph()
    base_graph.add_node(Node(id="check-config", content="git diff"))
    base_graph.add_node(Node(id="check-logs", content="tail logs"))
    base_graph.add_node(Node(id="restart-svc", content="systemctl restart app"))
    embed_fn = make_mock_embed_fn([n.content for n in base_graph.nodes()])

    adapter = OpenClawCrabPathAdapter(
        str(graph_path),
        str(index_path),
        embed_fn=embed_fn,
    )
    adapter.graph = base_graph
    adapter.index.build(adapter.graph, adapter.embed_fn)
    adapter.save()
    return adapter


def test_adapter_init_with_empty_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OpenClawCrabPathAdapter(
            str(Path(tmpdir) / "graph.json"),
            str(Path(tmpdir) / "index.json"),
            embed_fn=None,
        )
        graph, index = adapter.load()
        assert graph.node_count == 0
        assert index.vectors == {}


def test_seed_from_text_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = _build_graph_and_index(tmpdir)
        adapter.load()
        seeds = adapter.seed("git diff", top_k=2)

        assert "check-config" in seeds
        assert seeds["check-config"] > 0.0
        assert seeds["check-config"] >= seeds.get("check-logs", 0.0)


def test_seed_with_memory_search_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OpenClawCrabPathAdapter(
            str(Path(tmpdir) / "graph.json"),
            str(Path(tmpdir) / "index.json"),
            embed_fn=None,
        )
        adapter.graph.add_node(Node(id="node-a", content="a"))
        adapter.graph.add_node(Node(id="node-b", content="b"))
        adapter.save()
        adapter.load()

        seeds = adapter.seed("irrelevant", memory_search_ids=["node-a", "missing"])
        assert seeds == {"node-a": 0.25}


def test_activate_returns_fired_nodes():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OpenClawCrabPathAdapter(
            str(Path(tmpdir) / "graph.json"),
            str(Path(tmpdir) / "index.json"),
            embed_fn=None,
        )
        adapter.graph.add_node(Node(id="a", content="start"))
        adapter.graph.add_node(Node(id="b", content="middle"))
        adapter.graph.add_node(Node(id="c", content="end"))
        adapter.graph.add_edge(Edge(source="a", target="b", weight=1.5))
        adapter.graph.add_edge(Edge(source="b", target="c", weight=1.5))

        result = adapter.activate({"a": 1.0}, max_steps=3, top_k=5)
        fired_ids = [n.id for n, _ in result.fired]

        assert fired_ids == ["c", "b", "a"]
        assert result.steps == 3


def test_context_extraction_and_ordering():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OpenClawCrabPathAdapter(
            str(Path(tmpdir) / "graph.json"),
            str(Path(tmpdir) / "index.json"),
            embed_fn=None,
        )
        adapter.graph.add_node(Node(id="high", content="high energy", threshold=0.01))
        adapter.graph.add_node(Node(id="mid", content="middle energy", threshold=0.01))
        adapter.graph.add_node(Node(id="low", content="low energy", threshold=0.01))
        adapter.graph.add_node(Node(id="inhibit", content="inhibit"))

        adapter.graph.add_edge(Edge(source="high", target="mid", weight=0.6))
        adapter.graph.add_edge(Edge(source="mid", target="low", weight=0.3))
        adapter.graph.add_edge(Edge(source="high", target="inhibit", weight=-1.0))

        result = adapter.activate({"high": 3.0}, max_steps=4, top_k=3)
        ctx = adapter.context(result)

        assert ctx["contents"] == [
            "high energy",
            "middle energy",
            "low energy",
        ]
        assert ctx["guardrails"] == ["inhibit"]
        assert ctx["fired_ids"] == ["high", "mid", "low"]
        assert ctx["fired_scores"][0] > ctx["fired_scores"][1] > ctx["fired_scores"][2]


def test_learn_positive_and_negative():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OpenClawCrabPathAdapter(
            str(Path(tmpdir) / "graph.json"),
            str(Path(tmpdir) / "index.json"),
            embed_fn=None,
        )
        adapter.graph.add_node(Node(id="a", content="step one"))
        adapter.graph.add_node(Node(id="b", content="step two"))
        adapter.graph.add_edge(Edge(source="a", target="b", weight=1.0))

        result = adapter.activate({"a": 2.0}, max_steps=3, decay=0.0)
        before = adapter.graph.get_edge("a", "b").weight

        adapter.learn(result, outcome=1.0)
        after_success = adapter.graph.get_edge("a", "b").weight
        assert after_success > before

        adapter.learn(result, outcome=-1.0)
        after_negative = adapter.graph.get_edge("a", "b").weight
        assert after_negative < after_success


def test_snapshot_save_load_roundtrip(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OpenClawCrabPathAdapter(
            str(Path(tmpdir) / "graph.json"),
            str(Path(tmpdir) / "index.json"),
            embed_fn=None,
        )
        adapter.graph.add_node(Node(id="a", content="seed"))
        adapter.graph.add_edge(Edge(source="a", target="a", weight=1.0))
        adapter.save()

        snapshot_file = Path(tmpdir) / "session.events.db"
        monkeypatch.setenv("CRABPATH_SNAPSHOT_PATH", str(snapshot_file))
        adapter.snapshot_path = str(snapshot_file)

        result = adapter.activate({"a": 1.0}, max_steps=2)
        adapter.snapshot("session-1", 10, result)

        raw = snapshot_file.read_text(encoding="utf-8").strip()
        rows = [json.loads(raw_line) for raw_line in raw.splitlines()]
        assert len(rows) == 1
        assert rows[0]["session_id"] == "session-1"
        assert rows[0]["turn_id"] == 10

        mapped = map_correction_to_snapshot("session-1", turn_window=1)
        assert mapped is not None
        assert mapped["turn_id"] == 10
        assert mapped["session_id"] == "session-1"


def test_feedback_mapping_with_delayed_correction(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OpenClawCrabPathAdapter(
            str(Path(tmpdir) / "graph.json"),
            str(Path(tmpdir) / "index.json"),
            embed_fn=None,
        )
        adapter.graph.add_node(Node(id="a", content="seed"))
        adapter.save()

        snapshot_file = Path(tmpdir) / "session.events.db"
        monkeypatch.setenv("CRABPATH_SNAPSHOT_PATH", str(snapshot_file))
        adapter.snapshot_path = str(snapshot_file)

        for turn in range(1, 6):
            result = adapter.activate({"a": 1.0}, max_steps=2)
            adapter.snapshot("session-delayed", turn, result)

        # Feedback arrives at turn 10 -> should match the most recent snapshot
        # inside a 5-turn window (turn 5).
        monkeypatch.setenv("CRABPATH_CORRECTION_TURN", "10")
        mapped = map_correction_to_snapshot("session-delayed", turn_window=5)

        assert mapped is not None
        assert mapped["turn_id"] == 5
        assert mapped["turns_since_fire"] == 5
        assert (
            auto_outcome(corrections_count=1, turns_since_fire=mapped["turns_since_fire"]) == -1.0
        )
        assert auto_outcome(corrections_count=0, turns_since_fire=5) == 0.3
