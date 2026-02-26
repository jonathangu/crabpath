"""Tests for IO helpers."""

from __future__ import annotations

from crabpath import EmbeddingIndex, Graph, Node
from crabpath._io import run_query


def test_run_query_falls_back_to_keyword_scoring_on_bad_embedding_output():
    graph = Graph()
    graph.add_node(Node(id="check-config", content="git diff"))
    graph.add_node(Node(id="check-logs", content="tail logs"))

    index = EmbeddingIndex()
    index.vectors = {"check-config": [1.0, 0.0], "check-logs": [0.5, 0.5]}

    firing = run_query(graph, index, "git", top_k=2, embed_fn=lambda batch: [])

    assert [node.id for node, _ in firing.fired] == ["check-config"]
