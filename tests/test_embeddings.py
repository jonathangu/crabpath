"""Tests for the EmbeddingIndex API."""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

from crabpath import EmbeddingIndex, Graph, Node
from crabpath.embeddings import _cosine


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def make_mock_embed_fn(texts: list[str]):
    """Build a deterministic binary bag-of-words embedder from known text.

    Each word maps to one dimension. Embeddings are binary and L2-normalized.
    """

    vocabulary: list[str] = []
    for text in texts:
        for token in sorted(_tokenize(text)):
            if token not in vocabulary:
                vocabulary.append(token)

    dimension_lookup = {token: idx for idx, token in enumerate(vocabulary)}
    dim = len(vocabulary)

    def embed_fn(batch: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in batch:
            present = _tokenize(text)
            vector = [0.0] * dim

            for token in present:
                index = dimension_lookup.get(token)
                if index is not None:
                    vector[index] = 1.0

            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0:
                vectors.append(vector)
            else:
                vectors.append([v / norm for v in vector])
        return vectors

    return embed_fn


def test_build_index():
    graph = Graph()
    graph.add_node(Node(id="n1", content="red apple", metadata={"category": "fruit"}))
    graph.add_node(Node(id="n2", content="green pear"))
    graph.add_node(Node(id="n3", content="blue berry"))

    embed_fn = make_mock_embed_fn([n.content for n in graph.nodes()])

    index = EmbeddingIndex()
    index.build(graph, embed_fn)

    assert set(index.vectors.keys()) == {"n1", "n2", "n3"}
    for node in graph.nodes():
        assert index.vectors[node.id] == embed_fn([node.content])[0]


def test_seed_basic():
    graph = Graph()
    graph.add_node(Node(id="n1", content="quick brown fox"))
    graph.add_node(Node(id="n2", content="brown bear"))
    graph.add_node(Node(id="n3", content="green grass"))

    embed_fn = make_mock_embed_fn([n.content for n in graph.nodes()])
    index = EmbeddingIndex()
    index.build(graph, embed_fn)

    seeds = index.seed("quick brown", embed_fn)

    assert set(seeds.keys()) == {"n1", "n2"}
    assert seeds["n1"] > seeds["n2"] > 0


def test_seed_min_score():
    graph = Graph()
    graph.add_node(Node(id="n1", content="quick brown fox"))
    graph.add_node(Node(id="n2", content="brown bear"))
    graph.add_node(Node(id="n3", content="green grass"))

    embed_fn = make_mock_embed_fn([n.content for n in graph.nodes()])
    index = EmbeddingIndex()
    index.build(graph, embed_fn)

    seeds = index.seed("quick brown", embed_fn, min_score=0.8)

    assert set(seeds.keys()) == {"n1"}
    assert seeds["n1"] == 2.0


def test_seed_energy_scaling():
    graph = Graph()
    graph.add_node(Node(id="n1", content="quick brown"))
    graph.add_node(Node(id="n2", content="quick"))
    graph.add_node(Node(id="n3", content="blue"))

    embed_fn = make_mock_embed_fn([n.content for n in graph.nodes()])
    index = EmbeddingIndex()
    index.build(graph, embed_fn)

    query = "quick brown"
    query_vector = embed_fn([query])[0]
    top_sim = _cosine(query_vector, embed_fn(["quick brown"])[0])
    weak_sim = _cosine(query_vector, embed_fn(["quick"])[0])

    seeds = index.seed(query, embed_fn, energy=2.5, top_k=2)

    assert math.isclose(seeds["n1"], 2.5)
    assert math.isclose(seeds["n2"], 2.5 * (weak_sim / top_sim))
    assert "n3" not in seeds


def test_seed_top_k():
    graph = Graph()
    graph.add_node(Node(id="n1", content="alpha beta gamma"))
    graph.add_node(Node(id="n2", content="alpha beta"))
    graph.add_node(Node(id="n3", content="alpha"))
    graph.add_node(Node(id="n4", content="beta gamma delta"))

    embed_fn = make_mock_embed_fn([n.content for n in graph.nodes()])
    index = EmbeddingIndex()
    index.build(graph, embed_fn)

    seeds = index.seed("alpha beta gamma", embed_fn, top_k=2)

    assert len(seeds) == 2
    assert list(seeds.keys())[0] == "n1"
    assert list(seeds.keys())[1] == "n2"


def test_seed_empty_graph():
    graph = Graph()
    embed_fn = make_mock_embed_fn([])
    index = EmbeddingIndex()
    index.build(graph, embed_fn)

    seeds = index.seed("anything", embed_fn)
    assert seeds == {}


def test_save_load_roundtrip():
    graph = Graph()
    graph.add_node(Node(id="n1", content="slow turtle"))
    graph.add_node(Node(id="n2", content="fast rabbit"))
    graph.add_node(Node(id="n3", content="happy dog"))

    embed_fn = make_mock_embed_fn([n.content for n in graph.nodes()])
    index = EmbeddingIndex()
    index.build(graph, embed_fn)
    expected = index.seed("fast animal", embed_fn)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    index.save(str(path))
    restored = EmbeddingIndex.load(str(path))
    actual = restored.seed("fast animal", embed_fn)

    assert restored.vectors == index.vectors
    assert actual == expected

    path.unlink()


def test_cosine_similarity():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert math.isclose(_cosine([1.0, 1.0], [1.0, 1.0]), 1.0)
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
