"""
CrabPath Embeddings — Semantic seeding via vector similarity.

Embeds node content and finds the best seeds for a query
using cosine similarity. Supports any embedding function.

Ships with an OpenAI adapter. Bring your own for local models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .graph import Graph, Node


@dataclass
class EmbeddingIndex:
    """A vector index over graph nodes for semantic seeding.

    Build once, save to disk, reload on startup.
    Reindex when nodes change.
    """

    vectors: dict[str, list[float]] = field(default_factory=dict)  # node_id -> embedding
    dim: int = 0

    def build(
        self,
        graph: Graph,
        embed_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 100,
    ) -> None:
        """Embed all node content in the graph.

        Args:
            graph: The CrabPath graph.
            embed_fn: Function that takes a list of strings and returns
                a list of embedding vectors. Signature: (texts) -> vectors.
            batch_size: How many texts to embed per API call.
        """
        nodes = graph.nodes()
        if not nodes:
            return

        # Prepare texts: id + content + error_class for richer matching
        texts = []
        ids = []
        for node in nodes:
            parts = [node.content]
            ec = node.metadata.get("error_class", "")
            if ec:
                parts.append(ec)
            texts.append(" ".join(parts))
            ids.append(node.id)

        # Batch embed
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = embed_fn(batch)
            all_vectors.extend(vectors)

        # Store
        self.vectors = dict(zip(ids, all_vectors))
        if all_vectors:
            self.dim = len(all_vectors[0])

    def seed(
        self,
        query: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
        top_k: int = 10,
        min_score: float = 0.25,
        energy: float = 2.0,
    ) -> dict[str, float]:
        """Find the best seed nodes for a query.

        Returns {node_id: energy} for the top-k most similar nodes
        above min_score. Energy is scaled so the best match gets
        the full `energy` value and others are proportional.

        Args:
            query: The task or question text.
            embed_fn: Embedding function.
            top_k: Maximum seeds to return.
            min_score: Minimum cosine similarity to include.
            energy: Energy for the top-scoring seed. Others scale proportionally.
        """
        if not self.vectors:
            return {}

        q_vec = embed_fn([query])[0]

        # Cosine similarity against all nodes
        scores = {}
        for node_id, node_vec in self.vectors.items():
            sim = _cosine(q_vec, node_vec)
            if sim >= min_score:
                scores[node_id] = sim

        if not scores:
            return {}

        # Top-k, scaled to energy
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_sim = ranked[0][1] if ranked else 1.0
        return {nid: (sim / max_sim) * energy for nid, sim in ranked}

    def upsert(
        self,
        node_id: str,
        content: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
    ) -> None:
        """Add or replace one vector in the index."""
        vector = embed_fn([content])[0]
        self.vectors[node_id] = vector
        if vector:
            self.dim = len(vector)

    def remove(self, node_id: str) -> None:
        """Remove one vector from the index."""
        self.vectors.pop(node_id, None)

    def raw_scores(
        self,
        query: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return raw cosine scores for all nodes, sorted descending."""
        if not self.vectors:
            return []

        q_vec = embed_fn([query])[0]
        scores: list[tuple[str, float]] = []
        for node_id, node_vec in self.vectors.items():
            scores.append((node_id, _cosine(q_vec, node_vec)))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str) -> None:
        """Save index to JSON."""
        data = {"dim": self.dim, "vectors": self.vectors}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> EmbeddingIndex:
        """Load index from JSON."""
        with open(path) as f:
            data = json.load(f)
        idx = cls()
        idx.dim = data.get("dim", 0)
        idx.vectors = data.get("vectors", {})
        return idx


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---- OpenAI adapter ----

def openai_embed(
    model: str = "text-embedding-3-small",
) -> Callable[[list[str]], list[list[float]]]:
    """Returns an embed_fn that uses the OpenAI API.

    Usage:
        embed = openai_embed()
        index.build(graph, embed)
        seeds = index.seed("my query", embed)
    """
    from openai import OpenAI

    client = OpenAI()

    def embed_fn(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    return embed_fn
