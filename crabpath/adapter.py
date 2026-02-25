"""OpenClaw adapter for CrabPath.

This module wraps graph loading, activation, seeding, learning, and snapshot
persistence into one session-oriented helper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .activation import Firing, activate as _activate, learn as _learn
from .embeddings import EmbeddingIndex
from .feedback import snapshot_path
from .graph import Graph, Node
from .neurogenesis import (
    NeurogenesisConfig,
    NoveltyResult,
    assess_novelty,
    connect_new_node,
    deterministic_auto_id,
)

EmbeddingFn = Callable[[list[str]], list[list[float]]]

MEMORY_SEARCH_ENERGY = 0.25


class OpenClawCrabPathAdapter:
    """Adapter used by OpenClaw session runtime.

    The adapter keeps the graph, embedding index, and snapshot path in one object.
    It intentionally stays lightweight and dependency free.
    """

    def __init__(
        self,
        graph_path: str,
        index_path: str,
        embed_fn: Optional[EmbeddingFn] = None,
    ) -> None:
        """Create an adapter bound to graph/index paths.

        Args:
            graph_path: JSON path for the graph artifact.
            index_path: JSON path for the embedding index.
            embed_fn: Optional embedding function to use with EmbeddingIndex.
        """
        self.graph_path = graph_path
        self.index_path = index_path
        self.embed_fn = embed_fn

        self.graph = Graph()
        self.index = EmbeddingIndex()
        self.snapshot_path = str(snapshot_path(graph_path))

    # -- Lifecycle ---------------------------------------------------------

    def load(self) -> tuple[Graph, EmbeddingIndex]:
        """Load graph + index from disk, or start empty if missing."""
        graph_file = Path(self.graph_path)
        if graph_file.exists():
            self.graph = Graph.load(self.graph_path)
        else:
            self.graph = Graph()

        index_file = Path(self.index_path)
        if index_file.exists():
            self.index = EmbeddingIndex.load(self.index_path)
        else:
            self.index = EmbeddingIndex()
        return self.graph, self.index

    # -- Seeding -----------------------------------------------------------

    def seed(
        self,
        query_text: str,
        memory_search_ids: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> dict[str, float]:
        """Build the seed map used for activation.

        The seed map combines:
          1) semantic seeds from the EmbeddingIndex (if available)
          2) `memory_search_ids` as weak symbolic seeds (default 0.25 each)
        """
        seeds: dict[str, float] = {}

        if self.embed_fn is not None and self.index.vectors:
            embed_seeds = self.index.seed(
                query_text,
                embed_fn=self.embed_fn,
                top_k=top_k,
            )
            seeds.update(embed_seeds)

        if memory_search_ids:
            for node_id in memory_search_ids:
                if self.graph.get_node(node_id) is None:
                    continue
                seeds[node_id] = max(seeds.get(node_id, 0.0), MEMORY_SEARCH_ENERGY)

        return seeds

    # -- Activation --------------------------------------------------------

    def activate(
        self,
        seeds: dict[str, float],
        max_steps: int = 3,
        decay: float = 0.1,
        top_k: int = 12,
    ) -> Firing:
        """Run one activation pass over the graph.

        Uses `reset=False` to retain warm state between turns by default.
        """
        return _activate(
            self.graph,
            seeds,
            max_steps=max_steps,
            decay=decay,
            top_k=top_k,
            reset=False,
        )

    def query(
        self,
        query_text: str,
        memory_search_ids: Optional[list[str]] = None,
        top_k: int = 8,
        config: NeurogenesisConfig = NeurogenesisConfig(),
    ) -> dict[str, Any]:
        """Query + activation with automatic neurogenesis.

        Always-on behavior:
        - build seeds from semantic + memory search ids
        - run raw novelty detection
        - optionally create an auto node and re-seed graph/index
        - activate
        - lightly learn from co-firing nodes when auto node was involved
        - persist graph and index
        """
        seeds = self.seed(query_text, memory_search_ids=memory_search_ids, top_k=top_k)

        raw_scores: list[tuple[str, float]] = []
        if self.embed_fn is not None and self.index.vectors:
            raw_scores = self.index.raw_scores(query_text, self.embed_fn, top_k=top_k)

        novelty = assess_novelty(
            query_text=query_text,
            raw_scores=raw_scores,
            config=config,
        )

        auto_node_id: str | None = None
        auto_created = False
        if novelty.should_create and self.embed_fn is not None:
            auto_node_id = deterministic_auto_id(query_text)
            existing_node = self.graph.get_node(auto_node_id)
            now = time.time()

            if existing_node is None:
                auto_created = True
                existing_node = Node(
                    id=auto_node_id,
                    content=query_text.strip(),
                    threshold=0.8,
                    metadata={
                        "source": "auto",
                        "created_ts": now,
                        "auto_probationary": True,
                        "auto_seed_count": 1,
                    },
                )
                self.graph.add_node(existing_node)
            else:
                existing_node.metadata["auto_probationary"] = True
                existing_node.metadata["auto_seed_count"] = int(
                    existing_node.metadata.get("auto_seed_count", 0)
                ) + 1

            existing_node.metadata["last_seen_ts"] = now

            self.index.upsert(auto_node_id, existing_node.content, self.embed_fn)
            connect_new_node(
                graph=self.graph,
                new_node_id=auto_node_id,
                current_seed_ids=list(seeds.keys()),
                weights=0.15,
            )

            # Ensure the new node is also seeded for this query.
            seeds[auto_node_id] = max(seeds.get(auto_node_id, 0.0), 0.25)

        firing = self.activate(seeds, max_steps=3, decay=0.1, top_k=top_k)

        # Light auto-learning for the auto concept and co-firing concepts.
        if auto_node_id is not None:
            fired_ids = [node.id for node, _ in firing.fired]
            if auto_node_id in fired_ids:
                self.learn(firing, outcome=0.1)

        self.save()
        context = self.context(firing)
        context["auto_node"] = {
            "node_id": auto_node_id,
            "created": auto_created,
            "should_create": novelty.should_create,
            "top_score": novelty.top_score,
            "band": novelty.band,
            "metadata": dict(self.graph.get_node(auto_node_id).metadata)
            if auto_node_id and self.graph.get_node(auto_node_id)
            else None,
        }
        context["novelty"] = {
            "should_create": novelty.should_create,
            "top_score": novelty.top_score,
            "band": novelty.band,
            "blocked": novelty.blocked,
        }
        return context

    # -- Context -----------------------------------------------------------

    def context(self, firing_result: Firing) -> dict[str, Any]:
        """Create context payload from a firing result.

        Returns:
            {
                "contents": content strings ordered by firing energy desc,
                "guardrails": inhibited node ids,
                "fired_ids": ids ordered by firing energy desc,
                "fired_scores": firing energies ordered by same order,
            }
        """
        ranked = sorted(firing_result.fired, key=lambda item: item[1], reverse=True)
        return {
            "contents": [node.content for node, _ in ranked],
            "guardrails": list(firing_result.inhibited),
            "fired_ids": [node.id for node, _ in ranked],
            "fired_scores": [score for _, score in ranked],
        }

    # -- Learning ----------------------------------------------------------

    def learn(self, firing_result: Firing, outcome: float) -> None:
        """Apply STDP-style learning for the firing outcome."""
        _learn(self.graph, firing_result, outcome=outcome)

    # -- Snapshotting ------------------------------------------------------

    def snapshot(self, session_id: str, turn_id: int | str, firing_result: Firing) -> dict[str, Any]:
        """Persist metadata about one assistant turn for delayed feedback."""
        record = {
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": time.time(),
            "fired_ids": [node.id for node, _ in firing_result.fired],
            "fired_scores": [score for _, score in firing_result.fired],
            "fired_at": firing_result.fired_at,
            "inhibited": list(firing_result.inhibited),
            "attributed": False,
        }

        path = Path(self.snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    # -- Save ----------------------------------------------------------------

    def save(self) -> None:
        """Persist graph and index to their paths."""
        self.graph.save(self.graph_path)
        self.index.save(self.index_path)
