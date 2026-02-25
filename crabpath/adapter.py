"""Canonical CrabPath runtime adapter.

This module defines the canonical agent interface used by OpenClaw and
other consumers: graph + embedding index + router + learning + maintenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .activation import Firing, activate as _activate, learn as _learn
from .decay import DecayConfig, apply_decay
from .embeddings import EmbeddingIndex
from .feedback import snapshot_path
from .graph import Graph, Node
from .learning import LearningConfig, LearningResult, RewardSignal, make_learning_step
from .neurogenesis import (
    NeurogenesisConfig,
    assess_novelty,
    connect_new_node,
    deterministic_auto_id,
)
from .router import Router
from .traversal import TraversalConfig, TraversalTrajectory, render_context, traverse

EmbeddingFn = Callable[[list[str]], list[list[float]]]


MEMORY_SEARCH_ENERGY = 0.25
FALLBACK_VECTOR_DIM = 32


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _fallback_hash_vector(text: str) -> list[float]:
    vector: list[float] = [0.0] * FALLBACK_VECTOR_DIM
    for token in set(_tokenize_text(text)):
        bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:2], 16) % FALLBACK_VECTOR_DIM
        vector[bucket] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


@dataclass
class QueryResult:
    context: str
    nodes: list[str]
    chars: int
    novel_node: str | None
    band: str
    best_cosine: float
    trajectory_steps: list[Any]
    candidate_node: str | None = None
    created_novel_node: bool = False

    def __getitem__(self, key: str) -> Any:
        mapping: dict[str, Any] = {
            "context": self.context,
            "nodes": self.nodes,
            "chars": self.chars,
            "novel_node": self.novel_node,
            "band": self.band,
            "best_cosine": self.best_cosine,
            "trajectory_steps": self.trajectory_steps,
            "auto_node": {
                "node_id": self.novel_node if self.created_novel_node else self.candidate_node,
                "created": self.created_novel_node,
                "band": self.band,
                "top_score": self.best_cosine,
                "metadata": None,
                "should_create": self.created_novel_node,
            },
            "novelty": {
                "band": self.band,
                "top_score": self.best_cosine,
                "should_create": self.created_novel_node,
                "blocked": self.band == "blocked",
            },
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class ConsolidationResult:
    pruned_edges: int
    pruned_nodes: int


class CrabPathAgent:
    """Canonical interface for querying and learning over a memory graph."""

    def __init__(
        self,
        graph_path: str,
        index_path: str | None = None,
        embed_fn: EmbeddingFn | None = None,
        router: Router | None = None,
        neurogenesis_config: NeurogenesisConfig | None = None,
    ) -> None:
        graph_path_obj = Path(graph_path)
        self.graph_path = str(graph_path_obj)
        self.index_path = (
            str(graph_path_obj.with_suffix(".index.json")) if index_path is None else index_path
        )
        self.embed_fn = embed_fn
        self.router = router or Router()
        self.neurogenesis_config = neurogenesis_config or NeurogenesisConfig()
        self.learning_config = LearningConfig()
        self.decay_config = DecayConfig()

        self.graph = Graph()
        self.index = EmbeddingIndex()
        self.snapshot_path = str(snapshot_path(self.graph_path))

        self._last_trajectory: TraversalTrajectory | None = None
        self._last_raw_scores: list[tuple[str, float]] = []
        self._pending_index_refresh = True
        self._episode_id = f"agent::{graph_path_obj}"

        self.load()

    def _embedding_fn(self) -> EmbeddingFn:
        if self.embed_fn is not None:
            return self.embed_fn

        return lambda texts: [_fallback_hash_vector(text) for text in texts]

    def _ensure_index_built(self, top_k: int = 8) -> None:
        if self.index.vectors:
            return
        if not self.graph.nodes():
            return
        embed_fn = self._embedding_fn()
        try:
            self.index.build(self.graph, embed_fn, batch_size=max(1, top_k))
            self._pending_index_refresh = False
        except Exception as exc:
            import warnings

            warnings.warn(
                f"CrabPath: index build failed: {exc}. Falling back to best-effort behavior.",
                stacklevel=2,
            )
            # Keep going with best-effort behavior in constrained environments.
            self._pending_index_refresh = True

    def load(self) -> tuple[Graph, EmbeddingIndex]:
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

        self._pending_index_refresh = not bool(self.index.vectors)
        return self.graph, self.index

    def save(self) -> None:
        self.graph.save(self.graph_path)
        self.index.save(self.index_path)

    def seed(
        self,
        query_text: str,
        memory_search_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> dict[str, float]:
        seeds: dict[str, float] = {}
        if self.embed_fn is not None or self.index.vectors:
            self._ensure_index_built(top_k=top_k)
            if self.index.vectors:
                try:
                    seeds.update(
                        self.index.seed(
                            query_text,
                            embed_fn=self._embedding_fn(),
                            top_k=top_k,
                        )
                    )
                except TypeError:
                    # Fall back to keyword matching when index scores are malformed.
                    self._pending_index_refresh = True
                except Exception as exc:
                    # Keep the query running even if embedding index access fails.
                    self._pending_index_refresh = True
                    _ = str(exc)

        if not seeds:
            raw_scores = self._keyword_raw_scores(query_text, top_k=top_k)
            for node_id, score in raw_scores:
                seeds[node_id] = score * 2.0

        if memory_search_ids:
            for node_id in memory_search_ids:
                if self.graph.get_node(node_id) is None:
                    continue
                seeds[node_id] = max(seeds.get(node_id, 0.0), MEMORY_SEARCH_ENERGY)

        return seeds

    def _keyword_raw_scores(self, query_text: str, top_k: int = 8) -> list[tuple[str, float]]:
        query_tokens = set(_tokenize_text(query_text))
        if not query_tokens:
            return []

        scored: list[tuple[str, float]] = []
        for node in self.graph.nodes():
            node_tokens = set(_tokenize_text(f"{node.id} {node.content} {node.summary}"))
            if not node_tokens:
                continue
            intersection = query_tokens.intersection(node_tokens)
            if not intersection:
                continue
            score = len(intersection) / max(1, len(query_tokens))
            scored.append((node.id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _get_raw_scores(self, query_text: str, top_k: int = 8) -> list[tuple[str, float]]:
        if self.embed_fn is not None or self.index.vectors:
            self._ensure_index_built(top_k=top_k)
            if self.index.vectors:
                try:
                    return self.index.raw_scores(
                        query_text,
                        embed_fn=self._embedding_fn(),
                        top_k=top_k,
                    )
                except Exception as exc:
                    import warnings

                    warnings.warn(
                        f"CrabPath: embedding raw score retrieval failed: {exc}. Falling back to keyword scoring.",
                        stacklevel=2,
                    )
                    return []

        return self._keyword_raw_scores(query_text, top_k=top_k)

    def activate(
        self,
        seeds: dict[str, float],
        max_steps: int = 3,
        decay: float = 0.1,
        top_k: int = 12,
    ) -> Firing:
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
        top_k: int = 8,
        max_hops: int = 3,
        memory_search_ids: list[str] | None = None,
        config: NeurogenesisConfig | None = None,
    ) -> QueryResult:
        raw_scores = self._get_raw_scores(query_text, top_k=top_k)
        self._last_raw_scores = raw_scores

        novelty = assess_novelty(
            query_text=query_text,
            raw_scores=raw_scores,
            config=config or self.neurogenesis_config,
        )

        seeds = self.seed(query_text, memory_search_ids=memory_search_ids, top_k=top_k)

        auto_node_id: str | None = None
        auto_created = False
        if novelty.should_create and not novelty.blocked:
            auto_node_id = deterministic_auto_id(query_text)
            existing = self.graph.get_node(auto_node_id)
            now = time.time()

            if existing is None:
                auto_created = True
                existing = Node(
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
                self.graph.add_node(existing)
            else:
                existing.metadata["auto_probationary"] = True
                existing.metadata["auto_seed_count"] = (
                    int(existing.metadata.get("auto_seed_count", 0)) + 1
                )

            existing.metadata["last_seen_ts"] = now
            try:
                self.index.upsert(auto_node_id, query_text, self._embedding_fn())
            except TypeError as exc:
                # Index type assumptions can vary during bootstrap; ignore this miss.
                self._pending_index_refresh = True
                _ = str(exc)

            seeds[auto_node_id] = max(seeds.get(auto_node_id, 0.0), 0.25)

        cfg = TraversalConfig(max_hops=max_hops)
        trajectory = traverse(
            query_text,
            self.graph,
            self.router,
            config=cfg,
            embedding_index=self.index,
            seed_nodes=list(seeds.items()),
        )

        if auto_node_id and auto_created:
            connect_targets = dict.fromkeys(list(trajectory.context_nodes) + list(seeds.keys()))
            connect_new_node(
                graph=self.graph,
                new_node_id=auto_node_id,
                current_seed_ids=list(connect_targets.keys()),
                weights=0.15,
            )

        context = render_context(trajectory, self.graph, max_chars=4096)
        nodes = list(trajectory.visit_order)
        self._last_trajectory = trajectory

        result = QueryResult(
            context=context,
            nodes=nodes,
            chars=len(context),
            novel_node=auto_node_id if auto_created else None,
            candidate_node=auto_node_id if novelty.should_create else None,
            band=novelty.band,
            best_cosine=novelty.top_score,
            trajectory_steps=list(trajectory.steps),
            created_novel_node=auto_created,
        )

        self.save()
        return result

    def context(self, firing_result: Firing) -> dict[str, Any]:
        ranked = sorted(firing_result.fired, key=lambda item: item[1], reverse=True)
        return {
            "contents": [node.content for node, _ in ranked],
            "guardrails": list(firing_result.inhibited),
            "fired_ids": [node.id for node, _ in ranked],
            "fired_scores": [score for _, score in ranked],
        }

    def learn(self, reward: float | Firing, outcome: float | None = None) -> LearningResult:
        if isinstance(reward, Firing):
            if outcome is None:
                raise TypeError("Activation-based learn requires outcome")
            _learn(self.graph, reward, outcome=outcome)
            self.save()
            return LearningResult(updates=[], baseline=0.0, avg_reward=float(outcome))

        if self._last_trajectory is None:
            raise RuntimeError("No trajectory cached. Call query() first.")

        reward_signal = RewardSignal(
            episode_id=self._episode_id,
            final_reward=float(reward),
        )
        learning_result = make_learning_step(
            self.graph,
            self._last_trajectory.steps,
            reward_signal,
            self.learning_config,
        )
        self.save()
        return learning_result

    def learn_implicit(self, reward: float = 0.1) -> LearningResult:
        return self.learn(reward)

    def decay(self, turns: int = 1) -> dict[str, float]:
        changed = apply_decay(self.graph, turns_elapsed=turns, config=self.decay_config)
        if changed:
            self.save()
        return changed

    def consolidate(self, config: dict[str, Any] | None = None) -> ConsolidationResult:
        min_weight = 0.05 if config is None else float(config.get("min_weight", 0.05))
        stats = self.graph.consolidate(min_weight=min_weight)
        self.save()
        return ConsolidationResult(
            pruned_edges=int(stats.get("pruned_edges", 0)),
            pruned_nodes=int(stats.get("pruned_nodes", 0)),
        )

    def snapshot(
        self, session_id: str, turn_id: int | str, firing_result: Firing
    ) -> dict[str, Any]:
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


OpenClawCrabPathAdapter = CrabPathAgent
