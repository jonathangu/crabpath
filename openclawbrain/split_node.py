"""Runtime node splitting utilities for maintenance-time graph refactors."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .graph import Edge, Graph, Node
from .index import VectorIndex


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value into a fixed range."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _node_authority(node: Node | None) -> str:
    """Get node authority with safe fallback."""
    if node is None or not isinstance(node.metadata, dict):
        return "overlay"
    authority = node.metadata.get("authority")
    if authority in {"constitutional", "canonical", "overlay"}:
        return authority
    return "overlay"


def _outgoing_variance(weights: list[float]) -> float:
    """Compute sample-free variance for outgoing edge weights."""
    count = len(weights)
    if count <= 1:
        return 0.0
    mean = sum(weights) / count
    return sum((weight - mean) ** 2 for weight in weights) / count


@dataclass
class SplitCandidate:
    """Scoring details for a node considered for splitting."""

    node_id: str
    score: float
    reasons: dict[str, float]
    needs_confirmation: bool


@dataclass
class SplitResult:
    """Result payload for ``split_node()``."""

    parent_id: str
    child_ids: list[str]
    edges_rewired: int
    siblings_added: int


def suggest_splits(
    graph: Graph,
    index: VectorIndex,
    *,
    min_content_chars: int = 800,
    max_candidates: int = 10,
    llm_fn: Callable[[str, str], str] | None = None,
) -> list[SplitCandidate]:
    """Score candidate nodes that may benefit from runtime splitting."""
    _ = index
    _ = llm_fn

    candidates: list[SplitCandidate] = []
    for node in graph.nodes():
        if _node_authority(node) == "constitutional":
            continue

        content_len = len(node.content)
        content_score = _clamp((content_len - min_content_chars) / max(1, 3000 - min_content_chars), 0.0, 1.0)

        incoming = graph.incoming(node.id)
        outgoing = graph.outgoing(node.id)
        degree = len(incoming) + len(outgoing)
        hub_score = _clamp(math.log1p(degree) / math.log1p(30), 0.0, 1.0)

        is_merged = isinstance(node.metadata, dict) and (
            "merged_from" in node.metadata or str(node.id).startswith("merged:")
        )
        merge_origin_score = 1.0 if is_merged else 0.0

        outgoing_variance = _clamp(_outgoing_variance([edge.weight for _target, edge in outgoing]), 0.0, 1.0)
        reasons = {
            "content_length": content_score,
            "hub_degree": hub_score,
            "merge_origin": merge_origin_score,
            "edge_weight_variance": outgoing_variance,
        }

        score = (
            reasons["content_length"] * 0.40
            + reasons["hub_degree"] * 0.25
            + reasons["merge_origin"] * 0.15
            + reasons["edge_weight_variance"] * 0.20
        )

        candidates.append(
            SplitCandidate(
                node_id=node.id,
                score=score,
                reasons=reasons,
                needs_confirmation=_node_authority(node) == "canonical",
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_candidates]


def _node_embedding(
    graph: Graph,
    index: VectorIndex,
    embed_fn: Callable[[str], list[float]],
    node_id: str,
    cache: dict[str, list[float] | None],
) -> list[float] | None:
    """Resolve an embedding from index first, then fallback to the callback."""
    if node_id in cache:
        return cache[node_id]

    vector = index._vectors.get(node_id)
    if vector is not None:
        cache[node_id] = list(vector)
        return cache[node_id]

    node = graph.get_node(node_id)
    if node is None:
        cache[node_id] = None
        return None

    value = list(embed_fn(node.content))
    cache[node_id] = value
    return value


def split_node(
    graph: Graph,
    index: VectorIndex,
    node_id: str,
    chunks: list[str],
    *,
    embed_fn: Callable[[str], list[float]],
    sibling_weight: float = 0.3,
) -> SplitResult:
    """Split a node into children and rewire all incident edges."""
    parent = graph.get_node(node_id)
    if parent is None:
        raise ValueError(f"parent node not found: {node_id}")
    if _node_authority(parent) == "constitutional":
        raise ValueError("constitutional nodes cannot be split")

    cleaned_chunks = [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    if not cleaned_chunks:
        raise ValueError("chunks must contain at least one non-empty section")

    child_ids: list[str] = []
    child_vectors: list[list[float]] = []
    for idx, chunk in enumerate(cleaned_chunks):
        child_id = f"split:{node_id}:{idx}"
        child_ids.append(child_id)
        summary = chunk.splitlines()[0] if chunk.splitlines() else ""
        child_vector = list(embed_fn(chunk))
        graph.add_node(Node(id=child_id, content=chunk, summary=summary, metadata={"parent": node_id, "source": "split_node"}))
        index.upsert(child_id, child_vector)
        child_vectors.append(child_vector)

    edges_rewired = 0
    embedding_cache: dict[str, list[float] | None] = {}

    outgoing_edges = [(target_id, edge) for target_id, edge in list(graph._edges.get(node_id, {}).items()) if graph.get_node(target_id) is not None]
    for target_id, edge in outgoing_edges:
        if edge.kind == "inhibitory":
            for child_id in child_ids:
                graph.add_edge(Edge(source=child_id, target=target_id, weight=edge.weight, kind=edge.kind, metadata=dict(edge.metadata)))
                edges_rewired += 1
            continue

        target_embedding = _node_embedding(graph, index, embed_fn, target_id, embedding_cache)
        best_child = child_ids[0]
        best_similarity = -1.0
        for child_index, (child_id, child_embedding) in enumerate(zip(child_ids, child_vectors)):
            sim = VectorIndex.cosine(child_embedding, target_embedding) if target_embedding else 0.0
            if sim > best_similarity:
                best_similarity = sim
                best_child = child_id

        if best_similarity < 0.2:
            split_weight = edge.weight / len(child_ids)
            for child_id in child_ids:
                graph.add_edge(Edge(source=child_id, target=target_id, weight=split_weight, kind=edge.kind, metadata=dict(edge.metadata)))
                edges_rewired += 1
        else:
            graph.add_edge(Edge(source=best_child, target=target_id, weight=edge.weight, kind=edge.kind, metadata=dict(edge.metadata)))
            edges_rewired += 1

    incoming_edges = [(source.id, edge) for source, edge in graph.incoming(node_id)]
    for source_id, edge in incoming_edges:
        if edge.kind == "inhibitory":
            for child_id in child_ids:
                graph.add_edge(Edge(source=source_id, target=child_id, weight=edge.weight, kind=edge.kind, metadata=dict(edge.metadata)))
                edges_rewired += 1
            continue

        source_embedding = _node_embedding(graph, index, embed_fn, source_id, embedding_cache)
        best_child = child_ids[0]
        best_similarity = -1.0
        for child_id, child_embedding in zip(child_ids, child_vectors):
            sim = VectorIndex.cosine(child_embedding, source_embedding) if source_embedding else 0.0
            if sim > best_similarity:
                best_similarity = sim
                best_child = child_id

        if best_similarity < 0.2:
            split_weight = edge.weight / len(child_ids)
            for child_id in child_ids:
                graph.add_edge(Edge(source=source_id, target=child_id, weight=split_weight, kind=edge.kind, metadata=dict(edge.metadata)))
                edges_rewired += 1
        else:
            graph.add_edge(Edge(source=source_id, target=best_child, weight=edge.weight, kind=edge.kind, metadata=dict(edge.metadata)))
            edges_rewired += 1

    siblings_added = 0
    if len(child_ids) > 1:
        for source_offset in range(len(child_ids)):
            for target_offset in range(len(child_ids)):
                if source_offset == target_offset:
                    continue
                source_child = child_ids[source_offset]
                target_child = child_ids[target_offset]
                graph.add_edge(Edge(source=source_child, target=target_child, weight=sibling_weight, kind="sibling"))
                siblings_added += 1

    graph.remove_node(node_id)
    index.remove(node_id)

    return SplitResult(
        parent_id=node_id,
        child_ids=child_ids,
        edges_rewired=edges_rewired,
        siblings_added=siblings_added,
    )
