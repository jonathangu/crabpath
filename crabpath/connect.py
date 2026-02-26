"""Cross-file connection suggestions driven by lexical overlap or LLM feedback."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from .graph import Edge, Graph, Node


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _extract_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None

    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return None
    return None


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text or "")}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_file(node: Node | None) -> str | None:
    if node is None:
        return None
    file_value = node.metadata.get("file")
    return str(file_value) if isinstance(file_value, str) else None


def suggest_connections(
    graph: Graph,
    llm_fn: Callable[[str, str], str] | None = None,
    max_candidates: int = 20,
) -> list[tuple[str, str, float, str]]:
    """Suggest new cross-file edges based on content overlap.

    Finds pairs of nodes from different files that might be related.
    If ``llm_fn`` is provided, asks LLM to confirm and score the connection.
    Otherwise uses a simple word-overlap score.
    """
    if max_candidates <= 0:
        return []

    node_items = list(graph._nodes.items())
    scored_candidates: list[tuple[str, str, float]] = []

    for left_index, (source_id, source_node) in enumerate(node_items):
        source_file = _node_file(source_node)
        if not source_file:
            continue
        source_tokens = _tokenize(source_node.content)
        if not source_tokens:
            continue

        for target_id, target_node in node_items[left_index + 1 :]:
            target_file = _node_file(target_node)
            if target_file is None or source_file == target_file:
                continue

            target_tokens = _tokenize(target_node.content)
            if not target_tokens:
                continue

            shared = source_tokens & target_tokens
            if not shared:
                continue
            union = source_tokens | target_tokens
            if not union:
                continue
            overlap = len(shared) / len(union)
            scored_candidates.append((source_id, target_id, overlap))

    scored_candidates.sort(key=lambda item: (item[2], item[0], item[1]), reverse=True)
    scored_candidates = scored_candidates[:max_candidates]

    suggested: list[tuple[str, str, float, str]] = []
    for source_id, target_id, overlap in scored_candidates:
        source_node = graph.get_node(source_id)
        target_node = graph.get_node(target_id)
        if source_node is None or target_node is None:
            continue

        source_file = _node_file(source_node) or "unknown"
        target_file = _node_file(target_node) or "unknown"

        if llm_fn is None:
            suggested.append((source_id, target_id, overlap, f"word overlap score: {overlap:.4f}"))
            continue

        system = (
            "Given two document chunks from different files, decide if they should be connected. "
            'Return JSON: {"should_connect": true/false, "weight": 0.0-1.0, "reason": "brief"}'
        )
        user = (
            f"Chunk A (from {source_file}): {source_node.content}\n\n"
            f"Chunk B (from {target_file}): {target_node.content}"
        )

        try:
            payload = _extract_json(llm_fn(system, user))
            if payload is None:
                continue
            if not bool(payload.get("should_connect", False)):
                continue
            reason = str(payload.get("reason", "") or "")
            weight = _safe_float(payload.get("weight"), overlap)
            weight = max(0.0, min(1.0, weight))
            suggested.append((source_id, target_id, weight, reason))
        except (Exception, SystemExit):
            continue

    return suggested


def apply_connections(graph: Graph, connections: list[tuple[str, str, float, str]]) -> int:
    """Apply suggested connections as new edges. Returns the number of edges added."""
    added = 0
    seen: set[tuple[str, str]] = set()
    for source_id, target_id, weight, reason in connections:
        if source_id == target_id:
            continue
        if (source_id, target_id) in seen:
            continue
        if graph.get_node(source_id) is None or graph.get_node(target_id) is None:
            continue

        source_node = graph.get_node(source_id)
        target_node = graph.get_node(target_id)
        source_file = _node_file(source_node) or None
        target_file = _node_file(target_node) or None
        if source_file is None or target_file is None or source_file == target_file:
            continue

        existing = graph._edges.get(source_id, {}).get(target_id)
        graph.add_edge(
            Edge(
                source=source_id,
                target=target_id,
                weight=weight,
                kind="cross_file",
                metadata={"reason": reason, "kind": "cross_file", "source": "connect"},
            )
        )
        seen.add((source_id, target_id))
        if existing is None:
            added += 1

    return added
