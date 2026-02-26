"""Policy updates for graph edges and Hebbian co-firing."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable

from .graph import Edge, Graph, Node


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


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


def _first_line(text: str) -> str:
    return (text.splitlines() or [""])[0]


def _unique_node_id(content: str, graph: Graph) -> str:
    base = f"auto:{hashlib.sha1(content.encode('utf-8')).hexdigest()[:10]}"
    candidate = base
    suffix = 0
    while graph.get_node(candidate) is not None:
        suffix += 1
        candidate = f"{base}:{suffix}"
    return candidate


def maybe_create_node(
    graph: Graph,
    query_text: str,
    fired_nodes: list[str],
    llm_fn: Callable[[str, str], str] | None = None,
) -> str | None:
    """If no good nodes were found for this query, ask LLM to create a new node."""
    if llm_fn is None:
        return None

    system = (
        "A query found no relevant documents. Should a new memory node be created from this "
        'query? Return JSON: {"should_create": true/false, "content": "...", "reason": "..."}'
    )
    nearby = ", ".join(
        f"{node_id}: {node.content[:80]}"
        for node_id in fired_nodes
        if (node := graph.get_node(node_id)) is not None
    )
    if not nearby:
        nearby = "<no recent fired nodes>"
    user = f"Query: {query_text}\n\nRecent context: {nearby}"

    try:
        payload = _extract_json(llm_fn(system, user)) or {}
        if not bool(payload.get("should_create", False)):
            return None

        content = str(payload.get("content", query_text)).strip() or query_text
        reason = str(payload.get("reason", "")).strip()
        summary = str(payload.get("summary", _first_line(content))).strip() or _first_line(content)
        node_id = _unique_node_id(content, graph)
        graph.add_node(
            Node(
                id=node_id,
                content=content,
                summary=_first_line(summary),
                metadata={"source": "llm-create", "query": query_text, "reason": reason},
            )
        )
        for source_id in fired_nodes[:5]:
            if source_id == node_id or graph.get_node(source_id) is None:
                continue
            graph.add_edge(Edge(source=source_id, target=node_id, weight=0.15))
            graph.add_edge(Edge(source=node_id, target=source_id, weight=0.15))
        return node_id
    except (Exception, SystemExit):
        return None


@dataclass
class LearningConfig:
    """Hyperparameters for policy-like updates."""

    learning_rate: float = 0.1
    discount: float = 0.95
    hebbian_increment: float = 0.06
    weight_bounds: tuple[float, float] = (-1.0, 1.0)


def _clip_weight(weight: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], weight))


def hebbian_update(
    graph: Graph,
    fired_nodes: list[str],
    config: LearningConfig | None = None,
) -> None:
    """Apply co-firing updates between all observed node pairs.

    Every fired node pair in the trajectory strengthens their edge by
    ``hebbian_increment``. Edges are created if missing.
    """
    cfg = config or LearningConfig()
    if len(fired_nodes) < 2:
        return

    for i, source_id in enumerate(fired_nodes):
        for target_id in fired_nodes[i + 1 :]:
            edge = graph._edges.get(source_id, {}).get(target_id)
            if edge is None:
                graph.add_edge(
                    Edge(
                        source=source_id,
                        target=target_id,
                        weight=cfg.hebbian_increment,
                        kind="sibling",
                    )
                )
            else:
                edge.weight = _clip_weight(
                    edge.weight + cfg.hebbian_increment,
                    cfg.weight_bounds,
                )
                graph._edges[source_id][target_id] = edge


def apply_outcome(
    graph: Graph,
    fired_nodes: list[str],
    outcome: float,
    config: LearningConfig | None = None,
) -> dict:
    """Apply outcome-based policy updates over the full fired trajectory.

    Positive outcome strengthens traversed edges; negative outcome weakens them.
    Negative outcomes may create inhibitory edges when missing.
    Returns a mapping of ``"source->target"`` to weight delta.
    """
    cfg = config or LearningConfig()
    updates: dict[str, float] = defaultdict(float)

    if len(fired_nodes) < 2:
        hebbian_update(graph, fired_nodes, cfg)
        return dict(updates)

    sign = 1.0 if outcome >= 0 else -1.0

    for idx in range(len(fired_nodes) - 1):
        source_id = fired_nodes[idx]
        target_id = fired_nodes[idx + 1]
        edge = graph._edges.get(source_id, {}).get(target_id)
        delta = cfg.learning_rate * (cfg.discount ** (idx + 1)) * sign

        if edge is None:
            graph.add_edge(
                Edge(
                    source=source_id,
                    target=target_id,
                    weight=_clip_weight(delta, cfg.weight_bounds),
                    kind="inhibitory" if sign < 0 else "sibling",
                )
            )
            updates[f"{source_id}->{target_id}"] = delta
            continue

        edge.weight = _clip_weight(edge.weight + delta, cfg.weight_bounds)
        if sign < 0 and edge.weight <= 0:
            edge.kind = "inhibitory"
        graph._edges[source_id][target_id] = edge
        updates[f"{source_id}->{target_id}"] = delta

    hebbian_update(graph, fired_nodes, cfg)
    return dict(updates)
