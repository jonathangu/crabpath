"""Policy updates for graph edges and Hebbian co-firing."""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable

from .graph import Edge, Graph, Node
from ._util import _extract_json, _first_line
from .decay import apply_decay


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
    """Hyperparameters for policy-like updates.

    learning_rate controls outcome update magnitude. Default is ``0.1``. For
    small graphs (<100 nodes) use ``0.05``-``0.1``. For large graphs (>1000
    nodes) use ``0.01``-``0.05``. Higher values are faster but more volatile.

    promotion_threshold is the minimum co-firing count before Hebbian creates an
    edge. Default is ``2``. Lower values create more edges, higher values are
    more selective.
    """

    learning_rate: float = 0.1
    discount: float = 0.95
    hebbian_increment: float = 0.06
    promotion_threshold: int = 2
    weight_bounds: tuple[float, float] = (-1.0, 1.0)


_apply_outcome_call_count = 0


def _clip_weight(weight: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], weight))


def hebbian_update(
    graph: Graph,
    fired_nodes: list[str],
    config: LearningConfig | None = None,
    max_hebbian_pairs: int = 20,
) -> None:
    """Apply co-firing updates between observed node pairs.

    By default, every observed pair receives ``hebbian_increment``. When
    ``max_hebbian_pairs`` is set and the fire trajectory is long, only the
    closest firing-order pairs are updated, capped to ``max_hebbian_pairs`` total
    updates.
    """
    cfg = config or LearningConfig()
    if len(fired_nodes) < 2:
        return

    if max_hebbian_pairs <= 0:
        return

    max_pairs = max_hebbian_pairs
    if len(fired_nodes) <= math.sqrt(2 * max_pairs):
        pair_indexes = [(i, j) for i in range(len(fired_nodes)) for j in range(i + 1, len(fired_nodes))]
    else:
        candidates: list[tuple[int, int, int]] = []
        for source_idx in range(len(fired_nodes)):
            for target_idx in range(source_idx + 1, len(fired_nodes)):
                distance = target_idx - source_idx
                candidates.append((distance, source_idx, target_idx))

        candidates.sort(key=lambda item: (item[0], item[1]))
        pair_indexes = [(source_idx, target_idx) for _, source_idx, target_idx in candidates[:max_pairs]]

    for source_idx, target_idx in pair_indexes:
        co_firing_count = target_idx - source_idx + 1
        if co_firing_count < cfg.promotion_threshold:
            continue

        source_id = fired_nodes[source_idx]
        target_id = fired_nodes[target_idx]
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
    auto_decay: bool = False,
    decay_interval: int = 10,
) -> dict:
    """Apply outcome-based policy updates over the full fired trajectory.

    Positive outcome strengthens traversed edges; negative outcome weakens them.
    Negative outcomes may create inhibitory edges when missing.
    Returns a mapping of ``"source->target"`` to weight delta.

    When ``auto_decay`` is true, the graph is decayed after every
    ``decay_interval`` successful calls to this function.
    """
    global _apply_outcome_call_count
    cfg = config or LearningConfig()
    updates: dict[str, float] = defaultdict(float)

    if len(fired_nodes) < 2:
        hebbian_update(graph, fired_nodes, cfg)
        _apply_outcome_call_count += 1
        if auto_decay:
            if decay_interval <= 0:
                decay_interval = 1
            if _apply_outcome_call_count % decay_interval == 0:
                apply_decay(graph)
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
        if sign < 0:
            edge.kind = "inhibitory"
        graph._edges[source_id][target_id] = edge
        updates[f"{source_id}->{target_id}"] = delta

    hebbian_update(graph, fired_nodes, cfg)

    _apply_outcome_call_count += 1
    if auto_decay:
        if decay_interval <= 0:
            decay_interval = 1
        if _apply_outcome_call_count % decay_interval == 0:
            apply_decay(graph)

    return dict(updates)
