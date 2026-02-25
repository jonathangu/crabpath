"""
CrabPath — A neuron-inspired memory graph. Zero dependencies.

A node is a neuron: it accumulates energy, fires when threshold is crossed,
and sends weighted signals (positive or negative) to its connections.
It leaves a trace when it fires — a decaying record of recent activity.

Nodes hold content. Edges are weighted pointers. That's it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Node:
    """A neuron in the memory graph.

    - content: what this neuron "knows" (a fact, rule, action, whatever)
    - threshold: fires when potential >= threshold
    - potential: current accumulated energy (transient state)
    - trace: decaying record of recent firing (0 = cold, higher = recently active)
    - metadata: your bag of whatever — types, tags, timestamps, priors.
      CrabPath has no opinions about what goes in here.
    """

    id: str
    content: str
    threshold: float = 1.0
    potential: float = 0.0
    trace: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A weighted directed connection between neurons.

    - weight > 0: excitatory (adds energy to target)
    - weight < 0: inhibitory (removes energy from target)
    """

    source: str
    target: str
    weight: float = 1.0


class Graph:
    """A weighted directed graph. Plain dicts. No dependencies."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str], Edge] = {}
        self._outgoing: dict[str, list[str]] = {}
        self._incoming: dict[str, list[str]] = {}

    # -- Nodes --

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node
        self._outgoing.setdefault(node.id, [])
        self._incoming.setdefault(node.id, [])

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        for tgt in list(self._outgoing.get(node_id, [])):
            self._edges.pop((node_id, tgt), None)
            lst = self._incoming.get(tgt, [])
            if node_id in lst:
                lst.remove(node_id)
        for src in list(self._incoming.get(node_id, [])):
            self._edges.pop((src, node_id), None)
            lst = self._outgoing.get(src, [])
            if node_id in lst:
                lst.remove(node_id)
        self._nodes.pop(node_id, None)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)

    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    # -- Edges --

    def add_edge(self, edge: Edge) -> None:
        key = (edge.source, edge.target)
        existed = key in self._edges
        self._edges[key] = edge
        if not existed:
            self._outgoing.setdefault(edge.source, []).append(edge.target)
            self._incoming.setdefault(edge.target, []).append(edge.source)

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        return self._edges.get((source, target))

    def _remove_edge(self, source: str, target: str) -> bool:
        key = (source, target)
        if self._edges.pop(key, None) is None:
            return False
        if target in self._incoming and source in self._incoming[target]:
            self._incoming[target].remove(source)
        if source in self._outgoing and target in self._outgoing[source]:
            self._outgoing[source].remove(target)
        return True

    def outgoing(self, node_id: str) -> list[tuple[Node, Edge]]:
        """All outgoing connections from a node: [(target_node, edge), ...]"""
        result = []
        for tgt in self._outgoing.get(node_id, []):
            node = self._nodes.get(tgt)
            edge = self._edges.get((node_id, tgt))
            if node and edge:
                result.append((node, edge))
        return result

    def incoming(self, node_id: str) -> list[tuple[Node, Edge]]:
        """All incoming connections to a node: [(source_node, edge), ...]"""
        result = []
        for src in self._incoming.get(node_id, []):
            node = self._nodes.get(src)
            edge = self._edges.get((src, node_id))
            if node and edge:
                result.append((node, edge))
        return result

    def edges(self) -> list[Edge]:
        return list(self._edges.values())

    def consolidate(self, min_weight: float = 0.05) -> dict[str, int]:
        pruned_edges = 0
        pruned_nodes = 0

        for source, target in list(self._edges):
            edge = self._edges.get((source, target))
            if edge is None:
                continue
            if abs(edge.weight) < min_weight:
                if self._remove_edge(source, target):
                    pruned_edges += 1

        for node_id in list(self._nodes):
            if self._incoming.get(node_id) or self._outgoing.get(node_id):
                continue
            node = self._nodes.get(node_id)
            if node is not None and node.metadata.get("protected") is True:
                continue
            self.remove_node(node_id)
            pruned_nodes += 1

        return {"pruned_edges": pruned_edges, "pruned_nodes": pruned_nodes}

    def merge_nodes(self, keep_id: str, remove_id: str) -> bool:
        if keep_id == remove_id or keep_id not in self._nodes or remove_id not in self._nodes:
            return False

        edges_to_move = {}
        for target in list(self._outgoing.get(remove_id, [])):
            edge = self.get_edge(remove_id, target)
            if edge is not None:
                edges_to_move[(remove_id, target)] = edge.weight
        for source in list(self._incoming.get(remove_id, [])):
            edge = self.get_edge(source, remove_id)
            if edge is not None:
                edges_to_move[(source, remove_id)] = edge.weight

        for (source, target), weight in edges_to_move.items():
            new_source = keep_id if source == remove_id else source
            new_target = keep_id if target == remove_id else target
            if new_source == source and new_target == target:
                continue

            existing = self.get_edge(new_source, new_target)
            if existing is not None:
                if abs(weight) > abs(existing.weight):
                    existing.weight = weight
            else:
                self.add_edge(Edge(source=new_source, target=new_target, weight=weight))

            self._remove_edge(source, target)

        self.remove_node(remove_id)
        return True

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # -- State --

    def reset_potentials(self) -> None:
        """Set all node potentials to 0."""
        for node in self._nodes.values():
            node.potential = 0.0

    def warm_nodes(self, min_trace: float = 0.01) -> list[tuple[Node, float]]:
        """Nodes with non-trivial trace, sorted by trace descending.

        Useful for checking "what's been active recently?" without
        running a full activation pass.
        """
        warm = [(n, n.trace) for n in self._nodes.values() if n.trace >= min_trace]
        warm.sort(key=lambda x: x[1], reverse=True)
        return warm

    # -- Persistence --

    def save(self, path: str) -> None:
        """Save graph to a JSON file."""
        data = {
            "nodes": [_node_to_dict(n) for n in self._nodes.values()],
            "edges": [_edge_to_dict(e) for e in self._edges.values()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Graph:
        """Load graph from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        g = cls()
        for nd in data["nodes"]:
            g.add_node(Node(**nd))
        for ed in data["edges"]:
            g.add_edge(Edge(**ed))
        return g

    def __repr__(self) -> str:
        return f"Graph(nodes={self.node_count}, edges={self.edge_count})"


def _node_to_dict(n: Node) -> dict:
    d: dict[str, Any] = {"id": n.id, "content": n.content}
    if n.threshold != 1.0:
        d["threshold"] = n.threshold
    if n.potential != 0.0:
        d["potential"] = n.potential
    if n.trace != 0.0:
        d["trace"] = n.trace
    if n.metadata:
        d["metadata"] = n.metadata
    return d


def _edge_to_dict(e: Edge) -> dict:
    d: dict[str, Any] = {"source": e.source, "target": e.target}
    if e.weight != 1.0:
        d["weight"] = e.weight
    return d
