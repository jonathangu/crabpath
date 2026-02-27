"""Core in-memory graph primitives for CrabPath."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .index import VectorIndex


@dataclass
class Node:
    """A single memory unit."""

    id: str
    content: str
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Directed connection between nodes."""

    source: str
    target: str
    weight: float = 0.5
    kind: str = "sibling"
    metadata: dict[str, Any] = field(default_factory=dict)


class Graph:
    """Directed weighted graph used by all CrabPath operations."""

    def __init__(self) -> None:
        """Create an empty graph."""
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, dict[str, Edge]] = {}

    def add_node(self, node: Node) -> None:
        """Add or replace a node by ``node.id``.

        Args:
            node: Node to add.
        """
        self._nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add or replace a directed edge.

        The graph auto-creates the source bucket when needed and clamps ``edge.weight``
        to ``[-1.0, 1.0]``.
        """
        source = edge.source
        if source not in self._edges:
            self._edges[source] = {}
        self._edges[source][edge.target] = Edge(
            source=edge.source,
            target=edge.target,
            weight=max(-1.0, min(1.0, edge.weight)),
            kind=edge.kind,
            metadata=dict(edge.metadata),
        )

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all incident edges."""
        self._nodes.pop(node_id, None)
        self._edges.pop(node_id, None)
        for source in list(self._edges):
            self._edges[source].pop(node_id, None)
            if not self._edges[source]:
                self._edges.pop(source, None)

    def remove_node_cascade(self, node_id: str) -> None:
        """Remove a node and all incident edges."""
        self.remove_node(node_id)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge if present."""
        source_edges = self._edges.get(source)
        if source_edges is None:
            return
        source_edges.pop(target, None)
        if not source_edges:
            self._edges.pop(source, None)

    def get_node(self, id: str) -> Node | None:
        """Return a node by id, or ``None`` when absent."""
        return self._nodes.get(id)

    def nodes(self) -> list[Node]:
        """Return all nodes in insertion-like id order."""
        return list(self._nodes.values())

    def outgoing(self, id: str) -> list[tuple[Node, Edge]]:
        """Return ``(node, edge)`` pairs for edges whose source is ``id``."""
        if id not in self._edges:
            return []
        result: list[tuple[Node, Edge]] = []
        for edge in self._edges[id].values():
            node = self._nodes.get(edge.target)
            if node is not None:
                result.append((node, edge))
        return result

    def incoming(self, id: str) -> list[tuple[Node, Edge]]:
        """Return ``(node, edge)`` pairs for edges whose target is ``id``."""
        result: list[tuple[Node, Edge]] = []
        for source_edges in self._edges.values():
            edge = source_edges.get(id)
            if edge is None:
                continue
            node = self._nodes.get(edge.source)
            if node is not None:
                result.append((node, edge))
        return result

    def node_count(self) -> int:
        """Return number of nodes in the graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return number of directed edges in the graph."""
        return sum(len(edges) for edges in self._edges.values())

    def save(self, path: str) -> None:
        """Persist graph structure to JSON at ``path``."""
        payload = {
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "summary": node.summary,
                    "metadata": node.metadata,
                }
                for node in self.nodes()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "kind": edge.kind,
                    "metadata": edge.metadata,
                }
                for source_edges in self._edges.values()
                for edge in source_edges.values()
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Graph":
        """Load graph data from JSON persisted by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(
                Node(
                    id=node_data["id"],
                    content=node_data["content"],
                    summary=node_data.get("summary", ""),
                    metadata=node_data.get("metadata", {}),
                )
            )
        for edge_data in data.get("edges", []):
            graph.add_edge(
                Edge(
                    source=edge_data["source"],
                    target=edge_data["target"],
                    weight=edge_data.get("weight", 0.5),
                    kind=edge_data.get("kind", "sibling"),
                    metadata=edge_data.get("metadata", {}),
                )
            )
        return graph


def remove_from_state(graph: Graph, index: VectorIndex, node_id: str) -> None:
    """Remove a node from both graph and vector index."""
    graph.remove_node_cascade(node_id)
    index.remove(node_id)
