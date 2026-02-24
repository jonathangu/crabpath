"""
CrabPath Memory Graph — Core data structures.

The memory graph is a directed, typed, signed multigraph where:
- Nodes represent typed memory chunks (facts, rules, tools, actions, sequences, episodes, etc.)
- Edges encode relationships (association, sequence, causation, inhibition, etc.)
- Edge weights are learned from outcomes
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import networkx as nx
except ImportError:
    nx = None  # Defer error to runtime


# ─── Node Types ───────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    FACT = "fact"
    RULE = "rule"
    TOOL = "tool"
    ACTION = "action"
    SEQUENCE = "sequence"
    EPISODE = "episode"
    ERROR_CLASS = "error_class"
    PREFERENCE = "preference"
    HUB = "hub"


# ─── Edge Types ───────────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    ASSOCIATION = "association"
    SEQUENCE = "sequence"          # temporal: A then B
    CAUSATION = "causation"        # A caused B
    CONTINGENCY = "contingency"    # if A then B
    INHIBITION = "inhibition"      # A blocks B (negative weight)
    PREFERENCE = "preference"      # biases choices
    ABSTRACTION = "abstraction"    # generalizes to
    TOOL_APPLIES = "tool_applies"  # tool applicable to context


# ─── Node ─────────────────────────────────────────────────────────────────────

@dataclass
class MemoryNode:
    """A typed memory node in the CrabPath graph."""
    id: str
    node_type: NodeType
    content: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    prior: float = 0.0  # base-level activation (recency × frequency)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Quarantine state (immune system)
    quarantined: bool = False
    quarantine_reason: str = ""


# ─── Edge ─────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEdge:
    """A typed, weighted edge in the CrabPath graph."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0  # positive for excitatory, negative for inhibitory
    condition: str = ""  # optional: when does this edge apply?
    success_count: int = 0
    failure_count: int = 0
    last_traversed: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


# ─── Memory Graph ─────────────────────────────────────────────────────────────

class MemoryGraph:
    """
    The CrabPath memory graph.
    
    A directed, typed, signed multigraph backed by NetworkX (in-memory)
    and SQLite (persistence + metadata).
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        if nx is None:
            raise ImportError("CrabPath requires networkx: pip install networkx")
        
        self.G = nx.DiGraph()
        self.db_path = Path(db_path) if db_path else None
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: dict[tuple[str, str, str], MemoryEdge] = {}
        
        if self.db_path:
            self._init_db()

    def _init_db(self):
        """Initialize SQLite storage."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                prior REAL DEFAULT 0.0,
                created_at REAL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                quarantined INTEGER DEFAULT 0,
                quarantine_reason TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                condition TEXT DEFAULT '',
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_traversed REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}',
                PRIMARY KEY (source, target, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_tags ON nodes(tags);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        conn.close()

    # ─── Node Operations ──────────────────────────────────────────────────

    def add_node(self, node: MemoryNode) -> None:
        """Add a memory node to the graph."""
        self._nodes[node.id] = node
        self.G.add_node(node.id, node_type=node.node_type.value)

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        self._nodes.pop(node_id, None)
        # Remove edges involving this node
        to_remove = [k for k in self._edges if node_id in (k[0], k[1])]
        for k in to_remove:
            self._edges.pop(k)
        if node_id in self.G:
            self.G.remove_node(node_id)

    def nodes_by_type(self, node_type: NodeType) -> list[MemoryNode]:
        """Get all nodes of a given type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    # ─── Edge Operations ──────────────────────────────────────────────────

    def add_edge(self, edge: MemoryEdge) -> None:
        """Add a typed edge to the graph."""
        key = (edge.source, edge.target, edge.edge_type.value)
        self._edges[key] = edge
        self.G.add_edge(
            edge.source, edge.target,
            edge_type=edge.edge_type.value,
            weight=edge.weight
        )

    def get_edges_from(self, node_id: str, edge_type: Optional[EdgeType] = None) -> list[MemoryEdge]:
        """Get outgoing edges from a node, optionally filtered by type."""
        edges = [e for e in self._edges.values() if e.source == node_id]
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_edges_to(self, node_id: str, edge_type: Optional[EdgeType] = None) -> list[MemoryEdge]:
        """Get incoming edges to a node, optionally filtered by type."""
        edges = [e for e in self._edges.values() if e.target == node_id]
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_inhibitors(self, node_id: str) -> list[MemoryEdge]:
        """Get all inhibitory edges targeting this node."""
        return self.get_edges_to(node_id, EdgeType.INHIBITION)

    # ─── Graph Stats ──────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def stats(self) -> dict:
        """Return graph statistics."""
        type_counts = {}
        for n in self._nodes.values():
            type_counts[n.node_type.value] = type_counts.get(n.node_type.value, 0) + 1
        
        edge_type_counts = {}
        for e in self._edges.values():
            edge_type_counts[e.edge_type.value] = edge_type_counts.get(e.edge_type.value, 0) + 1

        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "node_types": type_counts,
            "edge_types": edge_type_counts,
            "quarantined": sum(1 for n in self._nodes.values() if n.quarantined),
        }

    def __repr__(self) -> str:
        return f"MemoryGraph(nodes={self.node_count}, edges={self.edge_count})"
