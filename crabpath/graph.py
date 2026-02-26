"""
CrabPath — A neuron-inspired memory graph. Zero dependencies.

A node is a neuron: it accumulates energy, fires when threshold is crossed,
and sends weighted signals (positive or negative) to its connections.
It leaves a trace when it fires — a decaying record of recent activity.

Nodes hold content. Edges are weighted pointers. That's it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1


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
    summary: str = ""
    type: str = "fact"
    threshold: float = 1.0
    potential: float = 0.0
    trace: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    failure_count: int = 0
    cluster_id: str = ""
    created_at: float | None = None


@dataclass
class Edge:
    """A weighted directed connection between neurons.

    - weight > 0: excitatory (adds energy to target)
    - weight < 0: inhibitory (removes energy from target)
    """

    source: str
    target: str
    weight: float = 1.0
    decay_rate: float = 0.01
    last_followed_ts: float | None = None
    created_by: str = "manual"  # auto | manual | llm
    follow_count: int = 0
    skip_count: int = 0
    kind: str = "excitatory"
    evidence_count: int = 0
    provenance: str = ""
    last_modified_ts: float | None = None
    decay_group: str = ""


@dataclass
class ConsolidationConfig:
    min_edge_weight: float = 0.03
    max_node_chars: int = 1500
    min_fires_to_split: int = 10
    merge_cosine_threshold: float = 0.90
    merge_cofire_threshold: float = 0.80
    probation_max_turns: int = 50


@dataclass
class ConsolidationResult:
    edges_pruned: int
    nodes_pruned: int
    nodes_split: int
    nodes_merged: int


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_probationary(metadata: dict) -> bool:
    return bool(metadata.get("probationary")) or bool(metadata.get("auto_probationary"))


def prune_weak_edges(graph: Graph, min_weight: float) -> int:
    pruned = 0
    for edge in list(graph.edges()):
        if abs(edge.weight) >= min_weight:
            continue
        if graph._remove_edge(edge.source, edge.target):
            pruned += 1
    return pruned


def prune_orphan_nodes(graph: Graph, protected_ids: set[str] | None = None) -> int:
    protected = set(protected_ids or [])
    pruned = 0
    orphan_ids = [node.id for node in graph.nodes() if not graph._incoming.get(node.id)]
    for node_id in orphan_ids:
        if graph.is_node_protected(node_id) or node_id in protected:
            continue
        graph.remove_node(node_id)
        pruned += 1
    return pruned


def prune_probationary(graph: Graph, max_turns: int) -> int:
    now = time.time()
    pruned = 0
    for node in list(graph.nodes()):
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        if not _is_probationary(metadata):
            continue
        if _as_int(metadata.get("fired_count")) >= 3:
            continue

        created_ts = _as_float(metadata.get("created_ts"), default=0.0)
        if now - created_ts <= max_turns:
            continue
        graph.remove_node(node.id)
        pruned += 1
    return pruned


def should_split(node: Node, config: ConsolidationConfig) -> bool:
    if not isinstance(node, Node):
        return False
    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    fires = _as_int(metadata.get("fired_count"), default=0)
    return len(node.content) > config.max_node_chars and fires > config.min_fires_to_split


def split_node(graph: Graph, node_id: str, parts: list[dict]) -> list[str]:
    parent = graph.get_node(node_id)
    if parent is None:
        return []

    now = time.time()
    child_ids: list[str] = []

    parent_outgoing = list(graph._outgoing.get(node_id, []))
    for part in parts:
        child_id = str(part.get("id", f"{node_id}:split"))
        idx = 1
        while graph.get_node(child_id) is not None:
            child_id = f"{child_id}:{idx}"
            idx += 1

        child = Node(
            id=child_id,
            content=str(part.get("content", "")),
            summary=str(part.get("summary", "")),
            type=parent.type,
            metadata={
                "fired_count": 0,
                "last_fired_ts": 0.0,
                "created_ts": now,
                "protected": bool(parent.metadata.get("protected"))
                if isinstance(parent.metadata, dict)
                else False,
            },
        )
        graph.add_node(child)
        child_ids.append(child_id)

        graph.add_edge(Edge(source=node_id, target=child_id, weight=0.7))

    for target in parent_outgoing:
        parent_target_edge = graph.get_edge(node_id, target)
        if parent_target_edge is None:
            continue
        graph._remove_edge(node_id, target)
        for child_id in child_ids:
            existing = graph.get_edge(child_id, target)
            if existing is None:
                graph.add_edge(
                    Edge(
                        source=child_id,
                        target=target,
                        weight=parent_target_edge.weight,
                        decay_rate=parent_target_edge.decay_rate,
                        last_followed_ts=parent_target_edge.last_followed_ts,
                        created_by=parent_target_edge.created_by,
                        follow_count=parent_target_edge.follow_count,
                        skip_count=parent_target_edge.skip_count,
                    )
                )
            elif abs(parent_target_edge.weight) > abs(existing.weight):
                existing.weight = parent_target_edge.weight
                existing.decay_rate = parent_target_edge.decay_rate
                existing.last_followed_ts = parent_target_edge.last_followed_ts
                existing.created_by = parent_target_edge.created_by
                existing.follow_count += parent_target_edge.follow_count
                existing.skip_count += parent_target_edge.skip_count

    parent.summary = f"See: {', '.join(child_ids)}"
    parent.content = parent.summary

    return child_ids


def should_merge(
    graph: Graph,
    node_a_id: str,
    node_b_id: str,
    cofire_count: int,
    total_fires: int,
    cosine_sim: float,
    config: ConsolidationConfig,
) -> bool:
    if total_fires <= 0:
        return False
    if node_a_id == node_b_id:
        return False
    if graph.get_node(node_a_id) is None or graph.get_node(node_b_id) is None:
        return False
    return (
        _as_float(cosine_sim, default=0.0) > config.merge_cosine_threshold
        and _as_float(cofire_count, default=0.0) / total_fires > config.merge_cofire_threshold
    )


def consolidate(
    graph: Graph,
    config: ConsolidationConfig | None = None,
    protected_ids: set[str] | None = None,
) -> ConsolidationResult:
    if config is None:
        config = ConsolidationConfig()

    protected_ids = set(protected_ids or [])
    prior_protection = {}
    for node_id in protected_ids:
        node = graph.get_node(node_id)
        if node is None:
            continue
        prior_protection[node_id] = bool(node.metadata.get("protected"))
        node.metadata["protected"] = True

    stats = graph.consolidate(min_weight=config.min_edge_weight)
    for node_id, was_protected in prior_protection.items():
        node = graph.get_node(node_id)
        if node is None:
            continue
        if was_protected:
            node.metadata["protected"] = True
        else:
            node.metadata.pop("protected", None)

    edges_pruned = int(stats.get("pruned_edges", 0))
    nodes_pruned = int(stats.get("pruned_nodes", 0))
    nodes_pruned += prune_probationary(graph, config.probation_max_turns)

    return ConsolidationResult(
        edges_pruned=edges_pruned,
        nodes_pruned=nodes_pruned,
        nodes_split=0,
        nodes_merged=0,
    )


class Graph:
    """A weighted directed graph. Plain dicts. No dependencies."""

    VALID_EDGE_CREATORS = {"auto", "manual", "llm"}
    NODE_DEFAULT_TYPE = "fact"
    EDGE_DEFAULT_CREATED_BY = "manual"

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str], Edge] = {}
        self._outgoing: dict[str, list[str]] = {}
        self._incoming: dict[str, list[str]] = {}

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_guardrail_guard(node: Node) -> bool:
        if node.type != "guardrail":
            return False
        text = node.content.lower()
        return "never" in text or "always" in text

    def _is_node_protected(self, node: Node | None) -> bool:
        if node is None:
            return False
        if node.metadata.get("protected") is True:
            return True
        if self._is_guardrail_guard(node):
            node.metadata["protected"] = True
            return True
        return False

    def is_node_protected(self, node_id: str) -> bool:
        return self._is_node_protected(self.get_node(node_id))

    # -- Nodes --

    def add_node(self, node: Node) -> None:
        if not isinstance(node.metadata, dict):
            node.metadata = {}

        node.metadata.setdefault(
            "fired_count", self._coerce_int(node.metadata.get("fired_count"), 0)
        )
        node.metadata.setdefault(
            "last_fired_ts", self._coerce_float(node.metadata.get("last_fired_ts"), 0.0)
        )
        if "created_ts" not in node.metadata:
            node.metadata["created_ts"] = time.time()

        if self._is_node_protected(node):
            node.metadata["protected"] = True

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

    @staticmethod
    def normalize_node(record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            return {}

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        node_type = record.get("type")
        if not isinstance(node_type, str):
            node_type = metadata.get("type")
            if not isinstance(node_type, str):
                node_type = Graph.NODE_DEFAULT_TYPE

        summary = record.get("summary")
        if not isinstance(summary, str):
            summary_meta = metadata.get("summary")
            summary = summary_meta if isinstance(summary_meta, str) else ""

        def _coerce_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _coerce_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _coerce_optional_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        cluster_id = record.get("cluster_id", "")
        if not isinstance(cluster_id, str):
            cluster_id = ""

        return {
            "id": str(record.get("id", "")),
            "content": str(record.get("content", "")),
            "summary": summary,
            "type": node_type,
            "threshold": _coerce_float(record.get("threshold"), 1.0),
            "potential": _coerce_float(record.get("potential"), 0.0),
            "trace": _coerce_float(record.get("trace"), 0.0),
            "access_count": _coerce_int(record.get("access_count"), 0),
            "failure_count": _coerce_int(record.get("failure_count"), 0),
            "cluster_id": cluster_id,
            "created_at": _coerce_optional_float(record.get("created_at")),
            "metadata": metadata,
        }

    @staticmethod
    def normalize_edge(record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            return {}

        created_by = record.get("created_by", Graph.EDGE_DEFAULT_CREATED_BY)
        if created_by not in Graph.VALID_EDGE_CREATORS:
            created_by = Graph.EDGE_DEFAULT_CREATED_BY

        last_followed_ts = record.get("last_followed_ts")
        if last_followed_ts is not None:
            try:
                last_followed_ts = float(last_followed_ts)
            except (TypeError, ValueError):
                last_followed_ts = None

        def _coerce_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _coerce_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _coerce_optional_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return {
            "source": str(record.get("source", "")),
            "target": str(record.get("target", "")),
            "weight": _coerce_float(record.get("weight"), 1.0),
            "decay_rate": _coerce_float(record.get("decay_rate"), 0.01),
            "last_followed_ts": last_followed_ts,
            "created_by": created_by,
            "follow_count": _coerce_int(record.get("follow_count"), 0),
            "skip_count": _coerce_int(record.get("skip_count"), 0),
            "kind": str(record.get("kind", "excitatory")),
            "evidence_count": _coerce_int(record.get("evidence_count"), 0),
            "provenance": str(record.get("provenance", "")),
            "last_modified_ts": _coerce_optional_float(record.get("last_modified_ts")),
            "decay_group": str(record.get("decay_group", "")),
        }

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
            if self._is_node_protected(node):
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
                edges_to_move[(remove_id, target)] = edge
        for source in list(self._incoming.get(remove_id, [])):
            edge = self.get_edge(source, remove_id)
            if edge is not None:
                edges_to_move[(source, remove_id)] = edge

        for (source, target), edge in edges_to_move.items():
            new_source = keep_id if source == remove_id else source
            new_target = keep_id if target == remove_id else target
            if new_source == source and new_target == target:
                continue

            existing = self.get_edge(new_source, new_target)
            if existing is not None:
                if abs(edge.weight) > abs(existing.weight):
                    existing.weight = edge.weight
                    existing.decay_rate = edge.decay_rate
                    existing.last_followed_ts = edge.last_followed_ts
                    existing.created_by = edge.created_by
                existing.follow_count += edge.follow_count
                existing.skip_count += edge.skip_count
                existing.kind = edge.kind
                existing.evidence_count += edge.evidence_count
                existing.provenance = edge.provenance
                existing.last_modified_ts = edge.last_modified_ts
                existing.decay_group = edge.decay_group
            else:
                self.add_edge(
                    Edge(
                        source=new_source,
                        target=new_target,
                        weight=edge.weight,
                        decay_rate=edge.decay_rate,
                        last_followed_ts=edge.last_followed_ts,
                        created_by=edge.created_by,
                        follow_count=edge.follow_count,
                        skip_count=edge.skip_count,
                        kind=edge.kind,
                        evidence_count=edge.evidence_count,
                        provenance=edge.provenance,
                        last_modified_ts=edge.last_modified_ts,
                        decay_group=edge.decay_group,
                    )
                )

            self._remove_edge(source, target)

        self.remove_node(remove_id)
        return True

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def active_edge_count(self) -> int:
        return len([edge for edge in self._edges.values() if edge.weight != 0.0])

    def prune_zero_weight_edges(self) -> int:
        removed = 0
        for source, target in list(self._edges):
            edge = self._edges.get((source, target))
            if edge is not None and edge.weight == 0.0:
                if self._remove_edge(source, target):
                    removed += 1
        return removed

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
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=str(target.parent), suffix=".tmp"
        ) as f:
            json.dump(data, f, indent=2)
            temp_path = Path(f.name)

        os.replace(temp_path, target)

    @classmethod
    def _migrate_v0_to_v1(cls, data: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(data)
        migrated["schema_version"] = SCHEMA_VERSION
        return migrated

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        g = cls()
        for nd in data.get("nodes", []):
            g.add_node(Node(**cls.normalize_node(nd)))
        for ed in data.get("edges", []):
            g.add_edge(Edge(**cls.normalize_edge(ed)))
        return g

    @classmethod
    def load(cls, path: str) -> Graph:
        """Load graph from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid graph file: top-level object must be a dict")

        raw_schema_version = data.get("schema_version", 0)
        try:
            schema_version = int(raw_schema_version) if raw_schema_version is not None else 0
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid schema_version: {raw_schema_version}") from exc

        if schema_version == 0:
            data = cls._migrate_v0_to_v1(data)
        elif schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"Graph was saved with CrabPath v{schema_version}, you have v{SCHEMA_VERSION}. Please upgrade."
            )

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [_node_to_dict(n) for n in self._nodes.values()],
            "edges": [_edge_to_dict(e) for e in self._edges.values()],
        }

    def to_v2_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def __repr__(self) -> str:
        return f"Graph(nodes={self.node_count}, edges={self.edge_count})"


def _node_to_dict(n: Node) -> dict:
    d: dict[str, Any] = {"id": n.id, "content": n.content}
    if n.summary != "":
        d["summary"] = n.summary
    if n.type != Graph.NODE_DEFAULT_TYPE:
        d["type"] = n.type
    if n.threshold != 1.0:
        d["threshold"] = n.threshold
    if n.potential != 0.0:
        d["potential"] = n.potential
    if n.trace != 0.0:
        d["trace"] = n.trace
    if n.access_count != 0:
        d["access_count"] = n.access_count
    if n.failure_count != 0:
        d["failure_count"] = n.failure_count
    if n.cluster_id != "":
        d["cluster_id"] = n.cluster_id
    if n.created_at is not None:
        d["created_at"] = n.created_at
    if n.metadata:
        metadata_fields = set(n.metadata.keys())
        lifecycle_only = metadata_fields <= {"fired_count", "last_fired_ts", "created_ts"}
        if not lifecycle_only:
            d["metadata"] = n.metadata
    return d


def _edge_to_dict(e: Edge) -> dict:
    d: dict[str, Any] = {"source": e.source, "target": e.target}
    if e.weight != 1.0:
        d["weight"] = e.weight
    if e.decay_rate != 0.01:
        d["decay_rate"] = e.decay_rate
    if e.last_followed_ts is not None:
        d["last_followed_ts"] = e.last_followed_ts
    if e.created_by != "manual":
        d["created_by"] = e.created_by
    if e.follow_count != 0:
        d["follow_count"] = e.follow_count
    if e.skip_count != 0:
        d["skip_count"] = e.skip_count
    if e.kind != "excitatory":
        d["kind"] = e.kind
    if e.evidence_count != 0:
        d["evidence_count"] = e.evidence_count
    if e.provenance != "":
        d["provenance"] = e.provenance
    if e.last_modified_ts is not None:
        d["last_modified_ts"] = e.last_modified_ts
    if e.decay_group != "":
        d["decay_group"] = e.decay_group
    return d
