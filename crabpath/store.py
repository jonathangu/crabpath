"""State persistence helpers for CrabPath."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .graph import Edge, Graph, Node
from .hasher import HashEmbedder
from .index import VectorIndex


def save_state(
    graph: Graph,
    index: VectorIndex,
    path: str,
    *,
    embedder_name: str | None = None,
    embedder_dim: int | None = None,
) -> None:
    """Save graph and index together to one JSON file."""
    if embedder_name is None:
        embedder_name = "hash-v1"
    if embedder_dim is None:
        embedder_dim = HashEmbedder().dim

    payload = {
        "graph": {
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "summary": node.summary,
                    "metadata": node.metadata,
                }
                for node in graph.nodes()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "kind": edge.kind,
                    "metadata": edge.metadata,
                }
                for source_edges in graph._edges.values()
                for edge in source_edges.values()
            ],
        },
        "index": index._vectors,
        "meta": {
            "embedder_name": embedder_name,
            "embedder_dim": embedder_dim,
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node_count": graph.node_count(),
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state(path: str) -> tuple[Graph, VectorIndex, dict[str, object]]:
    """Load graph + index from one JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    graph = Graph()
    if not isinstance(payload, dict):
        raise SystemExit("state payload must be an object")
    graph_payload = payload.get("graph", payload)
    for node_data in graph_payload.get("nodes", []):
        graph.add_node(
            Node(
                id=node_data["id"],
                content=node_data["content"],
                summary=node_data.get("summary", ""),
                metadata=node_data.get("metadata", {}),
            )
        )

    for edge_data in graph_payload.get("edges", []):
        graph.add_edge(
            Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                weight=edge_data.get("weight", 0.5),
                kind=edge_data.get("kind", "sibling"),
                metadata=edge_data.get("metadata", {}),
            )
        )

    index = VectorIndex()
    index_payload = payload.get("index", {})
    if "index" in payload and not isinstance(index_payload, dict):
        raise SystemExit("index payload must be an object")
    if isinstance(index_payload, dict):
        for node_id, vector in index_payload.items():
            if not isinstance(vector, list):
                raise SystemExit("index payload vectors must be arrays")
            index.upsert(node_id, vector)
    meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    return graph, index, meta
