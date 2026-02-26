"""Thin, stdlib-only CLI wrapper for CrabPath workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import Edge, Graph, Node
from .index import VectorIndex
from .split import split_workspace
from .traverse import TraversalConfig, traverse
from .learn import apply_outcome
from .autotune import measure_health


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crabpath")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="split workspace and output node text payload")
    init.add_argument("--workspace", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--json", action="store_true")

    query = sub.add_parser("query", help="seed from index and traverse graph")
    query.add_argument("text")
    query.add_argument("--graph", required=True)
    query.add_argument("--index", required=True)
    query.add_argument("--top", type=int, default=10)
    query.add_argument("--json", action="store_true")
    query.add_argument("--query-vector", nargs="+", required=False)

    learn = sub.add_parser("learn", help="apply outcome update")
    learn.add_argument("--graph", required=True)
    learn.add_argument("--outcome", type=float, required=True)
    learn.add_argument("--fired-ids", required=True)
    learn.add_argument("--json", action="store_true")

    health = sub.add_parser("health", help="compute graph health")
    health.add_argument("--graph", required=True)
    health.add_argument("--json", action="store_true")

    return parser


def _load_payload(path: str) -> dict:
    if not Path(path).exists():
        raise SystemExit(f"missing graph file: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_graph(path: str) -> Graph:
    payload = _load_payload(path)
    graph_payload = payload["graph"] if "graph" in payload else payload
    graph = Graph()
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
    return graph


def _parse_vector(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None
    vector: list[float] = []
    for value in values:
        for chunk in value.split(","):
            if chunk:
                vector.append(float(chunk))
    return vector


def cmd_init(args: argparse.Namespace) -> int:
    graph, texts = split_workspace(args.workspace)
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
        "node_texts": texts,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"wrote {args.output}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    query_vec = _parse_vector(args.query_vector)
    if query_vec is None:
        raise SystemExit("query requires --query-vector in CLI mode")

    if not Path(args.index).exists():
        raise SystemExit(f"missing index file: {args.index}")
    index_payload = json.loads(Path(args.index).read_text(encoding="utf-8"))
    index = VectorIndex()
    for node_id, vector in index_payload.items():
        index.upsert(node_id, vector)

    seeds = index.search(query_vec, top_k=args.top)
    result = traverse(graph=graph, seeds=seeds, config=TraversalConfig(max_hops=15), route_fn=None)

    if args.json:
        print(
            json.dumps(
                {
                    "fired": result.fired,
                    "steps": [step.__dict__ for step in result.steps],
                    "context": result.context,
                }
            )
        )
    else:
        print(result.context)
        print()
        print("\"fired\":", result.fired)
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    fired_ids = [value.strip() for value in args.fired_ids.split(",") if value.strip()]
    apply_outcome(graph, fired_nodes=fired_ids, outcome=args.outcome)
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
    }
    Path(args.graph).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload["graph"], indent=2))
    else:
        print(f"updated {args.graph}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    health = measure_health(graph)
    payload = health.__dict__
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return cmd_init(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "learn":
        return cmd_learn(args)
    if args.command == "health":
        return cmd_health(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
