"""Pure graph-operations CLI for CrabPath."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from .connect import apply_connections, suggest_connections
from .graph import Edge, Graph, Node
from .index import VectorIndex
from .journal import (
    log_health,
    log_learn,
    log_query,
    log_replay,
    journal_stats,
    read_journal,
)
from .learn import apply_outcome
from .merge import apply_merge, suggest_merges
from .replay import extract_queries, extract_queries_from_dir, replay_queries
from .split import split_workspace
from .traverse import TraversalConfig, TraversalResult, traverse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crabpath")
    sub = parser.add_subparsers(dest="command", required=True)

    i = sub.add_parser("init")
    i.add_argument("--workspace", required=True)
    i.add_argument("--output", required=True)
    i.add_argument("--sessions")
    i.add_argument("--json", action="store_true")

    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--graph", required=True)
    q.add_argument("--index")
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--query-vector-stdin", action="store_true")
    q.add_argument("--json", action="store_true")

    l = sub.add_parser("learn")
    l.add_argument("--graph", required=True)
    l.add_argument("--outcome", type=float, required=True)
    l.add_argument("--fired-ids", required=True)
    l.add_argument("--json", action="store_true")

    m = sub.add_parser("merge")
    m.add_argument("--graph", required=True)
    m.add_argument("--json", action="store_true")

    c = sub.add_parser("connect")
    c.add_argument("--graph", required=True)
    c.add_argument("--json", action="store_true")

    r = sub.add_parser("replay")
    r.add_argument("--graph", required=True)
    r.add_argument("--sessions", nargs="+", required=True)
    r.add_argument("--json", action="store_true")

    h = sub.add_parser("health")
    h.add_argument("--graph", required=True)
    h.add_argument("--json", action="store_true")

    j = sub.add_parser("journal")
    j.add_argument("--last", type=int, default=10)
    j.add_argument("--stats", action="store_true")
    j.add_argument("--json", action="store_true")
    return parser


def _load_payload(path: str) -> dict:
    payload_path = Path(os.path.expanduser(path))
    if payload_path.is_dir():
        payload_path = payload_path / "graph.json"
    if not payload_path.exists():
        raise SystemExit(f"missing graph file: {path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _load_graph(path: str) -> Graph:
    payload = _load_payload(path)
    payload = payload["graph"] if "graph" in payload else payload
    graph = Graph()
    for node_data in payload.get("nodes", []):
        graph.add_node(
            Node(node_data["id"], node_data["content"], node_data.get("summary", ""), node_data.get("metadata", {}))
        )
    for edge_data in payload.get("edges", []):
        graph.add_edge(
            Edge(
                edge_data["source"],
                edge_data["target"],
                edge_data.get("weight", 0.5),
                edge_data.get("kind", "sibling"),
                edge_data.get("metadata", {}),
            )
        )
    return graph


def _graph_payload(graph: Graph) -> dict:
    return {
        "nodes": [
            {
                "id": n.id,
                "content": n.content,
                "summary": n.summary,
                "metadata": n.metadata,
            }
            for n in graph.nodes()
        ],
        "edges": [
            {"source": e.source, "target": e.target, "weight": e.weight, "kind": e.kind, "metadata": e.metadata}
            for source in graph._edges.values()
            for e in source.values()
        ],
    }


def _write_graph(path: str | Path, graph: Graph) -> None:
    destination = Path(path).expanduser()
    if destination.is_dir():
        destination = destination / "graph.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_graph_payload(graph), indent=2), encoding="utf-8")


def _load_query_vector_from_stdin() -> list[float]:
    data = sys.stdin.read().strip()
    if not data:
        raise SystemExit("query vector JSON required on stdin")
    payload = json.loads(data)
    if not isinstance(payload, list):
        raise SystemExit("query vector stdin payload must be a JSON array")
    return [float(v) for v in payload]


def _load_session_queries(session_paths: str | Iterable[str]) -> list[str]:
    if isinstance(session_paths, str):
        session_paths = [session_paths]
    queries: list[str] = []
    for session_path in session_paths:
        path = Path(session_path).expanduser()
        if path.is_dir():
            queries.extend(extract_queries_from_dir(path))
        elif path.is_file():
            queries.extend(extract_queries(path))
        else:
            raise SystemExit(f"invalid sessions path: {path}")
    return queries


def _keyword_seeds(graph: Graph, text: str, top_k: int) -> list[tuple[str, float]]:
    query_tokens = {
        token for token in re.split(r"\W+", text.lower()) if token
    }
    if not query_tokens:
        return []
    scores = [(node.id, len(query_tokens & {token for token in re.split(r"\W+", node.content.lower()) if token}) / len(query_tokens)) for node in graph.nodes()]
    scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scores[:top_k]


def _load_index(path: str) -> VectorIndex:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("index payload must be a JSON object")
    index = VectorIndex()
    for node_id, vector in payload.items():
        if not isinstance(vector, list):
            raise SystemExit("index payload vectors must be arrays")
        index.upsert(str(node_id), [float(v) for v in vector])
    return index


def _result_payload(result: TraversalResult) -> dict:
    return {
        "fired": result.fired,
        "steps": [step.__dict__ for step in result.steps],
        "context": result.context,
    }


def cmd_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).expanduser()
    if output_dir.suffix == ".json" and not output_dir.is_dir():
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, texts = split_workspace(args.workspace, llm_fn=None, llm_batch_fn=None)
    if args.sessions is not None:
        replay_queries(graph=graph, queries=_load_session_queries(args.sessions))

    graph_path = output_dir / "graph.json"
    text_path = output_dir / "texts.json"
    _write_graph(graph_path, graph)
    text_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({"graph": str(graph_path), "texts": str(text_path)}))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    if args.top <= 0:
        raise SystemExit("--top must be >= 1")

    if args.query_vector_stdin:
        if args.index is None:
            raise SystemExit("query-vector-stdin requires --index")
        query_vec = _load_query_vector_from_stdin()
        index = _load_index(args.index)
        seeds = index.search(query_vec, top_k=args.top)
    else:
        seeds = _keyword_seeds(graph, args.text, args.top)

    result = traverse(graph=graph, seeds=seeds, config=TraversalConfig(max_hops=15), query_text=args.text)
    log_query(query_text=args.text, fired_ids=result.fired, node_count=graph.node_count())
    print(json.dumps(_result_payload(result)) if args.json else result.context)
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    fired_ids = [value.strip() for value in args.fired_ids.split(",") if value.strip()]
    if not fired_ids:
        raise SystemExit("provide at least one fired id")

    apply_outcome(graph, fired_nodes=fired_ids, outcome=args.outcome)
    payload = {"graph": _graph_payload(graph)}
    Path(args.graph).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_learn(fired_ids=fired_ids, outcome=args.outcome)
    print(json.dumps(payload, indent=2) if args.json else f"updated {args.graph}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    suggestions = suggest_merges(graph)
    applied = []
    for source_id, target_id in suggestions:
        if graph.get_node(source_id) and graph.get_node(target_id):
            merged = apply_merge(graph, source_id, target_id)
            applied.append({"from": [source_id, target_id], "to": [merged]})
    _write_graph(args.graph, graph)
    payload = {"suggestions": [{"from": [s, t]} for s, t in suggestions], "applied": applied}
    print(json.dumps(payload) if args.json else f"Applied merges: {len(applied)}")
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    suggestions = suggest_connections(graph)
    added = apply_connections(graph=graph, connections=suggestions)
    _write_graph(args.graph, graph)
    payload = {
        "suggestions": [
            {"source_id": s, "target_id": t, "weight": w, "reason": r} for s, t, w, r in suggestions
        ],
        "added": added,
    }
    print(json.dumps(payload) if args.json else f"Added edges: {added}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    queries = _load_session_queries(args.sessions)
    stats = replay_queries(graph=graph, queries=queries, verbose=not args.json)
    log_replay(
        queries_replayed=stats["queries_replayed"],
        edges_reinforced=stats["edges_reinforced"],
        cross_file_created=stats["cross_file_edges_created"],
    )
    _write_graph(args.graph, graph)
    print(json.dumps(stats, indent=2) if args.json else f"Replayed {stats['queries_replayed']}/{len(queries)} queries, {stats['cross_file_edges_created']} cross-file edges created")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    from .autotune import measure_health

    payload = measure_health(graph).__dict__
    payload["nodes"] = graph.node_count()
    payload["edges"] = graph.edge_count()
    log_health(payload)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(
        "\n".join(
            [
                f"nodes: {payload['nodes']}",
                f"edges: {payload['edges']}",
                f"dormant_pct: {payload['dormant_pct']:.2f}",
                f"habitual_pct: {payload['habitual_pct']:.2f}",
                f"reflex_pct: {payload['reflex_pct']:.2f}",
                f"cross_file_edge_pct: {payload['cross_file_edge_pct']:.2f}",
                f"orphan_nodes: {payload['orphan_nodes']}",
            ]
        )
    )
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    if args.stats:
        print(json.dumps(journal_stats(), indent=2) if args.json else "\n".join(f"{k}: {v}" for k, v in journal_stats().items() if k != "avg_fired_per_query"))
        return 0
    entries = read_journal(last_n=args.last)
    print(
        json.dumps(entries, indent=2)
        if args.json
        else "\n".join(f"{idx+1:>2}. {entry.get('type')} @ {entry.get('iso', entry.get('ts', ''))}: {entry}" for idx, entry in enumerate(entries))
        or "No entries."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return {
        "init": cmd_init,
        "query": cmd_query,
        "learn": cmd_learn,
        "merge": cmd_merge,
        "connect": cmd_connect,
        "replay": cmd_replay,
        "health": cmd_health,
        "journal": cmd_journal,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
