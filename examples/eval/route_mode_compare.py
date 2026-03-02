#!/usr/bin/env python3
"""Compare retrieval metrics between route_mode=off and route_mode=edge+sim.

This is a lightweight, dependency-free harness for before/after routing checks on a
saved OpenClawBrain state. It reports, per query:
- prompt_context chars
- fired_nodes count

Usage:
  python examples/eval/route_mode_compare.py --state /path/to/state.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openclawbrain.graph import Edge
from openclawbrain.hasher import HashEmbedder
from openclawbrain.index import VectorIndex
from openclawbrain.prompt_context import build_prompt_context_ranked_with_stats
from openclawbrain.store import load_state
from openclawbrain.traverse import TraversalConfig, traverse


DEFAULT_QUERIES = [
    "how do we deploy safely",
    "rollback plan for failed release",
    "incident response escalation",
    "can we skip CI for hotfix",
    "where are production runbooks",
]


def _load_queries(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_QUERIES)

    query_path = Path(path).expanduser()
    if not query_path.exists():
        raise SystemExit(f"queries file not found: {query_path}")

    if query_path.suffix.lower() == ".json":
        data = json.loads(query_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            queries = [item.strip() for item in data if isinstance(item, str) and item.strip()]
        elif isinstance(data, dict) and isinstance(data.get("queries"), list):
            queries = [item.strip() for item in data["queries"] if isinstance(item, str) and item.strip()]
        else:
            raise SystemExit("json queries file must be either a list of strings or {'queries': [...]}.")
        if not queries:
            raise SystemExit("queries file contained no valid queries")
        return queries

    queries = [line.strip() for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not queries:
        raise SystemExit("queries file contained no valid queries")
    return queries


def _score_edge(
    edge: Edge,
    *,
    query_vector: list[float],
    index: VectorIndex,
    alpha_sim: float,
    use_relevance: bool,
) -> float:
    weight = float(edge.weight)
    relevance = 0.0
    if use_relevance:
        metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
        raw_relevance = metadata.get("relevance", 0.0)
        if isinstance(raw_relevance, (int, float)):
            relevance = float(raw_relevance)

    target_vector = index._vectors.get(edge.target)
    similarity = VectorIndex.cosine(query_vector, target_vector) if target_vector is not None else 0.0
    return weight + relevance + (alpha_sim * similarity)


def _route_fn_edge_sim(
    *,
    query_vector: list[float],
    index: VectorIndex,
    route_top_k: int,
    route_alpha_sim: float,
    route_use_relevance: bool,
):
    def _route(_source_id: str | None, candidates: list[Edge], _query_text: str) -> list[str]:
        ranked = sorted(
            ((edge.target, _score_edge(
                edge,
                query_vector=query_vector,
                index=index,
                alpha_sim=route_alpha_sim,
                use_relevance=route_use_relevance,
            )) for edge in candidates),
            key=lambda item: (-item[1], item[0]),
        )
        return [target_id for target_id, _score in ranked[:route_top_k]]

    return _route


def _query_metrics(
    *,
    query: str,
    query_vector: list[float],
    graph: Any,
    index: VectorIndex,
    top_k: int,
    max_hops: int,
    max_fired_nodes: int,
    max_prompt_context_chars: int,
    route_fn,
) -> dict[str, int]:
    seeds = index.search(query_vector, top_k=top_k)
    result = traverse(
        graph=graph,
        seeds=seeds,
        config=TraversalConfig(max_hops=max_hops, max_fired_nodes=max_fired_nodes),
        query_text=query,
        route_fn=route_fn,
    )
    _prompt_context, stats = build_prompt_context_ranked_with_stats(
        graph=graph,
        node_ids=result.fired,
        node_scores=result.fired_scores,
        max_chars=max_prompt_context_chars,
        include_node_ids=True,
    )
    return {
        "prompt_context_len": int(stats.get("prompt_context_len", 0)),
        "fired_nodes": len(result.fired),
    }


def _fmt_query(query: str, width: int = 44) -> str:
    if len(query) <= width:
        return query
    return query[: width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare route_mode off vs edge+sim on a saved state.")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument("--queries-file", help="Text (one query per line) or JSON query list")
    parser.add_argument("--top-k", type=int, default=4, help="Seed count from vector index")
    parser.add_argument("--max-hops", type=int, default=15, help="Traversal max hops")
    parser.add_argument("--max-fired-nodes", type=int, default=30, help="Traversal max fired nodes")
    parser.add_argument("--max-prompt-context-chars", type=int, default=30000, help="Prompt context char cap")
    parser.add_argument("--route-top-k", type=int, default=5)
    parser.add_argument("--route-alpha-sim", type=float, default=0.5)
    parser.add_argument("--no-route-use-relevance", action="store_true", help="Disable edge metadata relevance")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be > 0")
    if args.max_fired_nodes <= 0:
        raise SystemExit("--max-fired-nodes must be > 0")
    if args.max_prompt_context_chars <= 0:
        raise SystemExit("--max-prompt-context-chars must be > 0")
    if args.route_top_k <= 0:
        raise SystemExit("--route-top-k must be > 0")

    queries = _load_queries(args.queries_file)
    state_path = Path(args.state).expanduser()
    graph, index, meta = load_state(str(state_path))

    embedder_name = str(meta.get("embedder_name", "unknown"))
    if embedder_name != "hash-v1":
        print(
            f"warning: state embedder is '{embedder_name}', using hash-v1 query embeddings for this offline eval."
        )

    print(f"state={state_path}")
    print(f"queries={len(queries)} top_k={args.top_k} max_hops={args.max_hops} max_fired_nodes={args.max_fired_nodes}")
    print(f"max_prompt_context_chars={args.max_prompt_context_chars} route_top_k={args.route_top_k} route_alpha_sim={args.route_alpha_sim}")

    embed = HashEmbedder().embed
    rows: list[dict[str, Any]] = []

    for query in queries:
        qvec = embed(query)

        off_metrics = _query_metrics(
            query=query,
            query_vector=qvec,
            graph=graph,
            index=index,
            top_k=args.top_k,
            max_hops=args.max_hops,
            max_fired_nodes=args.max_fired_nodes,
            max_prompt_context_chars=args.max_prompt_context_chars,
            route_fn=None,
        )

        edge_sim_metrics = _query_metrics(
            query=query,
            query_vector=qvec,
            graph=graph,
            index=index,
            top_k=args.top_k,
            max_hops=args.max_hops,
            max_fired_nodes=args.max_fired_nodes,
            max_prompt_context_chars=args.max_prompt_context_chars,
            route_fn=_route_fn_edge_sim(
                query_vector=qvec,
                index=index,
                route_top_k=args.route_top_k,
                route_alpha_sim=args.route_alpha_sim,
                route_use_relevance=not args.no_route_use_relevance,
            ),
        )

        rows.append(
            {
                "query": query,
                "off_chars": off_metrics["prompt_context_len"],
                "edge_chars": edge_sim_metrics["prompt_context_len"],
                "delta_chars": edge_sim_metrics["prompt_context_len"] - off_metrics["prompt_context_len"],
                "off_fired": off_metrics["fired_nodes"],
                "edge_fired": edge_sim_metrics["fired_nodes"],
                "delta_fired": edge_sim_metrics["fired_nodes"] - off_metrics["fired_nodes"],
            }
        )

    header = (
        f"{'query':44}  {'off_chars':>9}  {'edge+sim_chars':>14}  {'delta':>7}"
        f"  {'off_fired':>9}  {'edge+sim_fired':>14}  {'delta':>7}"
    )
    print("\n" + header)
    print("-" * len(header))

    total_off_chars = 0
    total_edge_chars = 0
    total_off_fired = 0
    total_edge_fired = 0

    for row in rows:
        total_off_chars += int(row["off_chars"])
        total_edge_chars += int(row["edge_chars"])
        total_off_fired += int(row["off_fired"])
        total_edge_fired += int(row["edge_fired"])
        print(
            f"{_fmt_query(str(row['query'])):44}  {int(row['off_chars']):9d}  {int(row['edge_chars']):14d}"
            f"  {int(row['delta_chars']):7d}  {int(row['off_fired']):9d}  {int(row['edge_fired']):14d}"
            f"  {int(row['delta_fired']):7d}"
        )

    count = max(len(rows), 1)
    avg_off_chars = total_off_chars / count
    avg_edge_chars = total_edge_chars / count
    avg_off_fired = total_off_fired / count
    avg_edge_fired = total_edge_fired / count

    print("-" * len(header))
    print(
        f"{'AVERAGE':44}  {avg_off_chars:9.1f}  {avg_edge_chars:14.1f}  {(avg_edge_chars - avg_off_chars):7.1f}"
        f"  {avg_off_fired:9.2f}  {avg_edge_fired:14.2f}  {(avg_edge_fired - avg_off_fired):7.2f}"
    )


if __name__ == "__main__":
    main()
