#!/usr/bin/env python3
"""Lightweight prompt-context trim evaluation harness.

Runs retrieval via traverse (not daemon) and reports prompt_context trim behavior at
multiple char caps, including dropped authority distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openclawbrain.hasher import HashEmbedder
from openclawbrain.prompt_context import build_prompt_context_ranked_with_stats
from openclawbrain.store import load_state
from openclawbrain.traverse import TraversalConfig, traverse


DEFAULT_QUERIES = [
    "how do we deploy",
    "incident response runbook",
    "ci checks before merge",
    "rollback process",
    "on-call escalation",
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


def _parse_caps(raw_caps: str) -> list[int]:
    caps: list[int] = []
    for piece in raw_caps.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError
        caps.append(value)
    if not caps:
        raise ValueError
    return caps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompt_context trim behavior at fixed caps.")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument("--queries-file", help="Text (one query per line) or JSON query list")
    parser.add_argument("--top-k", type=int, default=6, help="Seed count from vector index")
    parser.add_argument("--max-hops", type=int, default=15)
    parser.add_argument("--max-fired-nodes", type=int, default=30)
    parser.add_argument("--caps", default="20000,30000", help="Comma-separated prompt_context caps")
    args = parser.parse_args()

    try:
        caps = _parse_caps(args.caps)
    except ValueError:
        raise SystemExit("--caps must be a comma-separated list of positive integers") from None

    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be > 0")
    if args.max_fired_nodes <= 0:
        raise SystemExit("--max-fired-nodes must be > 0")

    queries = _load_queries(args.queries_file)

    graph, index, meta = load_state(str(Path(args.state).expanduser()))
    embedder_name = str(meta.get("embedder_name", "unknown"))
    if embedder_name != "hash-v1":
        print(
            f"warning: state embedder is '{embedder_name}', using hash-v1 for eval queries (seed quality may differ)."
        )

    embed = HashEmbedder().embed

    aggregates: dict[int, dict[str, object]] = {
        cap: {
            "trimmed": 0,
            "total": 0,
            "dropped_count": 0,
            "dropped_authority": {},
        }
        for cap in caps
    }

    for query in queries:
        seeds = index.search(embed(query), top_k=args.top_k)
        result = traverse(
            graph=graph,
            seeds=seeds,
            config=TraversalConfig(max_hops=args.max_hops, max_fired_nodes=args.max_fired_nodes),
            query_text=query,
        )

        for cap in caps:
            _rendered, stats = build_prompt_context_ranked_with_stats(
                graph=graph,
                node_ids=result.fired,
                node_scores=result.fired_scores,
                max_chars=cap,
                include_node_ids=True,
            )
            entry = aggregates[cap]
            entry["total"] = int(entry["total"]) + 1
            entry["dropped_count"] = int(entry["dropped_count"]) + int(stats["prompt_context_dropped_count"])
            if stats["prompt_context_trimmed"]:
                entry["trimmed"] = int(entry["trimmed"]) + 1
            dropped_auth = entry["dropped_authority"]
            if isinstance(dropped_auth, dict):
                for authority, count in stats.get("prompt_context_dropped_authority_counts", {}).items():
                    dropped_auth[authority] = int(dropped_auth.get(authority, 0)) + int(count)

    print(f"state={Path(args.state).expanduser()}")
    print(f"queries={len(queries)} top_k={args.top_k} max_hops={args.max_hops} max_fired_nodes={args.max_fired_nodes}")
    for cap in sorted(caps):
        entry = aggregates[cap]
        total = int(entry["total"]) or 1
        trimmed = int(entry["trimmed"])
        trim_rate = (trimmed / total) * 100.0
        avg_dropped = float(int(entry["dropped_count"]) / total)
        dropped_authority = dict(sorted(dict(entry["dropped_authority"]).items()))
        print(f"cap={cap} trim_rate={trim_rate:.1f}% ({trimmed}/{total}) avg_dropped={avg_dropped:.2f}")
        print(f"cap={cap} dropped_authority={json.dumps(dropped_authority, sort_keys=True)}")


if __name__ == "__main__":
    main()
