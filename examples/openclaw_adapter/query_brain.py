#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from crabpath import load_state, traverse


EMBED_MODEL = "text-embedding-3-small"


def require_api_key() -> str:
    # The caller is responsible for injecting OPENAI_API_KEY into this process env.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is required in process env. "
            "Inject it in the agent process before running this script."
        )
    return api_key


def embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a CrabPath state.json with OpenAI embeddings")
    parser.add_argument("state", help="Path to state.json")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--top", type=int, default=4, help="Top-k vector matches")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    query_text = " ".join(args.query).strip()
    state_path = Path(args.state)
    if not state_path.exists():
        raise SystemExit(f"state file not found: {state_path}")

    api_key = require_api_key()
    client = OpenAI(api_key=api_key)

    graph, index, meta = load_state(str(state_path))
    query_vector = embed_query(client, query_text)

    expected_dim = meta.get("embedder_dim")
    if isinstance(expected_dim, int) and len(query_vector) != expected_dim:
        raise SystemExit(
            f"Embedding dimension mismatch: query={len(query_vector)} index={expected_dim}. "
            "Rebuild state.json with the same embedder model."
        )

    seeds = index.search(query_vector, top_k=args.top)
    result = traverse(graph=graph, seeds=seeds, query_text=query_text)

    output = {
        "state": str(state_path),
        "query": query_text,
        "seeds": seeds,
        "fired": result.fired,
        "context": result.context,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(f"state: {state_path}")
    print(f"query: {query_text}")
    print("fired:")
    for node_id in result.fired:
        print(f" - {node_id}")
    print("context:")
    print(result.context or "(no context)")


if __name__ == "__main__":
    main()
