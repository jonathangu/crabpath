#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from crabpath import VectorIndex, save_state, split_workspace
from crabpath._batch import batch_or_single_embed
from crabpath.autotune import measure_health
from crabpath.replay import extract_queries, extract_queries_from_dir, replay_queries


EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_DIM = 1536
FALLBACK_EMBED_DIM = 1536


def require_api_key() -> str:
    # The caller is responsible for injecting OPENAI_API_KEY into this process env.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is required in process env. "
            "Inject it in the agent process before running this script."
        )
    return api_key


def build_embed_batch_fn(client: OpenAI):
    def embed_batch(texts: list[tuple[str, str]]) -> dict[str, list[float]]:
        if not texts:
            return {}
        _, contents = zip(*texts)
        response = client.embeddings.create(model=EMBED_MODEL, input=list(contents))
        return {
            texts[idx][0]: response.data[idx].embedding
            for idx in range(len(response.data))
        }

    return embed_batch


def build_llm_fn(client: OpenAI):
    def llm_fn(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    return llm_fn


def load_session_queries(sessions_path: Path) -> list[str]:
    if not sessions_path.exists():
        raise SystemExit(f"sessions path not found: {sessions_path}")

    if sessions_path.is_dir():
        return extract_queries_from_dir(sessions_path)
    return extract_queries(sessions_path)


def resolve_state_path(output_path: Path) -> Path:
    if output_path.exists() and output_path.is_file() and output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / "state.json"


def run(workspace: Path, sessions: Path, output: Path) -> None:
    if not workspace.exists():
        raise SystemExit(f"workspace not found: {workspace}")

    api_key = require_api_key()
    client = OpenAI(api_key=api_key)
    embed_batch = build_embed_batch_fn(client)
    llm_fn = build_llm_fn(client)

    graph, texts = split_workspace(str(workspace), llm_fn=llm_fn, llm_batch_fn=None)
    session_queries = load_session_queries(sessions)
    replay_stats = replay_queries(graph=graph, queries=session_queries, verbose=False)

    embeddings = batch_or_single_embed(list(texts.items()), embed_batch_fn=embed_batch)
    index = VectorIndex()
    for node_id, vector in embeddings.items():
        index.upsert(node_id, vector)

    state_path = resolve_state_path(output)
    embedder_dim = len(next(iter(embeddings.values()))) if embeddings else DEFAULT_EMBEDDING_DIM
    save_state(
        graph=graph,
        index=index,
        path=str(state_path),
        embedder_name=EMBED_MODEL,
        embedder_dim=embedder_dim,
    )

    health = measure_health(graph)
    summary = {
        "state": str(state_path),
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "embeddings": len(embeddings),
        "embedder_name": EMBED_MODEL,
        "embedder_dim": embedder_dim,
        "replayed_sessions": len(session_queries),
        "replay_queries_replayed": replay_stats["queries_replayed"],
        "replay_edges_reinforced": replay_stats["edges_reinforced"],
        "replay_cross_file_edges_created": replay_stats["cross_file_edges_created"],
        "health": {
            "nodes": graph.node_count(),
            "edges": graph.edge_count(),
            "dormant_pct": health.dormant_pct,
            "habitual_pct": health.habitual_pct,
            "reflex_pct": health.reflex_pct,
            "cross_file_edge_pct": health.cross_file_edge_pct,
            "orphan_nodes": health.orphan_nodes,
        },
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a CrabPath state.json for OpenClaw adapters")
    parser.add_argument("workspace", help="Path to agent workspace markdown directory")
    parser.add_argument("sessions", help="Path to OpenClaw sessions file or directory")
    parser.add_argument("output", help="Output directory (or state.json path)")
    args = parser.parse_args()

    run(
        workspace=Path(args.workspace),
        sessions=Path(args.sessions),
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
