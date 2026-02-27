#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI
from crabpath import VectorIndex, save_state, split_workspace
from crabpath._batch import batch_or_single_embed
from crabpath.autotune import measure_health
from crabpath.replay import extract_queries, extract_queries_from_dir, replay_queries


EMBED_MODEL = "openai-text-embedding-3-small"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 100


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "This script must run inside the agent framework exec environment where OPENAI_API_KEY is injected."
        )
    return api_key


def build_embed_batch_fn(client: OpenAI):
    def embed_batch(texts: list[tuple[str, str]]) -> dict[str, list[float]]:
        if not texts:
            return {}

        results = {}
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            chunk = texts[start : start + EMBED_BATCH_SIZE]
            _, contents = zip(*chunk)
            response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=list(contents))
            for index, embedding in enumerate(response.data):
                results[chunk[index][0]] = embedding.embedding

        return results

    return embed_batch


def build_llm_fn(client: OpenAI):
    def llm_fn(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user[:4000]},
            ],
            max_completion_tokens=500,
        )
        return resp.choices[0].message.content
    return llm_fn

def load_session_queries(sessions_path: Path) -> list[str]:
    if not sessions_path.exists():
        raise SystemExit(f"sessions path not found: {sessions_path}")

    if sessions_path.is_dir():
        return extract_queries_from_dir(sessions_path)
    return extract_queries(sessions_path)


def resolve_output_paths(output_path: Path) -> tuple[Path, Path, Path]:
    if output_path.exists() and output_path.is_file() and output_path.suffix.lower() == ".json":
        state_path = output_path
        output_dir = output_path.parent
    else:
        output_dir = output_path
        state_path = output_dir / "state.json"

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, state_path, output_dir / "graph.json", output_dir / "index.json"


def write_graph(path: Path, graph, *, embedder_name: str, embedder_dim: int) -> None:
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
        "meta": {
            "embedder_name": embedder_name,
            "embedder_dim": embedder_dim,
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node_count": graph.node_count(),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(workspace: Path, sessions: Path, output: Path) -> None:
    if not workspace.exists():
        raise SystemExit(f"workspace not found: {workspace}")

    api_key = require_api_key()
    client = OpenAI(api_key=api_key)
    embed_batch = build_embed_batch_fn(client)

    graph, texts = split_workspace(str(workspace), llm_fn=None, llm_batch_fn=None)
    session_queries = load_session_queries(sessions)
    replay_stats = replay_queries(graph=graph, queries=session_queries, verbose=False)

    embeddings = batch_or_single_embed(list(texts.items()), embed_batch_fn=embed_batch)
    index = VectorIndex()
    for node_id, vector in embeddings.items():
        index.upsert(node_id, vector)

    output_dir, state_path, graph_path, index_path = resolve_output_paths(output)
    embedder_dim = EMBEDDING_DIM
    save_state(
        graph=graph,
        index=index,
        path=str(state_path),
        embedder_name=EMBED_MODEL,
        embedder_dim=embedder_dim,
    )
    write_graph(
        graph_path,
        graph,
        embedder_name=EMBED_MODEL,
        embedder_dim=embedder_dim,
    )
    index_path.write_text(json.dumps(index._vectors, indent=2), encoding="utf-8")

    health = measure_health(graph)
    health_payload = {
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "embedder_name": EMBED_MODEL,
        "embedder_dim": embedder_dim,
        "state_path": str(state_path),
        "graph_path": str(graph_path),
        "index_path": str(index_path),
        "embeddings": len(embeddings),
        "workspace_nodes": graph.node_count(),
        "replayed_sessions": len(session_queries),
        "replayed_queries": replay_stats["queries_replayed"],
        "reinforced_edges": replay_stats["edges_reinforced"],
        "cross_file_edges_created": replay_stats["cross_file_edges_created"],
    }
    health_payload["health"] = {
        "dormant_pct": health.dormant_pct,
        "habitual_pct": health.habitual_pct,
        "reflex_pct": health.reflex_pct,
        "cross_file_edge_pct": health.cross_file_edge_pct,
        "orphan_nodes": health.orphan_nodes,
    }
    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "state": str(state_path),
        "graph": str(graph_path),
        "index": str(index_path),
    }
    summary.update(health_payload)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a CrabPath state.json for OpenClaw adapters")
    parser.add_argument("workspace_path", help="Path to agent workspace markdown directory")
    parser.add_argument("sessions_path", help="Path to OpenClaw sessions file or directory")
    parser.add_argument("output_path", help="Output directory (or state.json path)")
    args = parser.parse_args()

    run(
        workspace=Path(args.workspace_path),
        sessions=Path(args.sessions_path),
        output=Path(args.output_path),
    )


if __name__ == "__main__":
    main()
