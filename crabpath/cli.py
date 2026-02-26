"""Thin, stdlib-only CLI wrapper for CrabPath workflows."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from .graph import Edge, Graph, Node
from .index import VectorIndex
from .replay import extract_queries, extract_queries_from_dir, replay_queries
from .split import split_workspace
from .traverse import TraversalConfig, traverse
from .learn import apply_outcome
from .autotune import measure_health
from .journal import log_health, log_learn, log_query, log_replay, read_journal, journal_stats

_EMBED_BATCH_SIZE = 50


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crabpath")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="split workspace and output node text payload")
    init.add_argument("--workspace", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--sessions", required=False)
    embed_init_group = init.add_mutually_exclusive_group(required=False)
    embed_init_group.add_argument("--embed-command")
    embed_init_group.add_argument("--embed-provider", choices=("openai", "ollama"))
    init.add_argument("--json", action="store_true")
    init.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    embed = sub.add_parser("embed", help="build index.json from texts.json using an embedding command")
    embed.add_argument("--texts", required=True)
    embed.add_argument("--output", required=True)
    embed_provider_group = embed.add_mutually_exclusive_group(required=True)
    embed_provider_group.add_argument("--command", dest="embed_command")
    embed_provider_group.add_argument("--provider", choices=("openai", "ollama"))
    embed.add_argument("--json", action="store_true")

    query = sub.add_parser("query", help="seed from index and traverse graph")
    query.add_argument("text")
    query.add_argument("--graph", required=True)
    query.add_argument("--index", required=False)
    query.add_argument("--top", type=int, default=10)
    query.add_argument("--json", action="store_true")
    query.add_argument("--query-vector", nargs="+", required=False)
    query.add_argument("--query-vector-stdin", action="store_true")
    query.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    learn = sub.add_parser("learn", help="apply outcome update")
    learn.add_argument("--graph", required=True)
    learn.add_argument("--outcome", type=float, required=True)
    learn.add_argument("--fired-ids", required=True)
    learn.add_argument("--json", action="store_true")
    learn.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    replay = sub.add_parser("replay", help="warm up graph from historical sessions")
    replay.add_argument("--graph", required=True)
    replay.add_argument("--sessions", nargs="+", required=True)
    replay.add_argument("--max-queries", type=int, default=None)
    replay.add_argument("--json", action="store_true")
    replay.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    health = sub.add_parser("health", help="compute graph health")
    health.add_argument("--graph", required=True)
    health.add_argument("--json", action="store_true")
    health.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    journal = sub.add_parser("journal", help="read recent journal entries or summary stats")
    journal.add_argument("--last", type=int, default=10)
    journal.add_argument("--stats", action="store_true")
    journal.add_argument("--json", action="store_true")
    journal.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    return parser


def _load_payload(path: str) -> dict:
    payload_path = Path(path)
    if payload_path.is_dir():
        payload_path = payload_path / "graph.json"
    if not payload_path.exists():
        raise SystemExit(f"missing graph file: {path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


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


def _load_session_queries(session_paths: list[str] | str) -> list[str]:
    if isinstance(session_paths, str):
        session_paths = [session_paths]

    queries: list[str] = []
    for session_path in session_paths:
        path = Path(session_path).expanduser()
        if not path.exists():
            raise SystemExit(f"missing sessions path: {path}")
        if path.is_dir():
            queries.extend(extract_queries_from_dir(path))
        elif path.is_file():
            queries.extend(extract_queries(path))
        else:
            raise SystemExit(f"invalid sessions path: {path}")
    return queries


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_text(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _load_query_vector_from_stdin() -> list[float]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("query vector JSON required on stdin")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit("query vector stdin payload must be a JSON array")
    vector: list[float] = []
    for value in data:
        vector.append(float(value))
    return vector


def _load_texts_payload(path: str) -> dict[str, str]:
    payload_path = Path(path).expanduser()
    if not payload_path.exists():
        raise SystemExit(f"missing texts file: {path}")
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("texts payload must be a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def _iter_text_batches(
    texts: list[tuple[str, str]],
    batch_size: int = _EMBED_BATCH_SIZE,
) -> list[list[tuple[str, str]]]:
    if batch_size <= 0:
        raise SystemExit("embed batch size must be positive")

    batches: list[list[tuple[str, str]]] = []
    for idx in range(0, len(texts), batch_size):
        batches.append(texts[idx : idx + batch_size])
    return batches


def _provider_script(provider: str) -> str:
    if provider == "openai":
        return """import json
import sys

from openai import OpenAI


try:
    client = OpenAI()
    for line in sys.stdin:
        obj = json.loads(line)
        resp = client.embeddings.create(model='text-embedding-3-small', input=[obj['text']])
        print(json.dumps({'id': obj['id'], 'embedding': resp.data[0].embedding}))
except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)
"""
    if provider == "ollama":
        return """import json
import sys

import requests


for line in sys.stdin:
    obj = json.loads(line)
    resp = requests.post('http://localhost:11434/api/embeddings', json={'model': 'nomic-embed-text', 'prompt': obj['text']})
    print(json.dumps({'id': obj['id'], 'embedding': resp.json()['embedding']}))
"""
    raise SystemExit(f"unsupported provider: {provider}")


def _build_embed_command(
    embed_command: str | None,
    embed_provider: str | None,
) -> tuple[list[str], str | None]:
    if embed_command is None and embed_provider is None:
        raise SystemExit("provide either --command or --provider for embedding")
    if embed_command is not None and embed_provider is not None:
        raise SystemExit("provide only one of --command or --provider")
    if embed_command is not None:
        return shlex.split(embed_command), None

    script_code = _provider_script(embed_provider)
    fd, script_path = tempfile.mkstemp(prefix="crabpath_embed_", suffix=".py")
    os.close(fd)
    Path(script_path).write_text(script_code, encoding="utf-8")
    return [sys.executable, script_path], script_path


def _run_embedding_batch(
    command: list[str],
    batch: list[tuple[str, str]],
) -> dict[str, list[float]]:
    payload = "\n".join(json.dumps({"id": node_id, "text": text}) for node_id, text in batch) + "\n"
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_data, stderr_data = proc.communicate(payload)
    if proc.returncode != 0:
        message = (stderr_data or "").strip() or f"exit code {proc.returncode}"
        raise SystemExit(f"embed command failed: {message}")

    if not stdout_data.strip():
        raise SystemExit("embed command returned no embeddings")

    expected_ids = [node_id for node_id, _ in batch]
    results: dict[str, list[float]] = {}
    for line in stdout_data.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise SystemExit(f"embedding service error: {data['error']}")
        if "id" not in data or "embedding" not in data:
            raise SystemExit("embedding output must contain id and embedding fields")
        vector = [float(value) for value in data["embedding"]]
        results[str(data["id"])] = vector

    missing = [node_id for node_id in expected_ids if node_id not in results]
    if missing:
        raise SystemExit(f"embed output missing ids: {', '.join(missing)}")
    return results


def _build_index_from_texts(
    texts: dict[str, str],
    embed_command: str | None,
    embed_provider: str | None,
) -> VectorIndex:
    text_items = list(texts.items())
    index = VectorIndex()
    if not text_items:
        return index

    command, temp_script = _build_embed_command(embed_command=embed_command, embed_provider=embed_provider)
    batches = _iter_text_batches(texts=text_items)
    try:
        total_batches = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            print(
                f"Embedding batch {batch_index}/{total_batches} ({min(batch_index * _EMBED_BATCH_SIZE, len(texts))}/{len(texts)})",
                file=sys.stderr,
            )
            results = _run_embedding_batch(command=command, batch=batch)
            for node_id, vector in results.items():
                index.upsert(node_id, vector)
    finally:
        if temp_script is not None and Path(temp_script).exists():
            Path(temp_script).unlink()

    return index


def _keyword_seeds(graph: Graph, text: str, top_k: int) -> list[tuple[str, float]]:
    query_tokens = _tokenize_text(text)
    if not query_tokens or top_k <= 0:
        return []

    scores: list[tuple[str, float]] = []
    for node in graph.nodes():
        node_tokens = _tokenize_text(node.content)
        overlap = len(query_tokens & node_tokens)
        scores.append((node.id, overlap / len(query_tokens)))

    if not scores:
        return []
    scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scores[:top_k]


def cmd_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).expanduser()
    if output_dir.suffix == ".json" and not output_dir.is_dir():
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, texts = split_workspace(args.workspace)
    if args.sessions is not None:
        queries = _load_session_queries(args.sessions)
        replay_queries(graph=graph, queries=queries)

    graph_path = output_dir / "graph.json"
    texts_path = output_dir / "texts.json"
    payload = {
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
    }
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    texts_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")
    if args.embed_command is not None or args.embed_provider is not None:
        index = _build_index_from_texts(
            texts=texts,
            embed_command=args.embed_command,
            embed_provider=args.embed_provider,
        )
        index_path = output_dir / "index.json"
        index.save(str(index_path))

    if args.json:
        payload = {"graph": str(graph_path), "texts": str(texts_path)}
        if args.embed_command is not None or args.embed_provider is not None:
            payload["index"] = str(output_dir / "index.json")
        print(json.dumps(payload))
    else:
        print(f"graph_path: {graph_path}")
        print(f"texts_path: {texts_path}")
        if args.embed_command is not None or args.embed_provider is not None:
            print(f"index_path: {output_dir / 'index.json'}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    texts = _load_texts_payload(args.texts)
    index = _build_index_from_texts(
        texts=texts,
        embed_command=args.embed_command,
        embed_provider=args.provider,
    )
    output_path = Path(args.output).expanduser()
    if output_path.suffix != ".json" and output_path.exists() is False:
        output_path = output_path / "index.json"
    if output_path.is_dir() or output_path.suffix != ".json":
        output_path = output_path / "index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output_path))
    if args.json:
        print(json.dumps({"index": str(output_path), "count": len(texts)}))
    else:
        print(f"index_path: {output_path}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    query_vec = _parse_vector(args.query_vector)
    using_stdin_vector = bool(args.query_vector_stdin)

    if args.query_vector_stdin and args.query_vector:
        raise SystemExit("use only one of --query-vector or --query-vector-stdin")

    if query_vec is not None:
        index_path = args.index
    elif using_stdin_vector:
        if not args.index:
            raise SystemExit("query-vector-stdin requires --index")
        query_vec = _load_query_vector_from_stdin()
        index_path = args.index
    else:
        query_vec = None
        index_path = args.index

    if query_vec is not None:
        if not index_path:
            raise SystemExit("query vector mode requires --index")
        if not Path(index_path).exists():
            raise SystemExit(f"missing index file: {index_path}")
        index_payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        index = VectorIndex()
        for node_id, vector in index_payload.items():
            index.upsert(node_id, vector)
        seeds = index.search(query_vec, top_k=args.top)
    else:
        seeds = _keyword_seeds(graph=graph, text=args.text, top_k=args.top)

    result = traverse(graph=graph, seeds=seeds, config=TraversalConfig(max_hops=15), route_fn=None)
    if not args.no_log:
        log_query(query_text=args.text, fired_ids=result.fired, node_count=graph.node_count())

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
    if not args.no_log:
        log_learn(fired_ids=fired_ids, outcome=args.outcome)
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


def cmd_replay(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    queries = _load_session_queries(args.sessions)
    if args.max_queries is not None:
        if args.max_queries <= 0:
            queries = []
        else:
            queries = queries[: args.max_queries]

    stats = replay_queries(graph=graph, queries=queries, verbose=not args.json)
    if not args.no_log:
        log_replay(
            queries_replayed=stats["queries_replayed"],
            edges_reinforced=stats["edges_reinforced"],
            cross_file_created=stats["cross_file_edges_created"],
        )

    graph_path = Path(args.graph).expanduser()
    if graph_path.is_dir():
        graph_path = graph_path / "graph.json"
    payload = {
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
    }
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(
            f"Replayed {stats['queries_replayed']}/{len(queries)} queries, "
            f"{stats['cross_file_edges_created']} cross-file edges created"
        )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    health = measure_health(graph)
    payload = health.__dict__
    payload["nodes"] = graph.node_count()
    payload["edges"] = graph.edge_count()
    if not args.no_log:
        log_health(payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            "nodes: {nodes}\nedges: {edges}\ndormant_pct: {dormant_pct:.2f}\nhabitual_pct: {habitual_pct:.2f}\n"
            "reflex_pct: {reflex_pct:.2f}\ncross_file_edge_pct: {cross_file_edge_pct:.2f}\norphan_nodes: {orphan_nodes}".format(
                **payload
            )
        )
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    if args.last is not None and args.last <= 0:
        raise SystemExit("last must be a positive integer")

    if args.stats:
        payload = journal_stats()
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0

        print(f"total_entries: {payload['total_entries']}")
        print(f"queries: {payload['queries']}")
        print(f"learns: {payload['learns']}")
        print(f"positive_outcomes: {payload['positive_outcomes']}")
        print(f"negative_outcomes: {payload['negative_outcomes']}")
        print(f"avg_fired_per_query: {payload['avg_fired_per_query']:.4f}")
        return 0

    entries = read_journal(last_n=args.last)
    if args.json:
        print(json.dumps(entries, indent=2))
        return 0

    if not entries:
        print("No entries.")
        return 0

    for idx, entry in enumerate(entries, start=1):
        kind = entry.get("type", "unknown")
        timestamp = entry.get("iso", entry.get("ts", ""))
        if kind == "query":
            detail = f"query={entry.get('query')!r}"
            detail += f", fired={entry.get('fired_count', 0)}"
        elif kind == "learn":
            detail = f"outcome={entry.get('outcome', 0)}"
        elif kind == "replay":
            detail = (
                f"queries_replayed={entry.get('queries_replayed', 0)}, "
                f"edges_reinforced={entry.get('edges_reinforced', 0)}, "
                f"cross_file_created={entry.get('cross_file_created', 0)}"
            )
        else:
            detail = ", ".join(
                f"{key}={value}"
                for key, value in entry.items()
                if key not in {"type", "ts", "iso"}
            )
        print(f"{idx:>2}. {kind} @ {timestamp}: {detail}")
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
    if args.command == "replay":
        return cmd_replay(args)
    if args.command == "health":
        return cmd_health(args)
    if args.command == "journal":
        return cmd_journal(args)
    if args.command == "embed":
        return cmd_embed(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
