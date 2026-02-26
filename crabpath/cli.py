"""Pure-callback CLI for CrabPath graph operations."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path

from ._batch import batch_or_single_embed
from .autotune import measure_health
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
from .score import score_retrieval
from .split import generate_summaries, split_workspace
from .traverse import TraversalConfig, TraversalResult, traverse


_EMBED_BATCH_SIZE = 50


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crabpath")
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("--workspace", required=True)
    i.add_argument("--output", required=True)
    i.add_argument("--sessions")
    i.add_argument("--embed-command")
    i.add_argument("--route-command")
    i.add_argument("--llm-split", choices=("auto", "always", "never"), default="auto")
    i.add_argument("--llm-split-max-files", type=int, default=30)
    i.add_argument("--llm-split-min-chars", type=int, default=4000)
    i.add_argument("--llm-summary", choices=("auto", "always", "never"), default="auto")
    i.add_argument("--llm-summary-max-nodes", type=int, default=200)
    i.add_argument("--parallel", type=int, default=8)
    i.add_argument("--json", action="store_true")
    e = sub.add_parser("embed")
    e.add_argument("--texts", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--command", dest="embed_command", required=True)
    e.add_argument("--json", action="store_true")
    e.add_argument("--parallel", type=int, default=8)
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--graph", required=True)
    q.add_argument("--index")
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--query-vector", nargs="+")
    q.add_argument("--query-vector-stdin", action="store_true")
    q.add_argument("--parallel", type=int, default=8)
    q.add_argument("--route-command")
    q.add_argument("--embed-command")
    q.add_argument("--auto-merge", action="store_true")
    q.add_argument("--json", action="store_true")
    l = sub.add_parser("learn")
    l.add_argument("--graph", required=True)
    l.add_argument("--outcome", type=float)
    l.add_argument("--scores")
    l.add_argument("--fired-ids", required=True)
    l.add_argument("--auto-merge", action="store_true")
    l.add_argument("--json", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--graph", required=True)
    m.add_argument("--json", action="store_true")
    c = sub.add_parser("connect")
    c.add_argument("--graph", required=True)
    c.add_argument("--max-candidates", type=int, default=20)
    c.add_argument("--json", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--graph", required=True)
    r.add_argument("--sessions", nargs="+", required=True)
    r.add_argument("--max-queries", type=int)
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
    payload_path = Path(path).expanduser()
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
        graph.add_node(Node(node_data["id"], node_data["content"], node_data.get("summary", ""), node_data.get("metadata", {})))
    for edge_data in payload.get("edges", []):
        graph.add_edge(Edge(edge_data["source"], edge_data["target"], edge_data.get("weight", 0.5), edge_data.get("kind", "sibling"), edge_data.get("metadata", {})))
    return graph


def _graph_payload(graph: Graph) -> dict:
    return {
        "nodes": [{"id": n.id, "content": n.content, "summary": n.summary, "metadata": n.metadata} for n in graph.nodes()],
        "edges": [{"source": e.source, "target": e.target, "weight": e.weight, "kind": e.kind, "metadata": e.metadata}
                  for source in graph._edges.values() for e in source.values()],
    }


def _write_graph(path: str | Path, graph: Graph) -> None:
    destination = Path(path).expanduser()
    if destination.is_dir():
        destination = destination / "graph.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_graph_payload(graph), indent=2), encoding="utf-8")


def _parse_vector(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(chunk) for value in values for chunk in value.split(",") if chunk]


def _parse_score_payload(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --scores payload: {exc}")
    scores = payload.get("scores", payload) if isinstance(payload, dict) else None
    if not isinstance(scores, dict):
        raise SystemExit("--scores payload must be a JSON object mapping node ids to scores")
    parsed = {}
    for node_id, raw_score in scores.items():
        if isinstance(node_id, str):
            try:
                parsed[node_id] = float(raw_score)
            except (TypeError, ValueError):
                pass
    return parsed


def _run_command(command: list[str], payload: str) -> tuple[int, str, str]:
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate(payload)
    return proc.returncode, out or "", err or ""


def _run_llm_command(command: list[str] | str, system_prompt: str, user_prompt: str) -> str:
    cmd = command if isinstance(command, list) else shlex.split(command)
    code, out, err = _run_command(cmd, json.dumps({"system": system_prompt, "user": user_prompt}))
    if code != 0:
        raise SystemExit((err.strip() or f"exit code {code}")[:120])
    return out.strip()


def _run_llm_batch(command: list[str] | str, requests: list[dict]) -> list[dict]:
    cmd = command if isinstance(command, list) else shlex.split(command)
    return [{"id": request.get("id", ""), "response": _run_llm_command(cmd, str(request.get("system", "")), str(request.get("user", "")))} for request in requests]


def _run_route_command(command: list[str] | str, query_text: str, candidates: list[dict[str, str | float]]) -> list[str]:
    cmd = command if isinstance(command, list) else shlex.split(command)
    code, out, err = _run_command(cmd, json.dumps({"query": query_text, "candidates": candidates}))
    if code != 0:
        raise SystemExit((err.strip() or f"exit code {code}")[:120])
    selected = json.loads((out or "").strip()).get("selected")
    if not isinstance(selected, list):
        raise SystemExit("route command output must contain JSON field selected")
    return [str(x) for x in selected]


def _run_embedding_batch(command: list[str] | str, batch: list[tuple[str, str]]) -> dict[str, list[float]]:
    cmd = command if isinstance(command, list) else shlex.split(command)
    payload = "\n".join(json.dumps({"id": node_id, "text": text}) for node_id, text in batch) + "\n"
    code, out, err = _run_command(cmd, payload)
    if code != 0:
        raise SystemExit((err.strip() or f"exit code {code}")[:120])
    output = {}
    for line in (out or "").splitlines():
        data = json.loads(line)
        if "id" not in data or "embedding" not in data:
            raise SystemExit("embedding output must contain id and embedding")
        output[str(data["id"])] = [float(v) for v in data["embedding"]]
    if not output:
        raise SystemExit("embed command returned no embeddings")
    return output


def _build_index_from_texts(
    texts: dict[str, str],
    parallel: int,
    embed_command: str | None = None,
    embed_batch_fn: Callable[[list[tuple[str, str]]], dict[str, list[float]]] | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> VectorIndex:
    if parallel <= 0:
        raise SystemExit("--parallel must be >= 1")
    index = VectorIndex()
    if not texts:
        return index

    if embed_command is None and embed_batch_fn is None and embed_fn is None:
        return index
    if embed_command is not None and (embed_batch_fn is not None or embed_fn is not None):
        raise SystemExit("--embed-command cannot be used with local embedding callbacks")

    text_items = list(texts.items())

    def _embed_one(text: str) -> list[float]:
        if embed_fn is not None:
            return embed_fn(text)
        if embed_command is None:
            raise SystemExit("local embedding requires an embed callback")
        result = _run_embedding_batch(embed_command, [("query", text)])
        if "query" not in result:
            raise SystemExit("embedding output missing query response")
        return result["query"]

    def _embed_batch(items: list[tuple[str, str]]) -> dict[str, list[float]]:
        if embed_batch_fn is not None:
            return embed_batch_fn(items)
        if embed_command is None:
            raise SystemExit("local embedding requires an embed callback")
        return _run_embedding_batch(embed_command, items)

    batches = [text_items[i:i + _EMBED_BATCH_SIZE] for i in range(0, len(text_items), _EMBED_BATCH_SIZE)]

    def _submit_batch(batch: list[tuple[str, str]]) -> dict[str, list[float]]:
        return batch_or_single_embed(
            texts=batch,
            embed_fn=_embed_one,
            embed_batch_fn=_embed_batch,
            max_workers=parallel,
        )

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        payloads = [pool.submit(_submit_batch, batch) for batch in batches]
        for future in payloads:
            try:
                batch_payload = future.result()
                for node_id, vector in batch_payload.items():
                    index.upsert(node_id, vector)
            except (Exception, SystemExit) as exc:
                print(f"Warning: embedding batch failed: {exc}", file=sys.stderr)
    return index


def _resolve_local_embed_batch_fn() -> Callable[[list[tuple[str, str]]], dict[str, list[float]]]:
    from .embeddings import local_embed_batch_fn
    return local_embed_batch_fn


def _embed_query_text(
    query_text: str,
    embed_command: str | None = None,
    embed_batch_fn: Callable[[list[tuple[str, str]]], dict[str, list[float]]] | None = None,
) -> list[float]:
    if embed_batch_fn is not None:
        result = embed_batch_fn([("query", query_text)])
        if "query" not in result:
            raise SystemExit("query embedding output missing query id")
        return result["query"]
    if embed_command is None:
        raise SystemExit("query embedding requires --embed-command")
    result = _run_embedding_batch(embed_command, [("query", query_text)])
    if "query" not in result:
        raise SystemExit("query embedding output missing query id")
    return result["query"]


def _load_query_vector_from_stdin() -> list[float]:
    data = sys.stdin.read().strip()
    if not data:
        raise SystemExit("query vector JSON required on stdin")
    payload = json.loads(data)
    if not isinstance(payload, list):
        raise SystemExit("query vector stdin payload must be a JSON array")
    return [float(v) for v in payload]


def _load_texts(path: str) -> dict[str, str]:
    payload_path = Path(path).expanduser()
    if not payload_path.exists():
        raise SystemExit(f"missing texts file: {path}")
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("texts payload must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def _load_session_queries(session_paths: list[str] | str) -> list[str]:
    if isinstance(session_paths, str):
        session_paths = [session_paths]
    queries = []
    for session_path in session_paths:
        path = Path(session_path).expanduser()
        if path.is_dir():
            queries.extend(extract_queries_from_dir(path))
        elif path.is_file():
            queries.extend(extract_queries(path))
        else:
            raise SystemExit(f"invalid sessions path: {path}")
    return queries


def _route_candidates(graph: Graph, candidate_ids: list[str]) -> list[dict[str, str | float]]:
    out = []
    for node_id in candidate_ids:
        node = graph.get_node(node_id)
        if node is not None:
            weight = max((edge.weight for _, edge in graph.incoming(node_id)), default=0.5)
            out.append({"id": node.id, "content": node.content, "weight": weight})
    return out


def _keyword_seeds(graph: Graph, text: str, top_k: int) -> list[tuple[str, float]]:
    query_tokens = {token for token in text.lower().replace("_", " ").split() if token}
    if not query_tokens:
        return []
    scores = [(node.id, len(query_tokens & {t for t in node.content.lower().replace("_", " ").split() if t}) / len(query_tokens)) for node in graph.nodes()]
    scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scores[:top_k]


def _prepare_llm_fns(route_command: str | None) -> tuple:
    if route_command is None:
        return None, None
    cmd = shlex.split(route_command)
    return (
        lambda system, user: _run_llm_command(cmd, system, user),
        lambda requests: _run_llm_batch(cmd, requests),
    )


def cmd_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).expanduser()
    if output_dir.suffix == ".json" and not output_dir.is_dir():
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel <= 0:
        raise SystemExit("--parallel must be >= 1")
    if args.llm_split_min_chars < 0 or args.llm_split_max_files < 0 or args.llm_summary_max_nodes < 0:
        raise SystemExit("llm thresholds must be >= 0")

    llm_fn, llm_batch_fn = _prepare_llm_fns(args.route_command if args.llm_split != "never" or args.llm_summary != "never" else None)
    count = {"file": 0}

    def should_use(relative_path: str, content: str) -> bool:
        if not llm_fn:
            return False
        if args.llm_split == "always":
            return True
        if args.llm_split == "auto":
            if args.llm_split_max_files <= 0 or relative_path.startswith(".") or len(content) < args.llm_split_min_chars:
                return False
            if sum(1 for line in content.splitlines() if line.startswith("## ")) > 1:
                return False
            if count["file"] >= args.llm_split_max_files:
                return False
            count["file"] += 1
            return True
        return False

    graph, texts = split_workspace(
        args.workspace,
        llm_fn=llm_fn,
        llm_batch_fn=llm_batch_fn,
        should_use_llm_for_file=should_use,
        llm_parallelism=args.parallel,
    )
    if args.sessions is not None:
        replay_queries(graph=graph, queries=_load_session_queries(args.sessions))

    graph_path = output_dir / "graph.json"
    text_path = output_dir / "texts.json"
    _write_graph(graph_path, graph)
    text_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")

    if args.llm_summary != "never" and llm_fn is not None:
        targets = None if args.llm_summary == "always" else {
            node.id for node in sorted(graph.nodes(), key=lambda n: len(n.content), reverse=True)[: args.llm_summary_max_nodes]
        }
        for node in graph.nodes():
            if targets is None or node.id in targets:
                pass
        for node_id, summary in generate_summaries(
            graph,
            llm_fn=llm_fn,
            llm_batch_fn=llm_batch_fn,
            llm_node_ids=targets,
            llm_parallelism=args.parallel,
        ).items():
            graph.get_node(node_id).summary = summary if graph.get_node(node_id) else summary
        _write_graph(graph_path, graph)

    local_embed_batch_fn = None
    if args.embed_command is None:
        local_embed_batch_fn = _resolve_local_embed_batch_fn()
        if local_embed_batch_fn is not None:
            print("Using local embeddings (all-MiniLM-L6-v2)", file=sys.stderr)

    if args.embed_command:
        _build_index_from_texts(texts, args.parallel, embed_command=args.embed_command).save(str(output_dir / "index.json"))
    elif local_embed_batch_fn is not None:
        try:
            _build_index_from_texts(texts, args.parallel, embed_batch_fn=local_embed_batch_fn).save(str(output_dir / "index.json"))
        except ImportError:
            print("Warning: local embeddings unavailable, continuing without index", file=sys.stderr)

    if args.json:
        print(json.dumps({"graph": str(graph_path), "texts": str(text_path), "index": str(output_dir / "index.json") if (output_dir / "index.json").exists() else None}))
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    texts = _load_texts(args.texts)
    index = _build_index_from_texts(texts, args.parallel, embed_command=args.embed_command)
    output = Path(args.output).expanduser()
    if output.suffix != ".json" or output.is_dir():
        output = output / "index.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output))
    if args.json:
        print(json.dumps({"index": str(output), "count": len(texts)}))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    if args.parallel <= 0:
        raise SystemExit("--parallel must be >= 1")
    if args.query_vector_stdin and args.query_vector:
        raise SystemExit("use only one of --query-vector or --query-vector-stdin")

    local_embed_batch_fn = None
    if args.embed_command is None and args.index is not None:
        local_embed_batch_fn = _resolve_local_embed_batch_fn()
        if local_embed_batch_fn is not None:
            print("Using local embeddings (all-MiniLM-L6-v2)", file=sys.stderr)

    query_vec = _parse_vector(args.query_vector)
    if query_vec is None:
        if args.query_vector_stdin:
            if not args.index:
                raise SystemExit("query-vector-stdin requires --index")
            query_vec = _load_query_vector_from_stdin()
        elif args.embed_command is not None and args.index is not None:
            query_vec = _embed_query_text(query_text=args.text, embed_command=args.embed_command)
        elif local_embed_batch_fn is not None and args.index is not None:
            try:
                query_vec = _embed_query_text(args.text, embed_batch_fn=local_embed_batch_fn)
            except ImportError:
                query_vec = None

    if query_vec is not None:
        if not args.index:
            raise SystemExit("query vector mode requires --index")
        raw = json.loads(Path(args.index).read_text(encoding="utf-8"))
        index = VectorIndex()
        for node_id, vector in raw.items():
            index.upsert(node_id, vector)
        seeds = index.search(query_vec, top_k=args.top)
    else:
        seeds = _keyword_seeds(graph, args.text, args.top)

    llm_fn, llm_batch_fn = _prepare_llm_fns(args.route_command)
    route_fn = None
    if llm_fn is not None:
        route_fn = lambda query_text, ids: _safe_route_command(graph, args.route_command, query_text, ids)

    result = traverse(graph=graph, seeds=seeds, config=TraversalConfig(max_hops=15), query_text=args.text, route_fn=route_fn)
    scores = {}
    merges = []
    if llm_fn is not None and result.fired:
        scores = score_retrieval(
            args.text,
            [(node_id, graph.get_node(node_id).content) for node_id in result.fired if graph.get_node(node_id)],
            llm_fn=llm_fn,
            llm_batch_fn=llm_batch_fn,
        )

    if args.auto_merge and llm_fn is not None:
        for source_id, target_id in suggest_merges(graph, llm_fn=llm_fn, llm_batch_fn=llm_batch_fn)[:5]:
            if graph.get_node(source_id) and graph.get_node(target_id):
                merged = apply_merge(graph, source_id, target_id)
                merges.append({"from": [source_id, target_id], "to": [merged]})

    if merges:
        _write_graph(args.graph, graph)
    log_query(query_text=args.text, fired_ids=result.fired, node_count=graph.node_count())

    payload = {"fired": result.fired, "steps": [step.__dict__ for step in result.steps], "context": result.context, "scores": scores, "merges": merges, "created_node_id": None}
    print(json.dumps(payload) if args.json else result.context)
    return 0


def _safe_route_command(graph: Graph, command: str | None, query_text: str | None, candidate_ids: list[str]) -> list[str]:
    if command is None:
        return candidate_ids
    try:
        return _run_route_command(command, query_text or "", _route_candidates(graph, candidate_ids))
    except (Exception, SystemExit):
        print("Warning: route command failed; falling back to all candidates", file=sys.stderr)
        return candidate_ids


def cmd_learn(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    fired_ids = [v.strip() for v in args.fired_ids.split(",") if v.strip()]

    parsed_scores = _parse_score_payload(args.scores)
    if parsed_scores:
        values = [parsed_scores[n] for n in fired_ids if n in parsed_scores]
        if values:
            outcome = max(-1.0, min(1.0, (sum(values) / len(values)) * 2.0 - 1.0))
        elif args.outcome is not None:
            outcome = args.outcome
        else:
            raise SystemExit("no matching scored nodes; provide --outcome")
    elif args.outcome is not None:
        outcome = args.outcome
    else:
        raise SystemExit("provide either --outcome or --scores")

    apply_outcome(graph, fired_nodes=fired_ids, outcome=outcome)
    merges = []
    if args.auto_merge:
        for source_id, target_id in suggest_merges(graph)[:5]:
            if graph.get_node(source_id) and graph.get_node(target_id):
                merged_id = apply_merge(graph, source_id, target_id)
                merges.append({"from": [source_id, target_id], "to": [merged_id]})

    payload = {"graph": _graph_payload(graph), "merges": merges}
    Path(args.graph).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_learn(fired_ids=fired_ids, outcome=outcome)
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
    suggestions = suggest_connections(graph, max_candidates=args.max_candidates)
    added = apply_connections(graph=graph, connections=suggestions)
    _write_graph(args.graph, graph)
    payload = {"suggestions": [{"source_id": s, "target_id": t, "weight": w, "reason": r} for s, t, w, r in suggestions], "added": added}
    print(json.dumps(payload) if args.json else f"Added edges: {added}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    queries = _load_session_queries(args.sessions)
    if args.max_queries is not None:
        queries = queries[: args.max_queries] if args.max_queries > 0 else []
    stats = replay_queries(graph=graph, queries=queries, verbose=not args.json)
    log_replay(queries_replayed=stats["queries_replayed"], edges_reinforced=stats["edges_reinforced"], cross_file_created=stats["cross_file_edges_created"])
    _write_graph(args.graph, graph)
    print(json.dumps(stats, indent=2) if args.json else f"Replayed {stats['queries_replayed']}/{len(queries)} queries, {stats['cross_file_edges_created']} cross-file edges created")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    payload = measure_health(graph).__dict__
    payload["nodes"] = graph.node_count()
    payload["edges"] = graph.edge_count()
    log_health(payload)
    print(json.dumps(payload, indent=2) if args.json else f"nodes: {payload['nodes']}\nedges: {payload['edges']}\ndormant_pct: {payload['dormant_pct']:.2f}\nhabitual_pct: {payload['habitual_pct']:.2f}\nreflex_pct: {payload['reflex_pct']:.2f}\ncross_file_edge_pct: {payload['cross_file_edge_pct']:.2f}\norphan_nodes: {payload['orphan_nodes']}")
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    if args.stats:
        print(json.dumps(journal_stats(), indent=2) if args.json else "\n".join(f"{k}: {v}" for k, v in journal_stats().items() if k != "avg_fired_per_query") )
        return 0
    entries = read_journal(last_n=args.last)
    print(json.dumps(entries, indent=2) if args.json else "\n".join(f"{idx+1:>2}. {entry.get('type')} @ {entry.get('iso', entry.get('ts', ''))}: {entry}" for idx, entry in enumerate(entries)) or "No entries.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return {
        "init": cmd_init,
        "embed": cmd_embed,
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
