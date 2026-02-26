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
import urllib.request
from pathlib import Path

from .graph import Edge, Graph, Node
from .index import VectorIndex
from .connect import apply_connections, suggest_connections
from .replay import extract_queries, extract_queries_from_dir, replay_queries
from .split import generate_summaries, split_workspace
from .traverse import TraversalConfig, traverse
from .learn import apply_outcome, maybe_create_node
from .merge import apply_merge, suggest_merges
from .score import score_retrieval
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
    init.add_argument("--route-provider", choices=("openai", "ollama"))
    init.add_argument("--llm-split", choices=("auto", "always", "never"), default="auto")
    init.add_argument("--llm-split-max-files", type=int, default=30)
    init.add_argument("--llm-split-min-chars", type=int, default=4000)
    init.add_argument("--llm-summary", choices=("auto", "always", "never"), default="auto")
    init.add_argument("--llm-summary-max-nodes", type=int, default=200)
    init.add_argument("--no-embed", action="store_true", help="skip embedding during init")
    init.add_argument("--no-route", action="store_true", help="skip route provider detection during init")
    init.add_argument("--json", action="store_true")
    init.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    embed = sub.add_parser("embed", help="build index.json from texts.json using an embedding command")
    embed.add_argument("--texts", required=True)
    embed.add_argument("--output", required=True)
    embed_provider_group = embed.add_mutually_exclusive_group(required=False)
    embed_provider_group.add_argument("--command", dest="embed_command")
    embed_provider_group.add_argument("--provider", choices=("openai", "ollama"))
    embed.add_argument("--json", action="store_true")

    query = sub.add_parser("query", help="seed from index and traverse graph")
    query.add_argument("text")
    query.add_argument("--graph", required=True)
    query.add_argument("--index", required=False)
    query.add_argument("--top", type=int, default=10)
    query.add_argument("--no-embed-query", action="store_true", help="skip automatic query embedding when --index is provided")
    query.add_argument("--json", action="store_true")
    query.add_argument("--query-vector", nargs="+", required=False)
    query.add_argument("--query-vector-stdin", action="store_true")
    query_route_group = query.add_mutually_exclusive_group(required=False)
    query_route_group.add_argument("--route-command")
    query_route_group.add_argument("--route-provider", choices=("openai", "ollama"))
    query.add_argument("--no-route", action="store_true", help="skip LLM routing")
    query.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    learn = sub.add_parser("learn", help="apply outcome update")
    learn.add_argument("--graph", required=True)
    learn.add_argument("--outcome", type=float, required=False)
    learn.add_argument("--scores", required=False)
    learn.add_argument("--fired-ids", required=True)
    learn.add_argument("--json", action="store_true")
    learn.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    merge = sub.add_parser("merge", help="suggest and apply node merges")
    merge.add_argument("--graph", required=True)
    merge.add_argument("--route-provider", choices=("openai", "ollama"))
    merge.add_argument("--json", action="store_true")
    merge.add_argument("--no-log", action="store_true", help="disable query/learn/replay/health logging")

    connect = sub.add_parser("connect", help="suggest and apply cross-file edges")
    connect.add_argument("--graph", required=True)
    connect.add_argument("--route-provider", choices=("openai", "ollama"))
    connect.add_argument("--max-candidates", type=int, default=20)
    connect.add_argument("--json", action="store_true")

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


def _graph_path_arg(path: str) -> Path:
    graph_path = Path(path).expanduser()
    if graph_path.is_dir():
        graph_path = graph_path / "graph.json"
    return graph_path


def _graph_payload(graph: Graph) -> dict:
    return {
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


def _write_graph(graph_path: str | Path, graph: Graph) -> None:
    destination = _graph_path_arg(str(graph_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_graph_payload(graph), indent=2), encoding="utf-8")


def _parse_vector(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None
    vector: list[float] = []
    for value in values:
        for chunk in value.split(","):
            if chunk:
                vector.append(float(chunk))
    return vector


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

    parsed: dict[str, float] = {}
    for node_id, raw_score in scores.items():
        if not isinstance(node_id, str):
            continue
        try:
            parsed[node_id] = float(raw_score)
        except (TypeError, ValueError):
            continue
    return parsed


def _auto_detect_provider(check_env_only: bool = False) -> str | None:
    """Auto-detect the best available provider.
    Checks: OPENAI_API_KEY env, ~/.env, macOS keychain, then Ollama."""
    if os.getenv("CRABPATH_NO_AUTO_DETECT"):
        return None

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return "openai"

    if check_env_only:
        return None

    def _strip_inline_comment(value: str) -> str:
        in_single = False
        in_double = False
        escaped = False
        for idx, char in enumerate(value):
            if escaped:
                escaped = False
                continue
            if char == "\\" and not in_single:
                escaped = True
                continue
            if char == "'" and not in_double:
                in_single = not in_single
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                continue
            if char == "#" and not in_single and not in_double:
                return value[:idx].rstrip()
        return value

    env_candidates = [
        ".env",
        ".zshrc",
        ".zprofile",
        ".bash_profile",
        ".bashrc",
    ]
    home = Path.home()
    for filename in env_candidates:
        env_file = home / filename
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export "):].strip()

            if not line.startswith("OPENAI_API_KEY="):
                continue

            raw_value = line.split("=", 1)[1].strip()
            raw_value = _strip_inline_comment(raw_value).strip()
            value = raw_value.strip().strip('"').strip("'").strip()
            if value:
                os.environ["OPENAI_API_KEY"] = value
                return "openai"

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "openai", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            os.environ["OPENAI_API_KEY"] = result.stdout.strip()
            return "openai"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return "ollama"
    except Exception:
        pass

    return None


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


def _route_provider_script(provider: str) -> str:
    if provider == "openai":
        return """import json
import re
import sys

from openai import OpenAI


client = OpenAI()
req = json.loads(sys.stdin.read())
candidates = '\\n'.join(f'- {c[\"id\"]}: {c[\"content\"][:100]}' for c in req['candidates'])
resp = client.chat.completions.create(
    model='gpt-5-mini',
    messages=[
        {'role': 'system', 'content': 'You are a memory router. Given a query and candidates, select which are most relevant. Return JSON: {\"selected\": [\"id1\", \"id2\"]}'},
        {'role': 'user', 'content': f'Query: {req[\"query\"]}\\n\\nCandidates:\\n{candidates}'},
    ],
    temperature=0.0,
    max_completion_tokens=200,
)

content = resp.choices[0].message.content
try:
    result = json.loads(content)
except:
    match = re.search(r'\\{[^}]+\\}', content)
    result = json.loads(match.group()) if match else {'selected': [c['id'] for c in req['candidates'][:3]]}
print(json.dumps(result))
"""
    if provider == "ollama":
        return """import json
import re
import sys

import requests


req = json.loads(sys.stdin.read())
candidates = '\\n'.join(f'- {c[\"id\"]}: {c[\"content\"][:100]}' for c in req['candidates'])
resp = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'llama3',
        'prompt': f'Select the most relevant candidates for this query. Return JSON {\"selected\": [\"id1\"]}\\n\\nQuery: {req[\"query\"]}\\n\\nCandidates:\\n{candidates}',
        'stream': False,
    },
)
content = resp.json()['response']
match = re.search(r'\\{[^}]+\\}', content)
result = json.loads(match.group()) if match else {'selected': [c['id'] for c in req['candidates'][:3]]}
print(json.dumps(result))
"""
    raise SystemExit(f"unsupported route provider: {provider}")


def _llm_provider_script(provider: str) -> str:
    if provider == "openai":
        return """import json
import sys

from openai import OpenAI


req = json.loads(sys.stdin.read())
client = OpenAI()
resp = client.chat.completions.create(
    model='gpt-5-mini',
    messages=[
        {'role': 'system', 'content': req['system']},
        {'role': 'user', 'content': req['user']},
    ],
    temperature=0.0,
    response_format={"type": "json_object"},
    max_completion_tokens=500,
)
content = resp.choices[0].message.content
print(content or '')
"""
    if provider == "ollama":
        return """import json
import re
import sys

import requests


req = json.loads(sys.stdin.read())
prompt = f"{req['system']}\\n\\n{req['user']}"
resp = requests.post(
    'http://localhost:11434/api/generate',
    json={'model': 'llama3', 'prompt': prompt, 'stream': False},
)
content = resp.json().get('response', '')
match = re.search(r'\\{.*\\}', content, re.S)
print(match.group(0) if match else content)
"""
    raise SystemExit(f"unsupported llm provider: {provider}")


def _build_llm_command(llm_provider: str) -> tuple[list[str], str | None]:
    if llm_provider is None:
        raise SystemExit("provider required for llm commands")
    script_code = _llm_provider_script(llm_provider)
    fd, script_path = tempfile.mkstemp(prefix="crabpath_llm_", suffix=".py")
    os.close(fd)
    Path(script_path).write_text(script_code, encoding="utf-8")
    return [sys.executable, script_path], script_path


def _run_llm_command(command: list[str], system_prompt: str, user_prompt: str) -> str:
    payload = {"system": system_prompt, "user": user_prompt}
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_data, stderr_data = proc.communicate(json.dumps(payload))
    if proc.returncode != 0:
        message = (stderr_data or "").strip() or f"exit code {proc.returncode}"
        raise SystemExit(f"llm command failed: {message}")
    return (stdout_data or "").strip()


def _build_route_command(route_command: str | None, route_provider: str | None) -> tuple[list[str], str | None]:
    if route_command is None and route_provider is None:
        raise SystemExit("provide either --route-command or --route-provider for routing")
    if route_command is not None and route_provider is not None:
        raise SystemExit("provide only one of --route-command or --route-provider")
    if route_command is not None:
        return shlex.split(route_command), None

    script_code = _route_provider_script(route_provider)
    fd, script_path = tempfile.mkstemp(prefix="crabpath_route_", suffix=".py")
    os.close(fd)
    Path(script_path).write_text(script_code, encoding="utf-8")
    return [sys.executable, script_path], script_path


def _run_route_command(
    command: list[str],
    query_text: str,
    candidates: list[dict[str, str | float]],
) -> list[str]:
    payload = {"query": query_text, "candidates": candidates}
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_data, stderr_data = proc.communicate(json.dumps(payload))
    if proc.returncode != 0:
        message = (stderr_data or "").strip() or f"exit code {proc.returncode}"
        raise SystemExit(f"route command failed: {message}")
    if not stdout_data.strip():
        raise SystemExit("route command returned no candidates")
    data = json.loads(stdout_data)
    selected = data.get("selected")
    if not isinstance(selected, list):
        raise SystemExit("route output must contain selected list")
    return [str(item) for item in selected]


def _build_route_candidates(graph: Graph, candidate_ids: list[str]) -> list[dict[str, str | float]]:
    payload: list[dict[str, str | float]] = []
    for node_id in candidate_ids:
        node = graph.get_node(node_id)
        if node is None:
            continue
        incoming = graph.incoming(node_id)
        weight = max((edge.weight for _, edge in incoming), default=0.5)
        payload.append({"id": node.id, "content": node.content, "weight": weight})
    return payload


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


def _embed_query_text(
    query_text: str,
    embed_command: str | None,
    embed_provider: str | None,
) -> list[float]:
    command, temp_script = _build_embed_command(embed_command=embed_command, embed_provider=embed_provider)
    try:
        results = _run_embedding_batch(command=command, batch=[("query", query_text)])
        if "query" not in results:
            raise SystemExit("query embedding response missing query id")
        return results["query"]
    finally:
        if temp_script is not None and Path(temp_script).exists():
            Path(temp_script).unlink()


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

    if args.no_route:
        if args.route_provider is not None:
            raise SystemExit("use --no-route without --route-provider")
        route_provider = None
    elif args.route_provider is not None:
        route_provider = args.route_provider
    else:
        route_provider = _auto_detect_provider()
        if route_provider:
            print(f"Auto-detected routing provider: {route_provider}", file=sys.stderr)
        else:
            print("Warning: no routing provider detected, continuing without LLM splitting/summaries", file=sys.stderr)

    llm_command = None
    llm_temp_script = None
    if route_provider is not None:
        llm_command, llm_temp_script = _build_llm_command(route_provider)
        print("Using LLM for: splitting, summaries", file=sys.stderr)

    if args.llm_split_max_files < 0:
        raise SystemExit("--llm-split-max-files must be >= 0")
    if args.llm_split_min_chars < 0:
        raise SystemExit("--llm-split-min-chars must be >= 0")
    if args.llm_summary_max_nodes < 0:
        raise SystemExit("--llm-summary-max-nodes must be >= 0")

    llm_enabled = route_provider is not None

    split_counter = {"count": 0}

    def should_split_with_llm(relative_path: str, content: str) -> bool:
        if not llm_enabled:
            return False
        if args.llm_split == "never":
            return False
        if args.llm_split == "always":
            return True
        if args.llm_split == "auto":
            if args.llm_split_max_files <= 0:
                return False
            if len(content) < args.llm_split_min_chars:
                return False
            header_count = sum(1 for line in content.splitlines() if line.startswith("## "))
            if header_count > 1:
                return False
            if split_counter["count"] >= args.llm_split_max_files:
                return False
            split_counter["count"] += 1
            return True
        return False

    def split_progress_fn(file_index: int, file_total: int, rel_path: str, mode: str) -> None:
        print(f"Splitting {file_index}/{file_total} ({mode}): {rel_path}", file=sys.stderr)

    def llm_fn(system_prompt: str, user_prompt: str) -> str:
        if llm_command is None:
            raise SystemExit("LLM command is unavailable")
        return _run_llm_command(llm_command, system_prompt, user_prompt)

    def summary_progress_fn(node_index: int, node_total: int) -> None:
        print(f"Summarizing {node_index}/{node_total}", file=sys.stderr)

    graph_path = output_dir / "graph.json"
    texts_path = output_dir / "texts.json"

    try:
        graph, texts = split_workspace(
            args.workspace,
            llm_fn=llm_fn if llm_enabled else None,
            should_use_llm_for_file=should_split_with_llm if llm_enabled else None,
            split_progress=split_progress_fn,
        )

        if args.sessions is not None:
            queries = _load_session_queries(args.sessions)
            replay_queries(graph=graph, queries=queries)

        _write_graph(graph_path, graph)
        texts_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")

        if args.llm_summary != "never" and llm_enabled:
            nodes = graph.nodes()
            if args.llm_summary == "auto":
                if args.llm_summary_max_nodes == 0:
                    summary_targets = set()
                else:
                    summary_targets = {
                        node.id
                        for node in sorted(nodes, key=lambda node: len(node.content), reverse=True)[
                            : args.llm_summary_max_nodes
                        ]
                    }
            else:
                summary_targets = None

            summaries = generate_summaries(graph, llm_fn=llm_fn, llm_node_ids=summary_targets)
            for index, node in enumerate(graph.nodes(), start=1):
                summary_progress_fn(index, len(graph.nodes()))
                node_summary = summaries.get(node.id)
                if node_summary is not None:
                    node.summary = node_summary

        if args.no_embed:
            if args.embed_command is not None or args.embed_provider is not None:
                raise SystemExit("use --no-embed without --embed-command or --embed-provider")
            embed_provider = None
            embed_command = None
        elif args.embed_command is not None or args.embed_provider is not None:
            embed_command = args.embed_command
            embed_provider = args.embed_provider
        else:
            embed_provider = _auto_detect_provider()
            embed_command = None
            if embed_provider:
                print(f"Auto-detected embedding provider: {embed_provider}")
            else:
                print("Warning: no embedding provider detected, continuing without embeddings")

        if args.embed_command is not None or embed_provider is not None:
            index = _build_index_from_texts(
                texts=texts,
                embed_command=embed_command,
                embed_provider=embed_provider,
            )
            index_path = output_dir / "index.json"
            index.save(str(index_path))

        if llm_enabled:
            connections = suggest_connections(graph=graph, llm_fn=llm_fn, max_candidates=20)
            apply_connections(graph=graph, connections=connections)
            _write_graph(graph_path, graph)

    finally:
        if llm_temp_script is not None and Path(llm_temp_script).exists():
            Path(llm_temp_script).unlink()

    if args.json:
        payload = {"graph": str(graph_path), "texts": str(texts_path)}
        if args.embed_command is not None or embed_provider is not None:
            payload["index"] = str(output_dir / "index.json")
        if route_provider is not None:
            payload["route_provider"] = route_provider
        print(json.dumps(payload))
    else:
        print(f"graph_path: {graph_path}")
        print(f"texts_path: {texts_path}")
        if route_provider is not None:
            print(f"route_provider: {route_provider}")
        if args.embed_command is not None or embed_provider is not None:
            print(f"index_path: {output_dir / 'index.json'}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    texts = _load_texts_payload(args.texts)
    if args.embed_command is None and args.provider is None:
        args.provider = _auto_detect_provider()
        if args.provider:
            print(f"Auto-detected embedding provider: {args.provider}")
        else:
            raise SystemExit(
                "provide either --command or --provider, or configure OPENAI_API_KEY / Ollama availability"
            )

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

    if query_vec is None and index_path and not args.no_embed_query:
        embed_provider = _auto_detect_provider()
        if embed_provider is not None:
            query_vec = _embed_query_text(args.text, embed_command=None, embed_provider=embed_provider)

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

    route_fn = None
    route_temp_script = None
    llm_command = None
    llm_temp_script = None
    scores: dict[str, float] = {}
    created_node_id: str | None = None
    if args.no_route:
        if args.route_command is not None or args.route_provider is not None:
            raise SystemExit("use --no-route without --route-command or --route-provider")
        route_provider = None
    else:
        route_provider = args.route_provider
        if args.route_command is None and route_provider is None:
            route_provider = _auto_detect_provider()
            if route_provider:
                print(f"Auto-detected routing provider: {route_provider}")

    if args.route_command is not None or route_provider is not None:
        route_command, route_temp_script = _build_route_command(
            route_command=args.route_command,
            route_provider=route_provider,
        )

        def route_fn_impl(query_text: str | None, candidate_ids: list[str]) -> list[str]:
            payload = _build_route_candidates(graph=graph, candidate_ids=candidate_ids)
            return _run_route_command(
                command=route_command,
                query_text=query_text or "",
                candidates=payload,
            )

        route_fn = route_fn_impl

    if not args.no_route and args.route_command is None and route_provider is not None:
        llm_command, llm_temp_script = _build_llm_command(route_provider)

    def llm_fn(system_prompt: str, user_prompt: str) -> str:
        if llm_command is None:
            raise SystemExit("no llm route command configured")
        return _run_llm_command(llm_command, system_prompt, user_prompt)

    try:
        result = traverse(
            graph=graph,
            seeds=seeds,
            config=TraversalConfig(max_hops=15),
            query_text=args.text,
            route_fn=route_fn,
        )
        fired_ids = result.fired
        context = result.context

        if llm_command is not None and result.fired:
            fired_pairs = [
                (node_id, graph.get_node(node_id).content)
                for node_id in result.fired
                if graph.get_node(node_id)
            ]
            if fired_pairs:
                scores = score_retrieval(args.text, fired_pairs, llm_fn=llm_fn)

        if not fired_ids and llm_command is not None:
            created_node_id = maybe_create_node(graph, args.text, [node_id for node_id, _ in seeds], llm_fn=llm_fn)
            if created_node_id:
                fired_ids = [created_node_id]
                context = graph.get_node(created_node_id).content if graph.get_node(created_node_id) else ""
    finally:
        if route_temp_script is not None and Path(route_temp_script).exists():
            Path(route_temp_script).unlink()
        if llm_temp_script is not None and Path(llm_temp_script).exists():
            Path(llm_temp_script).unlink()

    if not args.no_log:
        log_query(query_text=args.text, fired_ids=fired_ids, node_count=graph.node_count())

    if created_node_id:
        _write_graph(args.graph, graph)

    if args.json:
        print(
            json.dumps(
                {
                    "fired": fired_ids,
                    "steps": [step.__dict__ for step in result.steps],
                    "context": context,
                    "scores": scores,
                    "created_node_id": created_node_id,
                }
            )
        )
    else:
        print(context)
        print()
        print("\"fired\":", fired_ids)
        if scores:
            print("\"scores\":", scores)
        if created_node_id:
            print("created_node_id:", created_node_id)
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    fired_ids = [value.strip() for value in args.fired_ids.split(",") if value.strip()]
    parsed_scores = _parse_score_payload(args.scores)
    if parsed_scores:
        relevant_scores = [parsed_scores[node_id] for node_id in fired_ids if node_id in parsed_scores]
        if relevant_scores:
            avg_score = sum(relevant_scores) / len(relevant_scores)
            outcome = max(-1.0, min(1.0, avg_score * 2.0 - 1.0))
        elif args.outcome is not None:
            outcome = args.outcome
        else:
            raise SystemExit("no matching scored nodes; provide --outcome")
    elif args.outcome is not None:
        outcome = args.outcome
    else:
        raise SystemExit("provide either --outcome or --scores")

    apply_outcome(graph, fired_nodes=fired_ids, outcome=outcome)
    if not args.no_log:
        log_learn(fired_ids=fired_ids, outcome=outcome)
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
    graph_path = _graph_path_arg(args.graph)
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload["graph"], indent=2))
    else:
        print(f"updated {args.graph}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)

    route_provider = args.route_provider
    if args.route_provider is None:
        route_provider = _auto_detect_provider()
        if route_provider:
            print(f"Auto-detected merge provider: {route_provider}", file=sys.stderr)

    llm_temp_script = None
    llm_fn = None
    if route_provider is not None:
        llm_command, llm_temp_script = _build_llm_command(route_provider)

        def route_aware_llm_fn(system_prompt: str, user_prompt: str) -> str:
            return _run_llm_command(llm_command, system_prompt, user_prompt)

        llm_fn = route_aware_llm_fn

    try:
        suggestions = suggest_merges(graph, llm_fn=llm_fn)
        applied: list[dict[str, list[str]]] = []
        for source_id, target_id in suggestions:
            if graph.get_node(source_id) is None or graph.get_node(target_id) is None:
                continue
            merged_id = apply_merge(graph, source_id, target_id)
            applied.append({"from": [source_id, target_id], "to": [merged_id]})
    finally:
        if llm_temp_script is not None and Path(llm_temp_script).exists():
            Path(llm_temp_script).unlink()

    _write_graph(args.graph, graph)
    payload = {
        "suggestions": [{"from": [source_id, target_id]} for source_id, target_id in suggestions],
        "applied": applied,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"Suggested merges: {len(suggestions)}")
        for pair in suggestions:
            print(f"{pair[0]} + {pair[1]}")
        print(f"Applied merges: {len(applied)}")
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
    if args.command == "merge":
        return cmd_merge(args)
    if args.command == "connect":
        return cmd_connect(args)
    return 1


def cmd_connect(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)

    route_provider = args.route_provider
    if route_provider is None:
        route_provider = _auto_detect_provider()
        if route_provider:
            print(f"Auto-detected connect provider: {route_provider}", file=sys.stderr)

    llm_temp_script = None
    llm_fn = None
    if route_provider is not None:
        llm_command, llm_temp_script = _build_llm_command(route_provider)

        def route_aware_llm_fn(system_prompt: str, user_prompt: str) -> str:
            return _run_llm_command(llm_command, system_prompt, user_prompt)

        llm_fn = route_aware_llm_fn

    try:
        suggestions = suggest_connections(
            graph=graph,
            llm_fn=llm_fn,
            max_candidates=args.max_candidates,
        )
        added = apply_connections(graph=graph, connections=suggestions)
    finally:
        if llm_temp_script is not None and Path(llm_temp_script).exists():
            Path(llm_temp_script).unlink()

    _write_graph(args.graph, graph)
    payload = {
        "suggestions": [
            {"source_id": source_id, "target_id": target_id, "weight": weight, "reason": reason}
            for source_id, target_id, weight, reason in suggestions
        ],
        "added": added,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"Suggested connections: {len(suggestions)}")
        print(f"Added edges: {added}")
        for item in suggestions:
            print(f"{item[0]} -> {item[1]} ({item[2]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
