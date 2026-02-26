"""Workspace splitter utilities for constructing an initial graph."""

from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import re
from collections.abc import Callable
from collections.abc import Iterable
import threading
from pathlib import Path

from ._batch import batch_or_single
from .graph import Edge, Graph, Node


DEFAULT_EXCLUDES = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".cache",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "vendor",
    "target",
}

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)
SPLIT_PROMPT = (
    "Split this document into coherent semantic sections. Each section should be a "
    "concept or topic. Return JSON: {\"sections\": [\"section1 text\", \"section2 text\"]}"
)
SUMMARY_PROMPT = 'Write a one-line summary for this node. Return JSON: {"summary": "..."}'


def _extract_json(raw: str) -> dict | list | str:
    text = (raw or "").strip()
    if not text:
        return {}

    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _first_line(text: str) -> str:
    return (text.splitlines() or [""])[0]


def _extract_sections(raw: str) -> list[str]:
    try:
        payload = _extract_json(raw)
        sections = payload.get("sections") if isinstance(payload, dict) else None
        if isinstance(sections, list):
            parsed = [str(section).strip() for section in sections if str(section).strip()]
            if parsed:
                return parsed
    except (Exception, SystemExit):  # noqa: BLE001
        pass
    return []


def generate_summaries(
    graph: Graph,
    llm_fn: Callable[[str, str], str] | None = None,
    llm_batch_fn: Callable[[list[dict]], list[dict]] | None = None,
    llm_node_ids: set[str] | None = None,
    summary_progress: Callable[[int, int], None] | None = None,
    llm_parallelism: int = 8,
) -> dict[str, str]:
    """Generate one-line summaries for each node.

    If no LLM callback is provided, fall back to the first content line.
    """
    if llm_parallelism <= 0:
        raise ValueError("llm_parallelism must be >= 1")

    summaries: dict[str, str] = {}
    target_nodes = set(llm_node_ids) if llm_node_ids is not None else None
    nodes = list(graph.nodes())
    total_nodes = len(nodes)
    pending: list[tuple[str, str]] = []
    summary_lock = threading.Lock()
    completed = 0

    def _report(node_id: str | None = None) -> None:
        nonlocal completed
        if summary_progress is None:
            return
        with summary_lock:
            completed += 1
            current = completed
        summary_progress(current, total_nodes)

    def _summary_worker(content: str, response: str) -> str:
        payload = _extract_json(response)
        if isinstance(payload, dict):
            maybe_summary = payload.get("summary")
            if isinstance(maybe_summary, str):
                summary = maybe_summary.strip()
                if summary:
                    return summary
        return _first_line(content)

    requests: list[dict] = []
    has_llm = llm_fn is not None or llm_batch_fn is not None
    for node in nodes:
        if not has_llm or (target_nodes is not None and node.id not in target_nodes):
            summaries[node.id] = _first_line(node.content)
            _report(node.id)
            continue
        requests.append({"id": node.id, "system": SUMMARY_PROMPT, "user": node.content})
        pending.append((node.id, node.content))

    responses: list[dict] = []
    if requests:
        responses = batch_or_single(
            requests=requests,
            llm_fn=llm_fn,
            llm_batch_fn=llm_batch_fn,
            max_workers=llm_parallelism,
        )
    response_by_id = {str(item["id"]): str(item.get("response", "")) for item in responses}

    for node_id, content in pending:
        summary = _summary_worker(content, response_by_id.get(node_id, ""))
        summaries[node_id] = summary
        _report(node_id)
    return summaries


def _chunk_markdown(content: str) -> list[str]:
    """Split markdown content by level-2 headers.

    If there are no ``##`` headings, split on blank lines.
    """
    lines = content.splitlines()
    has_headers = any(line.startswith("## ") for line in lines)

    if not has_headers:
        parts = [part.strip() for part in content.split("\n\n") if part.strip()]
        return parts or [content]

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("## ") and current:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = [line]
        else:
            current.append(line)

    final = "\n".join(current).strip()
    if final:
        chunks.append(final)
    return chunks or [content]


def _sibling_weight(file_id: str, idx: int) -> float:
    """Return deterministic sibling baseline weight around ``0.5`` with tiny jitter."""
    digest = hashlib.sha256(f"{file_id}:{idx}".encode("utf-8")).hexdigest()[:8]
    jitter = (int(digest, 16) % 2001 - 1000) / 100000.0
    return max(0.4, min(0.6, 0.5 + jitter))


def _load_gitignore_patterns(workspace: Path) -> list[str]:
    """Load non-empty, non-comment patterns from ``.gitignore``.

    This is intentionally minimal but useful for production directories where default
    ignores are not enough.
    """
    gitignore = workspace / ".gitignore"
    if not gitignore.exists():
        return []
    raw_lines = gitignore.read_text(encoding="utf-8").splitlines()
    return [line.strip().replace("\\", "/") for line in raw_lines if line.strip() and not line.strip().startswith("#")]


def _match_gitignore(path_posix: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("!"):
            continue
        normalized = pattern.lstrip("./").replace("\\", "/")
        is_dir_pattern = normalized.endswith("/")
        if is_dir_pattern:
            normalized = normalized[:-1]
            if path_posix == normalized or path_posix.startswith(normalized + "/"):
                return True
            continue
        if pattern.startswith("/"):
            normalized = normalized[1:]
            if fnmatch.fnmatch(path_posix, normalized):
                return True
            continue
        if "/" in normalized:
            if fnmatch.fnmatch(path_posix, normalized):
                return True
            continue
        if fnmatch.fnmatch(Path(path_posix).name, normalized) or fnmatch.fnmatch(path_posix, f"*/{normalized}"):
            return True
    return False


def _normalize_excludes(exclude: Iterable[str] | None) -> set[str]:
    excludes = set(DEFAULT_EXCLUDES)
    if exclude is None:
        return excludes
    for item in exclude:
        value = item.strip()
        if value:
            excludes.add(value)
    return excludes


def _should_skip_path(relative_path: str, excludes: set[str], gitignore_patterns: list[str]) -> bool:
    if not relative_path:
        return False
    path = Path(relative_path)
    if any(part.startswith(".") for part in path.parts):
        return True
    normalized = str(path).replace("\\", "/").lstrip("./")
    for pattern in excludes:
        if path.name == pattern:
            return True
        if pattern.endswith("/"):
            if normalized == pattern[:-1] or normalized.startswith(pattern):
                return True
            continue
        if "*" in pattern or "/" in pattern:
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
            continue
        if fnmatch.fnmatch(path.name, pattern) or normalized == pattern:
            return True
    normalized = str(path).replace("\\", "/").lstrip("./")
    if _match_gitignore(normalized, gitignore_patterns):
        return True
    return False


def split_workspace(
    workspace_dir: str | Path,
    *,
    max_depth: int = 3,
    exclude: Iterable[str] | None = None,
    llm_fn: Callable[[str, str], str] | None = None,
    llm_batch_fn: Callable[[list[dict]], list[dict]] | None = None,
    should_use_llm_for_file: Callable[[str, str], bool] | None = None,
    split_progress: Callable[[int, int, str, str], None] | None = None,
    llm_parallelism: int = 8,
) -> tuple[Graph, dict[str, str]]:
    """Read markdown files from workspace and convert them into a graph.

    Args:
        workspace_dir: Directory containing markdown source files.

    Returns:
        ``(graph, texts)`` where each ``texts[node_id]`` is the chunk content for
        caller-provided embeddings.
    """
    workspace = Path(workspace_dir).expanduser()
    if not workspace.exists():
        raise FileNotFoundError(f"workspace not found: {workspace}")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    excludes = _normalize_excludes(exclude)
    gitignore_patterns = _load_gitignore_patterns(workspace)

    if llm_parallelism <= 0:
        raise ValueError("llm_parallelism must be >= 1")

    graph = Graph()
    texts: dict[str, str] = {}

    candidates: list[tuple[Path, str]] = []
    for dir_path, dir_names, file_names in os.walk(workspace):
        rel_dir = Path(dir_path).resolve().relative_to(workspace.resolve())
        depth = len(rel_dir.parts)
        if depth > max_depth:
            dir_names[:] = []
            continue
        if depth == max_depth:
            dir_names[:] = []

        for dir_name in sorted(list(dir_names)):
            rel = (rel_dir / dir_name).as_posix() if rel_dir.parts else dir_name
            if _should_skip_path(rel, excludes, gitignore_patterns):
                dir_names.remove(dir_name)

        for filename in sorted(file_names):
            rel = (rel_dir / filename).as_posix() if rel_dir.parts else filename
            file_path = Path(dir_path) / filename
            if not file_path.is_file() or file_path.suffix.lower() != ".md":
                continue
            if _should_skip_path(rel, excludes, gitignore_patterns):
                continue
            candidates.append((file_path, rel))

    split_plan: list[tuple[int, str, str, bool]] = []
    for file_path, rel in candidates:
        text = file_path.read_text(encoding="utf-8")
        use_llm = False
        if llm_fn is not None or llm_batch_fn is not None:
            if should_use_llm_for_file is None:
                use_llm = True
            else:
                try:
                    use_llm = bool(should_use_llm_for_file(rel, text))
                except Exception:
                    use_llm = False
        split_plan.append((len(split_plan), rel, text, use_llm))

    total_files = len(split_plan)
    text_chunks_by_index: list[list[str]] = [[] for _ in range(total_files)]
    split_lock = threading.Lock()
    split_count = 0

    def _report(relative_path: str, mode: str) -> None:
        nonlocal split_count
        if split_progress is None:
            return
        with split_lock:
            current = split_count + 1
            split_count = current
        split_progress(current, total_files, relative_path, mode)

    llm_requests: list[dict] = []
    llm_request_map: dict[str, tuple[int, str, str]] = {}
    for idx, rel, text, use_llm in split_plan:
        if not use_llm:
            text_chunks_by_index[idx] = _chunk_markdown(text)
            _report(rel, "heuristic")
            continue
        llm_requests.append({"id": rel, "system": SPLIT_PROMPT, "user": text})
        llm_request_map[rel] = (idx, rel, text)

    response_by_id: dict[str, str] = {}
    if llm_requests:
        responses = batch_or_single(
            requests=llm_requests,
            llm_fn=llm_fn,
            llm_batch_fn=llm_batch_fn,
            max_workers=llm_parallelism,
        )
        response_by_id = {str(item["id"]): str(item.get("response", "")) for item in responses}

    for rel, (idx, source_rel, text) in llm_request_map.items():
        response = response_by_id.get(rel, "")
        chunks = _extract_sections(response) if response else []
        mode = "llm" if chunks else "heuristic"
        if not chunks:
            chunks = _chunk_markdown(text)
        text_chunks_by_index[idx] = chunks
        _report(source_rel, mode)

    for idx, rel, _text, _use_llm in split_plan:
        chunks = text_chunks_by_index[idx]

        node_ids: list[str] = []
        for chunk_index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            node_id = f"{rel}::{chunk_index}"
            summary = chunk.splitlines()[0] if chunk.splitlines() else ""
            node = Node(
                id=node_id,
                content=chunk,
                summary=summary,
                metadata={"file": rel, "chunk": chunk_index, "kind": "markdown"},
            )
            graph.add_node(node)
            texts[node_id] = chunk
            node_ids.append(node_id)

        for source_offset, (source_id, target_id) in enumerate(zip(node_ids, node_ids[1:])):
            weight = _sibling_weight(rel, source_offset)
            graph.add_edge(Edge(source=source_id, target=target_id, weight=weight, kind="sibling"))
            graph.add_edge(Edge(source=target_id, target=source_id, weight=weight, kind="sibling"))

    return graph, texts
