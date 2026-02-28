"""Workspace splitter utilities for constructing an initial graph."""

from __future__ import annotations

import hashlib
import fnmatch
import os
import sys
import json
import re
from collections.abc import Callable
from collections.abc import Iterable
import threading
from pathlib import Path

from ._batch import batch_or_single
from .graph import Edge, Graph, Node
from ._util import _extract_json, _first_line


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
SPLIT_PROMPT = (
    "Split this document into coherent semantic sections. Each section should be a "
    "concept or topic. Return JSON: {\"sections\": [\"section1 text\", \"section2 text\"]}"
)
SUMMARY_PROMPT = 'Write a one-line summary for this node. Return JSON: {"summary": "..."}'
SUPPORTED_FILE_EXTENSIONS = (
    ".md",
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".rst",
    ".html",
)


def _extract_sections(raw: str) -> list[str]:
    """ extract sections."""
    try:
        payload = _extract_json(raw) or {}
        sections = payload.get("sections") if isinstance(payload, dict) else None
        if isinstance(sections, list):
            parsed = [str(section).strip() for section in sections if str(section).strip()]
            if parsed:
                return parsed
    except (Exception, SystemExit):  # noqa: BLE001
        pass
    return [] 


def _normalize_extensions(file_extensions: Iterable[str] | None = None) -> set[str]:
    """ normalize extensions."""
    if file_extensions is None:
        return set(SUPPORTED_FILE_EXTENSIONS)

    normalized = set()
    for extension in file_extensions:
        if not extension:
            continue
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    if not normalized:
        return set(SUPPORTED_FILE_EXTENSIONS)
    return normalized


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
        """ report."""
        nonlocal completed
        if summary_progress is None:
            return
        with summary_lock:
            completed += 1
            current = completed
        summary_progress(current, total_nodes)

    def _summary_worker(content: str, response: str) -> str:
        """ summary worker."""
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


def _chunk_python(content: str) -> list[str]:
    """ chunk python."""

    lines = content.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    saw_blocks = False

    def is_block_start(line: str) -> bool:
        """is block start."""
        return bool(re.match(r"^\s*(def|class)\s+\w+", line))

    for line in lines:
        if is_block_start(line):
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = [line]
            saw_blocks = True
            continue
        current.append(line)

    final = "\n".join(current).strip()
    if final:
        chunks.append(final)

    return chunks if saw_blocks else _chunk_markdown(content)


def _chunk_json(content: str) -> list[str]:
    """ chunk json."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return [content]

    if not isinstance(payload, dict):
        return [content]
    if len(payload) <= 1 or len(content) < 3200:
        return [content]

    chunks: list[str] = []
    for key, value in payload.items():
        chunks.append(f"{key}:\n{json.dumps(value, indent=2, ensure_ascii=False)}")
    return chunks


def _chunk_config_like(content: str) -> list[str]:
    """ chunk config like."""
    lines = content.splitlines()
    if len(content) < 3200 or len(lines) < 12:
        return [content]

    chunks: list[str] = []
    current: list[str] = []
    has_split = False
    for line in lines:
        is_key = bool(re.match(r"^\w[\w-]*\s*:", line)) and not line.startswith(" ")
        is_section = bool(re.match(r"^\[.+\]", line.strip()))
        if (is_key or is_section) and current:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
                has_split = True
            current = []
        current.append(line)

    tail = "\n".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks if has_split else [content]


_MAX_CHUNK_CHARS = 12_000  # ~3000 tokens, safely under embedding model limits
DEFAULT_LLM_SPLIT_MIN_CHARS = 7_000


def _rechunk_oversized(chunks: list[str], max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Re-split any chunks that exceed max_chars.

    Strategy: split on blank lines first, then on single newlines, then hard-cut.
    This ensures every chunk is under the embedding model's context limit.
    """
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
            continue

        # Try splitting on blank lines
        parts = [p.strip() for p in chunk.split("\n\n") if p.strip()]
        if len(parts) > 1:
            # Recursively rechunk the sub-parts
            result.extend(_rechunk_oversized(parts, max_chars))
            continue

        # Try splitting on single newlines
        lines = chunk.splitlines()
        if len(lines) > 1:
            current: list[str] = []
            current_len = 0
            for line in lines:
                if current_len + len(line) + 1 > max_chars and current:
                    result.append("\n".join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line) + 1
            if current:
                result.append("\n".join(current))
            continue

        # Hard cut as last resort
        for i in range(0, len(chunk), max_chars):
            result.append(chunk[i:i + max_chars])

    return result


def _sibling_weight(file_id: str, idx: int) -> float:
    """ sibling weight."""
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
    """ match gitignore."""
    for pattern in patterns:
        if pattern.startswith("!"):
            continue
        normalized = pattern.lstrip("./").replace("\\", "/")
        is_dir_pattern = normalized.endswith("/")
        if is_dir_pattern:
            normalized = normalized[:-1]
            # Match at root or any subdirectory level
            if path_posix == normalized or path_posix.startswith(normalized + "/") or ("/" + normalized + "/") in ("/" + path_posix + "/") or ("/" + normalized) in ("/" + path_posix):
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
    """ normalize excludes."""
    excludes = set(DEFAULT_EXCLUDES)
    if exclude is None:
        return excludes
    for item in exclude:
        value = item.strip()
        if value:
            excludes.add(value)
    return excludes


def _should_skip_path(relative_path: str, excludes: set[str], gitignore_patterns: list[str]) -> bool:
    """ should skip path."""
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
    file_extensions: Iterable[str] | None = None,
) -> tuple[Graph, dict[str, str]]:
    """Split workspace files and convert them into a graph.

    Args:
        workspace_dir: Directory containing source files.

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
    extensions = _normalize_extensions(file_extensions)
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
            if not file_path.is_file() or file_path.suffix.lower() not in extensions:
                continue
            if _should_skip_path(rel, excludes, gitignore_patterns):
                continue
            candidates.append((file_path, rel))

    split_plan: list[tuple[int, str, str, bool]] = []
    for file_path, rel in candidates:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            continue  # skip binary/non-UTF8 files
        use_llm = False
        if llm_fn is not None or llm_batch_fn is not None:
            if should_use_llm_for_file is None:
                use_llm = len(text) >= DEFAULT_LLM_SPLIT_MIN_CHARS
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
        """ report."""
        nonlocal split_count
        with split_lock:
            current = split_count + 1
            split_count = current
        message = f"Splitting {current}/{total_files}: {relative_path}"
        if split_progress is None:
            print(message, file=sys.stderr)
            return
        split_progress(current, total_files, relative_path, mode)

    llm_requests: list[dict] = []
    llm_request_map: dict[str, tuple[int, str, str]] = {}
    for idx, rel, text, use_llm in split_plan:
        if not use_llm:
            suffix = Path(rel).suffix.lower()
            if suffix == ".py":
                chunks = _chunk_python(text)
            elif suffix in {".txt", ".rst", ".md", ".html"}:
                chunks = _chunk_markdown(text)
            elif suffix == ".json":
                chunks = _chunk_json(text)
            elif suffix in {".yaml", ".yml", ".toml", ".cfg", ".ini"}:
                chunks = _chunk_config_like(text)
            else:
                chunks = _chunk_markdown(text)
            text_chunks_by_index[idx] = _rechunk_oversized(chunks)
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

    llm_total = len(llm_request_map)
    llm_index = 0
    for rel, (idx, source_rel, text) in llm_request_map.items():
        llm_index += 1
        print(f"LLM splitting {llm_index}/{llm_total}: {rel}", file=sys.stderr)
        response = response_by_id.get(rel, "")
        chunks = _extract_sections(response) if response else []
        mode = "llm" if chunks else "heuristic"
        if not chunks:
            suffix = Path(rel).suffix.lower()
            if suffix == ".py":
                chunks = _chunk_python(text)
            elif suffix in {".txt", ".rst", ".md", ".html"}:
                chunks = _chunk_markdown(text)
            elif suffix == ".json":
                chunks = _chunk_json(text)
            elif suffix in {".yaml", ".yml", ".toml", ".cfg", ".ini"}:
                chunks = _chunk_config_like(text)
            else:
                chunks = _chunk_markdown(text)
        # Ensure no chunk exceeds embedding model limits
        chunks = _rechunk_oversized(chunks)
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
