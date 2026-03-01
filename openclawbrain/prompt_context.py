"""Deterministic prompt-context block formatting for retrieval output."""

from __future__ import annotations

import re
from typing import Any

from .graph import Graph


def _as_int(value: Any) -> int | None:
    """Best-effort conversion to integer line numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _line_bounds(metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract start/end line values from common metadata fields."""
    start = _as_int(metadata.get("start_line"))
    end = _as_int(metadata.get("end_line"))

    if start is not None or end is not None:
        return start, end

    line_range = metadata.get("line_range")
    if isinstance(line_range, (list, tuple)):
        if len(line_range) >= 2:
            return _as_int(line_range[0]), _as_int(line_range[1])
        if len(line_range) == 1:
            value = _as_int(line_range[0])
            return value, value

    if isinstance(line_range, dict):
        start_value = _as_int(line_range.get("start"))
        end_value = _as_int(line_range.get("end"))
        if start_value is not None or end_value is not None:
            return start_value, end_value

    if isinstance(line_range, str):
        matches = [int(value) for value in re.findall(r"\d+", line_range)]
        if len(matches) >= 2:
            return matches[0], matches[1]
        if len(matches) == 1:
            return matches[0], matches[0]

    return None, None


def _source_path(metadata: dict[str, Any]) -> str | None:
    """Extract source-path metadata from common keys."""
    for key in ("path", "file", "source"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _citation(path: str | None, start_line: int | None, end_line: int | None) -> str | None:
    """Build source citation text when path metadata is available."""
    if not path:
        return None
    if start_line is not None and end_line is not None:
        return f"{path}#L{start_line}-L{end_line}"
    if start_line is not None:
        return f"{path}#L{start_line}"
    if end_line is not None:
        return f"{path}#L{end_line}"
    return path


def _sort_key(graph: Graph, node_id: str) -> tuple[int, str, int, str]:
    """Sort by source metadata when available, otherwise by node id."""
    node = graph.get_node(node_id)
    if node is None or not isinstance(node.metadata, dict):
        return (1, "", 0, node_id)

    path = _source_path(node.metadata)
    if path is None:
        return (1, "", 0, node_id)

    start_line, _end_line = _line_bounds(node.metadata)
    return (0, path, start_line if start_line is not None else 0, node_id)


def _format_entry(
    graph: Graph,
    node_id: str,
    *,
    include_node_ids: bool,
) -> str | None:
    """Render one deterministic context entry."""
    node = graph.get_node(node_id)
    if node is None:
        return None

    metadata = node.metadata if isinstance(node.metadata, dict) else {}
    path = _source_path(metadata)
    start_line, end_line = _line_bounds(metadata)
    citation = _citation(path, start_line, end_line)

    lines: list[str] = []
    if include_node_ids:
        lines.append(f"- node: {node_id}")
    else:
        lines.append("- snippet")
    if citation is not None:
        lines.append(f"  source: {citation}")

    content = node.content or ""
    if content:
        lines.extend(f"  {line}" for line in content.splitlines())
    else:
        lines.append("  ")
    return "\n".join(lines)


def build_prompt_context(
    graph: Graph,
    node_ids: list[str],
    *,
    max_chars: int | None = 20000,
    include_node_ids: bool = True,
) -> str:
    """Build a deterministic, cache-friendly context appendix block."""
    header = "[BRAIN_CONTEXT v1]"
    footer = "[/BRAIN_CONTEXT]"

    unique_ids = sorted(set(node_ids), key=lambda item: _sort_key(graph, item))
    entries = [
        entry
        for entry in (
            _format_entry(graph, node_id, include_node_ids=include_node_ids)
            for node_id in unique_ids
        )
        if entry is not None
    ]

    if entries:
        body = "\n\n".join(entries)
        rendered = f"{header}\n{body}\n{footer}"
    else:
        rendered = f"{header}\n{footer}"

    if max_chars is None:
        return rendered
    if max_chars <= 0:
        return ""
    if len(rendered) <= max_chars:
        return rendered

    minimum = len(header) + 1 + len(footer)
    if max_chars <= minimum:
        return rendered[:max_chars]

    body_budget = max_chars - minimum
    trimmed_body = ""
    for entry in entries:
        candidate = entry if not trimmed_body else f"{trimmed_body}\n\n{entry}"
        if len(candidate) <= body_budget:
            trimmed_body = candidate
            continue
        if not trimmed_body and body_budget > 0:
            trimmed_body = entry[:body_budget]
        break

    if trimmed_body:
        return f"{header}\n{trimmed_body}\n{footer}"
    return f"{header}\n{footer}"[:max_chars]
