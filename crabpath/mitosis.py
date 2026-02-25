"""
🦀 CrabPath Mitosis — Recursive Cell Division

Every file starts as one node. The cheap LLM splits it into 4 chunks.
All chunks get sibling edges at weight 1.0 (behave as one unit).
Over time, non-co-fired edges decay. When all sibling edges reconverge
to 1.0 (always co-fire), the node is functionally monolithic again —
re-split it with the LLM for finer granularity.

Lifecycle:
  file → 4 chunks (edges=1.0) → decay separates → some reconverge → re-split → repeat

The graph finds its own resolution. No heatmaps. No thresholds.
Just edges and decay.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .graph import Graph, Node, Edge


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MitosisConfig:
    """Configuration for cell division."""
    num_chunks: int = 4                    # Split each node into N chunks
    sibling_weight: float = 1.0            # Initial edge weight between siblings
    reconverge_threshold: float = 0.95     # Weight above which siblings are "merged"
    min_content_chars: int = 200           # Don't split nodes smaller than this
    parent_type: str = "workspace_file"    # Node type for parent placeholders
    chunk_type: str = "chunk"              # Node type for chunk nodes
    decay_rate: float = 0.01              # Edge decay rate for sibling edges


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Result of splitting a node into chunks."""
    parent_id: str
    chunk_ids: list[str]
    chunk_contents: list[str]
    edges_created: int
    generation: int  # How many times this content has been split


@dataclass
class MitosisState:
    """Tracks split history for reconvergence detection."""
    # parent_id -> list of chunk_ids
    families: dict[str, list[str]] = field(default_factory=dict)
    # parent_id -> generation count
    generations: dict[str, int] = field(default_factory=dict)
    # chunk_id -> parent_id
    chunk_to_parent: dict[str, str] = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "families": self.families,
                "generations": self.generations,
                "chunk_to_parent": self.chunk_to_parent,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> MitosisState:
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                families=data.get("families", {}),
                generations=data.get("generations", {}),
                chunk_to_parent=data.get("chunk_to_parent", {}),
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


# ---------------------------------------------------------------------------
# LLM Splitting
# ---------------------------------------------------------------------------

SPLIT_SYSTEM_PROMPT = (
    "You split documents into coherent sections. "
    "Given a document, divide it into exactly {n} sections. "
    "Each section should be a self-contained, coherent unit of related content. "
    "Prefer splitting at natural boundaries (headings, topic changes). "
    "Return JSON: {{\"sections\": [\"section1 content\", \"section2 content\", ...]}}"
)

SPLIT_USER_PROMPT = (
    "Split this document into exactly {n} coherent sections. "
    "Preserve ALL content verbatim — do not summarize, omit, or rephrase anything. "
    "Each section should be a self-contained topic group.\n\n"
    "---\n{content}\n---"
)


def _make_chunk_id(parent_id: str, index: int, content: str) -> str:
    """Deterministic chunk ID from parent + index + content hash."""
    h = sha256(content.encode("utf-8")).hexdigest()[:8]
    return f"{parent_id}::chunk-{index}-{h}"


def split_with_llm(
    content: str,
    num_chunks: int,
    llm_call: Callable[[str, str], str],
) -> list[str]:
    """Ask the cheap LLM to split content into N coherent sections.

    Args:
        content: The full text to split.
        num_chunks: Number of sections to produce.
        llm_call: Callable(system_prompt, user_prompt) -> raw LLM output string.

    Returns:
        List of section strings. Falls back to character-based splitting if LLM fails.
    """
    system = SPLIT_SYSTEM_PROMPT.format(n=num_chunks)
    user = SPLIT_USER_PROMPT.format(n=num_chunks, content=content)

    try:
        raw = llm_call(system, user)
        # Parse JSON from potentially markdown-wrapped output
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                if lines[-1].strip().endswith("```"):
                    lines = lines[1:-1]
                cleaned = "\n".join(lines).strip()

        parsed = json.loads(cleaned)
        sections = parsed.get("sections", [])

        if isinstance(sections, list) and len(sections) >= 2:
            # Validate: all sections are non-empty strings
            sections = [str(s).strip() for s in sections if str(s).strip()]
            if len(sections) >= 2:
                return sections
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        pass

    # Fallback: split by markdown headers, then by size
    return _fallback_split(content, num_chunks)


def _fallback_split(content: str, num_chunks: int) -> list[str]:
    """Split by markdown headers. If not enough headers, split by character count."""
    import re

    # Try splitting by ## headers first
    parts = re.split(r'\n(?=## )', content)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= num_chunks:
        # Merge smallest adjacent parts until we have num_chunks
        while len(parts) > num_chunks:
            # Find the two smallest adjacent parts
            min_combined = float('inf')
            min_idx = 0
            for i in range(len(parts) - 1):
                combined = len(parts[i]) + len(parts[i + 1])
                if combined < min_combined:
                    min_combined = combined
                    min_idx = i
            parts[min_idx] = parts[min_idx] + "\n\n" + parts[min_idx + 1]
            parts.pop(min_idx + 1)
        return parts

    # Not enough headers — split by character count
    chunk_size = max(len(content) // num_chunks, 1)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(content)
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks if chunks else [content]


# ---------------------------------------------------------------------------
# Core Mitosis Operations
# ---------------------------------------------------------------------------

def split_node(
    graph: Graph,
    node_id: str,
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
) -> SplitResult | None:
    """Split a node into N chunks using the cheap LLM.

    Creates chunk nodes, connects them as siblings (all-to-all, weight=1.0),
    and removes the original node (replacing with chunks).

    Returns None if the node is too small to split.
    """
    config = config or MitosisConfig()
    node = graph.get_node(node_id)
    if node is None:
        return None

    if len(node.content) < config.min_content_chars:
        return None

    # Split content
    sections = split_with_llm(node.content, config.num_chunks, llm_call)
    if len(sections) < 2:
        return None

    # Determine generation
    generation = state.generations.get(node_id, 0) + 1

    # Create chunk nodes
    chunk_ids = []
    for i, section_content in enumerate(sections):
        chunk_id = _make_chunk_id(node_id, i, section_content)
        chunk_node = Node(
            id=chunk_id,
            content=section_content,
            summary=section_content[:80].replace("\n", " "),
            type=config.chunk_type,
            threshold=node.threshold,
            metadata={
                "parent_id": node_id,
                "chunk_index": i,
                "generation": generation,
                "source": "mitosis",
                "created_ts": time.time(),
                "fired_count": 0,
                "last_fired_ts": 0.0,
            },
        )
        # Carry over protection from parent
        if node.metadata.get("protected"):
            chunk_node.metadata["protected"] = True

        graph.add_node(chunk_node)
        chunk_ids.append(chunk_id)

    # Create sibling edges (all-to-all between chunks)
    edges_created = 0
    for i, src_id in enumerate(chunk_ids):
        for j, tgt_id in enumerate(chunk_ids):
            if i != j:
                graph.add_edge(Edge(
                    source=src_id,
                    target=tgt_id,
                    weight=config.sibling_weight,
                    decay_rate=config.decay_rate,
                    created_by="auto",
                ))
                edges_created += 1

    # Transfer incoming/outgoing edges from parent to all chunks
    for src_node, edge in graph.incoming(node_id):
        for chunk_id in chunk_ids:
            graph.add_edge(Edge(
                source=edge.source,
                target=chunk_id,
                weight=edge.weight,
                decay_rate=edge.decay_rate,
                created_by="auto",
            ))

    for tgt_node, edge in graph.outgoing(node_id):
        for chunk_id in chunk_ids:
            graph.add_edge(Edge(
                source=chunk_id,
                target=edge.target,
                weight=edge.weight,
                decay_rate=edge.decay_rate,
                created_by="auto",
            ))

    # Remove the parent node
    graph.remove_node(node_id)

    # Update state
    state.families[node_id] = chunk_ids
    state.generations[node_id] = generation
    for chunk_id in chunk_ids:
        state.chunk_to_parent[chunk_id] = node_id

    return SplitResult(
        parent_id=node_id,
        chunk_ids=chunk_ids,
        chunk_contents=sections,
        edges_created=edges_created,
        generation=generation,
    )


def check_reconvergence(
    graph: Graph,
    state: MitosisState,
    config: MitosisConfig | None = None,
) -> list[str]:
    """Find families where all sibling edges have reconverged (weight ≈ 1.0).

    These nodes always co-fire — they're functionally one unit again.
    Returns list of parent_ids that should be re-split.
    """
    config = config or MitosisConfig()
    reconverged = []

    for parent_id, chunk_ids in state.families.items():
        # Check all chunks still exist
        alive = [cid for cid in chunk_ids if graph.get_node(cid) is not None]
        if len(alive) < 2:
            continue

        # Check all sibling edges are at or above threshold
        all_converged = True
        for i, src in enumerate(alive):
            for j, tgt in enumerate(alive):
                if i == j:
                    continue
                edge = graph.get_edge(src, tgt)
                if edge is None or edge.weight < config.reconverge_threshold:
                    all_converged = False
                    break
            if not all_converged:
                break

        if all_converged:
            reconverged.append(parent_id)

    return reconverged


def merge_and_resplit(
    graph: Graph,
    parent_id: str,
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
) -> SplitResult | None:
    """Merge reconverged chunks back together, then re-split with LLM.

    The LLM may split differently this time — that's the point.
    The graph refines its granularity through repeated division.
    """
    config = config or MitosisConfig()
    chunk_ids = state.families.get(parent_id, [])

    # Gather content from all living chunks (in order)
    chunks_with_index = []
    for cid in chunk_ids:
        node = graph.get_node(cid)
        if node is not None:
            idx = node.metadata.get("chunk_index", 0)
            chunks_with_index.append((idx, node.content, cid))

    if not chunks_with_index:
        return None

    # Reassemble in original order
    chunks_with_index.sort(key=lambda x: x[0])
    merged_content = "\n\n".join(content for _, content, _ in chunks_with_index)

    # Create a temporary merged node
    merged_node = Node(
        id=parent_id,
        content=merged_content,
        summary=merged_content[:80].replace("\n", " "),
        type=config.chunk_type,
        metadata={
            "source": "mitosis-merge",
            "generation": state.generations.get(parent_id, 0),
            "created_ts": time.time(),
            "fired_count": 0,
            "last_fired_ts": 0.0,
        },
    )

    # Remove old chunks
    for _, _, cid in chunks_with_index:
        graph.remove_node(cid)
        state.chunk_to_parent.pop(cid, None)

    # Add merged node and re-split
    graph.add_node(merged_node)
    return split_node(graph, parent_id, llm_call, state, config)


# ---------------------------------------------------------------------------
# Bootstrap: Carbon Copy Workspace Files
# ---------------------------------------------------------------------------

def bootstrap_workspace(
    graph: Graph,
    workspace_files: dict[str, str],
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
) -> list[SplitResult]:
    """Bootstrap CrabPath from workspace files.

    1. Each file becomes a node (carbon copy, verbatim)
    2. Each node is immediately split into N chunks by the cheap LLM
    3. Sibling chunks get all-to-all edges at weight 1.0

    Args:
        graph: The CrabPath graph to populate.
        workspace_files: Dict of {file_id: file_content}.
        llm_call: Callable(system_prompt, user_prompt) -> raw LLM output.
        state: MitosisState to track families.
        config: MitosisConfig.

    Returns:
        List of SplitResults, one per file.
    """
    config = config or MitosisConfig()
    results = []

    for file_id, content in workspace_files.items():
        # Step 1: Add the file as a monolithic node
        file_node = Node(
            id=file_id,
            content=content,
            summary=f"Workspace file: {file_id}",
            type=config.parent_type,
            metadata={
                "source": "workspace-bootstrap",
                "original_file": file_id,
                "created_ts": time.time(),
                "fired_count": 0,
                "last_fired_ts": 0.0,
            },
        )
        graph.add_node(file_node)

        # Step 2: Immediately split into chunks
        result = split_node(graph, file_id, llm_call, state, config)
        if result:
            results.append(result)

    # Step 3: Create cross-file edges between all chunks (low weight)
    # These allow the router to jump between files
    all_chunk_ids = []
    for result in results:
        all_chunk_ids.extend(result.chunk_ids)

    # Don't do all-to-all for cross-file — too dense.
    # Instead, rely on embedding seeding for cross-file routing.

    return results


# ---------------------------------------------------------------------------
# Maintenance: Run as part of the query loop
# ---------------------------------------------------------------------------

def mitosis_maintenance(
    graph: Graph,
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
) -> dict[str, Any]:
    """Check for reconvergence and re-split as needed.

    Call this periodically (e.g., every N queries alongside decay).
    """
    config = config or MitosisConfig()
    reconverged = check_reconvergence(graph, state, config)

    resplit_results = []
    for parent_id in reconverged:
        result = merge_and_resplit(graph, parent_id, llm_call, state, config)
        if result:
            resplit_results.append(result)

    return {
        "reconverged_families": len(reconverged),
        "resplit_count": len(resplit_results),
        "resplit_results": resplit_results,
    }
