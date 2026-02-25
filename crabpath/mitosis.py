"""
🦀 CrabPath Mitosis — LLM-Driven Graph Self-Organization

One cheap LLM does everything:
  1. Routing — "which node answers this query?"
  2. Splitting — "where are the natural boundaries?"
  3. Merging — "should these chunks be one node?"
  4. Neurogenesis — "this is new. Create a node."

No magic numbers. No cosine thresholds. No hardcoded chunk counts.
The LLM is the judgment. Decay is the only mechanical process.

Lifecycle:
  file → coherent sections (LLM decides how many) → decay separates
  → LLM merges what belongs together → LLM splits what grew too big
  → LLM creates nodes for novel concepts → repeat forever
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable

from .graph import Graph, Node, Edge


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MitosisConfig:
    """Configuration for graph self-organization."""
    sibling_weight: float = 0.65           # Initial sibling weight (habitual, not reflex)
    min_content_chars: int = 200           # Don't split nodes smaller than this
    chunk_type: str = "chunk"              # Node type for chunk nodes
    decay_rate: float = 0.01              # Edge decay rate for sibling edges


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Result of splitting a node."""
    parent_id: str
    chunk_ids: list[str]
    chunk_contents: list[str]
    edges_created: int
    generation: int


@dataclass
class MergeResult:
    """Result of merging nodes."""
    merged_id: str
    source_ids: list[str]
    content: str


@dataclass
class NeurogenesisResult:
    """Result of creating a new node."""
    node_id: str
    content: str
    connected_to: list[str]


@dataclass
class MaintenanceAction:
    """A single structural action proposed by the LLM."""
    action: str  # "split" | "merge" | "create" | "none"
    target_ids: list[str] = field(default_factory=list)
    reason: str = ""
    # For splits: the sections
    sections: list[str] = field(default_factory=list)
    # For creates: the new content
    new_content: str = ""
    # For merges: the merged content
    merged_content: str = ""


@dataclass
class MitosisState:
    """Tracks split history for the graph."""
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
# LLM Prompts — the LLM is the judgment for everything
# ---------------------------------------------------------------------------

SPLIT_SYSTEM_PROMPT = (
    "You split documents into coherent sections. "
    "Given a document, divide it into natural, self-contained sections. "
    "You decide how many sections — use as many as the content demands. "
    "Each section should be a coherent unit of related content. "
    "Prefer splitting at natural boundaries (headings, topic changes). "
    "Return JSON: {\"sections\": [\"section1 content\", \"section2 content\", ...]}"
)

SPLIT_USER_PROMPT = (
    "Split this document into coherent sections. "
    "You decide how many — split where the natural boundaries are. "
    "Preserve ALL content verbatim — do not summarize, omit, or rephrase anything.\n\n"
    "---\n{content}\n---"
)

MERGE_SYSTEM_PROMPT = (
    "You are a memory graph organizer. Given a set of content chunks that "
    "frequently co-activate (always needed together), decide if they should "
    "be merged into one node. "
    "Return JSON: {\"should_merge\": true/false, \"reason\": \"brief\", "
    "\"merged_content\": \"combined content if merging\"}"
)

MERGE_USER_PROMPT = (
    "These chunks always fire together — every query that needs one needs all of them. "
    "Should they be one node? Or do they serve different purposes that happen to overlap?\n\n"
    "{chunks}"
)

MAINTAIN_SYSTEM_PROMPT = (
    "You are a memory graph maintainer. Given the current state of some nodes "
    "and recent query patterns, decide what structural changes are needed. "
    "Options: split a large node, merge co-firing nodes, create a new node "
    "for an uncovered concept, or do nothing. "
    "Return JSON: {\"actions\": [{\"action\": \"split|merge|create|none\", "
    "\"target_ids\": [...], \"reason\": \"brief\"}]}"
)

NEUROGENESIS_SYSTEM_PROMPT = (
    "You are a memory graph builder. A query came in that doesn't match "
    "any existing node well. Decide if this represents a genuinely new concept "
    "that deserves its own node, or if it's noise/greeting/trivial. "
    "Return JSON: {\"should_create\": true/false, \"reason\": \"brief\", "
    "\"content\": \"node content if creating\", \"summary\": \"one line summary\"}"
)

NEUROGENESIS_USER_PROMPT = (
    "Query: {query}\n\n"
    "Closest existing nodes (with similarity scores):\n{existing}\n\n"
    "Should this be a new node? Or is it covered / too trivial?"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(parent_id: str, index: int, content: str) -> str:
    """Deterministic chunk ID from parent + index + content hash."""
    h = sha256(content.encode("utf-8")).hexdigest()[:8]
    return f"{parent_id}::chunk-{index}-{h}"


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from potentially markdown-wrapped LLM output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            if lines[-1].strip().endswith("```"):
                lines = lines[1:-1]
            cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _fallback_split(content: str) -> list[str]:
    """Split by markdown headers. No target count — use natural structure."""
    import re

    # Split by ## headers
    parts = re.split(r'\n(?=## )', content)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        return parts

    # Try # headers
    parts = re.split(r'\n(?=# )', content)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        return parts

    # Last resort: split by double newlines, merge small ones
    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        # Merge very small adjacent parts (< 100 chars)
        merged = [parts[0]]
        for p in parts[1:]:
            if len(merged[-1]) < 100:
                merged[-1] = merged[-1] + "\n\n" + p
            else:
                merged.append(p)
        if len(merged) >= 2:
            return merged

    return [content]


# ---------------------------------------------------------------------------
# Core Operations — all LLM-driven
# ---------------------------------------------------------------------------

def split_with_llm(
    content: str,
    llm_call: Callable[[str, str], str],
) -> list[str]:
    """Ask the cheap LLM to split content into coherent sections.

    The LLM decides how many sections. No magic number.
    Falls back to structural splitting if LLM fails.
    """
    try:
        raw = llm_call(SPLIT_SYSTEM_PROMPT, SPLIT_USER_PROMPT.format(content=content))
        parsed = _parse_json_response(raw)
        sections = parsed.get("sections", [])

        if isinstance(sections, list) and len(sections) >= 2:
            sections = [str(s).strip() for s in sections if str(s).strip()]
            if len(sections) >= 2:
                return sections
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        pass

    return _fallback_split(content)


def should_merge(
    chunks: list[tuple[str, str]],
    llm_call: Callable[[str, str], str],
) -> tuple[bool, str, str]:
    """Ask the LLM if co-firing chunks should be merged.

    Args:
        chunks: List of (chunk_id, chunk_content) pairs.
        llm_call: The cheap LLM.

    Returns:
        (should_merge, reason, merged_content)
    """
    chunks_text = "\n\n".join(
        f"--- {cid} ---\n{content[:500]}" for cid, content in chunks
    )

    try:
        raw = llm_call(MERGE_SYSTEM_PROMPT, MERGE_USER_PROMPT.format(chunks=chunks_text))
        parsed = _parse_json_response(raw)

        do_merge = parsed.get("should_merge", False)
        reason = parsed.get("reason", "")
        merged = parsed.get("merged_content", "")

        if do_merge and not merged:
            # LLM said merge but didn't provide content — concatenate
            merged = "\n\n".join(content for _, content in chunks)

        return bool(do_merge), str(reason), str(merged)
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        return False, "llm_error", ""


def should_create_node(
    query: str,
    existing_matches: list[tuple[str, float, str]],
    llm_call: Callable[[str, str], str],
) -> tuple[bool, str, str, str]:
    """Ask the LLM if a query warrants a new node (neurogenesis).

    Args:
        query: The query that didn't match well.
        existing_matches: List of (node_id, score, summary) of closest matches.
        llm_call: The cheap LLM.

    Returns:
        (should_create, reason, content, summary)
    """
    existing_text = "\n".join(
        f"  {nid} (score={score:.2f}): {summary[:80]}"
        for nid, score, summary in existing_matches
    )

    try:
        raw = llm_call(
            NEUROGENESIS_SYSTEM_PROMPT,
            NEUROGENESIS_USER_PROMPT.format(query=query, existing=existing_text),
        )
        parsed = _parse_json_response(raw)

        do_create = parsed.get("should_create", False)
        reason = parsed.get("reason", "")
        content = parsed.get("content", query)
        summary = parsed.get("summary", query[:80])

        return bool(do_create), str(reason), str(content), str(summary)
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        return False, "llm_error", "", ""


# ---------------------------------------------------------------------------
# Graph Operations
# ---------------------------------------------------------------------------

def split_node(
    graph: Graph,
    node_id: str,
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
    embed_callback: Callable[[str, str], None] | None = None,
) -> SplitResult | None:
    """Split a node into coherent sections using the cheap LLM.

    The LLM decides how many sections. No hardcoded count.
    """
    config = config or MitosisConfig()
    node = graph.get_node(node_id)
    if node is None:
        return None

    if len(node.content) < config.min_content_chars:
        return None

    # LLM decides the split
    sections = split_with_llm(node.content, llm_call)
    if len(sections) < 2:
        return None

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
        if node.metadata.get("protected"):
            chunk_node.metadata["protected"] = True

        graph.add_node(chunk_node)
        if embed_callback is not None:
            embed_callback(chunk_id, section_content)
        chunk_ids.append(chunk_id)

    # Sibling edges (all-to-all at weight 1.0)
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

    # Transfer parent edges to all chunks
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

    # Remove parent
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


def find_co_firing_families(
    graph: Graph,
    state: MitosisState,
) -> list[tuple[str, list[str]]]:
    """Find families where all sibling edges are strong (co-firing).

    Returns list of (parent_id, chunk_ids) that may need merging.
    """
    co_firing = []

    for parent_id, chunk_ids in state.families.items():
        alive = [cid for cid in chunk_ids if graph.get_node(cid) is not None]
        if len(alive) < 2:
            continue

        # Check if all sibling edges are strong
        all_strong = True
        for i, src in enumerate(alive):
            for j, tgt in enumerate(alive):
                if i == j:
                    continue
                edge = graph.get_edge(src, tgt)
                if edge is None or edge.weight < 0.90:
                    all_strong = False
                    break
            if not all_strong:
                break

        if all_strong:
            co_firing.append((parent_id, alive))

    return co_firing


def merge_nodes(
    graph: Graph,
    parent_id: str,
    chunk_ids: list[str],
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
    embed_callback: Callable[[str, str], None] | None = None,
) -> MergeResult | None:
    """Ask the LLM if co-firing chunks should merge. If yes, merge them.

    After merging, the merged node may get split again if it's big enough —
    but the LLM might split it differently this time.
    """
    config = config or MitosisConfig()

    chunks = []
    for cid in chunk_ids:
        node = graph.get_node(cid)
        if node is not None:
            chunks.append((cid, node.content))

    if len(chunks) < 2:
        return None

    # Ask the LLM
    do_merge, reason, merged_content = should_merge(chunks, llm_call)

    if not do_merge:
        return None

    if not merged_content:
        # Reassemble in order
        ordered = []
        for cid in chunk_ids:
            node = graph.get_node(cid)
            if node:
                idx = node.metadata.get("chunk_index", 0)
                ordered.append((idx, node.content))
        ordered.sort(key=lambda x: x[0])
        merged_content = "\n\n".join(c for _, c in ordered)

    # Create merged node
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
    for cid in chunk_ids:
        graph.remove_node(cid)
        state.chunk_to_parent.pop(cid, None)

    graph.add_node(merged_node)
    state.families.pop(parent_id, None)
    if embed_callback is not None:
        embed_callback(parent_id, merged_content)

    return MergeResult(
        merged_id=parent_id,
        source_ids=chunk_ids,
        content=merged_content,
    )


def create_node(
    graph: Graph,
    query: str,
    existing_matches: list[tuple[str, float, str]],
    llm_call: Callable[[str, str], str],
    fired_node_ids: list[str] | None = None,
) -> NeurogenesisResult | None:
    """Ask the LLM if a query warrants a new node. If yes, create it.

    No cosine thresholds. The LLM decides if the concept is novel.
    """
    do_create, reason, content, summary = should_create_node(
        query, existing_matches, llm_call
    )

    if not do_create:
        return None

    h = sha256(content.encode("utf-8")).hexdigest()[:12]
    node_id = f"auto:{h}"

    if graph.get_node(node_id) is not None:
        return None  # Already exists

    node = Node(
        id=node_id,
        content=content,
        summary=summary,
        type="fact",
        threshold=0.8,
        metadata={
            "source": "neurogenesis-llm",
            "query": query,
            "reason": reason,
            "created_ts": time.time(),
            "fired_count": 0,
            "last_fired_ts": 0.0,
        },
    )
    graph.add_node(node)

    # Connect to whatever fired during this query
    connected = []
    for nid in (fired_node_ids or [])[:5]:
        if nid != node_id and graph.get_node(nid) is not None:
            graph.add_edge(Edge(source=nid, target=node_id, weight=0.15, created_by="auto"))
            graph.add_edge(Edge(source=node_id, target=nid, weight=0.15, created_by="auto"))
            connected.append(nid)

    return NeurogenesisResult(
        node_id=node_id,
        content=content,
        connected_to=connected,
    )


# ---------------------------------------------------------------------------
# Bootstrap: Carbon Copy Workspace Files
# ---------------------------------------------------------------------------

def bootstrap_workspace(
    graph: Graph,
    workspace_files: dict[str, str],
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
    embed_callback: Callable[[str, str], None] | None = None,
) -> list[SplitResult]:
    """Bootstrap CrabPath from workspace files.

    1. Each file becomes a node (carbon copy, verbatim)
    2. The cheap LLM splits each into coherent sections (it decides how many)
    3. Sibling chunks get all-to-all edges at weight 1.0
    """
    config = config or MitosisConfig()
    results = []

    for file_id, content in workspace_files.items():
        file_node = Node(
            id=file_id,
            content=content,
            summary=f"Workspace file: {file_id}",
            type="workspace_file",
            metadata={
                "source": "workspace-bootstrap",
                "original_file": file_id,
                "created_ts": time.time(),
                "fired_count": 0,
                "last_fired_ts": 0.0,
            },
        )
        graph.add_node(file_node)

        result = split_node(
            graph,
            file_id,
            llm_call,
            state,
            config,
            embed_callback=embed_callback,
        )
        if result:
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Maintenance: The Unified Loop
# ---------------------------------------------------------------------------

def mitosis_maintenance(
    graph: Graph,
    llm_call: Callable[[str, str], str],
    state: MitosisState,
    config: MitosisConfig | None = None,
    embed_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run the full maintenance cycle. One cheap LLM does everything.

    1. Find co-firing families → ask LLM if they should merge
    2. Find big leaf nodes → ask LLM to split them
    3. After merging, re-split if the merged node is big enough

    Call periodically (every N queries, alongside decay).
    """
    config = config or MitosisConfig()

    merges = []
    splits = []

    # 1. Find co-firing families and ask LLM about merging
    co_firing = find_co_firing_families(graph, state)
    for parent_id, chunk_ids in co_firing:
        merge_result = merge_nodes(
            graph,
            parent_id,
            chunk_ids,
            llm_call,
            state,
            config,
            embed_callback=embed_callback,
        )
        if merge_result:
            merges.append(merge_result)

            # After merging, re-split if big enough
            merged_node = graph.get_node(merge_result.merged_id)
            if merged_node and len(merged_node.content) >= config.min_content_chars:
                split_result = split_node(
                    graph,
                    merge_result.merged_id,
                    llm_call,
                    state,
                    config,
                    embed_callback=embed_callback,
                )
                if split_result:
                    splits.append(split_result)

    # 2. Find big leaf nodes (not in any family) that could be split
    family_chunks = set()
    for chunk_ids in state.families.values():
        family_chunks.update(chunk_ids)

    for node in graph.nodes():
        if node.id in family_chunks:
            continue  # Already part of a family
        if node.id in state.families:
            continue  # Is a tracked parent
        if len(node.content) >= config.min_content_chars * 3:
            # Big standalone node — ask LLM to split
            split_result = split_node(
                graph,
                node.id,
                llm_call,
                state,
                config,
                embed_callback=embed_callback,
            )
            if split_result:
                splits.append(split_result)

    return {
        "merges": len(merges),
        "splits": len(splits),
        "merge_results": merges,
        "split_results": splits,
    }
