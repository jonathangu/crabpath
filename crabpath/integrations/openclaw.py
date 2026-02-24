"""
CrabPath ↔ OpenClaw integration.

Bootstrap a CrabPath graph from an OpenClaw workspace:
- Memory files (MEMORY.md, memory/*.md)
- Learning harness corrections (learning/db/learning.db)
- Workspace config files (AGENTS.md, TOOLS.md, SOUL.md, USER.md)
- Active tasks (active-tasks.md)

This module parses OpenClaw workspace artifacts into typed CrabPath
nodes and edges, providing a warm start for the memory graph.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from ..graph import EdgeType, MemoryEdge, MemoryGraph, MemoryNode, NodeType


def import_workspace(
    graph: MemoryGraph,
    workspace: Path,
    learning_db: Optional[str] = None,
) -> dict:
    """
    Import an OpenClaw workspace into a CrabPath graph.

    Args:
        graph: Target MemoryGraph
        workspace: Path to OpenClaw workspace directory
        learning_db: Optional path to learning harness SQLite DB

    Returns:
        Dict with import statistics
    """
    stats = {"nodes_created": 0, "edges_created": 0, "sources": []}

    # 1. Import memory files as Fact nodes
    memory_dir = workspace / "memory"
    if memory_dir.exists():
        for md_file in sorted(memory_dir.glob("*.md")):
            n = _import_memory_file(graph, md_file)
            stats["nodes_created"] += n
        stats["sources"].append("memory/*.md")

    # 2. Import MEMORY.md sections as structured Facts
    memory_md = workspace / "MEMORY.md"
    if memory_md.exists():
        n = _import_structured_memory(graph, memory_md)
        stats["nodes_created"] += n
        stats["sources"].append("MEMORY.md")

    # 3. Import learning harness corrections as Rules + Error Classes
    if learning_db:
        db_path = Path(learning_db)
    else:
        db_path = workspace / "learning" / "db" / "learning.db"
    
    if db_path.exists():
        n_nodes, n_edges = _import_learning_harness(graph, db_path)
        stats["nodes_created"] += n_nodes
        stats["edges_created"] += n_edges
        stats["sources"].append("learning.db")

    # 4. Import tool docs as Tool nodes
    tools_md = workspace / "TOOLS.md"
    if tools_md.exists():
        n = _import_tools(graph, tools_md)
        stats["nodes_created"] += n
        stats["sources"].append("TOOLS.md")

    # 5. Import active tasks as Hub nodes
    active_tasks = workspace / "active-tasks.md"
    if active_tasks.exists():
        n = _import_active_tasks(graph, active_tasks)
        stats["nodes_created"] += n
        stats["sources"].append("active-tasks.md")

    # 6. Build co-occurrence edges between nodes that share tags
    n_edges = _build_tag_edges(graph)
    stats["edges_created"] += n_edges

    return stats


def _import_memory_file(graph: MemoryGraph, path: Path) -> int:
    """Import a daily memory file, splitting by ## sections."""
    content = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"^## ", content, flags=re.MULTILINE)
    count = 0

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
    date_tag = date_match.group(1) if date_match else path.stem

    for section in sections[1:]:  # skip header
        lines = section.strip().split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        if not body:
            continue

        node_id = f"memory-{date_tag}-{_slugify(title)}"
        
        # Detect type from content
        node_type = NodeType.FACT
        tags = [date_tag]
        
        if any(kw in title.lower() for kw in ["gate", "rule", "hard gate", "never"]):
            node_type = NodeType.RULE
            tags.append("gate")
        elif any(kw in title.lower() for kw in ["fix", "deploy", "build", "pipeline"]):
            tags.append("engineering")

        # Extract tags from content
        for tag in ["deploy", "config", "browser", "codex", "cron", "memory", "learning"]:
            if tag in body.lower():
                tags.append(tag)

        node = MemoryNode(
            id=node_id,
            node_type=node_type,
            content=body[:2000],  # cap content size
            summary=title,
            tags=list(set(tags)),
            prior=0.3,
        )
        graph.add_node(node)
        count += 1

    return count


def _import_structured_memory(graph: MemoryGraph, path: Path) -> int:
    """Import MEMORY.md, splitting by ## sections as high-level Facts/Hubs."""
    content = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"^## ", content, flags=re.MULTILINE)
    count = 0

    for section in sections[1:]:
        lines = section.strip().split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        if not body or len(body) < 20:
            continue

        node_id = f"longterm-{_slugify(title)}"
        
        # Long-term memory items are hubs (they connect to many things)
        node = MemoryNode(
            id=node_id,
            node_type=NodeType.HUB,
            content=body[:2000],
            summary=title,
            tags=["long-term"],
            prior=0.6,  # higher prior for long-term memory
        )
        graph.add_node(node)
        count += 1

    return count


def _import_learning_harness(graph: MemoryGraph, db_path: Path) -> tuple[int, int]:
    """Import corrections from the learning harness as Rule and ErrorClass nodes."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    nodes_created = 0
    edges_created = 0

    try:
        # Import active learnings
        rows = conn.execute("""
            SELECT id, category, subcategory, summary, details, severity, 
                   teacher, created_at
            FROM learnings 
            WHERE active = 1
        """).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return 0, 0

    for row in rows:
        learning_id = row["id"]
        severity = row["severity"] or "REF"
        category = row["category"] or "general"
        subcategory = row["subcategory"] or ""
        summary = row["summary"] or ""
        details = row["details"] or ""

        # GATE learnings become Rule nodes
        if severity == "GATE":
            node_type = NodeType.RULE
            tags = ["gate", "correction", category.lower()]
        else:
            node_type = NodeType.FACT
            tags = ["ref", "correction", category.lower()]

        if subcategory:
            tags.append(subcategory.lower())

        node = MemoryNode(
            id=f"learning-{learning_id}",
            node_type=node_type,
            content=f"{summary}\n\n{details}".strip(),
            summary=summary[:200],
            tags=list(set(tags)),
            prior=0.7 if severity == "GATE" else 0.4,
        )
        graph.add_node(node)
        nodes_created += 1

        # Create error class hub if category exists
        error_class_id = f"errorclass-{_slugify(category)}"
        if not graph.get_node(error_class_id):
            ec_node = MemoryNode(
                id=error_class_id,
                node_type=NodeType.ERROR_CLASS,
                content=f"Error class: {category}",
                summary=category,
                tags=["error-class"],
                prior=0.5,
            )
            graph.add_node(ec_node)
            nodes_created += 1

        # Edge: error class → learning
        edge = MemoryEdge(
            source=error_class_id,
            target=f"learning-{learning_id}",
            edge_type=EdgeType.ASSOCIATION,
            weight=0.8,
        )
        graph.add_edge(edge)
        edges_created += 1

    conn.close()
    return nodes_created, edges_created


def _import_tools(graph: MemoryGraph, path: Path) -> int:
    """Import TOOLS.md sections as Tool nodes."""
    content = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"^## ", content, flags=re.MULTILINE)
    count = 0

    for section in sections[1:]:
        lines = section.strip().split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        if not body or len(body) < 20:
            continue

        node = MemoryNode(
            id=f"tool-{_slugify(title)}",
            node_type=NodeType.TOOL,
            content=body[:2000],
            summary=title,
            tags=["tool"],
            prior=0.5,
        )
        graph.add_node(node)
        count += 1

    return count


def _import_active_tasks(graph: MemoryGraph, path: Path) -> int:
    """Import active-tasks.md items as Hub nodes."""
    content = path.read_text(encoding="utf-8", errors="replace")
    # Simple: extract numbered items
    items = re.findall(r"^\d+\.\s+\*\*(.+?)\*\*\s*[-—]?\s*(.*)", content, re.MULTILINE)
    count = 0

    for title, desc in items:
        node = MemoryNode(
            id=f"task-{_slugify(title)}",
            node_type=NodeType.HUB,
            content=f"{title}: {desc}".strip(),
            summary=title,
            tags=["active-task"],
            prior=0.8,  # active tasks get high prior
        )
        graph.add_node(node)
        count += 1

    return count


def _build_tag_edges(graph: MemoryGraph) -> int:
    """Build association edges between nodes that share tags."""
    # Index nodes by tag
    tag_index: dict[str, list[str]] = {}
    for nid, node in graph._nodes.items():
        for tag in node.tags:
            if tag in ("long-term", "correction", "gate", "ref"):
                continue  # skip overly broad tags
            tag_index.setdefault(tag, []).append(nid)

    edges_created = 0
    seen = set()

    for tag, node_ids in tag_index.items():
        if len(node_ids) > 20:
            continue  # skip tags that connect too many nodes (hub bias)
        
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                pair = (min(a, b), max(a, b))
                if pair in seen:
                    continue
                seen.add(pair)

                edge = MemoryEdge(
                    source=a,
                    target=b,
                    edge_type=EdgeType.ASSOCIATION,
                    weight=0.3,  # weak initial weight
                )
                graph.add_edge(edge)
                edges_created += 1

    return edges_created


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text[:60]
