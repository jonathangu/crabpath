"""
🦀 CrabPath Migration — Bootstrap + Playback

For new users migrating to CrabPath:
  1. Load workspace injection files (carbon copy)
  2. Split via cheap LLM
  3. Optionally replay session logs to warm up the graph

The replay accelerates graph formation — instead of waiting for
100+ live queries to form cross-file edges, replay your recent
history and get a pre-warmed graph immediately.

Usage:
  from crabpath.migrate import migrate

  graph, state = migrate(
      workspace_dir="~/.openclaw/workspace",
      session_logs=["session1.jsonl", "session2.jsonl"],  # optional
      llm_call=my_cheap_llm,
  )
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ._structural_utils import split_fallback_sections
from .autotune import suggest_config
from .decay import DecayConfig, apply_decay
from .graph import Graph
from .mitosis import MitosisConfig, MitosisState, bootstrap_workspace
from .synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    decay_proto_edges,
    edge_tier_stats,
    record_cofiring,
    record_skips,
)

logger = logging.getLogger(__name__)


class CrabPathError(ValueError):
    """Raised for migration-time data and workspace errors."""


def _is_system_noise(text: str) -> bool:
    """Return true for non-query transcripts like system/assistant/tool lines."""
    lowered = text.strip().lower()
    noise_prefixes = ("system:", "[system", "assistant:", "[assistant", "tool:", "[tool")
    return lowered.startswith(noise_prefixes) or len(lowered) < 3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MigrateConfig:
    """Configuration for migration."""

    # Which files to include
    injection_files: list[str] | None = None  # None = auto-detect
    include_memory: bool = True  # Include memory/*.md
    include_docs: bool = False  # Include docs/*.md

    # Splitting
    mitosis_config: MitosisConfig | None = None

    # Edge formation
    synapse_config: SynaptogenesisConfig | None = None

    # Decay during replay
    decay_config: DecayConfig | None = None
    decay_interval: int = 10  # Decay every N replayed queries

    # Replay
    max_replay_queries: int = 500  # Cap on replayed queries


# ---------------------------------------------------------------------------
# Default injection files
# ---------------------------------------------------------------------------

DEFAULT_INJECTION_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
    "MEMORY.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
]


# ---------------------------------------------------------------------------
# File gathering
# ---------------------------------------------------------------------------


def gather_files(
    workspace_dir: str | Path,
    config: MigrateConfig | None = None,
) -> dict[str, str]:
    """Gather workspace files for bootstrap."""
    config = config or MigrateConfig()
    workspace = Path(workspace_dir).expanduser()
    files = {}

    # Injection files
    injection = config.injection_files or DEFAULT_INJECTION_FILES
    for fname in injection:
        p = workspace / fname
        if p.exists():
            key = fname.replace(".md", "").lower().replace("/", "-")
            content = p.read_text()
            if content.strip():
                files[key] = content

    # Memory files
    if config.include_memory:
        memory_dir = workspace / "memory"
        if memory_dir.exists():
            for p in sorted(memory_dir.glob("*.md")):
                key = f"memory-{p.stem}"
                content = p.read_text()
                if content.strip() and len(content) > 100:
                    files[key] = content

    # Doc files
    if config.include_docs:
        docs_dir = workspace / "docs"
        if docs_dir.exists():
            for p in sorted(docs_dir.glob("*.md")):
                key = f"docs-{p.stem}"
                content = p.read_text()
                if content.strip() and len(content) > 100:
                    files[key] = content

    return files


# ---------------------------------------------------------------------------
# Session log parsing
# ---------------------------------------------------------------------------


def parse_session_logs(
    log_paths: list[str | Path],
    max_queries: int = 500,
) -> list[str]:
    """Extract user queries from session log files.

    Supports JSONL format with {"role": "user", "content": "..."} entries.
    Also supports plain text (one query per line).
    """
    queries = []
    skipped_records = 0

    def _extract_message_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if not isinstance(value, list):
            return ""

        parts: list[str] = []
        for block in value:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text.strip())
        return " ".join(part for part in parts if part)

    def _extract_query_from_record(record: dict[str, Any]) -> str:
        role = str(record.get("role", ""))
        content = record.get("content", "")

        if role == "user" and content:
            text = _extract_message_text(content)
            if text:
                return text

        query = record.get("query", "")
        if isinstance(query, str):
            return query.strip()
        return ""

    for path in log_paths:
        p = Path(path).expanduser()
        if not p.exists():
            continue

        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Try JSONL
                    extracted = False
                    is_json = False
                    try:
                        record = json.loads(line)
                        is_json = True
                        if not isinstance(record, dict):
                            raise ValueError("Session log JSON records must be objects")

                        # OpenClaw transcript format: type='message'
                        # with nested message dict.
                        if record.get("type") == "message":
                            message = record.get("message")
                            if isinstance(message, dict):
                                query = _extract_query_from_record(message)
                            elif isinstance(message, str):
                                try:
                                    message_payload = json.loads(message)
                                except json.JSONDecodeError:
                                    try:
                                        message_payload = ast.literal_eval(message)
                                    except (ValueError, SyntaxError) as inner_exc:
                                        raise ValueError(
                                            "Unable to decode OpenClaw message payload"
                                        ) from inner_exc
                                if not isinstance(message_payload, dict):
                                    raise ValueError("OpenClaw message payload must be a dict")
                                query = _extract_query_from_record(message_payload)
                            else:
                                raise ValueError("OpenClaw message record missing message payload")

                            if query and len(query) > 5 and not _is_system_noise(query):
                                queries.append(query[:500])
                            extracted = True
                            continue

                        # Skip other OpenClaw record types (session,
                        # model_change, thinking_level_change, custom, etc.)
                        if record.get("type") in (
                            "session",
                            "model_change",
                            "thinking_level_change",
                            "custom",
                        ):
                            continue

                        query = _extract_query_from_record(record)
                        if query and len(query) > 5:
                            candidate = query
                            if not _is_system_noise(candidate):
                                queries.append(candidate[:500])
                                extracted = True
                    except json.JSONDecodeError:
                        is_json = False
                    except (KeyError, ValueError):
                        skipped_records += 1
                        continue

                    # Plain text fallback — only for non-JSON lines
                    if (
                        not extracted
                        and not is_json
                        and len(line) > 5
                        and not line.startswith("#")
                    ):
                        queries.append(line[:500])

        except OSError:
            continue

        if len(queries) >= max_queries:
            break

    if skipped_records:
        logger.debug("parse_session_logs skipped %s malformed records", skipped_records)

    return queries[:max_queries]



# ---------------------------------------------------------------------------
# Replay — warm up the graph with historical queries
# ---------------------------------------------------------------------------


def replay_queries(
    graph: Graph,
    queries: list[str],
    router_fn: Callable[[str, list[tuple[str, float, str]]], list[str]],
    synapse_state: SynaptogenesisState,
    config: MigrateConfig | None = None,
) -> dict[str, Any]:
    """Replay historical queries to warm up edge formation.

    Uses a simple keyword-matching router if no real router provided.
    """
    config = config or MigrateConfig()
    syn_config = config.synapse_config or SynaptogenesisConfig()
    decay_config = config.decay_config or DecayConfig(half_life_turns=80)

    total_cofires = 0
    total_promotions = 0
    total_reinforcements = 0
    total_skips = 0

    for qi, query in enumerate(queries, 1):
        # Build candidates from all nodes
        q_words = set(query.lower().split())
        candidates = []
        for node in graph.nodes():
            n_words = set(node.content.lower().split())
            overlap = len(q_words & n_words)
            score = min(overlap / max(len(q_words), 1), 1.0)
            if score > 0.05:
                candidates.append(
                    (
                        node.id,
                        score,
                        node.summary or node.content[:80],
                    )
                )

        candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = candidates[:10]

        # Route
        selected = router_fn(query, candidates)

        # Co-firing
        if len(selected) >= 2:
            result = record_cofiring(graph, selected, synapse_state, syn_config)
            total_cofires += 1
            total_promotions += result["promoted"]
            total_reinforcements += result["reinforced"]

        # Skip penalty
        if selected:
            candidate_ids = [c[0] for c in candidates]
            total_skips += record_skips(graph, selected[0], candidate_ids, selected, syn_config)

        # Periodic decay
        if qi % config.decay_interval == 0:
            apply_decay(graph, turns_elapsed=config.decay_interval, config=decay_config)
            decay_proto_edges(synapse_state, syn_config)

    return {
        "queries_replayed": len(queries),
        "cofiring_events": total_cofires,
        "promotions": total_promotions,
        "reinforcements": total_reinforcements,
        "skips": total_skips,
    }


# ---------------------------------------------------------------------------
# Default keyword router for replay
# ---------------------------------------------------------------------------

_TRIVIAL = {"hello", "hi", "thanks", "yes", "no", "ok", "sure", "bye", ""}


def keyword_router(
    query: str,
    candidates: list[tuple[str, float, str]],
) -> list[str]:
    """Simple keyword-matching multi-select router for replay."""
    if not candidates:
        return []

    q_words = set(query.lower().split())
    if q_words.issubset(_TRIVIAL):
        return []

    scored = []
    for nid, embed_score, summary in candidates:
        s_words = set(summary.lower().split())
        overlap = len(q_words & s_words)
        combined = (overlap / max(len(q_words), 1)) * 0.6 + embed_score * 0.4
        if combined > 0.1 or overlap >= 1:
            scored.append((nid, combined))

    if not scored:
        best = max(candidates, key=lambda c: c[1])
        return [best[0]] if best[1] > 0.15 else []

    scored.sort(key=lambda x: x[1], reverse=True)
    threshold = scored[0][1] * 0.4
    return [nid for nid, sc in scored if sc >= threshold][:5]


# ---------------------------------------------------------------------------
# Fallback LLM for splitting (header-based, no API needed)
# ---------------------------------------------------------------------------


def fallback_llm_split(system: str, user: str) -> str:
    """Split by markdown headers. No API call needed."""
    content = user.split("---\n", 1)[-1].rsplit("\n---", 1)[0] if "---" in user else user
    sections = split_fallback_sections(
        content,
        min_header_chars=50,
        min_paragraph_chars=50,
        merge_short_paragraphs=0,
    )
    return json.dumps({"sections": sections})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def migrate(
    workspace_dir: str | Path = "~/.openclaw/workspace",
    session_logs: list[str | Path] | None = None,
    llm_call: Callable[[str, str], str] | None = None,
    router_fn: Callable | None = None,
    config: MigrateConfig | None = None,
    verbose: bool = False,
    embed_callback: Callable[[str, str], None] | None = None,
) -> tuple[Graph, dict[str, Any]]:
    """Migrate a workspace to CrabPath.

    1. Gather workspace files
    2. Bootstrap graph (carbon copy + LLM split)
    3. Optionally replay session logs to warm up edges

    Args:
        workspace_dir: Path to the workspace.
        session_logs: Optional list of session log files for replay.
        llm_call: Cheap LLM for splitting. Falls back to header-based if None.
        router_fn: Router for replay. Falls back to keyword matching if None.
        config: Migration config.
        verbose: Print progress.

    Returns:
        (graph, info_dict) — the bootstrapped graph and migration stats.
    """
    explicit_config = config is not None
    config = config or MigrateConfig()
    llm = llm_call or fallback_llm_split
    router = router_fn or keyword_router

    # 1. Gather files
    files = gather_files(workspace_dir, config)
    if verbose:
        print(f"📁 Gathered {len(files)} files ({sum(len(v) for v in files.values()):,} chars)")

    if not files:
        raise CrabPathError(f"No workspace files found in {workspace_dir}")

    if not explicit_config:
        autotune_defaults = suggest_config(files)
        config.mitosis_config = MitosisConfig(
            sibling_weight=autotune_defaults["sibling_weight"],
            min_content_chars=autotune_defaults["min_content_chars"],
        )
        config.synapse_config = SynaptogenesisConfig(
            promotion_threshold=autotune_defaults["promotion_threshold"],
            hebbian_increment=autotune_defaults["hebbian_increment"],
            skip_factor=autotune_defaults["skip_factor"],
            max_outgoing=autotune_defaults["max_outgoing"],
        )
        config.decay_config = DecayConfig(half_life_turns=autotune_defaults["decay_half_life"])
        config.decay_interval = autotune_defaults["decay_interval"]

    # 2. Bootstrap
    graph = Graph()
    mit_state = MitosisState()
    mit_config = config.mitosis_config or MitosisConfig(sibling_weight=0.25)
    syn_state = SynaptogenesisState()

    results = bootstrap_workspace(
        graph,
        files,
        llm,
        mit_state,
        mit_config,
        embed_callback=embed_callback,
    )

    if embed_callback is not None:
        for node in graph.nodes():
            embed_callback(node.id, node.content)

    bootstrap_info = {
        "files": len(files),
        "total_chars": sum(len(v) for v in files.values()),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "families": len(mit_state.families),
        "splits": [
            {
                "parent": r.parent_id,
                "chunks": len(r.chunk_ids),
            }
            for r in results
        ],
    }

    if verbose:
        print(f"🦀 Bootstrap: {graph.node_count} nodes, {graph.edge_count} edges")

    # 3. Replay session logs (optional)
    replay_info = None
    if session_logs:
        queries = parse_session_logs(session_logs, config.max_replay_queries)
        if queries:
            if verbose:
                print(f"🔄 Replaying {len(queries)} queries from session logs...")

            replay_info = replay_queries(graph, queries, router, syn_state, config)

            if verbose:
                print(f"   Promotions: {replay_info['promotions']}")
                print(f"   Reinforcements: {replay_info['reinforcements']}")
                print(f"   Cross-file co-fires: {replay_info['cofiring_events']}")

    tiers = edge_tier_stats(graph)

    info = {
        "bootstrap": bootstrap_info,
        "replay": replay_info,
        "final": {
            "nodes": graph.node_count,
            "edges": graph.edge_count,
            "tiers": tiers,
        },
        "states": {
            "mitosis": mit_state,
            "synapse": syn_state,
        },
    }

    if verbose:
        print(f"✅ Migration complete: {graph.node_count} nodes, {graph.edge_count} edges")
        print(f"   Tiers: {tiers}")

    return graph, info
