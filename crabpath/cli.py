"""Command-line interface for CrabPath.

All output is machine-readable JSON to keep agents simple:
- stdout carries success payloads
- stderr carries structured errors
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__
from ._io import (
    build_firing,
    build_health_rows,
    build_snapshot,
    graph_stats,
    load_graph,
    load_index,
    load_mitosis_state,
    load_query_stats,
    load_snapshot_rows,
    run_query,
    split_csv,
)
from .autotune import HEALTH_TARGETS, measure_health
from .controller import ControllerConfig, MemoryController
from .embeddings import EmbeddingIndex, auto_embed
from .feedback import auto_outcome, map_correction_to_snapshot, snapshot_path
from .graph import Graph
from .legacy.activation import learn as _learn
from .lifecycle_sim import SimConfig, run_simulation, workspace_scenario
from .migrate import MigrateConfig, fallback_llm_split, migrate, parse_session_logs
from .mitosis import MitosisConfig, MitosisState, split_node
from .synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    record_cofiring,
    record_correction,
)

DEFAULT_GRAPH_PATH = "crabpath_graph.json"
DEFAULT_INDEX_PATH = "crabpath_embeddings.json"
DEFAULT_TOP_K = 12
DEFAULT_WORKSPACE_PATH = "~/.openclaw/workspace"
DEFAULT_INIT_WORKSPACE_PATH = "."
DEFAULT_DATA_DIR = "~/.crabpath"


class CLIError(Exception):
    """Raised for user-facing CLI errors."""


class JSONArgumentParser(argparse.ArgumentParser):
    """Argparse parser that prints JSON errors and exits with code 1."""

    def error(self, message: str) -> None:  # pragma: no cover - exercised via CLI tests
        print(json.dumps({"error": message}), file=sys.stderr)
        raise SystemExit(1)


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload))


def _emit_error(message: str) -> int:
    print(json.dumps({"error": message}), file=sys.stderr)
    return 1


def _load_graph(path: str) -> Graph:
    try:
        return load_graph(path)
    except (FileNotFoundError, ValueError) as exc:
        raise CLIError(str(exc)) from exc


def _load_index(path: str) -> EmbeddingIndex:
    try:
        return load_index(path)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc


def _load_query_stats(path: str | None) -> dict[str, Any]:
    try:
        return load_query_stats(path)
    except (FileNotFoundError, ValueError) as exc:
        raise CLIError(str(exc)) from exc


def _load_mitosis_state(path: str | None) -> MitosisState:
    try:
        return load_mitosis_state(path)
    except (FileNotFoundError, ValueError) as exc:
        raise CLIError(str(exc)) from exc


def _load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return load_snapshot_rows(path)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc


def _format_health_target(target: tuple[float | None, float | None]) -> str:
    min_v, max_v = target
    if min_v is None and max_v is None:
        return "*"
    if min_v is None:
        return f"<= {max_v}"
    if max_v is None:
        return f">= {min_v}"
    return f"{min_v} - {max_v}"


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", default=False)


def _format_metric_value(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"

    if metric.endswith("_pct") or metric == "context_compression":
        return f"{value:.2f}%"

    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _build_health_report_lines(
    graph: Graph,
    health: Any,
    has_query_stats: bool,
    *,
    with_status: bool = False,
) -> list[str]:
    rows = build_health_rows(health, has_query_stats)
    lines: list[str] = [
        f"Graph Health: {graph.node_count} nodes, {graph.edge_count} edges",
        "-" * 46,
    ]
    for metric, target in HEALTH_TARGETS.items():
        row = next((item for item in rows if item["metric"] == metric), None)
        if row is None:
            continue
        value = row["value"]
        status_state = row["status"]
        status = (
            "⚠️"
            if status_state == "warn"
            else "❌"
            if status_state in {"low", "high"}
            else "✅"
        )
        value_text = (
            _format_metric_value(metric, float(value))
            if value is not None
            else "n/a (collect query stats)"
        )
        if with_status:
            target_text = _format_health_target(target)
            lines.append(
                f"{metric:24} | {value_text:>20} | target {target_text:15} | {status}"
            )
        else:
            lines.append(
                f"{metric}: {value_text} (target {_format_health_target(target)}) {status}"
            )
    return lines


def _build_health_payload(
    args: argparse.Namespace,
    graph: Graph,
    health: Any,
    has_query_stats: bool,
) -> dict[str, Any]:
    rows = build_health_rows(health, has_query_stats)
    for row in rows:
        if row["status"] == "warn":
            row["status"] = "⚠️"
        elif row["status"] in {"low", "high"}:
            row["status"] = "❌"
        else:
            row["status"] = "✅"

    return {
        "ok": True,
        "graph": args.graph,
        "query_stats_provided": has_query_stats,
        "mitosis_state": args.mitosis_state,
        "metrics": rows,
    }


def cmd_health(args: argparse.Namespace) -> dict[str, Any] | str:
    graph = _load_graph(args.graph)
    state = _load_mitosis_state(args.mitosis_state)
    query_stats = _load_query_stats(args.query_stats)
    has_query_stats = args.query_stats is not None
    health = measure_health(graph, state, query_stats)
    if args.json:
        return _build_health_payload(args, graph, health, has_query_stats)

    return "\n".join(
        _build_health_report_lines(
            graph,
            health,
            has_query_stats,
            with_status=True,
        )
    )


def _snapshot_path(path_value: str | None) -> Path:
    if path_value is None:
        raise CLIError("--snapshots is required for evolve")
    return Path(path_value)



def _format_timeline(snapshots: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Evolution timeline"]
    if not snapshots:
        return "No snapshots yet."

    previous: dict[str, Any] | None = None
    for idx, snapshot in enumerate(snapshots, start=1):
        timestamp = snapshot.get("timestamp")
        try:
            ts = float(timestamp) if timestamp is not None else 0.0
            label = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError, OverflowError):
            label = "invalid-timestamp"

        nodes = int(snapshot.get("nodes", 0))
        edges = int(snapshot.get("edges", 0))
        cross_file = int(snapshot.get("cross_file_edges", 0))
        tiers = snapshot.get("tier_counts", {})
        dormant = int((tiers or {}).get("dormant", 0))
        habitual = int((tiers or {}).get("habitual", 0))
        reflex = int((tiers or {}).get("reflex", 0))

        if previous is None:
            lines.append(
                f"#{idx:>2} {label} | nodes {nodes} | edges {edges} "
                f"| cross-file {cross_file} | tiers d={dormant} h={habitual} r={reflex}"
            )
        else:
            delta_nodes = nodes - int(previous.get("nodes", 0))
            delta_edges = edges - int(previous.get("edges", 0))
            delta_cross = cross_file - int(previous.get("cross_file_edges", 0))
            prev_tiers = previous.get("tier_counts", {})
            prev_dormant = int((prev_tiers or {}).get("dormant", 0))
            prev_habitual = int((prev_tiers or {}).get("habitual", 0))
            prev_reflex = int((prev_tiers or {}).get("reflex", 0))
            delta_dormant = dormant - prev_dormant
            delta_habitual = habitual - prev_habitual
            delta_reflex = reflex - prev_reflex

            lines.append(
                f"#{idx:>2} {label} | nodes {nodes} ({delta_nodes:+d}) "
                f"| edges {edges} ({delta_edges:+d}) "
                f"| cross-file {cross_file} ({delta_cross:+d}) "
                f"| tiers d={dormant} ({delta_dormant:+d}) "
                f"h={habitual} ({delta_habitual:+d}) "
                f"r={reflex} ({delta_reflex:+d})"
            )

        previous = snapshot

        if previous is None or previous.get("timestamp") is None:
            continue
    return "\n".join(lines)


def cmd_evolve(args: argparse.Namespace) -> dict[str, Any] | str:
    graph = _load_graph(args.graph)
    path = _snapshot_path(args.snapshots)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(graph)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")

    if not args.report:
        return {"ok": True, "snapshot": snapshot, "snapshots": str(path)}

    snapshots = _load_snapshot_rows(path)
    return _format_timeline(snapshots)


def _safe_embed_fn() -> Optional[Callable[[list[str]], list[list[float]]]]:
    try:
        return auto_embed()
    except Exception:
        return None


def _tokenize_query(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if token}


def _seed_scores(graph: Graph, query_text: str, top_k: int) -> list[tuple[str, float]]:
    query_tokens = _tokenize_query(query_text)
    if not query_tokens:
        return []

    scored: list[tuple[str, float]] = []
    for node in graph.nodes():
        node_tokens = set(_tokenize_query(f"{node.id} {node.summary} {node.content}"))
        overlap = len(query_tokens.intersection(node_tokens))
        if overlap == 0:
            continue
        score = overlap / max(len(query_tokens), 1)
        scored.append((node.id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _explain_traversal(
    graph: Graph,
    query_text: str,
    top_k: int,
) -> dict[str, Any]:
    score_map = _seed_scores(graph, query_text, top_k)
    if not score_map:
        return {
            "query": query_text,
            "seed_scores": [],
            "candidate_rankings": [],
            "inhibition_effects": [],
            "candidates_considered": 0,
            "selected_nodes": [],
            "fired_with_reasoning": [],
            "traversal_path": [],
            "selected_node_reasons": [],
        }

    controller = MemoryController(graph, config=ControllerConfig.default())
    result = controller.query(query_text, llm_call=None)
    if not result.selected_nodes:
        return {
            "query": query_text,
            "seed_scores": [
                {"node_id": node_id, "score": score} for node_id, score in score_map
            ],
            "candidate_rankings": [],
            "inhibition_effects": [],
            "candidates_considered": 0,
            "selected_nodes": [],
            "fired_with_reasoning": [],
            "traversal_path": [],
            "selected_node_reasons": [],
        }

    candidate_rankings = []
    traversal_path = []
    inhibition_effects = []
    selected_node_reasons: list[dict[str, Any]] = [
        {
            "node_id": result.selected_nodes[0],
            "step": 0,
            "reason": "seed node selected by query overlap",
        }
    ]

    outgoing_cache: dict[str, dict[str, float]] = {}
    for step_index, step in enumerate(result.trajectory):
        source = str(step.get("from_node", ""))
        target = str(step.get("to_node", ""))
        edge_weight = float(step.get("edge_weight", 0.0))
        outgoing = outgoing_cache.get(source)
        if outgoing is None:
            outgoing = {
                target_node.id: edge.weight
                for target_node, edge in graph.outgoing(source)
                if edge is not None
            }
            outgoing_cache[source] = outgoing

        ranked = []
        for target_id, suppressed_score in step.get("candidates", []):
            if not isinstance(target_id, str):
                continue
            target_score = (
                float(suppressed_score) if isinstance(suppressed_score, int | float) else 0.0
            )
            base_score = float(outgoing.get(target_id, 0.0))
            suppressed_by = target_score - base_score

            candidate = {
                "node_id": target_id,
                "base_score": base_score,
                "suppressed_score": target_score,
                "suppressed_by": suppressed_by,
            }
            if suppressed_by < 0:
                edge = graph.get_edge(source, target_id)
                if edge is not None and edge.weight < 0:
                    candidate["edge_weight"] = edge.weight
                    inhibition_effects.append(
                        {
                            "from": source,
                            "to": target_id,
                            "weight": edge.weight,
                            "base_score": base_score,
                            "suppressed_score": target_score,
                            "suppressed_delta": target_score - base_score,
                        }
                    )
            ranked.append(candidate)

        ranked.sort(key=lambda item: item["suppressed_score"], reverse=True)
        ranked_selected = next(
            (candidate for candidate in ranked if candidate["node_id"] == target),
            None,
        )
        if ranked_selected is None and ranked:
            ranked_selected = ranked[0]

        for candidate in ranked:
            if candidate["node_id"] == target:
                if step_index == 0:
                    reason = "selected highest score to leave seed"
                else:
                    reason = f"selected highest score at hop {step_index}"
            else:
                if candidate["node_id"] in {item["to"] for item in inhibition_effects}:
                    reason = "rejected due to suppression"
                else:
                    reason = "rejected by routing comparator"
            candidate["reason"] = reason

        candidate_rankings.append(
            {
                "step": step_index,
                "from": source,
                "candidates": ranked,
                "selected": target,
            }
        )
        traversal_path.append(
            {
                "step": step_index,
                "from": source,
                "to": target,
                "edge_weight": edge_weight,
            }
        )
        selected_node_reasons.append(
            {
                "node_id": target,
                "step": step_index + 1,
                "reason": (
                    "selected by inhibited score"
                    if target not in set(
                        item["node_id"] for item in selected_node_reasons
                    )
                    else "seed selection fallback"
                ),
            }
        )

    score_map = _seed_scores(graph, query_text, top_k)
    return {
        "query": query_text,
        "seed_scores": [
            {"node_id": node_id, "score": score} for node_id, score in score_map
        ],
        "candidate_rankings": candidate_rankings,
        "inhibition_effects": inhibition_effects,
        "candidates_considered": result.candidates_considered,
        "traversal_path": traversal_path,
        "selected_node_reasons": selected_node_reasons,
        "selected_nodes": result.selected_nodes,
        "fired_with_reasoning": selected_node_reasons,
    }


def _format_explain_trace(trace: dict[str, Any]) -> str:
    selected_nodes = trace.get("selected_nodes")
    if not isinstance(selected_nodes, list):
        selected_nodes = []

    seed_scores = trace.get("seed_scores")
    if not isinstance(seed_scores, list):
        seed_scores = []

    candidate_rankings = trace.get("candidate_rankings")
    if not isinstance(candidate_rankings, list):
        candidate_rankings = []

    inhibition_effects = trace.get("inhibition_effects")
    if not isinstance(inhibition_effects, list):
        inhibition_effects = []

    selected_node_reasons = trace.get("selected_node_reasons")
    if not isinstance(selected_node_reasons, list):
        selected_node_reasons = []

    selected_nodes_text = ", ".join(
        str(node_id) for node_id in selected_nodes if isinstance(node_id, str)
    )

    lines = [
        f"Query: {trace['query']}",
        f"Candidates considered: {trace['candidates_considered']}",
        f"Selected nodes: {selected_nodes_text or 'none'}",
        "",
        "Seed scores:",
    ]
    if seed_scores:
        for entry in seed_scores:
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"- {entry.get('node_id', '')}: {float(entry.get('score', 0.0)):.3f}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "Routing steps:"])
    for step in candidate_rankings:
        if not isinstance(step, dict):
            continue
        selected = step.get("selected") or "none"
        lines.append(f"Step {step.get('step')} from {step.get('from')} -> {selected}")
        for candidate in step.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            marker = "✓" if candidate.get("node_id") == selected else " "
            lines.append(
                f"  {marker} {candidate.get('node_id')} "
                f"base={float(candidate.get('base_score', 0.0)):.3f} "
                f"suppressed={float(candidate.get('suppressed_score', 0.0)):.3f} "
                f"suppressed_by={float(candidate.get('suppressed_by', 0.0)):.3f} "
                f"reason={candidate.get('reason')}"
            )

    if inhibition_effects:
        lines.extend(["", "Inhibition effects:"])
        for effect in inhibition_effects:
            if not isinstance(effect, dict):
                continue
            lines.append(
                f"- {effect.get('from')} -> {effect.get('to')} "
                f"delta={float(effect.get('suppressed_delta', 0.0)):.3f} "
                f"edge={float(effect.get('weight', 0.0)):.3f}"
            )

    lines.extend(["", "Selection rationale:"])
    for reason in selected_node_reasons:
        if not isinstance(reason, dict):
            continue
        lines.append(
            f"- step {reason.get('step')}: {reason.get('node_id')}: {reason.get('reason')}"
        )

    return "\n".join(lines)


def cmd_query(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    index = _load_index(args.index)
    embed_fn = _safe_embed_fn()
    firing = run_query(graph, index, args.query, top_k=args.top, embed_fn=embed_fn)
    payload = {
        "fired": [
            {"id": node.id, "content": node.content, "energy": score}
            for node, score in firing.fired
        ],
        "inhibited": list(firing.inhibited),
        "guardrails": list(firing.inhibited),
    }

    if args.explain:
        explanation = _explain_traversal(graph, args.query, args.top)
        payload["explain"] = explanation
        payload["seeds"] = explanation["seed_scores"]
        payload["candidates"] = explanation["candidate_rankings"]

    return payload


def cmd_explain(args: argparse.Namespace) -> dict[str, Any] | str:
    graph = _load_graph(args.graph)
    _load_index(args.index)
    trace = _explain_traversal(graph, args.query, DEFAULT_TOP_K)
    if args.json:
        return trace
    return _format_explain_trace(trace)


def cmd_learn(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)

    fired_ids = split_csv(args.fired_ids)
    try:
        outcome = float(args.outcome)
    except ValueError as exc:
        raise CLIError(f"invalid outcome: {args.outcome}") from exc

    before = {(edge.source, edge.target): edge.weight for edge in graph.edges()}
    try:
        firing = build_firing(graph, fired_ids)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
    _learn(graph, firing, outcome=outcome)

    after = {(edge.source, edge.target): edge.weight for edge in graph.edges()}
    edges_updated = 0
    for key, weight in after.items():
        if key not in before or before[key] != weight:
            edges_updated += 1

    graph.save(args.graph)
    return {"ok": True, "edges_updated": edges_updated}


def cmd_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    _load_graph(args.graph)
    fired_ids = split_csv(args.fired_ids)

    record = {
        "session_id": args.session,
        "turn_id": args.turn,
        "timestamp": time.time(),
        "fired_ids": fired_ids,
        "fired_scores": [1.0 for _ in fired_ids],
        "fired_at": {node_id: idx for idx, node_id in enumerate(fired_ids)},
        "inhibited": [],
        "attributed": False,
    }

    path = snapshot_path(args.graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"ok": True, "snapshot_path": str(path)}


def cmd_feedback(args: argparse.Namespace) -> dict[str, Any]:
    if args.session is not None:
        snapshot = map_correction_to_snapshot(
            session_id=args.session,
            turn_window=args.turn_window,
        )
        if snapshot is None:
            raise CLIError(f"no attributable snapshot found for session: {args.session}")

        turns_since_fire = snapshot.get("turns_since_fire", 0)
        return {
            "turn_id": snapshot.get("turn_id"),
            "fired_ids": snapshot.get("fired_ids", []),
            "turns_since_fire": turns_since_fire,
            "suggested_outcome": auto_outcome(
                corrections_count=1, turns_since_fire=int(turns_since_fire)
            ),
        }

    if args.query is None:
        raise CLIError("--query is required when not using --session")
    if args.trajectory is None:
        raise CLIError("--trajectory is required for manual feedback")
    if args.reward is None:
        raise CLIError("--reward is required for manual feedback")

    trajectory = split_csv(args.trajectory)
    graph = _load_graph(args.graph)
    syn_state = SynaptogenesisState()
    config = SynaptogenesisConfig()

    reward = float(args.reward)
    if reward < 0.0:
        results = record_correction(
            graph=graph,
            trajectory=trajectory,
            reward=reward,
            config=config,
        )
        payload = {
            "action": "record_correction",
            "reward": reward,
            "results": results,
        }
    else:
        results = record_cofiring(
            graph=graph,
            fired_nodes=trajectory,
            state=syn_state,
            config=config,
        )
        payload = {
            "action": "record_cofiring",
            "reward": 0.1,
            "results": results,
        }

    graph.save(args.graph)
    return {"ok": True, "query": args.query, "trajectory": trajectory, **payload}


def cmd_stats(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    return graph_stats(graph)


def cmd_migrate(args: argparse.Namespace) -> dict[str, Any]:
    config = MigrateConfig(
        include_memory=args.include_memory,
        include_docs=args.include_docs,
    )
    embed_fn = _safe_embed_fn()
    embeddings_index = EmbeddingIndex()
    embed_callback = None
    if args.output_embeddings is not None and embed_fn is not None:
        embeddings_index = _load_index(args.output_embeddings)

        def embed_callback(node_id: str, content: str) -> None:
            embeddings_index.upsert(node_id, content, embed_fn=embed_fn)

    try:
        graph, info = migrate(
            workspace_dir=args.workspace,
            session_logs=args.session_logs or None,
            config=config,
            embed_callback=embed_callback,
            verbose=False,
        )
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
    if "states" in info:
        info = dict(info)
        info.pop("states", None)

    graph_path = args.output_graph
    graph.save(graph_path)
    verbose = getattr(args, "verbose", False)
    if info.get("query_stats"):
        stats_path = str(Path(args.output_graph).with_suffix(".stats.json"))
        with open(stats_path, "w") as f:
            json.dump(info["query_stats"], f, indent=2)
        if verbose:
            print(f"📊 Query stats saved to {stats_path}")

    embeddings_path = args.output_embeddings
    if embeddings_path:
        if embed_callback is not None:
            embeddings_index.save(embeddings_path)
        else:
            EmbeddingIndex().save(embeddings_path)

    return {
        "ok": True,
        "graph_path": str(graph_path),
        "embeddings_path": str(embeddings_path) if embeddings_path else None,
        "info": info,
    }


def _build_temporary_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="crabpath-init-"))
    files = {
        "AGENTS.md": """# Atlas Harbor Ops
Use short safety checks before escalation and keep responses factual.""",
        "SOUL.md": """# Atlas Harbor Identity
Atlas Harbor is a fictional logistics platform for resilient shipping operations.""",
        "TOOLS.md": """# Atlas Harbor Tools
Use local tooling for route planning, event replay, and graph diagnostics.""",
        "USER.md": """# Atlas Harbor Users
Crew and dispatch coordinators rely on quick incident summaries.""",
        "MEMORY.md": """# Atlas Harbor Memory
Keep concise, timestamped notes on incidents and recovery outcomes.""",
    }
    for name, content in files.items():
        (workspace / name).write_text(f"{content}\n", encoding="utf-8")
    return workspace


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = Path(args.workspace).expanduser().resolve()
    temp_workspace = None
    if args.example:
        temp_workspace = _build_temporary_workspace()
        workspace_dir = temp_workspace

    graph_path = data_dir / "graph.json"
    embed_path = data_dir / "embed.json"

    session_logs = [args.sessions] if args.sessions else None
    stats_path = data_dir / "graph.stats.json"

    try:
        if args.no_embeddings:
            embed_fn = None
            embed_callback = None
            embeddings = None
        else:
            try:
                embed_fn = auto_embed()
            except RuntimeError as exc:
                raise CLIError(
                    f"No embedding provider found. {exc}\n\n"
                    "Embeddings are strongly recommended for accurate retrieval.\n"
                    "Set one of: OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY\n"
                    "Or run Ollama locally: ollama pull nomic-embed-text\n\n"
                    "To skip embeddings (keyword-only routing): crabpath init --no-embeddings"
                ) from exc
            embeddings = EmbeddingIndex()
            embed_callback = None
            if embed_fn is not None:

                def embed_callback(node_id: str, content: str) -> None:
                    embeddings.upsert(node_id, content, embed_fn=embed_fn)

        graph, info = migrate(
            workspace_dir=workspace_dir,
            session_logs=session_logs,
            config=MigrateConfig(),
            embed_callback=embed_callback,
            verbose=False,
        )
        if "states" in info:
            info = dict(info)
            info.pop("states", None)

        query_stats = info.get("query_stats")
        if query_stats is not None:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(query_stats, f, indent=2)

        graph.save(str(graph_path))
        if embeddings is not None:
            embeddings.save(str(embed_path))

        health = measure_health(graph, MitosisState(), {})
        health_payload = _build_health_payload(
            argparse.Namespace(
                graph=str(graph_path),
                mitosis_state=None,
            ),
            graph,
            health,
            False,
        )

        next_steps = [
            f"crabpath query '<query>' --graph {graph_path}",
            f"crabpath health --graph {graph_path} --json",
        ]
        if query_stats is not None:
            next_steps.append(
                f"crabpath health --graph {graph_path} --query-stats {stats_path} --json"
            )

        return {
            "ok": True,
            "data_dir": str(data_dir),
            "workspace": str(workspace_dir),
            "graph_path": str(graph_path),
            "embeddings_path": str(embed_path) if embeddings is not None else None,
            "stats_path": str(stats_path) if query_stats is not None else None,
            "migration": info,
            "health": health_payload["metrics"],
            "summary": {
                "nodes": graph.node_count,
                "edges": graph.edge_count,
                "files": info.get("bootstrap", {}).get("files", 0),
            },
            "next_steps": next_steps,
        }
    finally:
        if temp_workspace is not None:
            shutil.rmtree(temp_workspace, ignore_errors=True)


def cmd_install_hook(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    graph_path = data_dir / "graph.json"
    if not graph_path.exists():
        raise CLIError("No CrabPath graph found. Run crabpath init first.")

    agents_path = (
        Path(args.agent_workspace).expanduser().resolve() / "AGENTS.md"
    )
    if not agents_path.exists():
        raise CLIError(f"AGENTS.md not found: {agents_path}")

    existing = agents_path.read_text(encoding="utf-8")
    if "CrabPath" in existing:
        return {
            "ok": True,
            "agents_md": str(agents_path),
            "already_installed": True,
            "block_added": False,
            "data_dir": str(data_dir),
        }

    block = (
        "\n## CrabPath Memory Graph (auto-installed)\n\n"
        "Before answering questions about prior work, context, or decisions, query the CrabPath memory graph:\n"
        "```bash\n"
        f"crabpath query '<summary of user message>' --graph {graph_path} --top 8 --json\n"
        "```\n"
        "Use the returned context to supplement your memory files. CrabPath runs in shadow mode — it suggests context but never modifies your responses directly.\n\n"
        f"To check graph health: `crabpath health --graph {graph_path}`\n"
        "To remove: delete this section from AGENTS.md and run `pip uninstall crabpath`\n"
    )

    if not args.dry_run:
        with agents_path.open("a", encoding="utf-8") as f:
            f.write(block)
        block_added = True
    else:
        block_added = False

    return {
        "ok": True,
        "agents_md": str(agents_path),
        "block_added": block_added,
        "data_dir": str(data_dir),
    }


def cmd_extract_sessions(args: argparse.Namespace) -> dict[str, Any]:
    base = Path(args.agents_root).expanduser()
    if not base.exists():
        raise CLIError(f"sessions root not found: {base}")

    session_files = []
    for agent_dir in base.glob("*"):
        if not agent_dir.is_dir():
            continue
        sessions_dir = agent_dir / "sessions"
        if sessions_dir.is_dir():
            session_files.extend(sorted(sessions_dir.glob("*.jsonl")))

    if not session_files:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("", encoding="utf-8")
        return {
            "ok": True,
            "output": str(args.output),
            "sessions_scanned": 0,
            "queries_extracted": 0,
        }

    queries = parse_session_logs(session_files, max_queries=args.max_queries)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(queries) + ("\n" if queries else ""), encoding="utf-8")

    return {
        "ok": True,
        "output": str(output_path),
        "sessions_scanned": len(session_files),
        "queries_extracted": len(queries),
    }


def cmd_add(args: argparse.Namespace) -> dict[str, Any]:
    graph_path = Path(args.graph)
    if graph_path.exists():
        graph = Graph.load(args.graph)
    else:
        graph = Graph()

    from .graph import Edge, Node

    node_id = args.id
    if graph.get_node(node_id) is not None:
        # Update existing node
        node = graph.get_node(node_id)
        node.content = args.content
        if args.threshold is not None:
            node.threshold = args.threshold
        graph.save(args.graph)
        return {"ok": True, "action": "updated", "id": node_id}

    threshold = args.threshold if args.threshold is not None else 0.5
    graph.add_node(Node(id=node_id, content=args.content, threshold=threshold))

    # Connect to existing nodes if --connect provided
    edges_added = 0
    if args.connect:
        connect_ids = [c.strip() for c in args.connect.split(",") if c.strip()]
        for target_id in connect_ids:
            if graph.get_node(target_id) is not None and target_id != node_id:
                graph.add_edge(Edge(source=node_id, target=target_id, weight=0.5))
                graph.add_edge(Edge(source=target_id, target=node_id, weight=0.5))
                edges_added += 2

    graph.save(args.graph)
    return {"ok": True, "action": "created", "id": node_id, "edges_added": edges_added}


def cmd_remove(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    node = graph.get_node(args.id)
    if node is None:
        raise CLIError(f"node not found: {args.id}")
    graph.remove_node(args.id)
    graph.save(args.graph)
    return {"ok": True, "action": "removed", "id": args.id}


def cmd_consolidate(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    result = graph.consolidate(min_weight=args.min_weight)
    graph.save(args.graph)
    return {"ok": True, **result}


def cmd_split_node(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    state = MitosisState()
    index = _load_index(args.index)
    embed_fn = _safe_embed_fn()
    embed_callback = None
    if embed_fn is not None:

        def embed_callback(node_id: str, content: str) -> None:
            index.upsert(node_id, content, embed_fn=embed_fn)

    result = split_node(
        graph,
        node_id=args.node_id,
        llm_call=fallback_llm_split,
        state=state,
        config=MitosisConfig(),
        embed_callback=embed_callback,
    )

    if result is None:
        raise CLIError(f"could not split node: {args.node_id}")

    if args.save:
        graph.save(args.graph)
        if embed_callback is not None:
            index.save(args.index)

    return {
        "ok": True,
        "action": "split",
        "node_id": args.node_id,
        "chunk_ids": result.chunk_ids,
        "chunk_count": len(result.chunk_ids),
        "edges_created": result.edges_created,
    }


def cmd_sim(args: argparse.Namespace) -> dict[str, Any]:
    files, queries = workspace_scenario()
    selected_queries = queries[: args.queries]

    if not selected_queries:
        raise CLIError("queries must be a positive integer")

    config = SimConfig(
        decay_interval=args.decay_interval,
        decay_half_life=args.decay_half_life,
    )
    result = run_simulation(files, selected_queries, config=config)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))

    payload = {
        "ok": True,
        "queries": args.queries,
        "result": result,
    }
    if args.output:
        payload["output"] = args.output
    return payload


def _build_parser() -> JSONArgumentParser:
    parser = JSONArgumentParser(
        prog="crabpath",
        description="CrabPath CLI: JSON-in / JSON-out for agent use",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    q = subparsers.add_parser("query", help="Run query + activation against a graph")
    q.add_argument("query")
    q.add_argument("--top", type=int, default=DEFAULT_TOP_K)
    q.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    q.add_argument("--index", default=DEFAULT_INDEX_PATH)
    q.add_argument("--explain", action="store_true", default=False)
    _add_json_flag(q)
    q.set_defaults(func=cmd_query)

    learn = subparsers.add_parser("learn", help="Apply STDP on specified fired node ids")
    learn.add_argument("--outcome", required=True)
    learn.add_argument("--fired-ids", required=True)
    learn.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    _add_json_flag(learn)
    learn.set_defaults(func=cmd_learn)

    snap = subparsers.add_parser("snapshot", help="Persist a turn snapshot")
    snap.add_argument("--session", required=True)
    snap.add_argument("--turn", type=int, required=True)
    snap.add_argument("--fired-ids", required=True)
    snap.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    _add_json_flag(snap)
    snap.set_defaults(func=cmd_snapshot)

    fb = subparsers.add_parser("feedback", help="Find most attributable snapshot")
    fb.add_argument("--session", default=None)
    fb.add_argument("--turn-window", type=int, default=5)
    fb.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    fb.add_argument("--query", default=None)
    fb.add_argument("--trajectory", default=None)
    fb.add_argument("--reward", type=float, default=None)
    _add_json_flag(fb)
    fb.set_defaults(func=cmd_feedback)

    st = subparsers.add_parser("stats", help="Show simple graph stats")
    st.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    _add_json_flag(st)
    st.set_defaults(func=cmd_stats)

    mig = subparsers.add_parser("migrate", help="Bootstrap CrabPath from workspace files")
    mig.add_argument("--workspace", default=DEFAULT_WORKSPACE_PATH)
    mig.add_argument("--session-logs", action="append", default=[])
    mig.add_argument("--include-memory", dest="include_memory", action="store_true")
    mig.add_argument("--no-include-memory", dest="include_memory", action="store_false")
    mig.set_defaults(include_memory=True)
    mig.add_argument("--include-docs", action="store_true", default=False)
    mig.add_argument("--output-graph", default=DEFAULT_GRAPH_PATH)
    mig.add_argument("--output-embeddings", default=None)
    mig.add_argument("--verbose", action="store_true", default=False)
    _add_json_flag(mig)
    mig.set_defaults(func=cmd_migrate)

    init = subparsers.add_parser(
        "init",
        help="Bootstrap graph and index into a data directory and print a summary",
    )
    init.add_argument("--workspace", default=DEFAULT_INIT_WORKSPACE_PATH)
    init.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    init.add_argument("--example", action="store_true", default=False)
    init.add_argument(
        "--sessions",
        default=None,
        help="Session logs directory or files for replay (auto-globs *.jsonl from directories)",
    )
    init.add_argument(
        "--no-embeddings",
        action="store_true",
        default=False,
        help="Skip embedding generation (use keyword-based routing instead)",
    )
    _add_json_flag(init)
    init.set_defaults(func=cmd_init)

    hook = subparsers.add_parser(
        "install-hook",
        help="Install CrabPath integration into an agent workspace",
    )
    hook.add_argument("--agent-workspace", required=True, help="Path to agent workspace directory")
    hook.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    hook.add_argument("--dry-run", action="store_true", default=False)
    _add_json_flag(hook)
    hook.set_defaults(func=cmd_install_hook)

    explain = subparsers.add_parser("explain", help="Explain MemoryController routing for a query")
    explain.add_argument("query")
    explain.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    explain.add_argument("--index", default=DEFAULT_INDEX_PATH)
    _add_json_flag(explain)
    explain.set_defaults(func=cmd_explain)

    extract = subparsers.add_parser(
        "extract-sessions",
        help="Extract OpenClaw user queries from saved session logs",
    )
    extract.add_argument("output", help="Output file path for extracted queries")
    extract.add_argument("--agents-root", default="~/.openclaw/agents")
    extract.add_argument("--max-queries", type=int, default=500)
    _add_json_flag(extract)
    extract.set_defaults(func=cmd_extract_sessions)

    split = subparsers.add_parser("split", help="Split a node into coherent chunks")
    split.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    split.add_argument("--index", default=DEFAULT_INDEX_PATH)
    split.add_argument("--node-id", required=True, dest="node_id")
    split.add_argument("--save", action="store_true")
    _add_json_flag(split)
    split.set_defaults(func=cmd_split_node)

    sim = subparsers.add_parser("sim", help="Run the lifecycle simulation")
    sim.add_argument("--queries", type=int, default=100)
    sim.add_argument("--decay-interval", type=int, default=5)
    sim.add_argument("--decay-half-life", type=int, default=80)
    sim.add_argument("--output", default=None)
    _add_json_flag(sim)
    sim.set_defaults(func=cmd_sim)

    health = subparsers.add_parser(
        "health", help="Measure graph health from graph state + optional stats"
    )
    health.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    health.add_argument("--mitosis-state", default=None)
    health.add_argument("--query-stats", default=None)
    _add_json_flag(health)
    health.set_defaults(func=cmd_health)

    add = subparsers.add_parser("add", help="Add or update a node in the graph")
    add.add_argument("--id", required=True, help="Node ID")
    add.add_argument("--content", required=True, help="Node content text")
    add.add_argument(
        "--threshold", type=float, default=None, help="Firing threshold (default: 0.5)"
    )
    add.add_argument("--connect", default=None, help="Comma-separated node IDs to connect to")
    add.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    _add_json_flag(add)
    add.set_defaults(func=cmd_add)

    rm = subparsers.add_parser("remove", help="Remove a node and all its edges")
    rm.add_argument("--id", required=True, help="Node ID to remove")
    rm.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    _add_json_flag(rm)
    rm.set_defaults(func=cmd_remove)

    cons = subparsers.add_parser("consolidate", help="Consolidate and prune weak connections")
    cons.add_argument("--min-weight", type=float, default=0.05)
    cons.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    _add_json_flag(cons)
    cons.set_defaults(func=cmd_consolidate)

    evolve = subparsers.add_parser("evolve", help="Append graph snapshot stats to a JSONL timeline")
    evolve.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    evolve.add_argument("--snapshots", required=True)
    evolve.add_argument("--report", action="store_true", default=False)
    _add_json_flag(evolve)
    evolve.set_defaults(func=cmd_evolve)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        result = args.func(args)
        if isinstance(result, str):
            if getattr(args, "json", False):
                _emit_json({"ok": True, "output": result})
            else:
                print(result)
            return 0
        _emit_json(result)
        return 0
    except CLIError as exc:
        return _emit_error(str(exc))
    except Exception as exc:  # pragma: no cover - avoid raw tracebacks for CLI users
        return _emit_error(f"internal error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
