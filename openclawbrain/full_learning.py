"""Full-learning replay and harvest helpers."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._util import _extract_json
from .inject import inject_batch
from .maintain import run_maintenance
try:
    from .openai_embeddings import OpenAIEmbedder
except Exception:
    OpenAIEmbedder = None

from .hasher import HashEmbedder

try:
    from .openai_llm import openai_llm_fn
except Exception:
    openai_llm_fn = None
from .replay import replay_queries
from .store import load_state, save_state


LEARNING_EVENTS_FILENAME = "learning_events.jsonl"
REPLAY_CHECKPOINT_FILENAME = "replay_checkpoint.json"

FEEDBACK_CORRECTION = [
    "wrong",
    "incorrect",
    "no",
    "not what",
    "fix",
    "don't",
    "never",
]
FEEDBACK_TEACHING = [
    "should",
    "must",
    "prefer",
    "remember",
    "policy",
    "always",
]
FEEDBACK_REINFORCE = [
    "thanks",
    "great",
    "perfect",
    "correct",
    "good",
]

LEARNING_EXTRACTOR_PROMPT = """Extract feedback signals from a compact conversation window.

Return ONLY JSON:
{
  "corrections": [{"content": str, "context": str, "severity": "high|medium|low"}],
  "teachings": [{"content": str, "context": str, "severity": "high|medium|low"}],
  "reinforcements": [{"content": str, "context": str, "severity": "high|medium|low"}]
}
"""


@dataclass(frozen=True)
class SessionTurn:
    """Single user/assistant turn from a session log."""

    role: str
    content: str
    line_no: int
    source: str
    ts: float | None = None


def learning_event_path(state_path: str) -> Path:
    """Default path for append-only full-learning events."""
    return Path(state_path).expanduser().parent / LEARNING_EVENTS_FILENAME


def default_checkpoint_path(state_path: str) -> Path:
    """Default path for resume checkpoint."""
    return Path(state_path).expanduser().parent / REPLAY_CHECKPOINT_FILENAME


def _resolve_text(payload: object) -> str | None:
    """Extract normalized text from text, list, or OpenAI message payload."""
    if isinstance(payload, str):
        text = payload.strip()
        return text or None
    if isinstance(payload, list):
        out: list[str] = []
        for item in payload:
            piece = _resolve_text(item)
            if piece:
                out.append(piece)
        return " ".join(out) if out else None
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            resolved = text.strip()
            if resolved:
                return resolved
        content = payload.get("content")
        return _resolve_text(content)
    return None


def _read_records(path: Path, start_line: int = 0) -> list[tuple[int, dict]]:
    """Read JSONL records from a session file.

    Returns an empty list (with a stderr warning) when the file is missing
    or is a broken symlink, so that callers can continue processing the
    remaining session files instead of aborting.
    """
    if not path.exists():
        print(f"warning: skipping missing session file: {path}", file=sys.stderr)
        return []

    records: list[tuple[int, dict]] = []
    try:
        fh = path.expanduser().open("r", encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        print(f"warning: skipping unreadable session file: {path} ({exc})", file=sys.stderr)
        return []
    for idx, raw in enumerate(fh, start=1):
        if idx <= start_line:
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append((idx, payload))
    return records


def _extract_ts(payload: dict) -> float | None:
    """Extract timestamp from session record."""
    for key in ("ts", "timestamp", "created_at", "time"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def collect_session_files(session_paths: str | Path | list[str] | list[Path]) -> list[Path]:
    """Resolve session paths and return JSONL files.

    Missing or broken-symlink files are skipped with a stderr warning so
    that long rebuilds survive rotated/deleted session files.  If *no*
    files remain after filtering, the function exits with a helpful
    message.
    """
    paths: list[Path] = []
    skipped: list[Path] = []
    if isinstance(session_paths, (str, Path)):
        session_paths = [session_paths]
    for item in session_paths:
        src = Path(item).expanduser()
        if src.is_dir():
            patterns = ("*.jsonl", "*.jsonl.reset.*", "*.jsonl.deleted.*")
            collected = sorted((match for pattern in patterns for match in src.glob(pattern)))
            paths.extend(collected)
            continue
        if src.is_file():
            paths.append(src)
            continue
        # Missing or broken symlink — warn and skip.
        print(f"warning: skipping missing session path: {src}", file=sys.stderr)
        skipped.append(src)
    if not paths and skipped:
        raise SystemExit(
            f"no valid session files found ({len(skipped)} missing/broken path(s) skipped)"
        )
    return paths


def _collect_turns(
    session_paths: str | Path | list[str] | list[Path],
    since_lines: dict[str, int] | None = None,
) -> tuple[list[tuple[str, list[SessionTurn]]], dict[str, int]]:
    """Collect role turns from session files.

    Returns:
        grouped turns per source and latest processed line per source.
    """
    files = collect_session_files(session_paths)
    grouped: list[tuple[str, list[SessionTurn]]] = []
    next_offsets: dict[str, int] = {}

    for path in files:
        path_name = str(path)
        start_line = int((since_lines or {}).get(path_name, 0))
        turns: list[SessionTurn] = []
        last_line = start_line

        for line_no, payload in _read_records(path, start_line=start_line):
            last_line = line_no
            record_message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
            if not isinstance(record_message, dict):
                continue
            role = str(record_message.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = _resolve_text(record_message.get("content"))
            if not content:
                continue
            turns.append(
                SessionTurn(
                    role=role,
                    content=content,
                    line_no=line_no,
                    source=path_name,
                    ts=_extract_ts(payload),
                )
            )
        grouped.append((path_name, turns))
        next_offsets[path_name] = last_line
    return grouped, next_offsets


def _turn_score(turn: SessionTurn) -> int:
    """Score turns for feedback extraction priority."""
    if turn.role != "user":
        return 0
    text = turn.content.lower()
    score = 0
    if len(text) >= 200:
        score += 1
    for token in FEEDBACK_CORRECTION:
        if token in text:
            score += 5
            break
    for token in FEEDBACK_TEACHING:
        if token in text:
            score += 3
            break
    for token in FEEDBACK_REINFORCE:
        if token in text:
            score += 1
            break
    return score


def select_feedback_windows(
    turns: list[SessionTurn],
    *,
    window_radius: int,
    max_windows: int,
    hard_max_turns: int,
) -> list[tuple[int, int]]:
    """Select feedback windows around likely feedback turns."""
    if not turns:
        return []
    if len(turns) <= hard_max_turns:
        return [(0, len(turns))]

    scored: list[tuple[int, int]] = []
    for index, turn in enumerate(turns):
        score = _turn_score(turn)
        if score > 0:
            scored.append((index, score))

    if not scored:
        head_end = max(1, len(turns) // 2)
        ranges = [(0, head_end), (max(0, len(turns) - head_end), len(turns))]
        if ranges[0][1] >= ranges[1][0]:
            return [(ranges[0][0], ranges[1][1])]
        return ranges

    scored.sort(key=lambda item: item[1], reverse=True)
    picks = [idx for idx, _ in scored[:max_windows]]
    ranges: list[tuple[int, int]] = []
    for idx in sorted(picks):
        start = max(0, idx - window_radius)
        end = min(len(turns), idx + window_radius + 1)
        if not ranges or start > ranges[-1][1]:
            ranges.append((start, end))
        else:
            prev_start, prev_end = ranges[-1]
            ranges[-1] = (prev_start, max(prev_end, end))
    return ranges


def _window_to_payload(
    *,
    session: str,
    window: list[SessionTurn],
    window_idx: int,
    total_windows: int,
    llm_fn: Callable[[str, str], str],
) -> list[dict]:
    """Extract learning signals from one window."""
    text = "\n\n".join(f"{turn.role.upper()}: {turn.content[:1200]}" for turn in window).strip()
    if not text:
        return []

    window_suffix = f" (window {window_idx}/{total_windows})" if total_windows > 1 else ""
    user_prompt = f"Session: {session}{window_suffix}\n\n{text}\n\nExtract learning signals."
    response = llm_fn(LEARNING_EXTRACTOR_PROMPT, user_prompt)
    payload = _extract_json(response)
    if not isinstance(payload, dict):
        return []

    pointer = f"{session}:{window[0].line_no}-{window[-1].line_no}"
    events: list[dict] = []

    for event_type, key in (("CORRECTION", "corrections"), ("TEACHING", "teachings"), ("REINFORCEMENT", "reinforcements")):
        raw = payload.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            normalized = content.strip()
            if not normalized:
                continue
            severity = str(item.get("severity", "medium")).strip().lower()
            if severity not in {"low", "medium", "high"}:
                severity = "medium"
            context = item.get("context")
            if not isinstance(context, str):
                context = ""
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            events.append(
                {
                    "type": event_type,
                    "content": normalized,
                    "content_hash": digest,
                    "severity": severity,
                    "context": context[:600],
                    "session": session,
                    "session_pointer": pointer,
                }
            )
    return events


def _event_key(event: dict) -> str:
    """Stable dedup key: (type, sha256(content), session pointer)."""
    return "|".join(
        (
            str(event.get("type")),
            str(event.get("content_hash")),
            str(event.get("session_pointer")),
        )
    )


def event_log_entries(path: Path | str) -> list[dict]:
    """Load learning events from append-only log."""
    path = Path(path)
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def dedupe_learning_events(existing: set[str], events: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate by (type, sha(content), session pointer)."""
    appended: list[dict] = []
    skipped = 0
    for event in events:
        key = _event_key(event)
        if key in existing:
            skipped += 1
            continue
        existing.add(key)
        event["event_hash"] = key
        appended.append(event)
    return appended, skipped


def append_learning_events(path: Path | str, events: list[dict]) -> None:
    """Append new events durably."""
    if not events:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, indent=None) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def _load_checkpoint(path: Path | str) -> dict[str, int]:
    """Load checkpoint JSON."""
    payload = {"sessions": {}}
    p = Path(path)
    if not p.exists():
        return {"sessions": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return payload
    if not isinstance(raw, dict):
        return payload
    sessions = raw.get("sessions")
    if not isinstance(sessions, dict):
        return payload
    return {"sessions": {k: int(v) for k, v in sessions.items() if isinstance(v, (int, float))}}


def _save_checkpoint(path: Path | str, session_offsets: dict[str, int]) -> None:
    """Save replay checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"version": 1, "sessions": session_offsets}, handle, indent=2)


def _persist_state(
    graph: object,
    index: object,
    meta: dict[str, object],
    state_path: str,
    backup: bool,
) -> None:
    """Persist state while honoring backup semantics."""
    target = Path(state_path).expanduser()
    backup_path = target.with_suffix(".bak")
    had_backup = backup_path.exists()
    save_state(graph=graph, index=index, path=state_path, meta=meta)
    if not backup and not had_backup and backup_path.exists():
        try:
            backup_path.unlink()
        except OSError:
            print(f"warning: could not remove backup file {backup_path}", file=sys.stderr)


def load_interactions_for_replay(
    session_paths: str | Path | list[str] | list[Path],
    since_lines: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Load interactions from sessions with line-aware resume support."""
    files = collect_session_files(session_paths)
    interactions: list[dict] = []
    offsets: dict[str, int] = {}

    for path in files:
        path_name = str(path)
        start_line = int((since_lines or {}).get(path_name, 0))
        last_line = start_line
        last_user_idx: int | None = None

        for line_no, payload in _read_records(path, start_line=start_line):
            last_line = line_no
            message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                last_user_idx = None
                continue

            record_ts = _extract_ts(payload)
            if role == "user":
                query = _resolve_text(message.get("content"))
                if not query:
                    last_user_idx = None
                    continue
                interactions.append({"query": query, "response": None, "tool_calls": [], "ts": record_ts})
                last_user_idx = len(interactions) - 1
                continue

            if last_user_idx is None:
                continue

            response = _resolve_text(message.get("content"))
            interactions[last_user_idx]["response"] = response
            tool_calls = []
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, list):
                tool_calls = [call for call in raw_calls if isinstance(call, dict)]
            interactions[last_user_idx]["tool_calls"] = tool_calls
            interactions[last_user_idx]["ts"] = record_ts
            last_user_idx = None

        offsets[path_name] = last_line
    return interactions, offsets


def _state_embedder_info(
    meta: dict[str, object],
) -> tuple[Callable[[list[tuple[str, str]]], dict[str, list[float]]], float]:
    """Resolve embed batch fn and default connect similarity."""
    embedder_name = str(meta.get("embedder_name", ""))
    if OpenAIEmbedder is not None and embedder_name == OpenAIEmbedder().name:
        embedder = OpenAIEmbedder()
        return embedder.embed_batch, 0.30
    embedder = HashEmbedder()
    return embedder.embed_batch, 0.0


def run_fast_learning(
    state_path: str,
    session_paths: str | list[str] | list[Path] | Path,
    *,
    workers: int = 4,
    window_radius: int = 8,
    max_windows: int = 6,
    hard_max_turns: int = 120,
    include_reinforcements: bool = True,
    llm_fn: Callable[[str, str], str] | None = None,
    embed_batch_fn: Callable[[list[tuple[str, str]],], dict[str, list[float]]] | None = None,
    checkpoint_path: str | None = None,
    resume: bool = False,
    ignore_checkpoint: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """Run LLM-backed transcript mining and inject new learning nodes."""
    if workers < 1:
        workers = 1
    if llm_fn is None:
        if openai_llm_fn is None:
            raise SystemExit("LLM required for fast-learning; install openai and set OPENAI_API_KEY")
        llm_fn = openai_llm_fn

    checkpoint = _load_checkpoint(checkpoint_path) if (checkpoint_path and resume) else {"sessions": {}}
    if ignore_checkpoint:
        checkpoint = {"sessions": {}}
    start_lines = checkpoint.get("sessions", {})

    grouped_turns, turn_offsets = _collect_turns(session_paths, since_lines=start_lines)
    windows: list[tuple[str, list[SessionTurn], int]] = []
    for source, turns in grouped_turns:
        for idx, (start, end) in enumerate(
            select_feedback_windows(
                turns,
                window_radius=window_radius,
                max_windows=max_windows,
                hard_max_turns=hard_max_turns,
            ),
            start=1,
        ):
            if start < end:
                windows.append((source, turns[start:end], idx))

    all_events: list[dict] = []
    if windows:
        request_ctx = [(source, idx + 1, len(windows), segment) for idx, (source, segment, _) in enumerate(windows)]
        if workers == 1:
            for source, idx, total, segment in request_ctx:
                all_events.extend(_window_to_payload(session=source, window=segment, window_idx=idx, total_windows=total, llm_fn=llm_fn))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _window_to_payload,
                        session=source,
                        window=segment,
                        window_idx=idx,
                        total_windows=total,
                        llm_fn=llm_fn,
                    ): source
                    for source, idx, total, segment in request_ctx
                }
                for future in concurrent.futures.as_completed(futures):
                    all_events.extend(future.result())

    if not include_reinforcements:
        all_events = [event for event in all_events if event.get("type") != "REINFORCEMENT"]

    log_path = learning_event_path(state_path)
    existing = {_event_key(item) for item in event_log_entries(log_path)}
    deduped, skipped = dedupe_learning_events(existing, all_events)
    append_learning_events(log_path, deduped)

    graph, index, meta = load_state(state_path)
    if embed_batch_fn is None:
        embed_batch_fn, connect_min_sim = _state_embedder_info(meta)
    else:
        connect_min_sim = 0.30

    nodes: list[dict[str, object]] = []
    for event in deduped:
        node_type = event.get("type")
        if node_type == "REINFORCEMENT":
            node_type = "TEACHING"
        content_hash = str(event.get("content_hash"))
        pointer = str(event.get("session_pointer"))
        pointer_hash = hashlib.sha256(pointer.encode("utf-8")).hexdigest()[:10]
        event["node_id"] = f"learning::{node_type.lower()}::{content_hash[:12]}::{pointer_hash}"
        nodes.append(
            {
                "id": event["node_id"],
                "type": node_type,
                "content": event.get("content", ""),
                "summary": str(event.get("content", "")).split("\n")[0][:120],
                "metadata": {
                    "source": "full_learning",
                    "type": str(node_type),
                    "session": event.get("session"),
                    "session_pointer": event.get("session_pointer"),
                    "severity": event.get("severity"),
                    "content_hash": event.get("content_hash"),
                },
            }
        )

    injected = {"injected": 0, "edges_added": 0, "inhibitory": 0, "skipped": 0}
    if nodes:
        injected = inject_batch(
            graph=graph,
            index=index,
            nodes=nodes,
            embed_batch_fn=embed_batch_fn,
            connect_top_k=3,
            connect_min_sim=connect_min_sim,
        )
        _persist_state(graph=graph, index=index, meta=meta, state_path=state_path, backup=backup)

    if checkpoint_path and resume:
        _save_checkpoint(checkpoint_path, turn_offsets)

    return {
        "windows": len(windows),
        "events_extracted": len(all_events),
        "events_appended": len(deduped),
        "events_skipped": skipped,
        "events_injected": int(injected.get("injected", 0)),
        "edges_added": int(injected.get("edges_added", 0)),
        "learning_events_path": str(log_path),
        "session_offsets": turn_offsets,
    }


def run_harvest(
    state_path: str,
    *,
    events_path: Path | str | None = None,
    tasks: list[str] | None = None,
    dry_run: bool = False,
    max_merges: int = 5,
    prune_below: float = 0.01,
    backup: bool = True,
) -> dict[str, Any]:
    """Run slow loop maintenance from learning events."""
    if tasks is None:
        tasks = ["split", "merge", "prune", "connect", "scale"]

    event_path = Path(events_path) if events_path is not None else learning_event_path(state_path)
    events = event_log_entries(event_path)
    graph, index, meta = load_state(state_path)

    correction_nodes = defaultdict(int)
    for event in events:
        if str(event.get("type", "")).upper() != "CORRECTION":
            continue
        node_id = event.get("node_id")
        if isinstance(node_id, str) and graph.get_node(node_id) is not None:
            correction_nodes[node_id] += 1

    damped_edges = 0
    for node_id, count in correction_nodes.items():
        if count <= 0:
            continue
        factor = 1.0 / (1.0 + (0.25 * count))
        for _target, edge in graph.outgoing(node_id):
            before = edge.weight
            edge.weight = before * factor
            if edge.weight != before:
                damped_edges += 1
        for _source, edge in graph.incoming(node_id):
            before = edge.weight
            edge.weight = before * factor
            if edge.weight != before:
                damped_edges += 1

    if not dry_run:
        _persist_state(graph=graph, index=index, meta=meta, state_path=state_path, backup=backup)

    embedder_name = meta.get("embedder_name")
    if OpenAIEmbedder is not None and embedder_name == OpenAIEmbedder().name:
        embed_fn = OpenAIEmbedder().embed
    else:
        embed_fn = HashEmbedder().embed

    maintenance_report = run_maintenance(
        state_path=state_path,
        tasks=tasks,
        embed_fn=embed_fn,
        dry_run=dry_run,
        max_merges=max_merges,
        prune_below=prune_below,
    )

    return {
        "events_seen": len(events),
        "correction_nodes": len(correction_nodes),
        "damped_edges": damped_edges,
        "maintenance": {
            "tasks_run": maintenance_report.tasks_run,
            "edges_before": maintenance_report.edges_before,
            "edges_after": maintenance_report.edges_after,
            "pruned_edges": maintenance_report.pruned_edges,
            "pruned_nodes": maintenance_report.pruned_nodes,
            "merges_applied": maintenance_report.merges_applied,
            "merges_proposed": maintenance_report.merges_proposed,
        },
        "learning_events_path": str(event_path),
    }
