"""Feedback helpers for delayed learning attribution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Optional
from typing import Callable

from .graph import Graph
from .synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    record_cofiring,
    record_correction,
)

DEFAULT_SNAPSHOT_PATH = "crabpath_events.db"


RETRIEVAL_SCORING_MODEL = "gpt-5-mini"
RETRIEVAL_HELPFULNESS_GATE = 0.1  # Aggressive: almost every scored query produces RL signal
DEFAULT_OPENAI_TIMEOUT = 8.0

_CORRECTION_START_PATTERNS = (
    "no",
    "wrong",
    "not that",
    "actually",
    "that's incorrect",
)

_CORRECTION_PHRASES = (
    "should be",
    "instead of",
    "i meant",
    "let me correct",
)

_NEGATION_PATTERNS = (
    "don't do",
    "never",
    "stop",
)

_NOT_COMMA_PATTERN = re.compile(r"\bnot\s+[^,]+,\s*[^,]+", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _strip_quotes(value: str) -> str:
    return value.strip().strip("\"'“”")


def detect_correction(current_message: str, previous_message: str) -> float:
    """Detect if `current_message` is correcting the prior assistant response.

    This is intentionally lightweight and keyword-driven. No LLM call is required.
    """
    current = _normalize_text(_strip_quotes(current_message or ""))
    previous = _normalize_text(_strip_quotes(previous_message or ""))

    if not current:
        return 0.0

    strong_corrections = (
        _CORRECTION_START_PATTERNS
        + ("that's wrong", "no, that's", "wrong answer", "that's not", "no.")
    )
    mild_corrections = _NEGATION_PATTERNS

    for phrase in strong_corrections:
        if current == phrase:
            return -1.0
        if current.startswith(f"{phrase} "):
            return -1.0
        if phrase in current:
            return -1.0

    for phrase in _CORRECTION_PHRASES:
        if phrase in current:
            return -1.0

    if _NOT_COMMA_PATTERN.search(current):
        return -0.5

    # Add some simple context from the previous message when available
    context = f"{previous} {current}"
    for phrase in mild_corrections:
        if phrase in context:
            return -0.5

    return 0.0


def score_retrieval(
    query: str,
    retrieved_nodes: list[tuple[str, str]],
    actual_response: str,
    llm_call: Optional[Callable[[str, str], str]] = None,
) -> dict[str, Any]:
    """Ask the LLM to score retrieved node relevance for a given response."""
    if not retrieved_nodes:
        return {"scores": {}, "overall": 0.0}

    def default_scoring_call(prompt: str, system: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "pip install openai required for LLM scoring. Use --no-score to skip."
            ) from exc

        client = OpenAI()
        response = client.chat.completions.create(
            model=RETRIEVAL_SCORING_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=1,  # gpt-5-mini only supports temperature=1
            response_format={"type": "json_object"},
            timeout=DEFAULT_OPENAI_TIMEOUT,
        )
        if not response.choices:
            return ""
        message = response.choices[0].message
        return str(message.content or "")

    scorer = llm_call or default_scoring_call
    node_lines: list[str] = []
    for node_id, content in retrieved_nodes:
        snippet = (content or "").replace("\n", " ")[:220]
        node_lines.append(f"- {node_id}: {snippet}")

    system = (
        "You score whether document chunks were useful for answering a query. For each chunk: "
        "1.0 = essential (query cannot be answered without it), 0.5 = helpful "
        "(provides useful context), 0.0 = irrelevant, -0.5 = misleading, -1.0 = actively "
        "wrong. Be generous — if a chunk is partially relevant, score 0.5. Output JSON: "
        '{"scores": {"node_id": float}, "overall": float}'
    )
    prompt = (
        f"Query: {query}\n"
        "\n"
        "Chunks retrieved:\n"
        f"{chr(10).join(node_lines)}\n"
        "Which chunks would help answer this query? Score each."
    )

    default_scores = {node_id: 0.0 for node_id, _ in retrieved_nodes}

    try:
        raw = scorer(prompt, system)
    except ImportError:
        raise
    except Exception:
        return {"scores": default_scores, "overall": 0.0}

    parsed = _parse_retrieval_scores(raw)
    node_scores = parsed.get("scores", {})
    normalized_scores: dict[str, float] = {}
    for node_id in [node_id for node_id, _ in retrieved_nodes]:
        normalized_scores[node_id] = _coerce_score(node_scores.get(node_id, 0.0))

    overall = parsed.get("overall", 0.0)
    return {
        "scores": normalized_scores,
        "overall": _coerce_score(overall),
    }


def _coerce_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return score


def _parse_retrieval_scores(raw: str) -> dict[str, float]:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    if not text:
        return {"scores": {}, "overall": 0.0}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"scores": {}, "overall": 0.0}

    if not isinstance(payload, dict):
        if isinstance(payload, list):
            parsed: dict[str, float] = {}
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                node_id = str(
                    entry.get("node_id")
                    or entry.get("node")
                    or entry.get("id")
                    or entry.get("target")
                )
                if not node_id:
                    continue
                score = entry.get("score")
                parsed[node_id] = _coerce_score(score)
            return {"scores": parsed, "overall": 0.0}
        return {"scores": {}, "overall": 0.0}

    parsed_scores: dict[str, float] = {}
    payload_scores = payload.get("scores", {})
    if isinstance(payload_scores, dict):
        for key, value in payload_scores.items():
            if not isinstance(key, str):
                continue
            parsed_scores[key] = _coerce_score(value)

    overall = _coerce_score(payload.get("overall", 0.0))

    if isinstance(payload, dict):
        return {"scores": parsed_scores, "overall": overall}

    return {"scores": {}, "overall": overall}


def no_reward_on_missing_signal(
    correction: float,
    retrieval_helpfulness: float | dict[str, float] | None = None,
    *,
    min_helpfulness: float = RETRIEVAL_HELPFULNESS_GATE,
) -> float | None:
    """Return a reward only when there is real feedback signal.

    If correction is negative, return that correction reward.
    If scoring exists, require it to clear a minimum helpfulness floor.
    """
    if correction < 0.0:
        return correction
    if retrieval_helpfulness is None:
        return None

    if isinstance(retrieval_helpfulness, dict):
        node_scores = list(retrieval_helpfulness.values())
    else:
        node_scores = [retrieval_helpfulness]

    if not node_scores:
        return None
    max_score = max(_coerce_score(score) for score in node_scores)
    if max_score > 0.5:
        return max_score
    if max_score >= min_helpfulness:
        return 0.3
    if max_score < min_helpfulness:
        return None
    return None


def auto_feedback(
    query: str,
    user_followup: str,
    trajectory: list[str],
    graph: Graph,
    syn_state: SynaptogenesisState,
    config: SynaptogenesisConfig | None = None,
) -> dict[str, Any]:
    """Apply automatic feedback from a correction signal plus shadow-mode logic."""
    config = config or SynaptogenesisConfig()
    correction = detect_correction(user_followup, query)

    if correction < 0.0:
        corrections = record_correction(
            graph=graph,
            trajectory=trajectory,
            reward=correction,
            config=config,
        )
        return {
            "query": query,
            "user_followup": user_followup,
            "trajectory": trajectory,
            "correction": correction,
            "action": "record_correction",
            "implicit_reward": correction,
            "results": corrections,
        }

    cofire_result = record_cofiring(
        graph=graph,
        fired_nodes=trajectory,
        state=syn_state,
        config=config,
    )
    reward = no_reward_on_missing_signal(correction)
    return {
        "query": query,
        "user_followup": user_followup,
        "trajectory": trajectory,
        "correction": 0.0,
        "action": "record_cofiring",
        "implicit_reward": reward,
        "results": cofire_result,
    }


def snapshot_path(graph_path: str | None = None) -> Path:
    """Resolve snapshot location.

    Args:
        graph_path: Optional graph path to scope the events file. If omitted,
            use CRABPATH_SNAPSHOT_PATH if set, else package default.
    """
    env_path = os.getenv("CRABPATH_SNAPSHOT_PATH")
    if env_path:
        return Path(env_path)
    if graph_path:
        return Path(graph_path).with_suffix(".events.db")
    return Path(DEFAULT_SNAPSHOT_PATH)


def _load_raw_snapshots(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _save_raw_snapshots(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _coerce_turn_id(record: dict[str, Any]) -> Optional[int]:
    turn_id = record.get("turn_id")
    try:
        return int(turn_id)
    except (TypeError, ValueError):
        return None


def map_correction_to_snapshot(
    session_id: str,
    turn_window: int = 5,
) -> Optional[dict[str, Any]]:
    """Find the latest attributable snapshot for a session.

    The window is interpreted as max turn distance from the most recent
    non-attributed snapshot.
    """
    path = snapshot_path()
    records = _load_raw_snapshots(path)
    candidates = [r for r in records if r.get("session_id") == session_id]
    if not candidates:
        return None

    parsed = []
    for record in candidates:
        turn = _coerce_turn_id(record)
        if turn is None:
            continue
        parsed.append((turn, record))
    if not parsed:
        return None

    parsed.sort(key=lambda item: item[0], reverse=True)
    latest_turn = parsed[0][0]
    env_turn = os.getenv("CRABPATH_CORRECTION_TURN")
    if env_turn is None:
        correction_turn = latest_turn
    else:
        try:
            correction_turn = int(env_turn)
        except ValueError:
            correction_turn = latest_turn

    for turn, record in parsed:
        if correction_turn - turn <= turn_window:
            record = dict(record)
            record["turns_since_fire"] = max(0, correction_turn - turn)
            return record

    return None


def auto_outcome(corrections_count: int, turns_since_fire: int) -> float:
    """Generate a coarse outcome score from delayed feedback signals.

    Args:
        corrections_count: Number of correction hits attached to this firing context.
        turns_since_fire: Distance from the firing turn to the feedback event.
    """
    if corrections_count > 0:
        return -1.0
    if turns_since_fire >= 5:
        return 0.3
    return 0.0
