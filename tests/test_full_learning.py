from __future__ import annotations

from pathlib import Path

from openclawbrain.full_learning import (
    SessionTurn,
    collect_session_files,
    dedupe_learning_events,
    select_feedback_windows,
    _read_records,
)


def test_select_feedback_windows_picks_likely_feedback_regions() -> None:
    """test select_feedback_windows picks windows around likely feedback turns."""
    turns = [
        SessionTurn(role="user", content="how do I deploy?", line_no=1, source="session.jsonl"),
        SessionTurn(role="assistant", content="yes deploy from branch.", line_no=2, source="session.jsonl"),
        SessionTurn(role="user", content="that is wrong", line_no=3, source="session.jsonl"),
        SessionTurn(role="assistant", content="rollback", line_no=4, source="session.jsonl"),
        SessionTurn(role="user", content="thanks", line_no=5, source="session.jsonl"),
        SessionTurn(role="assistant", content="all good", line_no=6, source="session.jsonl"),
        SessionTurn(role="user", content="please explain", line_no=7, source="session.jsonl"),
    ]

    windows = select_feedback_windows(
        turns,
        window_radius=1,
        max_windows=6,
        hard_max_turns=3,
    )
    assert windows == [(1, 6)]


def test_select_feedback_windows_merges_overlapping_regions() -> None:
    """test select_feedback_windows merges overlapping windows."""
    turns = [
        SessionTurn(role="user", content="hello", line_no=1, source="session.jsonl"),
        SessionTurn(role="user", content="that was wrong", line_no=2, source="session.jsonl"),
        SessionTurn(role="assistant", content="ok", line_no=3, source="session.jsonl"),
        SessionTurn(role="user", content="and again incorrect", line_no=4, source="session.jsonl"),
        SessionTurn(role="assistant", content="yes", line_no=5, source="session.jsonl"),
        SessionTurn(role="user", content="final", line_no=6, source="session.jsonl"),
    ]

    windows = select_feedback_windows(
        turns,
        window_radius=2,
        max_windows=6,
        hard_max_turns=3,
    )
    assert windows == [(0, 6)]


def test_dedupe_learning_events_is_idempotent() -> None:
    """test dedupe_learning_events skips events seen before by deterministic key."""
    events = [
        {
            "type": "CORRECTION",
            "content": "alpha",
            "content_hash": "cafebabe",
            "session_pointer": "s:1-10",
        },
        {
            "type": "CORRECTION",
            "content": "alpha",
            "content_hash": "cafebabe",
            "session_pointer": "s:1-10",
        },
        {
            "type": "TEACHING",
            "content": "alpha",
            "content_hash": "cafebabe",
            "session_pointer": "s:1-10",
        },
    ]
    existing = {"CORRECTION|cafebabe|s:1-10"}
    appended, skipped = dedupe_learning_events(existing=existing, events=events)

    assert skipped == 2
    assert len(appended) == 1
    assert appended[0]["type"] == "TEACHING"


def test_read_records_missing_file_returns_empty(tmp_path: Path) -> None:
    """_read_records returns empty list for a missing file."""
    missing = tmp_path / "gone.jsonl"
    assert _read_records(missing) == []


def test_read_records_broken_symlink_returns_empty(tmp_path: Path) -> None:
    """_read_records returns empty list for a broken symlink."""
    target = tmp_path / "real.jsonl"
    link = tmp_path / "broken.jsonl"
    link.symlink_to(target)  # target does not exist → broken symlink
    assert _read_records(link) == []


def test_collect_session_files_skips_missing_paths(tmp_path: Path) -> None:
    """collect_session_files skips missing files and keeps valid ones."""
    valid = tmp_path / "good.jsonl"
    valid.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")
    missing = tmp_path / "gone.jsonl"
    result = collect_session_files([str(valid), str(missing)])
    assert result == [valid]


def test_collect_session_files_broken_symlink_skipped(tmp_path: Path) -> None:
    """collect_session_files skips broken symlinks gracefully."""
    valid = tmp_path / "good.jsonl"
    valid.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")
    target = tmp_path / "real.jsonl"
    link = tmp_path / "broken.jsonl"
    link.symlink_to(target)
    result = collect_session_files([str(valid), str(link)])
    assert result == [valid]
