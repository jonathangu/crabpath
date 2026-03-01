from __future__ import annotations

from pathlib import Path

from openclawbrain.full_learning import (
    SessionTurn,
    collect_session_files,
    dedupe_learning_events,
    run_fast_learning,
    select_feedback_windows,
    _collect_turns,
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


def test_collect_turns_includes_allowlisted_tool_result_for_media_stub(tmp_path: Path) -> None:
    """_collect_turns includes toolResult text as deterministic tool turns."""
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                '{"role":"user","content":"[media attached: screenshot (image/png)]"}',
                '{"role":"assistant","content":"Let me inspect that image."}',
                '{"role":"toolResult","toolName":"image","content":"OCR: rollout plan says restart api first."}',
                '{"role":"assistant","content":"I see the rollout plan text now."}',
            ]
        ),
        encoding="utf-8",
    )

    grouped, offsets = _collect_turns(
        [session],
        include_tool_results=True,
        tool_result_allowlist={"image"},
        tool_result_max_chars=20000,
    )

    assert len(grouped) == 1
    _, turns = grouped[0]
    assert [turn.role for turn in turns] == ["user", "assistant", "tool", "assistant"]
    assert turns[2].content == "[toolResult:image] OCR: rollout plan says restart api first."
    assert offsets[str(session)] == 4


def test_run_fast_learning_invokes_progress_callback(tmp_path: Path) -> None:
    """run_fast_learning emits progress payloads during fast-learning."""
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"graph":{"nodes":[],"edges":[]},"index":{},"meta":{"embedder_name":"hash-v1","embedder_dim":1024}}',
        encoding="utf-8",
    )
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                '{"role":"user","content":"that guidance is wrong","ts":1.0}',
                '{"role":"assistant","content":"acknowledged","ts":1.1}',
            ]
        ),
        encoding="utf-8",
    )

    events: list[dict[str, object]] = []

    def fake_llm(_: str, __: str) -> str:
        return (
            '{"corrections":[{"content":"Do X instead.","context":"ctx","severity":"high"}],'
            '"teachings":[],"reinforcements":[]}'
        )

    run_fast_learning(
        state_path=str(state_path),
        session_paths=[sessions],
        workers=1,
        max_windows=1,
        llm_fn=fake_llm,
        checkpoint_every=0,
        checkpoint_every_seconds=0,
        on_progress=events.append,
        progress_every_windows=1,
        progress_every_seconds=0,
        backup=False,
    )

    assert events
    assert events[-1]["type"] == "progress"
    assert events[-1]["phase"] == "fast_learning"
    assert events[-1]["completed"] == events[-1]["total"]
    assert isinstance(events[-1]["elapsed_seconds"], float)
    assert isinstance(events[-1]["rate"], float)
