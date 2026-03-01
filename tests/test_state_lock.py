"""Tests for state write lock behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openclawbrain.state_lock import StateLockError, lock_path_for_state, state_write_lock

fcntl = pytest.importorskip("fcntl")


def test_state_lock_acquire_and_release(tmp_path: Path) -> None:
    """Exclusive lock should block other writers until released."""
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    lock_path = lock_path_for_state(state_path)

    with state_write_lock(state_path, command_hint="test-lock"):
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_state_lock_contention_error_message(tmp_path: Path) -> None:
    """Lock contention error should include operator guidance."""
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    lock_path = lock_path_for_state(state_path)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(StateLockError, match="daemon/socket_server"):
            with state_write_lock(state_path, command_hint="openclawbrain replay"):
                pass
        with pytest.raises(StateLockError, match="rebuild_then_cutover"):
            with state_write_lock(state_path, command_hint="openclawbrain replay"):
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
