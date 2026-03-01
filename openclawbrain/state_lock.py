"""Advisory file lock for state.json single-writer safety."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


FORCE_ENV_VARS = ("OPENCLAWBRAIN_STATE_LOCK_FORCE", "OCB_STATE_LOCK_FORCE")


class StateLockError(RuntimeError):
    """Raised when single-writer state lock cannot be acquired."""


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def force_lock_bypass_enabled(force: bool = False) -> bool:
    """Return True when CLI flag or env var requests bypass."""
    if force:
        return True
    return any(_truthy(os.environ.get(name)) for name in FORCE_ENV_VARS)


def lock_path_for_state(state_path: str | Path) -> Path:
    """Resolve lock file path next to state.json."""
    state = Path(state_path).expanduser()
    return state.parent / f"{state.name}.lock"


def _read_lock_owner(lock_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _contention_message(
    *,
    state_path: Path,
    lock_path: Path,
    command_hint: str | None,
) -> str:
    owner = _read_lock_owner(lock_path)
    owner_command = owner.get("command")
    owner_pid = owner.get("pid")
    owner_bits: list[str] = []
    if isinstance(owner_command, str) and owner_command:
        owner_bits.append(f"command={owner_command}")
    if isinstance(owner_pid, int):
        owner_bits.append(f"pid={owner_pid}")
    owner_text = f" lock owner: {', '.join(owner_bits)}." if owner_bits else ""
    command_text = f" for {command_hint}" if command_hint else ""
    env_hint = " or ".join(f"{name}=1" for name in FORCE_ENV_VARS)
    return (
        f"state write lock is already held{command_text} ({lock_path}) while accessing {state_path}."
        f"{owner_text} This usually means a running daemon/socket_server is the active writer. "
        "Stop the daemon before direct writes, or use rebuild_then_cutover to replay into a new state and cut over safely. "
        f"Experts can bypass with --force or {env_hint}."
    )


@contextmanager
def state_write_lock(
    state_path: str | Path,
    *,
    force: bool = False,
    command_hint: str | None = None,
) -> Iterator[None]:
    """Acquire single-writer lock around state.json mutations."""
    if force_lock_bypass_enabled(force):
        yield
        return

    if fcntl is None:  # pragma: no cover - non-POSIX fallback
        raise StateLockError("state lock requires POSIX fcntl.flock support")

    state = Path(state_path).expanduser()
    lock_path = lock_path_for_state(state)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StateLockError(
                _contention_message(state_path=state, lock_path=lock_path, command_hint=command_hint)
            ) from exc
        acquired = True
        owner = {
            "pid": os.getpid(),
            "command": command_hint or Path(sys.argv[0]).name,
            "acquired_at": time.time(),
        }
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(owner).encode("utf-8"))
        os.fsync(fd)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
