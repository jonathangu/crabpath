"""Event journal interface and JSONL implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..journal import log_event, read_journal


class EventStore(ABC):
    """Storage boundary for append-only event logs."""

    @abstractmethod
    def append(self, event: dict[str, object]) -> None:
        """Append one event record."""

    @abstractmethod
    def iter_since(self, since_ts: float | None) -> list[dict[str, object]]:
        """Return events with timestamp >= since_ts (or all events when None)."""

    @abstractmethod
    def read_last(self, n: int) -> list[dict[str, object]]:
        """Read last n event records."""


class JsonlEventStore(EventStore):
    """EventStore implementation backed by existing journal helpers."""

    def __init__(self, path: str) -> None:
        self.path = path

    def append(self, event: dict[str, object]) -> None:
        log_event(dict(event), journal_path=self.path)

    def iter_since(self, since_ts: float | None) -> list[dict[str, object]]:
        entries = read_journal(journal_path=self.path)
        if since_ts is None:
            return [dict(entry) for entry in entries if isinstance(entry, dict)]
        out: list[dict[str, object]] = []
        cutoff = float(since_ts)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ts = entry.get("ts")
            if isinstance(ts, (int, float)) and float(ts) >= cutoff:
                out.append(dict(entry))
        return out

    def read_last(self, n: int) -> list[dict[str, object]]:
        if n <= 0:
            return []
        entries = read_journal(journal_path=self.path, last_n=n)
        return [dict(entry) for entry in entries if isinstance(entry, dict)]
