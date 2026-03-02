"""Event store interfaces and JSONL-backed implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from ..journal import log_event, read_journal


class EventStore(ABC):
    """Abstract append-only event store."""

    @abstractmethod
    def append(self, event: dict[str, object]) -> None:
        """Append one event payload."""

    @abstractmethod
    def iter_since(self, ts: float) -> Iterator[dict[str, object]]:
        """Iterate events with timestamp >= ts."""

    @abstractmethod
    def read_last(self, n: int) -> list[dict[str, object]]:
        """Read the last N events."""


class JsonlEventStore(EventStore):
    """EventStore backed by the existing journal.jsonl format."""

    def __init__(self, path: str) -> None:
        self.path = str(Path(path).expanduser())

    def append(self, event: dict[str, object]) -> None:
        # log_event adds ts/iso fields and keeps canonical journal formatting.
        log_event(dict(event), journal_path=self.path)

    def iter_since(self, ts: float) -> Iterator[dict[str, object]]:
        cutoff = float(ts)
        for entry in read_journal(journal_path=self.path):
            entry_ts = entry.get("ts")
            if isinstance(entry_ts, (int, float)) and float(entry_ts) >= cutoff:
                yield entry

    def read_last(self, n: int) -> list[dict[str, object]]:
        return read_journal(journal_path=self.path, last_n=max(0, int(n)))
