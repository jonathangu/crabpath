"""State store interfaces and JSON-backed implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..graph import Graph
from ..index import VectorIndex
from ..store import load_state, save_state


class StateStore(ABC):
    """Abstract graph/index state store."""

    @abstractmethod
    def load(self) -> tuple[Graph, VectorIndex, dict[str, object]]:
        """Load graph, index, metadata."""

    @abstractmethod
    def save(self, graph: Graph, index: VectorIndex, meta: dict[str, object]) -> None:
        """Persist graph, index, metadata."""


class JsonStateStore(StateStore):
    """StateStore wrapper over existing state.json persistence helpers."""

    def __init__(self, path: str) -> None:
        self.path = str(Path(path).expanduser())

    def load(self) -> tuple[Graph, VectorIndex, dict[str, object]]:
        graph, index, meta = load_state(self.path)
        return graph, index, dict(meta)

    def save(self, graph: Graph, index: VectorIndex, meta: dict[str, object]) -> None:
        save_state(graph=graph, index=index, path=self.path, meta=meta)
