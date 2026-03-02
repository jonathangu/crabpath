"""State persistence interface and JSON implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..graph import Graph
from ..index import VectorIndex
from ..store import load_state, save_state


class StateStore(ABC):
    """Storage boundary for loading/saving graph+index state."""

    @abstractmethod
    def load(self, path: str) -> tuple[Graph, VectorIndex, dict[str, object]]:
        """Load state from the given path."""

    @abstractmethod
    def save(
        self,
        path: str,
        *,
        graph: Graph,
        index: VectorIndex,
        meta: dict[str, object] | None = None,
    ) -> None:
        """Persist state to the given path."""


class JsonStateStore(StateStore):
    """StateStore implementation backed by existing JSON helpers."""

    def load(self, path: str) -> tuple[Graph, VectorIndex, dict[str, object]]:
        return load_state(path)

    def save(
        self,
        path: str,
        *,
        graph: Graph,
        index: VectorIndex,
        meta: dict[str, object] | None = None,
    ) -> None:
        save_state(graph=graph, index=index, path=path, meta=meta)
