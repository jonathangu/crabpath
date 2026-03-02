"""Local ONNX embedder wrapper with normalized vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LOCAL_MODEL_TAG = "bge-small-en-v1.5"

_MODEL_CACHE: dict[str, object] = {}


def _normalize(vec: object) -> list[float]:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError("embedding output must be a 1D vector")
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return [float(v) for v in arr.tolist()]


@dataclass
class LocalEmbedder:
    """Fast local embedder powered by `fastembed`."""

    model_name: str = DEFAULT_LOCAL_MODEL
    _dim: int | None = None

    @property
    def name(self) -> str:
        model_tag = self.model_name.rsplit("/", 1)[-1]
        return f"local:{model_tag}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(""))
        return self._dim

    def _model(self):
        cached = _MODEL_CACHE.get(self.model_name)
        if cached is not None:
            return cached
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError("fastembed is required for local embeddings") from exc
        model = TextEmbedding(model_name=self.model_name)
        _MODEL_CACHE[self.model_name] = model
        return model

    def embed(self, text: str) -> list[float]:
        vectors = list(self._model().embed([text]))
        if not vectors:
            raise RuntimeError("local embedder returned no vectors")
        normalized = _normalize(vectors[0])
        self._dim = len(normalized)
        return normalized

    def embed_batch(self, texts: list[tuple[str, str]]) -> dict[str, list[float]]:
        if not texts:
            return {}
        ids = [node_id for node_id, _ in texts]
        contents = [content for _, content in texts]
        vectors = list(self._model().embed(contents))
        if len(vectors) != len(ids):
            raise RuntimeError("local embedder batch size mismatch")
        normalized_vectors = [_normalize(vector) for vector in vectors]
        if normalized_vectors:
            self._dim = len(normalized_vectors[0])
        return dict(zip(ids, normalized_vectors))
