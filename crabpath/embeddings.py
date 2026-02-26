"""Local embeddings — included with crabpath. No API key needed."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


def local_embed_fn(text: str) -> list[float]:
    """Embed a single text using all-MiniLM-L6-v2 (local, no API key)."""
    if not hasattr(local_embed_fn, "_model"):
        local_embed_fn._model = SentenceTransformer("all-MiniLM-L6-v2")
    return local_embed_fn._model.encode(text).tolist()


def local_embed_batch_fn(texts: list[tuple[str, str]]) -> dict[str, list[float]]:
    """Batch embed using all-MiniLM-L6-v2 (local, no API key)."""
    if not hasattr(local_embed_batch_fn, "_model"):
        local_embed_batch_fn._model = SentenceTransformer("all-MiniLM-L6-v2")
    model = local_embed_batch_fn._model
    ids = [nid for nid, _ in texts]
    contents = [content for _, content in texts]
    vectors = model.encode(contents).tolist()
    return dict(zip(ids, vectors))
