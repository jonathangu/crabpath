"""OpenAI text embedding wrapper."""

from __future__ import annotations

import os

import openai


# Conservative limit: OpenAI allows 300K tokens per request.
# ~4 chars per token, use 250K tokens (1M chars) as buffer.
_MAX_CHARS_PER_CALL = 200_000  # ~50K tokens, well under OpenAI's 300K limit
_MAX_CHARS_PER_TEXT = 15_000   # ~4000 tokens, safely under model's 8192 token limit


class OpenAIEmbedder:
    """OpenAI text embedding wrapper."""

    name: str = "openai-text-embedding-3-small"
    dim: int = 1536

    def __init__(self) -> None:
        """  init  ."""
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """embed."""
        payload = self.client.embeddings.create(model="text-embedding-3-small", input=text[:_MAX_CHARS_PER_TEXT] if len(text) > _MAX_CHARS_PER_TEXT else text)
        return [float(v) for v in payload.data[0].embedding]

    def embed_batch(self, texts: list[tuple[str, str]]) -> dict[str, list[float]]:
        """Embed a batch of (node_id, text) pairs, chunking to stay within API limits."""
        result: dict[str, list[float]] = {}
        chunk: list[tuple[str, str]] = []
        chunk_chars = 0

        for node_id, text in texts:
            text_len = len(text)
            if chunk and chunk_chars + text_len > _MAX_CHARS_PER_CALL:
                self._flush_chunk(chunk, result)
                chunk, chunk_chars = [], 0
            chunk.append((node_id, text))
            chunk_chars += text_len

        if chunk:
            self._flush_chunk(chunk, result)

        return result

    def _flush_chunk(
        self,
        chunk: list[tuple[str, str]],
        result: dict[str, list[float]],
    ) -> None:
        """Send one chunk to the OpenAI API and merge into result."""
        contents = [text[:_MAX_CHARS_PER_TEXT] for _, text in chunk]
        payload = self.client.embeddings.create(
            model="text-embedding-3-small", input=contents
        )
        vectors = [[float(v) for v in item.embedding] for item in payload.data]
        for idx, (node_id, _) in enumerate(chunk):
            result[node_id] = vectors[idx]
