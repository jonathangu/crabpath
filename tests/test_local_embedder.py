from __future__ import annotations

import importlib
import math
import sys
import types


def test_local_embedder_embed_is_normalized_and_sets_dim(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            for text in texts:
                scale = float(len(text) or 1)
                yield [3.0 * scale, 4.0 * scale, 0.0, 0.0]

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    module_name = "openclawbrain.local_embedder"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    module._MODEL_CACHE.clear()

    embedder = module.LocalEmbedder()
    vec = embedder.embed("hello")
    assert len(vec) == 4
    assert embedder.dim == 4
    assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-6)


def test_local_embedder_embed_batch_is_normalized(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            for idx, _text in enumerate(texts):
                yield [1.0 + idx, 0.0, 0.0]

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    module_name = "openclawbrain.local_embedder"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    module._MODEL_CACHE.clear()

    embedder = module.LocalEmbedder()
    vectors = embedder.embed_batch([("a", "alpha"), ("b", "beta")])
    assert set(vectors) == {"a", "b"}
    assert embedder.dim == 3
    assert math.isclose(sum(v * v for v in vectors["a"]), 1.0, rel_tol=1e-6)
    assert math.isclose(sum(v * v for v in vectors["b"]), 1.0, rel_tol=1e-6)
