from __future__ import annotations

from math import isclose
from pathlib import Path

from crabpath.index import VectorIndex


def test_vector_index_cosine_known_vectors_and_zero_vector_case() -> None:
    assert isclose(VectorIndex.cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert isclose(VectorIndex.cosine([1.0, 2.0], [2.0, 4.0]), 1.0)
    assert VectorIndex.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_vector_index_zero_vector_handling() -> None:
    assert VectorIndex.cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert VectorIndex.cosine([1.0, 2.0], []) == 0.0
    assert VectorIndex.cosine([1.0], [1.0, 2.0]) == 0.0


def test_vector_index_similarity_identity() -> None:
    assert VectorIndex.cosine([1.5, -2.5, 3.0], [1.5, -2.5, 3.0]) == 1.0


def test_vector_index_identical_vectors_are_max_similarity() -> None:
    index = VectorIndex()
    index.upsert("a", [1.0, 0.0, 0.0])
    index.upsert("b", [0.0, 1.0, 0.0])
    assert index.search([1.0, 0.0, 0.0], top_k=2)[0][0] == "a"


def test_vector_index_orthogonal_vectors() -> None:
    assert VectorIndex.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_vector_index_top_k_with_ties_is_deterministic() -> None:
    index = VectorIndex()
    index.upsert("first", [1.0, 0.0])
    index.upsert("second", [1.0, 0.0])
    index.upsert("third", [0.0, 1.0])

    top = index.search([1.0, 0.0], top_k=2)
    assert top[0][0] == "first"
    assert top[1][0] == "second"
    assert isclose(top[0][1], 1.0)
    assert isclose(top[1][1], 1.0)


def test_vector_index_empty_search() -> None:
    assert VectorIndex().search([1.0, 2.0], top_k=5) == []


def test_vector_index_top_k_non_positive() -> None:
    index = VectorIndex()
    index.upsert("a", [1.0])
    assert index.search([1.0], top_k=0) == []
    assert index.search([1.0], top_k=-2) == []


def test_vector_index_save_load_round_trip() -> None:
    index = VectorIndex()
    index.upsert("a", [0.1, 0.2, 0.3])
    index.upsert("b", [0.4, 0.5, 0.6])
    path = Path("/tmp/index_roundtrip.json")
    index.save(path.as_posix())
    loaded = VectorIndex.load(path.as_posix())

    assert loaded.search([0.1, 0.2, 0.3], top_k=1)[0][0] == "a"


def test_vector_index_upsert_overwrites_id() -> None:
    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    index.upsert("a", [0.0, 1.0])

    result = index.search([1.0, 0.0], top_k=1)
    assert result[0][0] == "a"
    assert isclose(result[0][1], 0.0)


def test_vector_index_remove_and_large_payload() -> None:
    index = VectorIndex()
    index.remove("missing")

    for idx in range(1000):
        index.upsert(f"n{idx}", [float(idx), float(idx + 1)])
    index.remove("n500")

    assert index.search([1000.0, 1001.0], top_k=1)[0][0] == "n999"
    assert all(node_id != "n500" for node_id, _ in index.search([0.0, 0.0], top_k=1000))
