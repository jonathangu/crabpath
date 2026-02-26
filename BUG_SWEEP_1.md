# Bug Sweep #1 Findings

Scope: full `crabpath/*.py` scan, with regression coverage in `tests/`.

## Findings and fixes

1. Embedding index silently truncates vectors when model output cardinality mismatches batch size
- Files:
  - `crabpath/embeddings.py` (`EmbeddingIndex.build`)
- Issue:
  - `build()` previously used `dict(zip(ids, all_vectors))`; if `embed_fn` returned fewer vectors than nodes, extra nodes were silently dropped without signal.
- Fix:
  - Added strict batch validation in `EmbeddingIndex._validate_vector_batch`.
  - `build()` now verifies output length per batch and raises `ValueError` on mismatch.
- Tests:
  - `tests/test_embeddings.py::test_build_raises_on_mismatched_embedding_batch_size`

2. Embedding helper assumes non-empty/matching vectors across seed/raw/ upsert paths
- Files:
  - `crabpath/embeddings.py` (`_single_vector`, `seed`, `raw_scores`, `upsert`)
- Issue:
  - Previous code indexed embedding responses with `[0]` without validation, producing uncaught index/type errors in malformed returns.
- Fix:
  - Added central validation helpers for vector type/shape/content checks.
  - Replaced direct indexing in `seed`, `raw_scores`, `upsert` with `_single_vector`.
- Tests:
  - `tests/test_embeddings.py::test_seed_raises_on_empty_embedding_output`
  - `tests/test_embeddings.py::test_raw_scores_raises_on_empty_embedding_output`
  - `tests/test_embeddings.py::test_upsert_raises_on_empty_embedding_output`

3. Malformed embedding output from index not consistently handled in adapter and IO paths
- Files:
  - `crabpath/adapter.py` (`seed`, novelty upsert block)
  - `crabpath/_io.py` (`run_query`)
- Issue:
  - `adapter.seed()` and `run_query()` handled only some embedding errors, but not `ValueError` from malformed embedding outputs.
- Fix:
  - Expanded catches to include `ValueError` for malformed embedding payloads and preserved fallback behavior.
- Tests:
  - `tests/test_adapter.py::test_seed_falls_back_to_keyword_when_index_returns_empty`
  - `tests/test_io.py::test_run_query_falls_back_to_keyword_scoring_on_bad_embedding_output`

## Verification

- Command run: `python3 -m pytest --tb=short -q`
- Result: `316 passed`
