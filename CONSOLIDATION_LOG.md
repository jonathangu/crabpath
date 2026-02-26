# Consolidation Log

## Scope
Deep consolidation sweep completed on the public CrabPath package.

## Changes

1. Added shared IO/query utilities in `crabpath/_io.py`.
   - Added:
     - `load_graph(path) -> Graph`
     - `load_index(path) -> EmbeddingIndex`
     - `load_query_stats(path) -> dict`
     - `load_mitosis_state(path) -> MitosisState`
     - `load_snapshot_rows(path) -> list[dict]`
     - `split_csv(value) -> list[str]`
     - `graph_stats(graph) -> dict`
     - `run_query(graph, index, query_text, top_k, embed_fn)`
     - `build_firing(graph, fired_ids)`
     - `build_snapshot(graph)`
     - `health_metric_available` / `health_metric_status`
     - `build_health_rows(health, has_query_stats)`
2. Updated `crabpath/cli.py` to consume shared helpers from `_io.py`.
   - Removed duplicated implementations for graph/index loading, query execution, snapshot building, health rows formatting, and stats computation.
   - Removed leftover local helper duplicates for these paths.
3. Updated `crabpath/mcp_server.py` to consume shared helpers from `_io.py`.
   - Removed duplicated load/query/health/snapshot/stats logic.
4. Removed `crabpath/consolidation.py` shim (all uses now come directly from `crabpath.graph`).
   - `tests/test_consolidation.py` already imports directly from `crabpath.graph`.
5. Added package entrypoint.
   - Added `crabpath/__main__.py`.
   - `python -m crabpath` now routes to `crabpath.cli.main()`.
6. Removed toy workspace artifact copy.
   - Deleted `examples/toy-workspace/` and kept `examples/toy_workspace/`.
7. Removed root `ignored` artifact file and added `/ignored` to `.gitignore`.
8. Tightened public exports.
   - Removed redundant/legacy names from `crabpath/__init__.py`:
     - removed `consolidation_should_merge`
     - removed exported `inhibition` module re-export.
     - updated CLI example to `python -m crabpath`.
9. Sanity-checked examples for obvious import/API drift and runability:
   - `examples/*.py` and `examples/openclaw_shadow_hook.sh` were reviewed and no stale imports were found.
10. Created/updated verification artifacts.
   - Added `CONSOLIDATION_LOG.md` (this file).

## Validation

- `python3 -m ruff check crabpath/ tests/ examples/ --fix`
- `python3 -m pytest --tb=short -q`

Result: `305 passed, 0 failed`.

## Notes / remaining

- External benchmark anchor was not present in the repo; left as TODO in the previous list.
