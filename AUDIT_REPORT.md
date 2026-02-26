# AUDIT_REPORT.md

## 1. Code Quality & Completeness

- 🔴 CRITICAL `crabpath/migrate.py` — `parse_session_logs()` calls `_is_system_noise()` from two branches, but no symbol with that name exists in module scope. This produces a runtime `NameError` during log parsing.
- 🔴 CRITICAL `crabpath/migrate.py` — the same parser wraps per-record processing in a broad `except Exception` and ignores failures, so malformed input yields silently empty results instead of surfaced parse errors.
- 🟡 IMPORTANT `crabpath/migrate.py` — `parse_session_logs()` is used as the primary ingestion path for session replay, yet there is no explicit contract test for invalid log schema in a happy-path test matrix, so failures are non-obvious in normal operation.
- 🟡 IMPORTANT `crabpath/feedback.py` — public conversion helpers such as `snapshot_path()` and scoring helpers are only partially tested via indirect usage; several internal helpers (`_normalize_text`, `_strip_quotes`, `_coerce_score`, `_parse_retrieval_scores`) have no direct behavioral tests and are not documented with public API comments.
- ℹ️ NOTE `rg -n "TODO|FIXME|HACK" crabpath tests` returned no matches, so unfinished-work markers are not used for tracking; this is an observation rather than a blocker.
- ℹ️ NOTE Several module-level helpers are unannotated or loosely typed via broad `Any` unions; this appears deliberate for schema-flexible persistence parsing, but it increases the chance of silent runtime coercion issues.
- ℹ️ NOTE Type coverage is uneven: classes and public functions in larger modules (`mcp_server.py`, `synaptogenesis.py`, `autotune.py`) have docstrings for high-level behavior, while smaller utility functions in those files are often undocumented.

## 2. Test Coverage Gaps

- 🔴 CRITICAL `tests/test_migrate.py::test_parse_jsonl_logs`, `tests/test_migrate.py::test_parse_openclaw_jsonl_message_field`, `tests/test_migrate.py::test_parse_max_queries`, `tests/test_migrate.py::test_migrate_with_replay` fail because of the missing `_is_system_noise` path.
- 🟡 IMPORTANT `crabpath/adapter.py` has no direct test module/import coverage.
- 🟡 IMPORTANT `crabpath/mcp_server.py` has no dedicated tests; MCP and CLI behavior can diverge without safety nets.
- 🟡 IMPORTANT `crabpath/_structural_utils.py` helper utilities are largely untested by direct unit tests.
- 🟡 IMPORTANT Embedding provider surface in `crabpath/embeddings.py` is under-tested: `test_embeddings.py` only asserts `_cosine`.
- 🟡 IMPORTANT CLI command-level coverage exists, but error-path behavior and schema-fallback behavior for many commands are lightly covered.
- 🟡 IMPORTANT `tests/test_neurogenesis.py` and `tests/test_shadow_mode_v2_gating.py` do not import core modules via stable package names, reducing discoverability of intended public surface.
- ℹ️ NOTE Untested public functions/methods identified by name:
  - `crabpath/adapter.py`: `QueryResult`, `ConsolidationResult`, `CrabPathAgent`
  - `crabpath/embeddings.py`: `openai_embed`, `gemini_embed`, `cohere_embed`, `ollama_embed`, `auto_embed`
  - `crabpath/_structural_utils.py`: `parse_markdown_json`, `split_fallback_sections`, `node_file_id`, `count_cross_file_edges`, `classify_edge_tier`
  - `crabpath/router.py`: `normalize_router_payload`
  - `crabpath/traversal.py`: `_build_router_context`, `render_context`
  - `crabpath/mcp_server.py`: all handler-level entry points
- ℹ️ NOTE `tests/test_embeddings.py` imports private `_cosine` directly, which is an implementation-detail test and brittle against internal refactors.

### Module-to-test mapping (requested matrix)
- `crabpath/graph.py` → `tests/test_graph.py`, `tests/test_consolidation.py`, `tests/test_router.py`, `tests/test_inhibition.py`, `tests/test_synaptogenesis.py`, `tests/test_mitosis.py`, `tests/test_simulator.py`
- `crabpath/decay.py` → `tests/test_decay.py`
- `crabpath/inhibition.py` → `tests/test_inhibition.py`, `tests/test_controller.py`
- `crabpath/learning.py` → `tests/test_learning.py`, `tests/test_simulator.py`, `tests/test_controller.py`
- `crabpath/autotune.py` → `tests/test_autotune.py`
- `crabpath/synaptogenesis.py` → `tests/test_synaptogenesis.py`
- `crabpath/mitosis.py` → `tests/test_mitosis.py`, `tests/test_bootstrap.py`
- `crabpath/router.py` → `tests/test_router.py`, `tests/test_traversal.py`, `tests/test_simulator.py`
- `crabpath/traversal.py` → `tests/test_traversal.py`, `tests/test_simulator.py`
- `crabpath/controller.py` → `tests/test_controller.py`
- `crabpath/feedback.py` → `tests/test_feedback.py`, `tests/test_adapter.py`
- `crabpath/embeddings.py` → `tests/test_embeddings.py`
- `crabpath/cli.py` → `tests/test_cli.py`
- `crabpath/migrate.py` → `tests/test_migrate.py`
- `crabpath/lifecycle_sim.py` → `tests/test_lifecycle_sim.py`, `tests/test_simulator.py`
- `crabpath/legacy/activation.py` → `tests/test_activation.py`
- `crabpath/shadow_logger.py` → `tests/test_shadow_logger.py`
- `crabpath/mcp_server.py` → not mapped
- `crabpath/adapter.py` → not mapped
- `crabpath/_structural_utils.py` → not mapped

## 3. Architecture Review

- ℹ️ NOTE Module structure is relatively cohesive: core graph primitives (`graph.py`) are cleanly separated from policies (`learning.py`, `synaptogenesis.py`, `mitosis.py`) and infrastructure (`cli.py`, `mcp_server.py`).
- ℹ️ NOTE No explicit circular-import evidence was found in the static read-through, but both CLI and MCP duplicate persistence and formatting logic (loading graph/index/query stats/snapshots/health), creating maintenance coupling.
- 🟡 IMPORTANT CLI exports a broad API surface in `crabpath/cli.py` (query/learn/split/sim/consolidate/health/migrate, etc.), while MCP (`crabpath/mcp_server.py`) omits at least `init`-style session/bootstrap and extract-session flows that CLI offers; parity is incomplete.
- ℹ️ NOTE `crabpath/__init__.py` appears to be the intended public surface and generally exports high-level classes (`Graph`, `Node`, `Edge`, `MemoryController`, config objects, etc.), which keeps downstream imports ergonomic.
- 🟡 IMPORTANT `mcp_server.py` and `cli.py` independently build similar payloads and error handling; divergence risk is non-trivial and should be reduced by shared service-layer helpers.

## 4. Design Doc ↔ Code Consistency

- 🟡 IMPORTANT `PAPER_CODE_AUDIT.md` documents schema-level drift between the paper descriptions and runtime Node/Edge fields (e.g., inhibitory weight and schema defaults). These are marked as known docs-to-code mismatches and should be resolved or explicitly versioned if the implementation is intentionally diverged.
- ℹ️ NOTE `AUDIT.md` already flags behavior differences in query semantics and JSON safety risks (e.g., `_asdict()` assumptions). The codebase appears not to have closed these in a single coherent fix set.
- 🟡 IMPORTANT `ARCHITECTURE_REVIEW.md` claims a unified execution path; current duplicate CLI/MCP parser pipelines indicate partial divergence despite intent.
- 🟡 IMPORTANT `CHANGELOG.md` and `README.md` advertise MCP+CLI parity plus command list; CLI-only commands (like `init`, `extract`) are absent in MCP tool registry and could confuse users following docs.
- ℹ️ NOTE `README.md` usage examples for `MemoryController` and `migrate` generally match current API names, but there is no automated doc-code verification in tests.

## 5. What's Missing

- 🟡 IMPORTANT Production hardening gap: no structured telemetry/trace IDs for failed migrate/replay parse events (currently swallowed), which blocks reliable incident triage.
- 🟡 IMPORTANT Production hardening gap: migration and replay path has no deterministic schema-version negotiation/checksummed artifact; reproducibility depends on implicit assumptions.
- 🟡 IMPORTANT A stranger-facing API gap: no API-versioned migration story for serialized graph/node schema changes across releases.
- 🟡 IMPORTANT arXiv-style reproducibility gap: formal ablation/sensitivity details and fixed RNG/seed controls are not consistently surfaced in public docs, and several stochastic paths remain only minimally documented.
- 🟡 IMPORTANT Distribution gap: package metadata exists, but there is no strict “public API contract” document mapping command/tool invariants, deprecation policy, and stability guarantees.
- 🟡 IMPORTANT Developer experience gap: no clear test coverage contract indicating which modules are intended public vs internal (private/legacy). This increases accidental reliance on internals.
- ℹ️ NOTE `crabpath/shadow_logger.py` and `crabpath/legacy/*` are useful, but legacy boundaries and API stability boundaries are not explicitly called out in docs.

## 6. The 5 Failing Tests (diagnosis from `python3 -m pytest tests/test_migrate.py -v`)

### Observed result
- Actual run: **4 fails, 10 pass**.
- User-facing expectation in prompt of 5 failing tests does not match current suite state.

### Root cause
- `tests/test_migrate.py` failures all stem from `crabpath/migrate.py:parse_session_logs()` calling `_is_system_noise` without a definition/import.
- The call is inside the broad `try/except` for parsing, so the `NameError` is swallowed and each affected line is dropped.
- This causes `queries == []` and invalid `info` payloads, which cascades into replay failures.

### Fix (minimal and concrete)
- Add a module-local predicate and/or import source:
  - `def _is_system_noise(value: str) -> bool: return value.strip().lower().startswith(("system", "assistant", "tool"))` (or equivalent policy, centralized in one place).
  - Or replace calls with inlined checks.
- Narrow exception handling in `parse_session_logs()` to capture malformed record values, but record and report parse failures instead of silently continuing on every exception.
- Add regression tests for the exact error path (a malformed record should be surfaced and parsing should continue only under explicit policy).

## Severity summary

| category | critical | important | nice | note |
| --- | ---: | ---: | ---: | ---: |
| count | 1 | 11 | 0 | 6 |
