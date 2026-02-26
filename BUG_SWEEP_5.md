# Bug Sweep #5 — Security, performance, and release readiness

## Summary

Findings were focused on public-attack surface and release hardening. Two security/input-validation issues and two release-readiness issues were fixed, each with regression tests.

## Findings and fixes

1. High — MCP file path traversal and unsafe path acceptance
- Severity: High
- Location: `crabpath/mcp_server.py`
- Issue: MCP handlers accepted raw `graph/index/output_*/snapshots/workspace` values from user JSON-RPC and passed them to file loaders/writers, allowing crafted paths like `../../`.
- Fix:
  - Added strict path coercion helpers in `crabpath/mcp_server.py` that reject `..` path components.
  - Applied validation before every file load/save in MCP handlers.
- Tests:
  - `tests/test_mcp_server.py::test_query_validation_blocks_path_traversal`

2. Medium — MCP numeric/string argument validation surfaced as internal errors
- Severity: Medium
- Location: `crabpath/mcp_server.py`
- Issue: Missing or malformed keys/types in MCP arguments could raise `KeyError`/`TypeError` and return JSON-RPC internal errors instead of structured input errors.
- Fix:
  - Added coercion helpers: `_coerce_str`, `_coerce_bool`, `_coerce_int`, `_coerce_float`, `_coerce_string_list`.
  - Updated `_handle_request` to map input/typing failures to `-32602`.
  - Updated MCP handlers to use required/optional coercion before processing.
- Tests:
  - `tests/test_mcp_server.py::test_query_validation_rejects_bad_inputs`

3. Medium — MCP `health` tool accepted schema mismatch for `query_stats`
- Severity: Medium
- Location: `crabpath/mcp_server.py`
- Issue: Tool schema exposed `query_stats` as `object`, while runtime expected a path string, causing invalid API usage and unexpected failures.
- Fix:
  - Added `_coerce_query_stats` to accept inline query stats dicts or file paths.
  - Updated health tool schema to allow `query_stats` as object or string path.
- Tests:
  - `tests/test_mcp_server.py::test_health_accepts_inline_query_stats`

4. Low — Release dependency version policy
- Severity: Low
- Location: `pyproject.toml`
- Issue: Optional dependency requirements were broad unconstrained ranges.
- Fix:
  - Added upper bounds to optional dependencies (`openai`, `google-generativeai`, `cohere`, `requests`, and `dev`).

5. Low — PEP 561 marker missing
- Severity: Low
- Location: `crabpath/py.typed`
- Issue: Package lacked `py.typed`, so static type checkers do not treat `crabpath` as typed by default.
- Fix:
  - Added `crabpath/py.typed`.

## Validation run

- `python3 -m pytest --tb=short -q` (360 passed)
- `python3 -m pip install -e . --break-system-packages` (editable install succeeds)

## Additional review notes (no code changes required)

- No hardcoded secrets, `eval`, `exec`, `pickle`, or `marshal` usage was found in library code.
- No obvious command/shell injection in MCP/CLI paths.
- `README.md`, `CONTRIBUTING.md`, and `CHANGELOG.md` were checked for obvious release-blocking inconsistencies; no additional blockers identified.
