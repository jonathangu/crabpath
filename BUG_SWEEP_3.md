# BUG SWEEP #3 — CLI, MCP, persistence, and public-facing polish

## Scope
Validated public-facing behavior for install/CLI/MCP/persistence/README/examples on current `crabpath` tree (2026-02-26).

## Findings

1. 🔴 `crabpath explain` crashes when no seed overlap exists
   - Symptom: `_format_explain_trace` consumers expected `selected_nodes`, but successful explain paths that returned no match omitted this key.
   - Fix: `crabpath/cli.py` now always includes `selected_nodes` and `fired_with_reasoning` in all `explain` trace shapes.
   - Test: `tests/test_cli.py::test_explain_command_without_seeds_returns_no_traceback`

2. 🟡 Example bootstrap signature mismatch in `examples/quickstart.py`
   - Symptom: script called `bootstrap_workspace` with old positional signature, causing runtime failure.
   - Fix: Updated to current workflow (`gather_files` + `bootstrap_workspace(graph, workspace_files, llm_call=..., state=...)`) and safer graph replay behavior.
   - Test: `tests/test_examples.py::test_quickstart_main_runs_with_temporary_workspace`

3. 🟡 Optional dependency hard-failures on example import
   - Symptom: `examples/langchain_adapter.py` and `examples/openai_agent.py` raised import-time `RuntimeError`/`ModuleNotFoundError` paths when optional providers were missing.
   - Fix:
     - `examples/openai_agent.py`: moved OpenAI import/client creation into `_build_openai_client()`; top-level import now clean.
     - `examples/langchain_adapter.py`: guarded optional imports and provided fallback stubs/clear runtime messaging.
   - Test: `tests/test_examples.py::test_examples_import_cleanly`

4. 🟡 Graph deserialization robustness gaps
   - Symptom: malformed graph payloads could crash `Graph.load` (invalid node/edge entries, missing fields, wrong container types).
   - Fix: `Graph.from_dict()` now skips malformed node/edge items, tolerates missing IDs/fields, and normalizes invalid `nodes`/`edges` containers to empty lists.
   - Test: `tests/test_graph.py::test_graph_load_ignores_invalid_records_and_preserves_valid_data`

5. 🟡 Atomic save hardening
   - Symptom: save path was replace-based but lacked an explicit fsync, weakening crash-safety.
   - Fix: `Graph.save()` flushes and fsyncs temp file before `os.replace` to reduce corruption risk.
   - Test: `tests/test_persistence.py::test_graph_save_atomic_replace_failure_preserves_existing_file`

6. 🟡 MCP server metadata inconsistency
   - Symptom: MCP initialize response used a hardcoded version string.
   - Fix: `crabpath/mcp_server.py` now reports package `__version__`.
   - Test: `tests/test_mcp_server.py::test_initialize_reports_current_package_version`

7. 🟢 `python -m crabpath` exit-code propagation
   - Symptom: module runner did not explicitly return CLI status.
   - Fix: `crabpath/__main__.py` now `sys.exit(main())` for correct exit codes.

8. 🟢 Error handling polish
   - Symptom: unexpected CLI exceptions could surface noisy tracebacks instead of structured errors.
   - Fix: `crabpath/cli.py` wraps command dispatch in a general exception handler and emits JSON error payload.

## Additional checks

1. Packaging metadata alignment
   - Added `tests/test_packaging.py`:
     - version in `crabpath/__init__.py` matches `pyproject.toml`
     - `crabpath` console script exists in `pyproject.toml`

2. CLI/help surface sanity
   - Manual `--help` runs verified for all public subcommands.
   - `--version`, `init --example --json`, `health`, `query`, `explain`, `stats --json` verified on test graphs.

3. MCP import and tool schema validation
   - Verified `crabpath.mcp_server` imports cleanly.
   - Tool schema objects are valid JSON-serializable dicts.

4. README checks
   - Quickstart API examples checked for consistency.
   - Headline test count updated to `327`.
   - Link probe found:
     - `https://jonathangu.com/crabpath/` → 200
     - `https://jonathangu.com` → 200
     - `https://en.wikipedia.org/wiki/Carcinisation` returned 403 in this environment (may be request-agent restricted, not necessarily dead).

5. Environment/installability
   - `pip install -e .` fails on this host without override due PEP 668 externally-managed policy.
   - `python3 -m pip install -e . --break-system-packages` succeeds and installs `crabpath 1.0.0`.
   - Installed entrypoint works: `crabpath --version`.

## Required command validation

- `python3 -m pytest --tb=short -q`
  - Result: `336 passed`
- `python3 -m crabpath --version`
- `python3 -m crabpath init --example --json 2>&1 | python3 -m json.tool`
