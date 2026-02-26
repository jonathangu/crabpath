# CrabPath Audit Report

## Findings

1. **CRITICAL** — `crabpath/learning.py:79,332-348`
   - **Issue:** Shared per-process baseline state was effectively global and could leak learning signal across controllers/requests when using default config objects.
   - **Fix:** Move baseline storage to `LearningConfig.baseline_state` and guard access with `_BASELINE_LOCK` so each config instance keeps isolated statistics.

2. **HIGH** — `crabpath/controller.py:6,391-393` ✅ **Fixed**
   - **Issue:** `update._asdict()` is checked for serialization, but `LearningResult.updates` uses dataclass objects that do not implement `_asdict`; this made learning summaries non-JSON-safe and changed returned API shape.
   - **Fix:** Use `dataclasses.asdict(update)` when object is a dataclass, else keep dict passthrough.

3. **MEDIUM** — `crabpath/controller.py:252-254`
   - **Issue:** `MemoryController.query()` builds `visited` and `selected_nodes` from only the first seed node, discarding other high-scoring seeds and skipping valid retrieval branches.
   - **Fix:** Start traversal from top-k seeds (or iterate seeds until first hit) and merge trajectories using a priority policy.

4. **MEDIUM** — `crabpath/traversal.py:35-53`
   - **Issue:** `_normalize_seed_nodes()` calls `float(score)` directly for mapping/list inputs; malformed or string-numeric scores can raise `ValueError`/`TypeError` and hard-fail traversal before normalization fallback.
   - **Fix:** Wrap score coercion in `try/except` and skip bad seeds with default weight `1.0`.

5. **MEDIUM** — `crabpath/traversal.py:145` and `crabpath/controller.py:272`
   - **Issue:** Both query pipelines compute different effective hop semantics: traversal uses up to `cfg.max_hops`, while controller uses `traversal_max_hops - 1`, producing inconsistent path lengths across APIs.
   - **Fix:** Normalize hop semantics in one shared policy (e.g., include/exclude start node consistently).

6. **MEDIUM** — `crabpath/learning.py:323-348`
   - **Issue:** `make_learning_step()` updates baseline even for empty/invalid trajectories, allowing baseline drift from non-actionable reward events.
   - **Fix:** Short-circuit empty trajectories to return zero-updates and avoid baseline mutation when `trajectory_steps` is empty.

7. **LOW** — `crabpath/router.py:327-353`
   - **Issue:** `decide_next()` catches all `Exception`, including unexpected runtime/typing regressions, then falls back to heuristic behavior. This can hide real failures.
   - **Fix:** Catch only expected router/parsing/runtime integration errors and re-raise unexpected exceptions.

8. **LOW** — `crabpath/feedback.py:337-349`
   - **Issue:** Snapshot path resolution trusts environment/file inputs directly (`CRABPATH_SNAPSHOT_PATH`, `graph_path`), enabling callers with path injection via configuration to redirect writes.
   - **Fix:** Restrict writable snapshot roots (or validate/expand and reject unexpected absolute/parent traversal paths by policy).

9. **MEDIUM (Test gap)** — `tests/test_controller.py:23`
   - **Issue:** Existing query tests never exercise multi-seed or mixed-seed ranking scenarios, so seed-selection regression in `query()` (issue #3) is unprotected.
   - **Fix:** Add tests with two+ matching seeds and assertions that traversal explores highest-value seed according to explicit policy.

10. **LOW (Test gap)** — `tests/test_router.py:355`
    - **Issue:** `decide_next()` decision path with `score_with_inhibition` output is not covered by direct tests, so inhibition-score influence on routing choices is unverified.
    - **Fix:** Add unit tests that assert ranking changes when inhibitory candidates are present.

11. **LOW (Test gap)** — `tests/test_learning.py:217`
    - **Issue:** No test covers `make_learning_step()` with malformed/empty `trajectory_steps`, so edge-case baseline-stability and input validation risks are untested.
    - **Fix:** Add tests for empty trajectory and mixed candidate formats to ensure controlled returns and no state drift.

## Security note

- No `eval()`, `exec()`, or `pickle`-based deserialization is present.
- External command execution is not used.
- Remaining file-write risks are configuration-path based and should be treated as deployment-hardening concerns.

## Test coverage health check

- New audit-only test added: `test_baseline_state_is_isolated_per_config` in `tests/test_learning.py`.
- Total lint command passed: `python3 -m ruff check crabpath/ tests/`
- Total test command passed: `python3 -m pytest --tb=short` (**269 passed**)

### Severity totals

- CRITICAL: 1
- HIGH: 0 (resolved by fix in `crabpath/controller.py`)
- MEDIUM: 4
- LOW: 4

### Overall code health score

**8 / 10** (good baseline stability and test signal, with a few actionable medium issues around query-seed semantics, trajectory edge cases, and additional test coverage)
