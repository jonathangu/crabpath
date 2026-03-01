## Unreleased

### Operator CLI: canonical brain-on command (issues #24, #25)
- Added `openclawbrain serve --state <path> [--socket-path <path>] [--foreground]` as the first-class foreground socket service command for OpenClaw operators.
- Startup now prints an operator banner with socket path, state path, status-check command, and `Ctrl-C` shutdown guidance.
- Updated operator docs to recommend `openclawbrain serve` as the canonical production startup path.

### Replay checkpoint UX overhaul (issues #24, #25)
- Added `openclawbrain replay --show-checkpoint` to print checkpoint status without opening JSON files.
  - Human output includes checkpoint path, schema version, fast-learning progress/status, replay progress/merge batches, and whether resume would take effect.
  - `--json` emits stable schema payloads with `type: checkpoint_status`.
- `openclawbrain replay` now prints a startup banner in text mode with checkpoint path, `resume`, `ignore_checkpoint`, and planned phases.
- Backward compatibility: if `fast_learning.sessions` or `replay.sessions` is missing, replay falls back to legacy top-level `sessions` offsets and emits warnings.
- Added checkpoint fixture-based tests for legacy/new schemas, replay/fast-learning legacy fallback warnings, and startup/checkpoint status output.
- Updated operator docs (`docs/ops-recipes.md`, `docs/setup-guide.md`) with `--show-checkpoint` and explicit resume semantics.

### Fast-learning progress visibility (issues #24, #25)
- `run_fast_learning()` now supports an optional `on_progress` callback with payloads shaped like:
  - `{"type":"progress","phase":"fast_learning","completed","total","elapsed_seconds","rate","eta_seconds","updated_at"}`
- `openclawbrain replay` now forwards fast-learning progress through the existing CLI progress emitter, including JSON progress lines (`type=progress`) and text-mode elapsed/rate/ETA details.
- Added tests for CLI JSON progress emission during fast-learning.
- Updated `docs/ops-recipes.md` with a fast-learning progress output example.

### Replay ops hardening + simple parallel replay v0 (issue #19)
- `openclawbrain replay` adds:
  - `--progress-every N` (periodic progress, JSONL events when `--json` is enabled)
  - `--checkpoint-every-seconds S` and `--checkpoint-every K` (frequent checkpoints during fast-learning and replay)
  - `--persist-state-every-seconds S` (periodic state persistence during replay)
  - `--stop-after-fast-learning` (cutover mode: inject/embed/persist, then exit before replay/harvest)
  - `--replay-workers N` (simple true-parallel replay mode; default `1`)
- Replay now supports line-offset checkpoint resume through replay sessions, and writes replay checkpoints incrementally during processing.
- Added simple deterministic parallel replay reducer: workers compute shard deltas from traversal results, reducer merges batches in deterministic order, and checkpoints are written after merges.
- Added tests for replay progress events, replay checkpoint+resume behavior, stop-after-fast-learning early exit, and parallel replay mode output.

### OpenAI LLM client stability (issue #17)
- `openai_llm.py`: made OpenAI clients thread-local for batch-thread safety, added client-level defaults `timeout=60` and `max_retries=2` when supported, and narrowed the `TypeError` fallback in `openai_llm_fn` to timeout-only compatibility only.

### Self-healing replay for rotated/missing session files
- `replay.py`: `extract_queries`, `extract_query_records`, and `extract_interactions` now return empty lists (with a stderr warning) when a session file is missing or a broken symlink, instead of raising `SystemExit`. This lets long full-learning rebuilds survive rotated or deleted session files.
- `full_learning.py`: `_read_records` and `collect_session_files` apply the same self-healing behavior — missing or broken-symlink paths are skipped with a warning. If *all* paths are invalid, `collect_session_files` still exits with a helpful message.
- Added tests for missing-file and broken-symlink cases in both modules.

### Bug fix: maintain.py `apply_connections` import
- **CRITICAL**: `maintain.py` was missing `from .connect import apply_connections`, causing a `NameError` crash whenever `connect_learning_nodes` ran during full-learning harvest. Added the import and a regression test.

### Best brain by default
- `replay` now defaults to full-learning (fast-learning + harvest). Use `--edges-only` for cheap edge-only replay.
- `init` now defaults to `--embedder auto --llm auto`, which tries OpenAI and falls back to hash/none gracefully.
- Added `--edges-only` flag for replay to explicitly request old edge-only behavior.
- Guarded `full_learning.py` OpenAI imports with try/except so the module loads without openai installed.
- `split.py`: `split_workspace` now auto-gates LLM file splitting by default; LLM is used by default only for files with >= `DEFAULT_LLM_SPLIT_MIN_CHARS` characters. Set `should_use_llm_for_file=lambda rel, text: True` to force LLM splitting on all files.

### Full-learning replay and harvest

- Added `openclawbrain replay --fast-learning` for LLM transcript mining + injection from session logs.
- Added `openclawbrain replay --full-learning` to run replay + fast-learning + slow maintenance.
- `--full-learning` now enables decay during replay by default and runs harvest with decay task (`decay,scale,split,merge,prune,connect`).
- Added `--decay-during-replay` flag to enable periodic decay during any replay run (not just full-learning).
- Added `--decay-interval N` flag (default 10) to control how often decay fires during replay.
- Added resumability options: `--resume`, `--checkpoint`, and `--ignore-checkpoint`.
- Added learning-context knobs: `--workers`, `--window-radius`, `--max-windows`, `--hard-max-turns`.
- Added explicit `--backup` control for replay-backed mutation persistence.
- Added new `openclawbrain harvest` command to consume `learning_events.jsonl` and apply `split/merge/prune/connect/scale`.
- Added dedupe/idempotent behavior for event log ingestion keyed by `(type, sha256(content), session_pointer)`.
- Documented sidecar full-learning workflow and updated `docs/setup-guide.md`, `docs/openclaw-integration.md`, `docs/FULL_REBUILD.md`.
- Added tests for window selection, dedupe/idempotency, replay + fast-learning injection behavior, and decay-during-replay edge reduction.

## v12.2.1 (2026-02-27)

### Bug fixes (from deployment feedback — GH #4, #5, #6)
- Fixed --llm openai hang on init (timeout + empty list guard)
- Added embed_batch progress output (prevents false SIGTERM)
- CLI maintain now passes embed_fn from state metadata
- Daemon maintain delegates to run_maintenance (all 7 tasks)
- Added --embedder arg to maintain CLI parser
- LLM timeout increased to 60s + split content truncated to 8K chars
- Stale socket cleanup on daemon startup
- Replay diagnostic output + meta leak prevention
- install_launchd.sh auto-detects Python path

### New features
- openclawbrain status command (one-command health overview)
- Injected nodes (CORRECTION/TEACHING/DIRECTIVE) preserved on rebuild
- docs/vs-plain-rag.md comparison (honest: BM25 wins cold start)
- docs/agents_hook.md (when to query/learn/inject/edit)
- docs/upgrade-guide.md (recommended upgrade sequences)
- BM25 baseline added to benchmark suite
- Ablation study (5 conditions x 10 seeds)
- _rechunk_oversized: large texts split properly, never truncated
- connect_learnings.py accepts --state flag

## v12.2.0 (2026-02-27)

### Learning rule
- Switched default learning rule from heuristic (apply_outcome) to policy-gradient (apply_outcome_pg). The PG update redistributes probability mass across all outgoing edges at each visited node (sum of updates ≈ 0), replacing the inflationary heuristic that only modified traversed edges. Simulation shows 10-15x better branch separation, 24x faster recovery after concept drift, and 30% less weight inflation.

### Decay
- Increased default decay half-life from 80 to 140 cycles. With PG's deflationary updates replacing the inflationary heuristic, edges need less aggressive decay. At half-life 140: edges queried 1x/day stabilize at 0.24 (habitual), abandoned edges still reach prune threshold by day 17.

### Maintenance
- Added runtime node splitting (`split_node`, `suggest_splits`). The inverse of merge — splits bloated multi-topic nodes into focused children with automatic edge rewiring. Integrates into maintenance pipeline between decay and merge.

### Self-regulation
- Added tier hysteresis by widening the habitual band to `0.15 - 0.6` (from `0.2 - 0.6`), reducing reflex/habitual thrashing.
- Added homeostatic decay with adaptive half-life (bounded to 60-300): reflex-edge ratio is tracked and used to speed up/slow down decay to maintain a 5-15% reflex band.
- Added synaptic scaling between decay and split: outgoing positive mass is softly capped per node (`budget=5.0`) to prevent hub dominance and preserve network balance.

### Autonomous learning API (`self_learn`)
- New `self_learn` daemon method, socket client method, and CLI subcommand. Enables agents to learn from self-observed outcomes without human feedback.
- Full spectrum: negative outcome + CORRECTION (penalize + inhibit), neutral + TEACHING (inject only), positive outcome + TEACHING (reinforce + inject).
- `self_correct` available as backward-compatible alias.
- Unified internal `_do_learn` core — `learn`, `correction`, and `self_learn` all delegate to one function. No duplicated logic.

### Bug fixes
- **CRITICAL**: `save_state` is now atomic (write to temp → fsync → rename). Keeps `.bak` of previous state. A crash during save can no longer corrupt the brain.
- `read_journal` skips malformed JSON lines instead of crashing the entire health/replay pipeline.
- `split_workspace` skips non-UTF8/binary files instead of failing the entire init pass.

## v12.1.1 (2026-02-27)

### Bug fixes (from issue #3)
- **embed_batch chunking**: `OpenAIEmbedder.embed_batch()` now chunks requests to stay within OpenAI's 300K token-per-request limit. Previously sent all texts in a single call, which failed on workspaces with >~750 nodes.
- **openai optional dependency**: added `openai` to `[project.optional-dependencies]` so `pip install openclawbrain[openai]` pulls it in. Previously required manual `pip install openai`.
- **--version flag**: `openclawbrain --version` now works. Previously errored out.

## v12.1.0 (2026-02-27)

### Socket transport and client tooling
- `socket_server.py`: asyncio Unix socket server wrapping the NDJSON daemon worker.
- `socket_client.py`: sync stdlib-only client with context-manager lifecycle and CLI entrypoint.
- Adapter scripts (`query_brain.py`, `learn_correction.py`): added `--socket` flag and auto-detection of `daemon.sock`.
- Added LaunchAgent plists and macOS install script integration for socket-server lifecycle.
- Added integration test coverage for socket wrapper and client path.

## v12.0.1 (2026-02-27)

### Documentation and operators
- Documented `openclawbrain daemon` as the long-lived production path, including NDJSON protocol, startup command, timing expectations (including production 504ms on Mac Mini M4 Pro), methods, and limitations.
- Clarified daemon capability gaps (`inject`, fired-log emission, socket interface) and added explicit roadmap items in `docs/architecture.md` and `docs/setup-guide.md`.

## v12.0.0 (2026-02-27)

### Rename and compatibility
- Rebranded package from legacy namespace to `openclawbrain`.
- `pyproject.toml` metadata updated: `name`, `version`, `description`, keywords, and scripts.
- Added compatibility entry points in `pyproject.toml` for `openclawbrain`, `ocb`, and OpenClawBrain compatibility alias.
- Added OpenClawBrain compatibility shim package that re-exports `openclawbrain` with deprecation warnings.
- Added default-state behavior for `~/.openclawbrain/` with fallback from legacy paths when migrating.
- Updated CLI state resolution and daemon/progress naming to the new package.
- Added backward-compatibility tests for shim imports and warning behavior.

## v11.2.0 (2026-02-27)

### New: Persistent worker daemon (`openclawbrain daemon`)
- `openclawbrain/daemon.py`: loads state once, keeps graph+index hot in memory
- JSON-RPC over stdin/stdout (NDJSON protocol)
- Methods: query, learn, maintain, health, info, save, reload, shutdown
- Query responses include timing: `embed_query_ms`, `traverse_ms`, `total_ms`
- Auto-save after N write ops, graceful SIGTERM/SIGINT handling
- Eliminates per-call state reload (~100-800ms savings)
- `examples/ops/client_example.py`: Python client demo

### New: Constitutional anchors (`openclawbrain anchor`)
- Authority levels: constitutional (never decay/prune/merge), canonical (slow decay), overlay (default)
- `openclawbrain maintain` respects authority levels during all structural ops
- CLI: `openclawbrain anchor --state PATH --node-id ID --authority constitutional|canonical`

### New: Incremental file sync (`openclawbrain sync`)
- `openclawbrain/sync.py`: detect file changes, re-embed only what changed
- Default authority map for common files (SOUL.md → constitutional, USER.md → canonical)
- CLI: `openclawbrain sync --state PATH --workspace DIR`

### New: Daily note compaction (`openclawbrain compact`)
- `openclawbrain/compact.py`: old notes → extract facts → inject into graph → shrink files
- Optional LLM summarization, deterministic fallback
- CLI: `openclawbrain compact --state PATH --memory-dir DIR`

### Fix: save_state() embedder metadata preservation
- Now reads existing meta before writing; no more silent hash-v1/1024 overwrite
- Dimension mismatch raises ValueError (prevents corrupt state)
- Fixes GitHub Issue #1 item #7 (CormorantAI production data corruption)

### Fix: Rebuild preserves injected nodes
- `--preserve-injected` flag (default True) in init_agent_brain.py and rebuild_all_brains.py
- Snapshots + restores CORRECTION/TEACHING nodes across rebuilds
- Fixes GitHub Issue #1 item #5

### Docs
- README overhaul: all 17 CLI commands, apply_outcome_pg section, "Why OpenClawBrain", write policy
- `docs/architecture.md`: persistent worker, context lifecycle, constitutional anchors
- `docs/setup-guide.md`: steps 7-9 (sync, anchors, compact)
- Production stats: 4 brains (MAIN, PELICAN, BOUNTIFUL, CORMORANT)
- `jonathangu.com/openclawbrain/gu2016/`: accessible policy gradient derivation page

### Tests
- 246 tests (was 220), 15 simulations

## v11.1.0 (2026-02-27)

### New: Maintenance engine (`openclawbrain maintain`)
- `openclawbrain/maintain.py`: `run_maintenance()`, `prune_edges()`, `prune_orphan_nodes()`
- CLI: `openclawbrain maintain --state PATH --tasks health,decay,prune,merge [--dry-run]`
- Scheduler-agnostic: call from cron, systemd, Airflow, or any batch runner

### New: Two-timescale architecture docs
- `docs/architecture.md`: online learning (fast loop) vs maintenance (slow loop)
- Integration contract: what frameworks provide, what OpenClawBrain produces
- Brain directory layout specification

### New: Framework-agnostic examples (`examples/ops/`)
- `callbacks.py`: `make_embed_fn()` and `make_llm_fn()` factory functions
  - Default: OpenAI text-embedding-3-small + gpt-5-mini
  - Fallback: HashEmbedder (zero-dep, for testing)
- `query_and_learn.py`: fast loop example with --embedder flag
- `run_maintenance.py`: slow loop example with --embedder and --llm flags

### New: True REINFORCE policy gradient (`apply_outcome_pg`)
- Full derivation: softmax score function ∇log π = (1/τ)(eₐ - π)
- Updates ALL outgoing edges at each visited node (not just traversed)
- Conservation property: weight updates sum to zero per node
- Temperature scaling, baseline support, discount along trajectory

### New: 12 simulations + expanded benchmarks
- PG vs heuristic ablation (weight conservation)
- Noise robustness (graceful to 20%, degrades at 30%)
- Static vs learning baseline
- Scaling analysis (sub-linear: 50→2000 nodes)
- Internal benchmark: 45 queries, 8 methods
- External benchmarks: MultiHop-RAG (n=1000) + HotPotQA (n=500) with OpenAI embeddings

### Paper updates
- Title: "Learned Graph Traversal for Retrieval Routing"
- Full PG derivation with numerical examples
- External benchmark results (cold traverse +7.7% MultiHop, +33.6% HotPotQA)
- De-marketing pass: removed metaphors, added crisp definitions
- Honest finding: naive online learning degrades on non-repeating queries

# CHANGELOG

## Unreleased

### Maintenance loop and ops docs
- Added `openclawbrain.maintain` with `run_maintenance`, `prune_edges`, and `prune_orphan_nodes`.
- Added `openclawbrain maintain` CLI command with `--state`, `--tasks`, `--dry-run`, `--max-merges`, and `--prune-below`.
- Added framework-agnostic examples in `examples/ops/` for fast loop and maintenance loop.
- Added `docs/architecture.md` documenting fast/slow loops, integration contracts, and scheduler guidance.
- Added `tests/test_maintain.py` covering maintenance execution, dry-run behavior, prune operations, merge effects, and report accounting.

### Daemon mutation + correction fixes (fixes #2)
- feat: daemon `inject` method for TEACHING/CORRECTION/DIRECTIVE nodes.
- feat: daemon `correction` method performs atomic penalize + inject and supersedes `learn_correction.py`.
- feat: in-memory fired-log ring buffer with per-chat-id tracking in daemon mode.
- fix: `query_brain.py` now respects embedder metadata instead of hardcoding OpenAI defaults.
- Mentions: fixes #2.

## v10.4.0 (2026-02-27)

### True REINFORCE policy gradient
- Added `apply_outcome_pg` in `openclawbrain.learn` for true policy-gradient updates:
  - numerically stable softmax over outgoing edges plus synthetic STOP
  - updates for all outgoing edges at each step using `(1[chosen] - π)` score gradient
  - optional baseline and temperature support
  - weight clipping + Hebbian co-firing preserved
- Exported `apply_outcome_pg` from `openclawbrain.__init__`.
- Extended `LearningConfig` with `temperature` and `baseline`.

### External benchmark additions
- Added external benchmark harness support for MultiHop-RAG and HotPotQA.
- Added OpenAI embedding cache support for benchmark scripts to support fast repeat runs.

## v10.3.0 (2026-02-27)

### Traversal tiers recalibrated from production data
- **Defaults changed:** `reflex_threshold: 0.8 -> 0.6`, `habitual_range: 0.3-0.8 -> 0.2-0.6`, `inhibitory_threshold: -0.3 -> -0.01`.
- **Production root-cause fix:** inhibitory edges were non-functional because the default threshold was `-0.3` while negatives were near `-0.035`.
- **Tests and sims updated:** adjusted tier-classification expectations and added coverage for `-0.02` as inhibitory.
- Test count increased to **204** in docs to match new coverage.

## v10.2.0 (2026-02-27)

### Traversal engine hardening
- **Traversal termination updated:** stops on hop ceiling, fired-node budget, and context-character budget.
- **Defaults changed:** `beam_width: 3 → 8`, `max_hops: 15 → 30`, `fire_threshold: 0.0 → 0.01`.
- **New control:** `max_fired_nodes` hard-cap added for traversal loops.
- **Context budget enforcement:** `max_context_chars` is enforced during traversal, not only at render time.
- **Default CLI / script behavior:** `query_brain.py` now defaults to `max_context_chars=20000` for logged session flow.
- Test count updated to **203** (from 200).

## v10.1.0 (2026-02-27)

### Real-time correction flow
- **Fired-node logging:** `query_brain.py --chat-id` persists fired nodes to `fired_log.jsonl` per conversation, with 7-day rolling prune. Corrections can now find what to penalize even hours after the original query.
- **Same-turn corrections:** new `learn_correction.py` reads fired log by chat_id, penalizes via `apply_outcome`, and optionally injects a CORRECTION node — all in one command.
- **Dedup with batch harvester:** `injected_corrections.jsonl` prevents double-injection when the 2-hour harvester later processes the same correction. `init_agent_brain.py` reads the dedup log on rebuild.

### CLI DX improvements
- **Query text output shows node IDs** — no more `--json` required for the learn workflow. Format: `node_id / ~~~ / content`.
- **`learn --json` returns summary** (`{"edges_updated": N, "max_weight_delta": X}`) instead of dumping the entire graph.
- **`health` text output is human-readable** — `Brain health: / Nodes / Edges / Reflex% Habitual% Dormant% / Orphans / Cross-file edges`.
- **`inject` auto-detects hash-v1 embedder** and defaults `--connect-min-sim` to 0.0 (was 0.3, causing zero connections for users without OpenAI).

### OpenAI integration
- `openclawbrain/openai_embeddings.py` — `OpenAIEmbedder` class wrapping `text-embedding-3-small`.
- `openclawbrain/openai_llm.py` — `openai_llm_fn` and `chat_completion` for GPT-5-mini routing/scoring.
- CLI flags: `--embedder openai` for init/query/inject, `--llm openai` for init/merge/connect.
- Zero new required dependencies — `openai` is an optional runtime import.

### Documentation overhaul
- **README rewritten:** value prop first, 5-minute quickstart with A→B learning story, "Correcting mistakes" and "Adding new knowledge" sections promoted to top.
- **TEACHING documented:** all three injection types (CORRECTION, TEACHING, DIRECTIVE) explained with examples. Previously only CORRECTION was documented — three production agents independently thought rebuild was required to add new knowledge.
- **Competition table:** OpenClawBrain vs Plain RAG vs Reflexion vs MemGPT.
- **New sections:** State lifecycle, Cost control, "How OpenClawBrain differs from related tools".
- **Session replay demoted** to "Optional: warm start" with skip note for new users.

### New examples
- `examples/cold_start/` — guided zero-session walkthrough with canned queries and expected output.
- `examples/correction_flow/` — generic fired-node logging pattern for any agent framework (hash embedder, no API key).
- `examples/sample_workspace/` expanded from 3 to 5 files (added `incidents.md`, `onboarding.md`) for cross-file routing demos.
- `examples/openclaw_adapter/agents_hook.md` updated with full inject + correction + chat-id logging template.

### Tests
- 200 tests (was 188), 8 simulations all passing.
- New tests: hash-embedder inject default, query text node IDs, health readable output, learn summary, correction flow integration (3 tests).

## v10.0.0
### What Changed
- Added CLI and API flows so OpenClawBrain can inject and persist correction/teaching signals without rebuilding the entire state.
- Expanded operational guidance and verification workflow so feedback and correction behavior can be validated deterministically.

- Added live injection APIs (`inject_node`, `inject_correction`, `inject_batch`) with CLI support via `openclawbrain inject`, enabling runtime updates during operation.
- Added direct graph node injection paths for TEACHING, DIRECTIVE, and CORRECTION workflows, plus lightweight injection stats payloads from the CLI.

- Expanded reproducibility docs with live injection verification steps so users can confirm graph updates and health signals after corrective actions.

## v9.3.1
- Hardened command entrypoints and replay/logging behavior for edge cases in graph indexing and state workflows.
- Improved docs around runtime injection, exports, and deterministic test paths to make troubleshooting and operator handoff faster.

## v9.3.0
- Introduced live injection primitives (`inject_*`) and correction-node inhibitory edge behavior.
- Added the `openclawbrain inject` command path and supporting test coverage for repeatable feedback workflows.

## v9.1.0
- Added adversarial tests and latency benchmark harness updates to surface stability issues before release.
- Added interaction extraction and benchmark cleanup paths so simulation artifacts are predictable across runs.

## v9.0.0
- Delivered 20 user-feedback fixes focused on inhibitory suppression, file-type handling, `max_chars`, and `doctor/info` behavior.
- Tuned graph traversal defaults and diagnostics so noisy retrieval cases now recover more gracefully in production-like inputs.

## v8.0.0
- Unified state format and added dimension validation to prevent accidental embedding mismatches.
- Documented core design tenets to make production constraints explicit for integrators.

## v7.0.0
- Brought `HashEmbedder` into the core implementation as a built-in default path.
- This removed external required dependencies from baseline operation while preserving callback flexibility.

## v6.0-6.1
- Removed subprocess-based paths and deduplicated helper logic to simplify execution flow.
- This reduced operational complexity and improved reliability for smaller deployments.

## v5.0-5.3
- Removed provider-specific integrations and standardized pure callback behavior.
- This made core graph operations less opinionated and easier to adapt across embedding/LLM stacks.

## v4.0-4.5
- Added LLM-based features and batch callback execution paths.
- Improved how route extraction and scoring can combine deterministic traversal with optional model guidance.

## v3.0.0
- Rewrote the graph engine core with zero dependencies.
- Added a stable, dependency-light baseline for local, offline usage and reproducible graph state files.
