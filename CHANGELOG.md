# CHANGELOG

## v10.0.0
### What Changed
- Added CLI and API flows so CrabPath can inject and persist correction/teaching signals without rebuilding the entire state.
- Expanded operational guidance and verification workflow so feedback and correction behavior can be validated deterministically.

- Added live injection APIs (`inject_node`, `inject_correction`, `inject_batch`) with CLI support via `crabpath inject`, enabling runtime updates during operation.
- Added direct graph node injection paths for TEACHING, DIRECTIVE, and CORRECTION workflows, plus lightweight injection stats payloads from the CLI.

- Expanded reproducibility docs with live injection verification steps so users can confirm graph updates and health signals after corrective actions.

## v9.3.1
- Hardened command entrypoints and replay/logging behavior for edge cases in graph indexing and state workflows.
- Improved docs around runtime injection, exports, and deterministic test paths to make troubleshooting and operator handoff faster.

## v9.3.0
- Introduced live injection primitives (`inject_*`) and correction-node inhibitory edge behavior.
- Added the `crabpath inject` command path and supporting test coverage for repeatable feedback workflows.

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
