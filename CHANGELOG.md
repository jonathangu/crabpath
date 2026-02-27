# CHANGELOG

## v10.0.0
- Added live injection API (`inject_node`, `inject_correction`, `inject_batch`) with CLI support via `crabpath inject`.
- Added direct graph node injection paths for TEACHING/DIRECTIVE/CORRECTION workflows and lightweight injection stats payloads.
- Expanded reproducibility docs with live injection verification steps.

## v9.3.1
- Hardened command entry-points and replay/logging behavior for edge cases in graph index + state workflows.
- Improved docs around runtime injection, exports, and deterministic test paths.

## v9.3.0
- Introduced live injection primitives (`inject_*`) and correction-node inhibitory edge behavior.
- Added `crabpath inject` command path and supporting test coverage.

## v9.1.0
- Added adversarial tests, latency benchmark harness updates, interaction extraction, and benchmark cleanup.

## v9.0.0
- Delivered 20 user-feedback fixes, including inhibitory suppression, file type handling, `max_chars`, and `doctor/info` behavior.

## v8.0.0
- Unified state format, added dimension validation, and documented core design tenets.

## v7.0.0
- Brought `HashEmbedder` into the core implementation.

## v6.0-6.1
- Removed subprocess code paths and deduplicated helper logic.

## v5.0-5.3
- Stripped provider-specific integrations and standardized pure callback behavior.

## v4.0-4.5
- Added LLM features and batch callback execution paths.

## v3.0.0
- Rewrote the graph engine core with zero dependencies.
