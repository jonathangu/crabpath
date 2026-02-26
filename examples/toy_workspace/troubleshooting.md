# Acme Bot Troubleshooting Guide

This file captures recurring operational issues and recommended mitigations.

## Symptom: Repeated Timeout Spikes

- Check upstream dependency health.
- Verify the deployment query path in `runbook.md`.
- If latency is from repeated retries, reduce seed breadth in query config.

## Symptom: Incorrect Rollback Recommendations

- Confirm rollbacks are encoded as protected nodes in the graph.
- Run a query against `architecture.md` references to confirm guardrail precedence.
- Rebuild bootstrap after editing guardrail sections.

## Symptom: Memory Saturation

- Inspect graph node growth and prune weak edges.
- Regenerate embeddings after large documentation updates.
- Ensure index persists via the save/load path in `api-reference.md`.

## Symptom: Unclear API Errors

- Verify endpoint request shape from `api-reference.md`.
- Re-run the sample queries from `README.md` in isolated mode.

A useful first check is whether the bootstrap source changed unexpectedly.
That recovery flow begins in `runbook.md` and ends with schema verification in
`api-reference.md`.
