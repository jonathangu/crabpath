# Acme Bot API Reference

This reference maps the most common calls expected from operators and automation.

## Graph Bootstrap API

- Input: workspace path and output graph path.
- Output: graph JSON with nodes, edges, and schema metadata.

## Query API

- Inputs: `query` string plus optional `top_k` and `max_depth`.
- Output: `selected_nodes`, `context`, and confidence-related stats.

## Persistence API

- `Graph.save(path)` writes JSON atomically.
- `Graph.load(path)` supports backward-compatible pre-version payloads.
- `EmbeddingIndex.save(path)` and load methods mirror graph persistence.

## Safety and Routing API

- `MemoryController.query` applies inhibition and decay controls.
- Guardrails are sourced from policy docs and `troubleshooting.md`.

## Tool Mapping

- For procedure operations, follow `runbook.md`.
- For architecture constraints, see `architecture.md`.
- For release and rollback behavior, check `troubleshooting.md` for edge cases.

Use these APIs together: architecture informs strategy, runbook defines execution,
troubleshooting captures failures, and this reference defines signatures.
