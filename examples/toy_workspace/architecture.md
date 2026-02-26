# Acme Bot Architecture

Acme Bot is organized around three layers: ingestion, memory graph, and decision control.
The core graph stores operational facts, policies, and procedural checkpoints as nodes.
Edges represent sequence, dependency, and guardrail relationships.

## Ingestion Layer

Workspace files are bootstrapped into nodes using section headings and metadata.
The bootstrap pipeline attaches source file, section heading, and line boundaries to
node metadata so downstream tooling can trace every answer.

See `runbook.md` for how the operations team updates these sections.

## Memory Layer

The memory graph stores nodes as persistent knowledge and dynamic weights as execution
confidence. Query seeds can originate from lexical matching, embeddings, or both.

For recovery actions, prefer nodes with high edge weight and recent trace activity.

## Safety Layer

Safety nodes are marked as guardrails and treated as protected during consolidation.
This keeps critical constraints from being pruned by automatic cleanup routines.

Cross-reference the guardrail guidance in `troubleshooting.md` before adding new edges.

## Decision Layer

MemoryController consumes graph context, applies inhibition/decay signals, and returns a
ranked trail of nodes as `selected_nodes`.

Detailed runtime contracts and endpoint expectations are in `api-reference.md`.
