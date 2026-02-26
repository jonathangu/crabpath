# Key Facts

The team maintains a compact policy-routing graph for context retrieval across long conversations. The graph is built from workspace markdown chunks and then used with traversal and inhibitory dynamics to avoid over-reliance on a single path.

Important current decisions:

- Bootstrap produces both `graph.json` and `embed.json` in the configured data directory.
- Session ingestion accepts legacy JSONL and OpenClaw nested message formats.
- `init` can be run against a built-in toy workspace, useful for smoke tests.
- Explain mode reports seed confidence, candidate ranking, inhibitory suppression, and path reasoning.
- Health checks are expected immediately after migration and before first query.

Known assumptions:

1. Graph schema version is tracked at persistence and checked on load.
2. Atomic writes avoid partial graph corruption.
3. Session extraction should prefer `user` content and skip system-like messages.

Operational memory:

- The workspace defaults stay predictable and explicit.
- If a workspace has weak lexical overlap, fallback seeds still provide usable behavior.
- Inhibition weights can suppress unlikely edges without deleting them.
- Query explainability should favor transparency over compactness.

Recent context summary:

Agents have been using this repo to test end-to-end flows: migrate, extract sessions, run health checks, and inspect explain output. The objective is to keep everything observable and rerunnable.
