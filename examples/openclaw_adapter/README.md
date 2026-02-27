# OpenClaw adapter

This adapter is for frameworks that manage API keys internally.

`OPENAI_API_KEY` must be available in `os.environ` at execution time. The framework process injects it before invoking these scripts; no key discovery, keychain lookup, or dotfile parsing is used.

There is no manual key configuration for these scripts. Provide workspace, sessions, and output paths and the scripts run end-to-end.

The adapter is the integration layer between the pure CrabPath library and the framework. It handles:

- Building a workspace graph with `openai-text-embedding-3-small` metadata
- Persisting `state.json` (and legacy `graph.json`/`index.json` for compatibility)
- Querying the graph via `query_brain.py`
- Replaying history and printing health diagnostics in `init_agent_brain.py`

Production notes:
- Uses real OpenAI embeddings via caller-supplied callbacks.
- Supports 3-agent operational deployments in production.
