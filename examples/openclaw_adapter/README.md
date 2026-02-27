# OpenClaw adapter

This adapter is for frameworks that manage API keys internally.

`OPENAI_API_KEY` must be available in `os.environ` at execution time. The framework process injects it before invoking these scripts; no key discovery, keychain lookup, or dotfile parsing is used.

There is no manual key configuration for these scripts. Provide workspace, sessions, and output paths and the scripts run end-to-end.

The adapter is the integration layer between the pure CrabPath library and the framework. It handles:

- Building a workspace graph with `openai-text-embedding-3-small` metadata
- Persisting `state.json` (and legacy `graph.json`/`index.json` for compatibility)
- Querying the graph via `query_brain.py`
- Replaying history and printing health diagnostics in `init_agent_brain.py`
- Rebuilding all production brains with learnings in `rebuild_all_brains.py`
- Connecting learning nodes to workspace nodes with `connect_learnings.py`

Script specifics:

- `rebuild_all_brains.py`: full rebuild from workspace + learning DB + sessions, then connect learning nodes to workspace nodes and run health checks.
- `connect_learnings.py`: standalone utility to connect learning nodes to workspace nodes for one `--agent` or explicit `--state`.
- `init_agent_brain.py`: simpler rebuild from workspace + sessions with optional learning DB integration:
  - `--learning-db <path>`: load active learnings from SQLite as learning nodes
  - `--connect-learnings` / `--no-connect-learnings`: enable or disable the learning/learning->workspace connection and correction inhibition step (defaults to enabled when `--learning-db` is passed)
  - `--batch-size <int>`: override OpenAI embedding batch size (default `100`)
- `query_brain.py`: query a brain with OpenAI embeddings.
- `agents_hook.md`: AGENTS.md integration block.

Production notes:
- Uses real OpenAI embeddings via caller-supplied callbacks.
- Supports 3-agent operational deployments in production.

## AGENTS.md integration block

To wire the adapter into an AGENTS.md workflow, see `agents_hook.md` for the hook block that defines query, learn, rebuild, and health commands.
