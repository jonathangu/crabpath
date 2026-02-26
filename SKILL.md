# CrabPath

CrabPath is a memory graph for AI agents with migration, query, and graph maintenance operations.

## Requirements

- **Python 3.10+** (macOS ships 3.9 — run `brew install python@3.12` if needed)
- **Virtual environment recommended** on modern macOS (PEP 668):
  ```bash
  python3.12 -m venv ~/.crabpath-env && source ~/.crabpath-env/bin/activate
  ```
- Zero pip dependencies (pure stdlib)
- Optional: `OPENAI_API_KEY` for semantic embeddings
- **Recommended LLM for routing: GPT-5-mini** — the router only outputs tiny JSON decisions; use the cheapest model that follows instructions. This is already the default.

## Safe Mode (ClawHub users)

If you're using ClawHub and want to try CrabPath without risk:

### Step 1: Install

```bash
# Activate your venv first if needed
source ~/.crabpath-env/bin/activate

pip install crabpath
# or clone: git clone https://github.com/jonathangu/crabpath
```

### Step 2: Bootstrap (zero risk — read-only)

```bash
# Requires OPENAI_API_KEY, GEMINI_API_KEY, or local Ollama for embeddings
crabpath init --workspace ~/.openclaw/workspace \
  --sessions ~/.openclaw/agents/main/sessions/
```

This reads your files, creates a graph with embeddings, and replays session history. It does NOT modify your workspace.

Without an embedding provider, use `--no-embeddings` (keyword-only routing, less accurate).

### Step 3: Check Health

```bash
# Structural health (always works)
crabpath health --graph graph.json

# Full health with query stats (after replay)
crabpath health --graph graph.json --query-stats graph.stats.json
```

Shows tier distribution, cross-file edges, orphan nodes. After replay, also shows query-dependent metrics.

### Step 4: Shadow Mode (safe — runs alongside, never modifies responses)

Add to your AGENTS.md:

```text
# After each user message, run:
crabpath query --graph graph.json --embeddings embed.json --query 'summary of user message'
```

This logs what CrabPath would retrieve. Compare with what your agent actually loaded.

CrabPath NEVER modifies your agent's responses in shadow mode.

### Step 5: Replay History (optional, accelerates learning)

```bash
# Pass a directory — CrabPath auto-finds all *.jsonl files
crabpath migrate --workspace ~/.openclaw/workspace \
  --session-logs ~/.openclaw/agents/main/sessions/ \
  --output-graph graph.json --verbose
```

Feed historical queries to warm up the graph faster. Supports OpenClaw session format natively, flat JSONL, and plain text. Pass a directory or individual files.

### Step 6: Track Evolution

```bash
crabpath evolve --graph graph.json --snapshots evolution.jsonl --report
```

See how your graph changes over time. Make timelapses.

### Step 7: Graduate to Active Mode (when ready)

When you trust CrabPath's retrievals, start using its output as supplementary context.
Still keep your static context loading as fallback.

Monitor with:

```bash
crabpath health --graph graph.json
```

### Safety Guarantees

- CrabPath NEVER modifies your workspace files.
- CrabPath NEVER modifies your agent's responses (in shadow mode).
- CrabPath NEVER sends data anywhere (all local, zero network calls except optional embeddings).
- Kill switch: delete the graph.json file and you're back to normal.
- All state is in graph.json — portable, inspectable, deletable.

### What You Need

- Python 3.10+
- Zero pip dependencies
- Optional: OPENAI_API_KEY or GEMINI_API_KEY for semantic embeddings (better retrieval)
- Without API key: keyword-based routing still works (just less precise)

## Installation

```bash
# Local package installation
python -m pip install -e .

# Run commands using Python module path
PYTHONPATH=. python -m crabpath.cli --help
```

The project defines a `crabpath` console script entry point in `pyproject.toml`, so running `pip install -e .` also enables:

```bash
crabpath --help
```

## CLI usage

- Query a graph:
  - `crabpath query "how do I use codex" --graph crabpath_graph.json`
- Learn from outcome:
  - `crabpath learn --graph crabpath_graph.json --outcome 1.0 --fired-ids deploy-config,cache-refresh`
- Add and remove nodes:
  - `crabpath add --id my-node --content "..." --graph crabpath_graph.json`
  - `crabpath remove --id my-node --graph crabpath_graph.json`
- Inspect graph stats:
  - `crabpath stats --graph crabpath_graph.json`
- Split nodes:
  - `crabpath split --graph crabpath_graph.json --node-id my-node --save`
- Run lifecycle simulation:
  - `crabpath sim --queries 20 --decay-interval 5 --decay-half-life 80`
- Migrate a workspace:
  - `crabpath migrate --workspace ~/.openclaw/workspace --output-graph crabpath_graph.json --output-embeddings crabpath_embeddings.json --include-memory --verbose`

All successful commands emit JSON to `stdout`.

## Migration

`crabpath migrate` runs `crabpath.migrate.migrate` with options:
- `--workspace` workspace directory
- `--session-logs` optional replay files
- `--include-memory`, `--include-docs`
- `--output-graph` destination graph file
- `--output-embeddings` destination embedding index file
- `--verbose`

The command returns an object including `ok`, `graph_path`, optional `embeddings_path`, and detailed migration `info`.

## Python API

Import common operations directly from package modules:

```python
from crabpath import migrate, Graph
from crabpath.mcp_server import main as mcp_main
from crabpath.lifecycle_sim import run_simulation
from crabpath.mitosis import split_node
```

Use `run_simulation` for deterministic self-organization experiments and `migrate` for bootstrap + optional replay.

The MCP server entrypoint lives at `crabpath/mcp_server.py` and exposes the same core operations as a JSON-RPC tool surface.
