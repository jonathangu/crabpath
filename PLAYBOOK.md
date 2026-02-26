# CrabPath Playbook: Install → Shadow → Active

## Prerequisites

- **Python 3.10+** required (macOS ships 3.9 — you need a newer version)
- **No pip dependencies** — CrabPath is pure stdlib Python
- Optional: `OPENAI_API_KEY` for semantic embeddings (better retrieval, not required)
- **Recommended model for LLM routing: GPT-5-mini** — cheap, fast, good enough for the router's JSON selection task. Set via `RouterConfig(model="gpt-5-mini")` (this is already the default)

### macOS setup (if you don't have Python 3.10+)

```bash
# Check your version
python3 --version

# If < 3.10, install via Homebrew
brew install python@3.12
```

### Create a virtual environment (recommended)

Modern Homebrew Python refuses bare `pip install` (PEP 668). Use a venv:

```bash
python3.12 -m venv ~/.crabpath-env
source ~/.crabpath-env/bin/activate
```

Add the activate line to your shell profile if you want it persistent:
```bash
echo 'source ~/.crabpath-env/bin/activate' >> ~/.zshrc
```

## Step 1: Install (1 minute)

```bash
pip install crabpath
```

Or from source:
```bash
git clone https://github.com/jonathangu/crabpath
cd crabpath
pip install -e .
```

Verify:
```bash
crabpath --version
```

## Step 2: Bootstrap your graph (1-3 minutes)

```bash
# Embeddings are ON by default — requires OPENAI_API_KEY, GEMINI_API_KEY, or local Ollama
crabpath init \
  --workspace ~/.openclaw/workspace \
  --sessions ~/.openclaw/agents/main/sessions/
```

This does everything in one command:
1. Reads your workspace files → creates nodes
2. Splits large files into coherent chunks
3. Replays session history to warm up edges
4. Generates embeddings for semantic retrieval
5. Saves graph, embeddings, and query stats to `~/.crabpath/`

**It does NOT modify your workspace.**

**Embedding providers** (checked in order):
- `OPENAI_API_KEY` → OpenAI text-embedding-3-small (recommended, ~$0.02 for 200 nodes)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` → Gemini text-embedding-004 (free tier available)
- Local Ollama → nomic-embed-text (free, runs on your machine)

If no provider is found, `init` will error with setup instructions.

**Fallback** (keyword-only routing, no API needed):
```bash
crabpath init --workspace ~/.openclaw/workspace --no-embeddings
```

**Session replay** supports:
- OpenClaw sessions: `{"type":"message","message":{"role":"user","content":[...]}}` — parsed natively
- Flat JSONL: `{"role":"user","content":"..."}` — also works
- Plain text: one query per line — also works
- Directories: pass a directory and CrabPath auto-finds all `*.jsonl` files

## Step 3: Health check

```bash
# Basic structural health (always works)
crabpath health --graph graph.json

# Full health with query stats (after replay)
crabpath health --graph graph.json --query-stats graph.stats.json
```

**Without `--query-stats`:** Shows structural metrics — tier distribution, cross-file edges, orphan nodes. These always work, even right after bootstrap.

**With `--query-stats`:** Also shows query-dependent metrics — avg nodes fired, context compression, promotion rate.

## Step 4: Shadow mode (safe — never modifies responses)

Shadow mode runs CrabPath alongside your agent. It logs what it *would* retrieve but doesn't change anything.

### For OpenClaw users

Add this to your `AGENTS.md`:

```text
## CrabPath Shadow (optional context check)
Before answering, run CrabPath query in shadow mode and compare suggested context.
Never apply CrabPath output directly to the user response unless explicitly approved.
```

Then query manually or via the MCP server:

```bash
# CLI query
crabpath query "how do I deploy" \
  --graph graph.json \
  --embeddings embed.json \
  --top 8

# Or start the MCP server for tool-based access
python -m crabpath.mcp_server --graph graph.json --embeddings embed.json
```

### Shadow logging

Queries are logged to `~/.crabpath/shadow.jsonl` when configured. Review with:

```bash
tail -f ~/.crabpath/shadow.jsonl
```

Or inspect programmatically:
```python
from crabpath import ShadowLog
print(ShadowLog().tail(10))
```

## Step 5: Graduate to active mode

Switch when:
- Shadow picks are stable and relevant across recent queries
- Health check shows green on structural metrics
- You've compared CrabPath retrieval with your static context loading

Then use CrabPath output as supplementary context. Keep static loading as fallback.

## Step 6: Monitor

```bash
# Health check (run daily)
crabpath health --graph graph.json --query-stats graph.stats.json

# Evolution tracking (weekly)
crabpath evolve --graph graph.json --snapshots evolution.jsonl --report
```

## Uninstall

CrabPath is fully self-contained. To remove:

```bash
# Remove the package
pip uninstall crabpath

# Remove your graph data (if you want a clean slate)
rm graph.json graph.stats.json embed.json evolution.jsonl

# Remove shadow logs
rm -rf ~/.crabpath/

# Remove the venv (if you created one)
rm -rf ~/.crabpath-env
```

No system files, no daemons, no config outside your working directory.

## Troubleshooting

### "pip install fails with externally-managed-environment"
You need a virtual environment. See the Prerequisites section above.

### "command not found: crabpath"
If using a venv, make sure it's activated: `source ~/.crabpath-env/bin/activate`

### "replay: null" or 0 queries replayed
- Make sure you're passing a **directory** or individual `.jsonl` **files**, not a bare path
- Check that the directory contains `.jsonl` files: `ls ~/.openclaw/agents/main/sessions/*.jsonl`
- CrabPath now warns if a directory has no parseable files

### Health check shows "n/a (collect query stats)"
Pass `--query-stats graph.stats.json` (generated automatically after replay). Without it, only structural metrics are shown — which is fine for a fresh graph.

### Brain death (all edges dormant)
Run `crabpath evolve --graph graph.json --snapshots evolution.jsonl --report` to diagnose. Usually means decay is too aggressive — the autotuner will suggest config changes.

### Kill switch
```bash
rm graph.json  # removes current memory graph
touch ~/.crabpath-shadow-disabled  # stops shadow scoring immediately
```
