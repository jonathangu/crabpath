# OpenClaw Integration (OpenClawBrain)

OpenClawBrain is built to be the long-term memory layer for **OpenClaw agents**.
Canonical docs and examples: https://openclawbrain.ai

If you’re already running OpenClaw, this guide shows the fastest path to:

- Build a brain (`state.json`) from your OpenClaw workspace
- Run the **persistent daemon service** (`openclawbrain.socket_server`) so queries are fast
- Wire your OpenClaw agent’s **AGENTS.md** to query → respond → learn
- Operate it like a runbook: launchd/systemd, maintenance cron, troubleshooting

---

## What OpenClawBrain does (and why you want it)

OpenClawBrain is a Python memory graph library that turns your workspace into a **learned retrieval policy**.
Instead of “top-k similarity every time”, it learns which context routes actually helped.

**What changes in practice:**

- Your agent stops resurfacing the same wrong chunks over and over.
- Corrections create **inhibitory edges** that actively suppress bad retrieval routes.
- You can add knowledge without rebuilding the whole index (`inject --type TEACHING`).
- With the socket server, the brain stays **hot in memory**, avoiding per-call reload overhead.

OpenClawBrain stores everything in a single portable file:

- `state.json` — graph + embeddings + metadata

---

## Prerequisites

- Python **3.10+**
- `pip install openclawbrain`
- OpenAI embeddings: `OPENAI_API_KEY` in the environment of whatever runs the daemon. Required for the recommended production setup; hash embeddings are the offline/testing fallback.

Install:

```bash
pip install openclawbrain
```

Sanity check:

```bash
openclawbrain --help
openclawbrain info --help
```

---

## Architecture (how it fits into OpenClaw)

OpenClaw today is file-and-instructions driven: the agent reads `AGENTS.md`, then runs whatever you tell it to.
OpenClawBrain plugs into that contract.

### Data flow

```text
User message
   ↓
OpenClaw agent
   ↓ (reads AGENTS.md)
OpenClawBrain query (daemon)
   ↓
Context chunks + fired node IDs
   ↓
Agent response
   ↓
Outcome (+1 good / -1 bad / correction)
   ↓
OpenClawBrain learn / inject
```

### Why the daemon matters

Without the socket service, every query path tends to do:

- start Python
- load `state.json`
- initialize index
- run query
- exit

With `openclawbrain.socket_server`, you pay the load cost once and queries become:

- NDJSON request → response

This is the “production shape” for OpenClaw integration.

Adapter scripts (`query_brain.py` and `learn_correction.py`) now auto-detect the daemon socket:

- If `~/.openclawbrain/<agent>/daemon.sock` exists, they use the in-memory socket transport (fast path).
- If the socket is missing, they fall back to loading `state.json` from disk directly (slower but still works).
- Both scripts also accept `--socket` when you want to force an explicit socket path.

---

## Step 1 — Build your first brain (`openclawbrain init`)

Pick where you want brains to live. Recommended layout:

```text
~/.openclawbrain/
  main/
    state.json
    journal.jsonl
    fired_log.jsonl
    injected_corrections.jsonl
```

Build from an OpenClaw workspace:

```bash
# Example workspace path (adjust)
WORKSPACE=~/.openclaw/workspace

# Where the brain lives
BRAIN_DIR=~/.openclawbrain/main

mkdir -p "$BRAIN_DIR"

# Create/overwrite state.json inside the output directory
openclawbrain init --workspace "$WORKSPACE" --output "$BRAIN_DIR" --embedder openai --llm openai

# Quick health signal
openclawbrain doctor --state "$BRAIN_DIR/state.json"
openclawbrain health  --state "$BRAIN_DIR/state.json"
```

Notes:

- First init is the “embed everything once” step.
- Subsequent edits should use `openclawbrain sync` (incremental) rather than full rebuild.

---

## Step 2 — Wire it into your OpenClaw agent’s `AGENTS.md` (copy/paste)

OpenClaw agents do what `AGENTS.md` says. The simplest integration is:

- **Before answering:** run a query, get context + fired IDs
- **After answering:** call `learn` with +1/-1 using the fired IDs
- **When corrected:** use the daemon `correction` method to atomically penalize and inject a durable correction node

Paste this block into your OpenClaw workspace `AGENTS.md` (edit `AGENT` and paths):

```md
## OpenClawBrain Memory Graph (auto-installed)

**Query** (before answering questions about prior work, context, decisions, corrections, lessons):
```bash
python3 ~/openclawbrain/examples/openclaw_adapter/query_brain.py ~/.openclawbrain/AGENT/state.json '<summary of user message>' --chat-id '<chat_id from inbound metadata>' --json
```
Always pass `--chat-id` so fired nodes are logged for later corrections.

**Learn** (after each response, using fired node IDs from query output):
- Good: `openclawbrain learn --state ~/.openclawbrain/AGENT/state.json --outcome 1.0 --fired-ids <ids>`
- Bad: `openclawbrain learn --state ~/.openclawbrain/AGENT/state.json --outcome -1.0 --fired-ids <ids>`

**Inject correction** (when corrected — same turn, don't wait for harvester):
```bash
echo '{"id":"correct-1","method":"correction","params":{"chat_id":"<chat_id>","outcome":-1.0,"content":"The correction text here"}}' \
  | openclawbrain daemon --state ~/.openclawbrain/AGENT/state.json
```
This applies negative feedback to the last query's fired nodes and injects a CORRECTION node with inhibitory edges in one request.

**Inject new knowledge** (when you learn something not in any workspace file):
```bash
echo '{"id":"inject-1","method":"inject","params":{"id":"teaching::<short-id>","content":"The new fact","type":"TEACHING"}}' \
  | openclawbrain daemon --state ~/.openclawbrain/AGENT/state.json
```

**Health:** `openclawbrain health --state ~/.openclawbrain/AGENT/state.json`

**Maintenance** (structural ops — runs automatically via harvester cron, but can also run manually):
```bash
openclawbrain maintain --state ~/.openclawbrain/AGENT/state.json --tasks health,decay,prune,merge
```
Dry-run first: add `--dry-run` to preview changes without applying.

**Sync workspace** (after editing files):
```
openclawbrain sync --state ~/.openclawbrain/AGENT/state.json --workspace /path/to/workspace
```

**Compact old notes** (weekly or via cron):
```
openclawbrain compact --state ~/.openclawbrain/AGENT/state.json --memory-dir /path/to/memory
```
```

That block is intentionally boring: it’s the contract OpenClaw already supports.

---

## Step 3 — Run the daemon in production (launchd + systemd)

### Option A: run it manually (smoke test)

```bash
python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json
```

The daemon speaks NDJSON over `stdin`/`stdout`.

- Protocol: each line is one JSON request with `id`, `method`, and `params`.
- Response: one JSON object with matching `id` and either `result` or `error`.
- Start-up cost is paid once (state loaded at process start); expected savings are 100-800ms on hot-path calls, with production measurement at ~504ms on Mac Mini M4 Pro.

Supported methods: `query`, `learn`, `maintain`, `health`, `info`, `save`, `reload`, `shutdown`, `inject`, `correction`.

Example request and reply:

```bash
echo '{"id":"req-1","method":"query","params":{"query":"how to deploy","top_k":4}}' | openclawbrain daemon --state ~/.openclawbrain/main/state.json
```

```json
{"id":"req-1","result":{"fired_nodes":["a"],"context":"...","embed_query_ms":1.1,"traverse_ms":2.4,"total_ms":3.5}}
```

- `inject` and `correction` are now available and are the preferred path for same-turn updates.
- The daemon is still NDJSON over stdio internally, with production transport now provided by `openclawbrain.socket_server`.

### Option B: launchd (macOS)

Create `~/Library/LaunchAgents/com.openclawbrain.daemon.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.openclawbrain.daemon</string>

  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/env</string>
    <string>python3</string>
    <string>-m</string>
    <string>openclawbrain.socket_server</string>
    <string>--state</string>
    <string>/Users/YOU/.openclawbrain/main/state.json</string>
    <string>--auto-save-interval</string>
    <string>10</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/Users/YOU/.openclawbrain/main/daemon.stdout.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/YOU/.openclawbrain/main/daemon.stderr.log</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>OPENAI_API_KEY</key>
    <string>YOUR_KEY_HERE</string>
  </dict>
</dict>
</plist>
```

Load it:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.openclawbrain.daemon.plist 2>/dev/null || true
launchctl load -w ~/Library/LaunchAgents/com.openclawbrain.daemon.plist
launchctl list | rg openclawbrain
```

Notes:

- Prefer injecting `OPENAI_API_KEY` via your own secure mechanism (1Password, Keychain, etc.).
- The daemon is a stdio worker. You typically run it behind a small supervisor that manages pipes.

### Option C: systemd (Linux)

Create `/etc/systemd/system/openclawbrain-daemon.service`:

```ini
[Unit]
Description=OpenClawBrain daemon (hot state.json worker)
After=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER
Environment=OPENAI_API_KEY=YOUR_KEY_HERE
ExecStart=/usr/bin/env python3 -m openclawbrain.socket_server --state /home/YOUR_USER/.openclawbrain/main/state.json --auto-save-interval 10
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openclawbrain-daemon
sudo systemctl status openclawbrain-daemon --no-pager
```

---

## Step 4 — Wire up the learning loop (query → feedback → learn)

In OpenClaw terms, you want a stable ritual:

1. **Query** before answering
2. Extract `fired` node IDs
3. After answering, apply a reward:
   - `+1.0` if the context helped
   - `-1.0` if it hurt (or user corrected you)
4. When corrected, also inject the correction text as a durable inhibitory node

### Minimal CLI loop (no daemon)

```bash
# Query
Q=$(openclawbrain query "how do we deploy" --state ~/.openclawbrain/main/state.json --top 4 --json)

# Pull fired IDs out (jq recommended)
FIRED=$(echo "$Q" | jq -r '.fired | join(",")')

# Learn (reward)
openclawbrain learn --state ~/.openclawbrain/main/state.json --outcome 1.0 --fired-ids "$FIRED"
```

### Correction flow (the one you actually want)

OpenClawBrain ships an OpenClaw adapter that logs fired IDs per chat.

- Query: `query_brain.py --chat-id ...` writes `fired_log.jsonl`
- Correction: `correction` daemon method (`method:"correction"`) penalizes recent fired IDs *and* injects a correction node

That’s the ergonomic way to do “same-turn correction” inside OpenClaw.

For full-history rebuilds, use fast/full replay:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --full-learning
```

This does:

- replay query edges from session history
- LLM transcript mining into `learning::` nodes (`--fast-learning` behavior)
- slow-learning maintenance pass (`harvest`) with split/merge/prune/connect/scale

`learning_events.jsonl` is an append-only sidecar under the brain directory used by harvest:

`~/.openclawbrain/main/learning_events.jsonl`

---

## Step 5 — Maintenance cron (keep the graph healthy)

The fast loop changes weights and adds nodes.
The slow loop keeps the graph compact and sane.

### What to run

Recommended maintenance command:

```bash
openclawbrain maintain --state ~/.openclawbrain/main/state.json --tasks health,decay,prune,merge
```

The explicit slow-learning path is:

```bash
openclawbrain harvest \
  --state ~/.openclawbrain/main/state.json \
  --events ~/.openclawbrain/main/learning_events.jsonl \
  --tasks split,merge,prune,connect,scale
```

Start conservative:

- First week: run `--tasks health,decay` only
- Then enable `prune,merge` when you’re comfortable with the behavior

### cron example

```cron
# Every 30 minutes: small hygiene pass
*/30 * * * * /usr/bin/env openclawbrain maintain --state ~/.openclawbrain/main/state.json --tasks health,decay,prune,merge >> ~/.openclawbrain/main/maintenance.log 2>&1

# Weekly: compact old daily notes into the brain (optional)
0 4 * * 0 /usr/bin/env openclawbrain compact --state ~/.openclawbrain/main/state.json --memory-dir ~/.openclaw/memory >> ~/.openclawbrain/main/compact.log 2>&1
```

---

## Troubleshooting

### “`openclawbrain: command not found`”

You installed into a different Python environment.

- Verify: `which openclawbrain`
- Fix: install into the same interpreter your agent uses:

```bash
python3 -m pip install --upgrade openclawbrain
python3 -m pip show openclawbrain
```

### “Embedder mismatch / dimension mismatch”

OpenClawBrain hard-fails when `state.json` embedder metadata doesn’t match.

Fix: rebuild the brain with the embedder you intend to use.

```bash
openclawbrain init --workspace ~/.openclaw/workspace --output ~/.openclawbrain/main
```

### “My queries are slow”

Common causes:

- You aren’t using the daemon (so `state.json` reload happens every call).
- You’re using hash embeddings and still want production-grade retrieval (or `embed_query_ms` is high from OpenAI calls).

Fixes:

- Use the socket server command (`python3 -m openclawbrain.socket_server`) when available.
- Use OpenAI embeddings for production routing/scoring behavior; only use hash embeddings for offline/testing fallback.

### “Daemon starts but OpenClaw can’t talk to it”

The daemon worker is an NDJSON stdio process, wrapped for production by `socket_server`.
If your integration layer cannot reach the socket, fall back to disk-path operation.

Two practical options:

1. Keep using the adapter scripts; they will auto-detect and use the socket when available.
2. For custom integrations, call `openclawbrain.socket_client` against `daemon.sock`.

### “Corrections aren’t sticking”

- Ensure you pass `--chat-id` on every query so fired nodes are logged.
- On correction, run daemon `correction` in the same turn.

### “state.json is getting big”

- That’s normal with real embeddings.
- Run `maintain` (prune/merge) and optionally `compact` old notes.

---

## Appendix: ASCII architecture diagram

```text
                         ┌──────────────────────────────────┐
                         │           OpenClaw Agent          │
                         │  (reads AGENTS.md instructions)   │
                         └───────────────┬───────────────────┘
                                         │
                                         │ query (summary)
                                         ▼
                         ┌──────────────────────────────────┐
                         │        OpenClawBrain daemon       │
                         │ openclawbrain.socket_server --state│
                         │  (NDJSON over stdin/stdout)       │
                         └───────────────┬───────────────────┘
                                         │
                          context + fired│ ids + timings
                                         ▼
                         ┌──────────────────────────────────┐
                         │        OpenClaw Agent reply       │
                         └───────────────┬───────────────────┘
                                         │
                          outcome (+/-)  │  correction text
                                         ▼
                         ┌──────────────────────────────────┐
                         │     learn / inject / maintain     │
                         │  (updates weights, adds nodes)    │
                         └──────────────────────────────────┘
```
