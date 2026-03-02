# OpenClaw Integration (OpenClawBrain)

OpenClawBrain is built to be the long-term memory layer for **OpenClaw agents**.
Canonical docs and examples: https://openclawbrain.ai
Primary operator runbook: [docs/operator-guide.md](operator-guide.md)
Operator recipes (cutover, parallel replay, prompt caching, media memory): [docs/ops-recipes.md](ops-recipes.md)
New-agent canonical SOP (workspace + dedicated brain + launchd + routing): [docs/new-agent-sop.md](new-agent-sop.md)
Packaged adapter CLIs (no repo clone required): `python3 -m openclawbrain.openclaw_adapter.query_brain ...`, `python3 -m openclawbrain.openclaw_adapter.capture_feedback ...`, `python3 -m openclawbrain.openclaw_adapter.learn_by_chat_id ...`, and `python3 -m openclawbrain.openclaw_adapter.learn_correction ...`

If you’re already running OpenClaw, this guide shows the fastest path to:

- Build a brain (`state.json`) from your OpenClaw workspace
- Run the **persistent daemon service** (`openclawbrain serve`) so queries stay fast
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
- Daemon query embedding mode defaults to `--embed-model auto`:
  - `hash-v1` states use hash query embeddings (offline, no OpenAI call).
  - Non-hash states use OpenAI query embeddings with `text-embedding-3-small` (`OPENAI_API_KEY` required in daemon environment).
  - Force hash mode: `--embed-model hash`.
  - Force explicit model: `--embed-model text-embedding-3-small` (or another compatible model).

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

## Secrets registry

Use pointer-only secret handling. Keep capability names, required key names, and local secret pointers, but never secret values.

- Canonical policy: [docs/secrets-and-capabilities.md](secrets-and-capabilities.md)
- Harvest pointers: `python3 -m openclawbrain.ops.harvest_secret_pointers --workspace ~/.openclaw/workspace`
- Leak audit (path + line only): `python3 -m openclawbrain.ops.audit_secret_leaks --workspace ~/.openclaw/workspace --strict`

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
Prompt appendix (`[BRAIN_CONTEXT]`) + internal fired-node tracking
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

With `openclawbrain serve`, you pay the load cost once and queries become:

- NDJSON request → response

This is the “production shape” for OpenClaw integration.

Adapter CLIs (`python3 -m openclawbrain.openclaw_adapter.query_brain` and `python3 -m openclawbrain.openclaw_adapter.capture_feedback`) now auto-detect the daemon socket:

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
# By default, init auto-detects OpenAI (uses it if OPENAI_API_KEY is set, falls back to hash otherwise)
openclawbrain init --workspace "$WORKSPACE" --output "$BRAIN_DIR"

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

- **Before answering:** run a query and append prompt-only `[BRAIN_CONTEXT]`
- **After answering:** learn by `chat_id` (no node IDs in prompt payload)
- **When corrected or taught:** call `capture_feedback` in the same turn (canonical path)

Paste this block into your OpenClaw workspace `AGENTS.md` (edit `AGENT` and paths):

```md
## OpenClawBrain Memory Graph (auto-installed)

**Query** (before answering questions about prior work, context, decisions, corrections, lessons):
```bash
python3 -m openclawbrain.openclaw_adapter.query_brain ~/.openclawbrain/AGENT/state.json '<summary of user message>' --chat-id '<chat_id from inbound metadata>' --format prompt --exclude-bootstrap --max-prompt-context-chars 12000
```
Always pass `--chat-id` so fired nodes are logged for later learning/corrections.
Use `--exclude-recent-memory <today-note> <yesterday-note>` only when those files are already loaded by OpenClaw in the same prompt and you want to avoid duplication.

**Capture feedback** (canonical always-on path; same turn, no "log this" phrasing needed):
```bash
python3 -m openclawbrain.openclaw_adapter.capture_feedback \
  --state ~/.openclawbrain/AGENT/state.json \
  --chat-id '<chat_id>' \
  --kind CORRECTION \
  --content "The correction text here" \
  --lookback 1 \
  --message-id '<stable-message-id>' \
  --json
```
Use `--dedup-key` (or `--message-id`) whenever possible so harvest/replay retries cannot double-inject. For positive reinforcement with a teaching:
`--kind TEACHING --content "..." --outcome 1.0 --dedup-key '<stable-id>'`.

Compatibility fallback:
- `learn_by_chat_id` still works for outcome-only updates.
- `learn_correction` still works for correction-specific flow.

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

### Always-on self-learning (default)

Use this as the default operator policy; no special user phrasing like "log this" should be required.

- If the user clearly corrects the agent, run `capture_feedback --kind CORRECTION --chat-id '<chat_id>' --lookback 1`.
- If the user teaches a durable rule/fact, run `capture_feedback --kind TEACHING` (optionally with `--outcome`).
- If intent is ambiguous (correction vs preference vs new teaching), ask one clarifying question before writing memory.
- Never log secrets, tokens, passwords, private keys, or other sensitive values.
- Always pass a stable `--dedup-key` (or `--message-id`) when available.

This remains compatible with tight prompts:
- Keep retrieval on `query_brain --format prompt` so prompt payload contains only `[BRAIN_CONTEXT]`.
- Learning/injection uses `capture_feedback` keyed by `chat_id`, with optional `--outcome`.
- Do not pass `fired_nodes` in prompt/tool payloads; the brain tracks them internally by `chat_id`.

### Prompt-context duplication control (recommended)

OpenClaw already loads bootstrap files (`AGENTS.md`, `SOUL.md`, `USER.md`, `MEMORY.md`, `active-tasks.md`) into the base prompt. If you also include them again from brain retrieval, token usage grows quickly with little value.

Use prompt format and exclusions to keep context “tight and right”:

- Prefer `--format prompt` so only `[BRAIN_CONTEXT]` is appended to the model prompt.
- Keep `--exclude-bootstrap` enabled (default in the adapter).
- Start with `--max-prompt-context-chars 8000` to `12000`; only increase when needed.
- Use `--exclude-recent-memory ...` only for explicit daily notes already injected into the same OpenClaw turn.

This aligns retrieval output with OpenClawBrain’s context-efficiency goal: preserve high-value retrieved nodes while minimizing repeated bootstrap content.

---

## Step 3 — Run the daemon in production (launchd + systemd)

### Option A: run it manually (smoke test)

```bash
openclawbrain serve --state ~/.openclawbrain/main/state.json
```

The daemon speaks NDJSON over `stdin`/`stdout`.

- Protocol: each line is one JSON request with `id`, `method`, and `params`.
- Response: one JSON object with matching `id` and either `result` or `error`.
- Start-up cost is paid once (state loaded at process start); expected savings are 100-800ms on hot-path calls, with production measurement at ~504ms on Mac Mini M4 Pro.

Supported methods: `query`, `learn`, `last_fired`, `learn_by_chat_id`, `capture_feedback`, `maintain`, `health`, `info`, `save`, `reload`, `shutdown`, `inject`, `correction`.

Daemon embed-model default and overrides:
- Default is `--embed-model auto`.
- `hash-v1` state metadata => hash query embeddings (offline).
- Non-hash state metadata => OpenAI query embeddings with `text-embedding-3-small`.
- Force modes with `--embed-model hash` or `--embed-model <model>`.

Example request and reply:

```bash
echo '{"id":"req-1","method":"query","params":{"query":"how to deploy","top_k":4}}' | openclawbrain daemon --state ~/.openclawbrain/main/state.json
```

```json
{"id":"req-1","result":{"fired_nodes":["a"],"context":"...","embed_query_ms":1.1,"traverse_ms":2.4,"total_ms":3.5}}
```

- `inject` and `correction` are now available and are the preferred path for same-turn updates.
- The daemon is still NDJSON over stdio internally, with production transport now provided by `openclawbrain serve`.

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
    <string>openclawbrain</string>
    <string>serve</string>
    <string>--state</string>
    <string>/Users/YOU/.openclawbrain/main/state.json</string>
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
ExecStart=/usr/bin/env openclawbrain serve --state /home/YOUR_USER/.openclawbrain/main/state.json
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

For full-history rebuilds, replay your sessions (full-learning is the default; explicit alias pair: `--full-learning` / `--full-pipeline`):

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions
```

Single-writer reminder:
- Rebuild/replay and daemon learning are both writers.
- If lock checks fail on LIVE state, prefer rebuild-then-cutover from [docs/operator-guide.md](operator-guide.md).
- Expert override: `--force` or `OPENCLAWBRAIN_STATE_LOCK_FORCE=1` (only when no conflicting writer is active).

Media note for OpenClaw logs:

- User-uploaded image/audio is often stored as a stub like `[media attached: ...]`.
- The meaningful OCR/transcript text usually appears later as `toolResult`.
- OpenClawBrain replay can attach allowlisted `toolResult` text back onto the
  media-stub user query and also expose it to fast-learning extraction windows.

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --include-tool-results \
  --tool-result-allowlist image,openai-whisper,openai-whisper-api,openai-whisper-local,summarize \
  --tool-result-max-chars 20000
```

This does:

- replay query edges from session history (with decay enabled by default)
- LLM transcript mining into `learning::` nodes (`--fast-learning` / `--extract-learning-events` behavior)
- slow-learning maintenance pass (`harvest`) with decay/scale/split/merge/prune/connect

For cheap edge-only replay (no LLM, no harvest), use `--edges-only`:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --edges-only
```

To enable decay during an edges-only replay, add `--decay-during-replay`:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --edges-only \
  --decay-during-replay \
  --decay-interval 10
```

`--decay-interval N` (default 10) controls how many learning steps between each decay pass.

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

Dimension mismatch usually means query embedding dimensions do not match the vectors stored in `state.json` (often from forcing the wrong daemon embed model).

Fixes:
- Run daemon in default auto mode (`--embed-model auto`), which follows state metadata safely.
- Rebuild state if needed with the embedder you intend to keep.
- Only force `--embed-model hash` or `--embed-model <model>` when you intentionally want that behavior and dimensions are known to match.

```bash
openclawbrain init --workspace ~/.openclaw/workspace --output ~/.openclawbrain/main
```

### “My queries are slow”

Common causes:

- You aren’t using the daemon (so `state.json` reload happens every call).
- You’re using hash embeddings and still want production-grade retrieval (or `embed_query_ms` is high from OpenAI calls).

Fixes:

- Use the canonical brain-on command (`openclawbrain serve --state ...`).
- Use OpenAI embeddings for production routing/scoring behavior; only use hash embeddings for offline/testing fallback.

### “Daemon starts but OpenClaw can’t talk to it”

The daemon worker is an NDJSON stdio process, wrapped for production by `openclawbrain serve` (implemented in `socket_server`).
If your integration layer cannot reach the socket, fall back to disk-path operation.

Two practical options:

1. Keep using the adapter scripts; they will auto-detect and use the socket when available.
2. For custom integrations, call `openclawbrain.socket_client` against `daemon.sock`.

### “Corrections aren’t sticking”

- Ensure you pass `--chat-id` on every query so fired nodes are logged.
- On correction/teaching, run daemon `capture_feedback` in the same turn with `dedup_key`/`message_id`.

### “state.json is getting big”

- That’s normal with real embeddings.
- Run `maintain` (prune/merge) and optionally `compact` old notes.

---

## Native OpenClaw Tool (opencormorant fork)

If you run the **opencormorant** fork of OpenClaw (`github.com/jonathangu/opencormorant`), there is a built-in `openclawbrain` tool that agents can call directly — no shell exec needed.

### What it provides
The tool connects to the daemon Unix socket (`~/.openclawbrain/<agent>/daemon.sock`) and routes to the correct agent automatically (main/pelican/bountiful).

It supports **all OpenClawBrain daemon methods** (plus a safe generic passthrough):

| Action | Description |
|---|---|
| `query` | Retrieve context + prompt appendix (`prompt_context`) |
| `learn` | Apply numeric outcome to fired IDs (weight updates) |
| `last_fired` | Return recent fired node IDs for `chat_id` |
| `learn_by_chat_id` | Apply numeric outcome using recent fired nodes for `chat_id` |
| `capture_feedback` | Canonical real-time CORRECTION/TEACHING/DIRECTIVE with optional `outcome` + dedup |
| `inject` | Inject TEACHING/CORRECTION/DIRECTIVE nodes |
| `maintain` | Run structural maintenance tasks |
| `health` | Health summary |
| `info` | Node/edge counts + embedder |
| `save` | Persist state immediately |
| `reload` | Reload state from disk |
| `correction` | Penalize recent fired path for `chat_id` + inject correction node |
| `self_learn` / `self_correct` | Autonomous learning/correction helpers |
| `shutdown` | Stop the daemon (**requires** `confirm="shutdown"`) |
| `call` | Generic validated passthrough: provide `method` + `params` |

### How to use it (agent perspective)
The tool is registered automatically. Agents can call it like any other tool.

Query:
```
openclawbrain(action="query", query="how do we deploy", chat_id="telegram:123", top_k=4)
```

Feedback capture (canonical):
```
openclawbrain(action="capture_feedback", chat_id="telegram:123", kind="CORRECTION", content="Actually we use blue-green deploys, not rolling", message_id="telegram:123:456")
```

Legacy correction:
```
openclawbrain(action="correction", chat_id="telegram:123", content="Actually we use blue-green deploys, not rolling")
```

Generic call (validated passthrough):
```
openclawbrain(action="call", method="maintain", params={"tasks":["health","decay"]})
```

Shutdown (explicit guard):
```
openclawbrain(action="shutdown", confirm="shutdown")
```

### Requirements
- OpenClawBrain daemon must be running (`openclawbrain serve --state ...`)
- The `daemon.sock` file must be accessible from the OpenClaw process

### Media understanding synergy
OpenClaw's built-in `tools.media` pipeline (audio transcription, image description) runs *before* the agent responds. When enabled, audio/image content is extracted to text and stored in session logs, so OpenClawBrain replay/full-learning can learn from media messages naturally.

To enable in your OpenClaw config:
```json
{
  "tools": {
    "media": {
      "audio": { "enabled": true },
      "image": { "enabled": true }
    }
  }
}
```

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
                         │ openclawbrain serve --state ...    │
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
