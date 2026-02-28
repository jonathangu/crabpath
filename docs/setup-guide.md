# OpenClawBrain Operator Setup Guide

## Prerequisites
- Python 3.10+
- `pip install openclawbrain`
- `OPENAI_API_KEY` in environment (required for the recommended OpenAI setup; optional only for hash offline/testing mode)
- A workspace directory with markdown files (your agent's knowledge base)

## Step 1: Build your first brain

```bash
openclawbrain init --workspace ./my-workspace --state ./brain/state.json --embedder openai --llm openai
openclawbrain doctor --state ./brain/state.json
openclawbrain info --state ./brain/state.json
```

We recommend OpenAI embeddings with GPT-5-mini routing for production use. Hash embeddings work offline but produce lower-quality retrieval.

## Initial learning: cold-start replay vs sidecar fast-learning

After init, replay your existing sessions once to seed graph edges from real usage:

```bash
openclawbrain replay --state ./brain/state.json --sessions ./sessions/
```

Then run transcript-backed fast-learning to extract durable correction/teaching nodes into the same brain directory:

```bash
openclawbrain replay \
  --state ./brain/state.json \
  --sessions ./sessions/ \
  --fast-learning \
  --workers 4 \
  --window-radius 8 \
  --max-windows 6 \
  --hard-max-turns 120 \
  --checkpoint ./brain/replay_checkpoint.json \
  --json
```

`fast-learning` stores extracted events in an append-only log:
- `./brain/learning_events.jsonl`

You can run this repeatedly; dedupe is by `(type, sha256(content), session_pointer)`, so repeated runs are idempotent.

For ongoing operation after startup, use `--ignore-checkpoint` only when you intentionally want to replay older chunks that were already ingested.

## Step 2: Wire up the fast loop (per-query)

Minimum per-query pattern: **query → log → learn**.  
Reference implementation: `examples/ops/query_and_learn.py`

```python
from examples.ops.callbacks import make_embed_fn, make_llm_fn

embed_fn = make_embed_fn("openai")  # or "hash" for offline/testing mode
llm_fn = make_llm_fn("gpt-5-mini")  # optional, for LLM-assisted merge
```

Use this when building your query handler:

- Run embeddings with `embed_fn(text)`
- Traverse the graph and return `fired_ids`
- Log query traces and learn outcomes:
  - `+1.0` for good output
  - `-1.0` for bad output
- Persist feedback with `openclawbrain learn`

For an AGENTS.md drop-in hook, use:

`examples/openclaw_adapter/agents_hook.md`

## Step 3: Wire up the slow loop (maintenance)

Run maintenance manually:

```bash
openclawbrain maintain --state ./brain/state.json --tasks health,decay,prune,merge
openclawbrain maintain --state ./brain/state.json --dry-run --json

# Slow-learning harvest (now supported):
openclawbrain harvest \
  --state ./brain/state.json \
  --events ./brain/learning_events.jsonl \
  --tasks split,merge,prune,connect,scale \
  --json
```

The harvest path is intentionally sidecar: it consumes OpenClaw replay artifacts and updates graph structure from them, without changing OpenClaw core memory files.

For production automation, run `harvest` after `replay --fast-learning` with a bounded schedule (daily/hourly depending on session volume).

## Decay during replay

`--full-learning` automatically enables decay during the replay pass. Unrelated edges weaken while actively traversed paths are reinforced. The harvest step also includes the `decay` task (`decay,scale,split,merge,prune,connect`).

To enable decay during a plain replay (without full-learning), pass `--decay-during-replay`:

```bash
openclawbrain replay --state ./brain/state.json --sessions ./sessions/ --decay-during-replay --decay-interval 10
```

`--decay-interval N` (default 10) controls how many learning steps occur between each decay pass.

## Performance knobs

- `--workers`: parallelize LLM window extraction (higher = faster, bounded by rate limits)
- `--window-radius`: context breadth around likely feedback turns
- `--max-windows`: max feedback windows sampled per file/session
- `--hard-max-turns`: hard cap total turns considered to keep extraction bounded

Recommended defaults:
- `--workers 4`
- `--window-radius 8`
- `--max-windows 6`
- `--hard-max-turns 120`
```

Schedule maintenance:

### Cron one-liner

```cron
0 2 * * * /usr/bin/python3 -m openclawbrain.cli maintain --state /opt/openclawbrain/brain/state.json --tasks health,decay,prune,merge
```

### systemd timer snippet

`/etc/systemd/system/openclawbrain-maintenance.service`

```ini
[Service]
Type=oneshot
ExecStart=/usr/bin/python3 -m openclawbrain.cli maintain --state /opt/openclawbrain/brain/state.json --tasks health,decay,prune,merge
```

`/etc/systemd/system/openclawbrain-maintenance.timer`

```ini
[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

### Existing batch pipeline

You can also invoke from your existing batch workflow/job:

```bash
python3 /opt/openclawbrain/examples/ops/run_maintenance.py --state /opt/openclawbrain/brain/state.json --tasks health,decay,prune,merge
```

## Step 4: Inject corrections and teachings

```bash
openclawbrain inject --state ./brain/state.json --id "correction::wrong-deploy" \
  --content "Never deploy on Fridays without notifying ops" --type CORRECTION

openclawbrain inject --state ./brain/state.json --id "teaching::deploy-window" \
  --content "Deploy window is Tuesday-Thursday 10am-2pm" --type TEACHING
```

- `CORRECTION`: creates inhibitory edges.
- `TEACHING`: creates supportive edges.

## Step 5: Rebuild (when workspace changes significantly)

```bash
python3 examples/openclaw_adapter/rebuild_all_brains.py --agent main
```

## Step 6: Monitor health

```bash
openclawbrain health --state ./brain/state.json
openclawbrain info --state ./brain/state.json
```

### Key metrics

- `dormant_pct`: target 70-95% (most edges should be dormant)
- `reflex_pct`: target 0-10% (only proven routes)
- `orphan_nodes`: should be 0

## Step 7: Set up file sync (incremental re-embedding)

```bash
openclawbrain sync --state ~/.openclawbrain/main/state.json --workspace ./my-workspace --embedder openai
```

## Step 8: Set constitutional anchors

```bash
openclawbrain anchor --state ~/.openclawbrain/main/state.json --node-id "SOUL.md::0" --authority constitutional
openclawbrain anchor --state ~/.openclawbrain/main/state.json --list
```

## Step 9: Compact old daily notes

```bash
openclawbrain compact --state ~/.openclawbrain/main/state.json --memory-dir ./memory --dry-run
```

## Brain directory layout

```
~/.openclawbrain/main/
├── state.json                 # Graph + index + metadata
├── journal.jsonl              # Append-only telemetry
├── fired_log.jsonl            # Per-chat fired nodes
└── injected_corrections.jsonl # Dedup ledger
```

## Callback pattern (the core principle)

OpenClawBrain never imports `openai` or any provider package directly. You construct callbacks and pass them in.

- `embed_fn`: `(text) -> vector`
- `llm_fn`: `(system, user) -> str`

Reference: `examples/ops/callbacks.py`

## Socket Server (recommended for production)

Run the production service as a Unix socket wrapper around the NDJSON daemon:

```bash
python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json
```

The socket server creates and manages:

- `~/.openclawbrain/<agent>/daemon.sock` (for example: `~/.openclawbrain/main/daemon.sock`)  
  (created automatically by `socket_server`)

Test it with:

```bash
python3 -m openclawbrain.socket_client --socket ~/.openclawbrain/main/daemon.sock --method health --params "{}"
```

### launchd (macOS)

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
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/YOU/.openclawbrain/main/daemon.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOU/.openclawbrain/main/daemon.stderr.log</string>
</dict>
</plist>
```

Load it:

```bash
launchctl load -w ~/Library/LaunchAgents/com.openclawbrain.daemon.plist
launchctl list | rg openclawbrain
```

### systemd (Linux)

Create `/etc/systemd/system/openclawbrain-daemon.service`:

```ini
[Unit]
Description=OpenClawBrain daemon worker
After=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER
ExecStart=/usr/bin/python3 -m openclawbrain.socket_server --state /home/YOUR_USER/.openclawbrain/main/state.json
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openclawbrain-daemon.service
```

### Client Library

Use the stdlib-only socket client directly from Python:

```python
from openclawbrain.socket_client import OCBClient

with OCBClient("~/.openclawbrain/main/daemon.sock") as client:
    health = client.health()
    result = client.query("how do we deploy", chat_id="telegram:123", top_k=4)
```

## Daemon setup (legacy / debug)

Keep the process warm and reuse loaded state when testing transport internals directly:

Run the daemon directly for smoke tests:

```bash
openclawbrain daemon --state ~/.openclawbrain/main/state.json
```

### launchd (macOS)

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
    <string>openclawbrain.daemon</string>
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
</dict>
</plist>
```

Load it:

```bash
launchctl load -w ~/Library/LaunchAgents/com.openclawbrain.daemon.plist
launchctl list | rg openclawbrain
```

### systemd (Linux)

Create `/etc/systemd/system/openclawbrain-daemon.service`:

```ini
[Unit]
Description=OpenClawBrain daemon worker
After=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER
ExecStart=/usr/bin/env python3 -m openclawbrain.daemon --state /home/YOUR_USER/.openclawbrain/main/state.json
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openclawbrain-daemon.service
```

## Testing the daemon

Use an `echo` pipe with NDJSON to validate request/response wiring:

```bash
echo '{"id":"health-1","method":"health","params":{}}' | openclawbrain daemon --state ~/.openclawbrain/main/state.json
```

You should receive a JSON reply on stdout with the same `id` and health fields.

Or use the socket client one-liner:

```bash
python3 -m openclawbrain.socket_client --socket ~/.openclawbrain/main/daemon.sock --method health --params "{}"
```

Daemon references (supported methods):

- `query`, `learn`, `maintain`, `health`, `info`, `save`, `reload`, `shutdown`, `inject`, `correction`.
- `inject` supports TEACHING/CORRECTION/DIRECTIVE node types.
- `correction` performs atomic penalize + inject in a single request.
