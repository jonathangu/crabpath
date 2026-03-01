# OpenClawBrain Memory Graph (auto-installed)

Always-on self-learning is the default: apply clear corrections in the same turn and persist durable teachings without waiting for "log this" phrasing.

**Query** (before answering questions about prior work, context, decisions, corrections, lessons):
```bash
python3 -m openclawbrain.openclaw_adapter.query_brain ~/.openclawbrain/AGENT/state.json '<summary of user message>' --chat-id '<chat_id from inbound metadata>' --format prompt --exclude-bootstrap --max-prompt-context-chars 12000
```
Always pass `--chat-id` so fired nodes are logged for later corrections.

**Capture feedback** (canonical always-on path):
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
This injects immediately and defaults correction outcome to `-1.0`; use `--outcome` for explicit reinforce/penalize behavior.
Use `--dedup-key` (or `--message-id`) to avoid duplicate injections from retries/replays.

**Learn** (legacy outcome-only path, still supported):
- Good: `python3 -m openclawbrain.openclaw_adapter.learn_by_chat_id --state ~/.openclawbrain/AGENT/state.json --chat-id '<chat_id>' --outcome 1.0 --lookback 1 --json`
- Bad: `python3 -m openclawbrain.openclaw_adapter.learn_by_chat_id --state ~/.openclawbrain/AGENT/state.json --chat-id '<chat_id>' --outcome -1.0 --lookback 1 --json`

**Inject new knowledge** (when you learn something not in any workspace file):
```bash
openclawbrain inject --state ~/.openclawbrain/AGENT/state.json \
  --id "teaching::<short-id>" --content "The new fact" --type TEACHING
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
