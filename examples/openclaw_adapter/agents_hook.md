# OpenClawBrain Memory Graph (auto-installed)

**Query** (before answering questions about prior work, context, decisions, corrections, lessons):
```bash
python3 -m openclawbrain.openclaw_adapter.query_brain ~/.openclawbrain/AGENT/state.json '<summary of user message>' --chat-id '<chat_id from inbound metadata>' --format prompt --exclude-bootstrap --max-prompt-context-chars 12000
```
Always pass `--chat-id` so fired nodes are logged for later corrections.

**Learn** (after each response, no fired node IDs needed):
- Good: `python3 -m openclawbrain.openclaw_adapter.learn_by_chat_id --state ~/.openclawbrain/AGENT/state.json --chat-id '<chat_id>' --outcome 1.0 --lookback 1 --json`
- Bad: `python3 -m openclawbrain.openclaw_adapter.learn_by_chat_id --state ~/.openclawbrain/AGENT/state.json --chat-id '<chat_id>' --outcome -1.0 --lookback 1 --json`

**Inject correction** (when corrected — same turn, don't wait for harvester):
```bash
python3 -m openclawbrain.openclaw_adapter.learn_correction \
  --state ~/.openclawbrain/AGENT/state.json \
  --chat-id '<chat_id>' --outcome -1.0 \
  --content "The correction text here"
```
This penalizes the last query's fired nodes AND injects a CORRECTION node with inhibitory edges.

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
