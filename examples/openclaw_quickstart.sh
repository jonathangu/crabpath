#!/usr/bin/env bash
set -euo pipefail

# OpenClawBrain + OpenClaw quickstart
#
# What this script does:
# - installs openclawbrain
# - builds a brain from an OpenClaw workspace
# - starts the OpenClawBrain daemon
# - queries the brain once
# - appends an AGENTS.md hook block (if not already present)

WORKSPACE_DEFAULT="$HOME/.openclaw/workspace"
BRAIN_DIR_DEFAULT="$HOME/.openclawbrain/main"

WORKSPACE="${WORKSPACE:-$WORKSPACE_DEFAULT}"
BRAIN_DIR="${BRAIN_DIR:-$BRAIN_DIR_DEFAULT}"
STATE="$BRAIN_DIR/state.json"

AGENTS_MD="$WORKSPACE/AGENTS.md"

echo "==> Workspace: $WORKSPACE"
echo "==> Brain dir:  $BRAIN_DIR"

if [[ ! -d "$WORKSPACE" ]]; then
  echo "ERROR: workspace not found: $WORKSPACE" >&2
  echo "Set WORKSPACE=/path/to/openclaw/workspace and re-run." >&2
  exit 1
fi

echo "==> Installing openclawbrain (into current python/pip env)"
python3 -m pip install --upgrade openclawbrain

mkdir -p "$BRAIN_DIR"

if [[ ! -f "$STATE" ]]; then
  echo "==> Building brain (openclawbrain init)"
  openclawbrain init --workspace "$WORKSPACE" --output "$BRAIN_DIR"
else
  echo "==> state.json already exists; syncing workspace (incremental)"
  openclawbrain sync --state "$STATE" --workspace "$WORKSPACE" || true
fi

echo "==> Doctor/health"
openclawbrain doctor --state "$STATE" || true
openclawbrain health  --state "$STATE" || true

echo "==> Starting daemon in background"
# The daemon is a stdio protocol; this launches it so you can see it working.
# In real OpenClaw integration, you usually run the daemon under launchd/systemd
# and talk to it from a small wrapper.
openclawbrain daemon --state "$STATE" --auto-save-interval 10 >"$BRAIN_DIR/daemon.stdout.log" 2>"$BRAIN_DIR/daemon.stderr.log" &
DAEMON_PID=$!

sleep 0.3
if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
  echo "ERROR: daemon failed to start. See:" >&2
  echo "  $BRAIN_DIR/daemon.stderr.log" >&2
  exit 1
fi

echo "==> Daemon PID: $DAEMON_PID"

# Smoke-test query using the adapter script (this does NOT use the daemon yet;
# it proves the state file is valid and query works end-to-end).
# For a true daemon client, see examples/ops/client_example.py.

CHAT_ID="${CHAT_ID:-local:quickstart}"
QUERY="${QUERY:-what are the hard rules in this workspace?}"

echo "==> Querying brain (adapter script)"
python3 "$HOME/openclawbrain/examples/openclaw_adapter/query_brain.py" \
  "$STATE" "$QUERY" --chat-id "$CHAT_ID" --json \
  | tee "$BRAIN_DIR/last_query.json"

echo "==> Ensuring AGENTS.md hook exists"
HOOK_HEADER="## OpenClawBrain Memory Graph (auto-installed)"

if [[ ! -f "$AGENTS_MD" ]]; then
  echo "ERROR: AGENTS.md not found at: $AGENTS_MD" >&2
  echo "Create it (OpenClaw convention) and re-run, or set WORKSPACE." >&2
  exit 1
fi

if rg -q --fixed-string "$HOOK_HEADER" "$AGENTS_MD"; then
  echo "    Hook already present: $AGENTS_MD"
else
  echo "    Appending hook block to: $AGENTS_MD"
  cat >>"$AGENTS_MD" <<'EOF'

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
python3 ~/openclawbrain/examples/openclaw_adapter/learn_correction.py \
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
EOF
fi

echo ""
echo "==> Done. Next steps:"
echo "- Edit the hook paths: replace AGENT + workspace paths with your real ones."
echo "- For production, run the daemon under launchd/systemd (see docs/openclaw-integration.md)."
echo "- Stop the quickstart daemon: kill $DAEMON_PID"
