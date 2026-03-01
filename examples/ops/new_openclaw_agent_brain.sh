#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Guided setup for a new OpenClaw agent workspace + dedicated OpenClawBrain.

Usage:
  examples/ops/new_openclaw_agent_brain.sh \
    --agent-id <agent_id> \
    --workspace <workspace_dir> \
    --brain-dir <brain_dir> \
    [--write-plist]

Notes:
- This script is intentionally guided (not fully automatic).
- It creates a workspace skeleton and runs `openclawbrain init`.
- It does not create Telegram bots and does not handle bot tokens.
USAGE
}

AGENT_ID=""
WORKSPACE_DIR=""
BRAIN_DIR=""
WRITE_PLIST="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-id)
      AGENT_ID="${2:-}"
      shift 2
      ;;
    --workspace)
      WORKSPACE_DIR="${2:-}"
      shift 2
      ;;
    --brain-dir)
      BRAIN_DIR="${2:-}"
      shift 2
      ;;
    --write-plist)
      WRITE_PLIST="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$AGENT_ID" || -z "$WORKSPACE_DIR" || -z "$BRAIN_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

WORKSPACE_DIR="${WORKSPACE_DIR/#\~/$HOME}"
BRAIN_DIR="${BRAIN_DIR/#\~/$HOME}"
STATE_PATH="$BRAIN_DIR/state.json"
PLIST_TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/com.openclawbrain.AGENT.plist.template"
PLIST_TARGET="$HOME/Library/LaunchAgents/com.openclawbrain.${AGENT_ID}.plist"

cat <<PLAN
Plan:
- Agent ID:          $AGENT_ID
- Workspace dir:     $WORKSPACE_DIR
- Brain dir:         $BRAIN_DIR
- State path:        $STATE_PATH
- Write launchd plist: $WRITE_PLIST
PLAN

echo
echo "[1/4] Creating workspace skeleton..."
mkdir -p "$WORKSPACE_DIR/memory"

if [[ ! -f "$WORKSPACE_DIR/AGENTS.md" ]]; then
  cat > "$WORKSPACE_DIR/AGENTS.md" <<'AGENTS'
# AGENTS

Replace this with your agent operating instructions.
AGENTS
fi

for f in SOUL.md USER.md MEMORY.md active-tasks.md memory/today.md; do
  touch "$WORKSPACE_DIR/$f"
done

echo "Created/updated workspace files under: $WORKSPACE_DIR"

echo
echo "[2/4] Initializing dedicated brain..."
mkdir -p "$BRAIN_DIR"
openclawbrain init --workspace "$WORKSPACE_DIR" --output "$BRAIN_DIR"

echo
echo "[3/4] Quick health check..."
openclawbrain doctor --state "$STATE_PATH"

echo
echo "[4/4] Next actions"
cat <<NEXT
- Pin the tight query line in: $WORKSPACE_DIR/AGENTS.md
- Start daemon manually for smoke test:
    openclawbrain serve --state "$STATE_PATH"
- Or configure launchd with template:
    $PLIST_TEMPLATE
NEXT

if [[ "$WRITE_PLIST" == "1" ]]; then
  echo
  echo "Generating launchd plist from template..."
  mkdir -p "$HOME/Library/LaunchAgents"
  LOG_PATH="$BRAIN_DIR/daemon.log"
  sed \
    -e "s|AGENT_ID|$AGENT_ID|g" \
    -e "s|STATE_PATH|$STATE_PATH|g" \
    -e "s|LOG_PATH|$LOG_PATH|g" \
    "$PLIST_TEMPLATE" > "$PLIST_TARGET"
  echo "Wrote: $PLIST_TARGET"
  cat <<LAUNCHD
To load it:
  launchctl bootstrap "gui/$(id -u)" "$PLIST_TARGET"
  launchctl kickstart -k "gui/$(id -u)/com.openclawbrain.$AGENT_ID"
  launchctl print "gui/$(id -u)/com.openclawbrain.$AGENT_ID"
LAUNCHD
fi

echo
echo "Done."
