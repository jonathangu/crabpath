#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: rebuild_then_cutover.sh <agent> <workspace_dir> <sessions_path...>

Example:
  rebuild_then_cutover.sh main ~/.openclaw/workspace \
    ~/.openclaw/agents/main/sessions \
    ~/.openclaw/agents/other/sessions/session-2026-03-01.jsonl
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

if ! command -v openclawbrain >/dev/null 2>&1; then
  echo "error: openclawbrain CLI not found in PATH" >&2
  exit 1
fi

AGENT="$1"
WORKSPACE_DIR="$2"
shift 2
SESSIONS=("$@")

ROOT="${OPENCLAWBRAIN_ROOT:-$HOME/.openclawbrain}"
LIVE="$ROOT/$AGENT"
TS="$(date '+%Y%m%d-%H%M%S')"
NEW="$ROOT/$AGENT.rebuild.$TS"
BACKUP="$ROOT/$AGENT.bak.$TS"
NEW_STATE="$NEW/state.json"
CHECKPOINT="$NEW/replay_checkpoint.json"
SOCKET="$LIVE/daemon.sock"
LAUNCH_LABEL="com.openclawbrain.$AGENT"
LAUNCH_TARGET="gui/$(id -u)/$LAUNCH_LABEL"

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "error: workspace directory not found: $WORKSPACE_DIR" >&2
  exit 1
fi

for p in "${SESSIONS[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "error: sessions path not found: $p" >&2
    exit 1
  fi
done

mkdir -p "$ROOT"

if [[ -e "$NEW" ]]; then
  echo "error: rebuild target already exists: $NEW" >&2
  exit 1
fi

echo "== 1) Build NEW brain directory =="
echo "ROOT: $ROOT"
echo "LIVE: $LIVE"
echo "NEW:  $NEW"
openclawbrain init --workspace "$WORKSPACE_DIR" --output "$NEW"

echo
echo "== 2) Fast-learning replay into NEW state =="
openclawbrain replay \
  --state "$NEW_STATE" \
  --sessions "${SESSIONS[@]}" \
  --fast-learning \
  --stop-after-fast-learning \
  --resume \
  --checkpoint "$CHECKPOINT"

echo
echo "Optional full-learning (safe options):"
cat <<EOF
# Option A: Run full-learning into NEW *before* cutover (slower, but single-writer safe):
# openclawbrain replay \\
#   --state "$NEW_STATE" \\
#   --sessions ${SESSIONS[*]} \\
#   --full-learning \\
#   --resume \\
#   --checkpoint "$CHECKPOINT"
#
# Option B: After cutover, rebuild again into a fresh directory (run this script again)
# and cut over a second time.
EOF

echo
echo "== 3) Verify NEW state =="
if [[ ! -f "$NEW_STATE" ]]; then
  echo "error: expected state file not found: $NEW_STATE" >&2
  exit 1
fi

DOCTOR_OUTPUT="$(openclawbrain doctor --state "$NEW_STATE" 2>&1 || true)"
printf '%s\n' "$DOCTOR_OUTPUT"
if ! grep -q "state_file_exists: PASS" <<<"$DOCTOR_OUTPUT"; then
  echo "error: doctor failed state_file_exists for $NEW_STATE" >&2
  exit 1
fi
if ! grep -q "state_json_valid: PASS" <<<"$DOCTOR_OUTPUT"; then
  echo "error: doctor failed state_json_valid for $NEW_STATE" >&2
  exit 1
fi

HAVE_LAUNCHCTL=0
SERVICE_PRESENT=0
if command -v launchctl >/dev/null 2>&1; then
  HAVE_LAUNCHCTL=1
  if launchctl list "$LAUNCH_LABEL" >/dev/null 2>&1; then
    SERVICE_PRESENT=1
  fi
fi

echo
echo "== 4) Stop daemon (if managed by launchd) =="
if [[ "$HAVE_LAUNCHCTL" -eq 1 && "$SERVICE_PRESENT" -eq 1 ]]; then
  if launchctl stop "$LAUNCH_LABEL" >/dev/null 2>&1; then
    echo "Stopped $LAUNCH_LABEL via launchctl stop."
  else
    echo "launchctl stop failed; trying kickstart -k to force restart cycle."
    launchctl kickstart -k "$LAUNCH_TARGET"
  fi
elif [[ "$HAVE_LAUNCHCTL" -eq 1 ]]; then
  echo "No launchd service found for $LAUNCH_LABEL; skipping stop."
else
  echo "launchctl not available; skipping stop. Manually stop the daemon before cutover if needed."
fi

echo
echo "== 5) Atomic cutover (swap dirs) =="
MOVED_LIVE=0
if [[ -d "$LIVE" ]]; then
  mv "$LIVE" "$BACKUP"
  MOVED_LIVE=1
fi

if ! mv "$NEW" "$LIVE"; then
  echo "error: failed to move $NEW -> $LIVE" >&2
  if [[ "$MOVED_LIVE" -eq 1 && -d "$BACKUP" ]]; then
    mv "$BACKUP" "$LIVE"
    echo "rollback: restored original LIVE from backup"
  fi
  exit 1
fi

echo "Cutover complete."
if [[ "$MOVED_LIVE" -eq 1 ]]; then
  echo "Backup created: $BACKUP"
else
  echo "No previous LIVE directory was present."
fi

echo
echo "== 6) Restart daemon (if managed by launchd) =="
if [[ "$HAVE_LAUNCHCTL" -eq 1 && "$SERVICE_PRESENT" -eq 1 ]]; then
  launchctl kickstart -k "$LAUNCH_TARGET"
  echo "Restarted $LAUNCH_LABEL."
elif [[ "$HAVE_LAUNCHCTL" -eq 1 ]]; then
  echo "No launchd service found for $LAUNCH_LABEL; skipping restart."
else
  echo "launchctl not available; start the daemon manually for agent '$AGENT'."
fi

echo
echo "== 7) Verify daemon socket =="
if [[ "$HAVE_LAUNCHCTL" -eq 1 && "$SERVICE_PRESENT" -eq 1 ]]; then
  for _ in {1..30}; do
    if [[ -S "$SOCKET" ]]; then
      echo "Socket ready: $SOCKET"
      break
    fi
    sleep 1
  done
  if [[ ! -S "$SOCKET" ]]; then
    echo "warning: socket not found after restart: $SOCKET" >&2
    echo "check service logs, then restart manually:"
    echo "  launchctl kickstart -k $LAUNCH_TARGET"
    exit 1
  fi
else
  echo "Socket check skipped (no managed launchd restart). Expected socket path: $SOCKET"
fi

echo
echo "Done."
echo "LIVE brain: $LIVE"
echo "Backup: $BACKUP"
echo "If needed, rollback:"
echo "  mv \"$LIVE\" \"$LIVE.failed.$TS\""
echo "  mv \"$BACKUP\" \"$LIVE\""
