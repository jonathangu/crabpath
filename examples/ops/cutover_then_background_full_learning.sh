#!/usr/bin/env bash
set -euo pipefail

STATE="${STATE:-$HOME/.openclawbrain/main/state.json}"
# For OpenClaw, sessions live under: ~/.openclaw/agents/<agent>/sessions
SESSIONS_DIR="${SESSIONS_DIR:-$HOME/.openclaw/agents/main/sessions}"
REPLAY_WORKERS="${REPLAY_WORKERS:-4}"
WORKERS="${WORKERS:-4}"
PROGRESS_EVERY="${PROGRESS_EVERY:-2000}"
CHECKPOINT_EVERY_SECONDS="${CHECKPOINT_EVERY_SECONDS:-60}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
PERSIST_STATE_EVERY_SECONDS="${PERSIST_STATE_EVERY_SECONDS:-300}"

usage() {
  cat <<'EOF'
Usage: cutover_then_background_full_learning.sh [options]

Options:
  --state PATH
  --sessions-dir PATH
  --replay-workers N
  --workers N
  --progress-every N
  --checkpoint-every-seconds N
  --checkpoint-every N
  --persist-state-every-seconds N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state) STATE="$2"; shift 2 ;;
    --sessions-dir) SESSIONS_DIR="$2"; shift 2 ;;
    --replay-workers) REPLAY_WORKERS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    --checkpoint-every-seconds) CHECKPOINT_EVERY_SECONDS="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --persist-state-every-seconds) PERSIST_STATE_EVERY_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "$STATE" ]]; then
  echo "state file not found: $STATE" >&2
  exit 1
fi
if [[ ! -d "$SESSIONS_DIR" ]]; then
  echo "sessions directory not found: $SESSIONS_DIR" >&2
  exit 1
fi

STATE_DIR="$(dirname "$STATE")"
CHECKPOINT="${CHECKPOINT:-$STATE_DIR/replay_checkpoint.json}"
SOCKET="$STATE_DIR/daemon.sock"

# Safety: replay writes state.json. If the daemon socket is live, assume the daemon is running
# and may write too. Enforce single-writer unless explicitly overridden.
if [[ -S "$SOCKET" && "${ALLOW_LIVE_REPLAY:-}" != "1" ]]; then
  echo "error: daemon socket detected at $SOCKET" >&2
  echo "Running replay against LIVE state while the daemon is active can clobber/split state." >&2
  echo "Stop the daemon first, or use examples/ops/rebuild_then_cutover.sh (no-drama rebuild)." >&2
  echo "To force anyway: ALLOW_LIVE_REPLAY=1 $0 ..." >&2
  exit 1
fi
timestamp="$(date '+%Y%m%d-%H%M%S')"
OUT_LOG="${OUT_LOG:-$STATE_DIR/replay-full-${timestamp}.out.log}"
ERR_LOG="${ERR_LOG:-$STATE_DIR/replay-full-${timestamp}.err.log}"

echo "1) Fast cutover pass (stop after fast-learning)"
openclawbrain replay \
  --state "$STATE" \
  --sessions "$SESSIONS_DIR" \
  --fast-learning \
  --stop-after-fast-learning \
  --resume \
  --checkpoint "$CHECKPOINT" \
  --workers "$WORKERS" \
  --checkpoint-every-seconds "$CHECKPOINT_EVERY_SECONDS" \
  --checkpoint-every "$CHECKPOINT_EVERY"

echo "2) Starting background full-learning replay with nohup"
nohup openclawbrain replay \
  --state "$STATE" \
  --sessions "$SESSIONS_DIR" \
  --full-learning \
  --resume \
  --checkpoint "$CHECKPOINT" \
  --replay-workers "$REPLAY_WORKERS" \
  --workers "$WORKERS" \
  --progress-every "$PROGRESS_EVERY" \
  --checkpoint-every-seconds "$CHECKPOINT_EVERY_SECONDS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --persist-state-every-seconds "$PERSIST_STATE_EVERY_SECONDS" \
  >"$OUT_LOG" 2>"$ERR_LOG" < /dev/null &

pid=$!

echo "Background replay started."
echo "  pid: $pid"
echo "  state: $STATE"
echo "  sessions_dir: $SESSIONS_DIR"
echo "  checkpoint: $CHECKPOINT"
echo "  stdout_log: $OUT_LOG"
echo "  stderr_log: $ERR_LOG"
