#!/usr/bin/env bash
set -euo pipefail

STATE="${STATE:-$HOME/.openclawbrain/main/state.json}"
# For OpenClaw, sessions live under: ~/.openclaw/agents/<agent>/sessions
SESSIONS_DIR="${SESSIONS_DIR:-$HOME/.openclaw/agents/main/sessions}"
DAYS="${DAYS:-14}"
REPLAY_WORKERS="${REPLAY_WORKERS:-4}"
WORKERS="${WORKERS:-4}"
PROGRESS_EVERY="${PROGRESS_EVERY:-2000}"
CHECKPOINT_EVERY_SECONDS="${CHECKPOINT_EVERY_SECONDS:-60}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
PERSIST_STATE_EVERY_SECONDS="${PERSIST_STATE_EVERY_SECONDS:-300}"

usage() {
  cat <<'EOF'
Usage: replay_last_days.sh [options]

Options:
  --state PATH
  --sessions-dir PATH
  --days N
  --replay-workers N    # edge replay workers
  --workers N           # fast-learning LLM workers
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
    --days) DAYS="$2"; shift 2 ;;
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

CHECKPOINT="${CHECKPOINT:-$(dirname "$STATE")/replay_checkpoint.json}"

session_files=()
while IFS= read -r -d '' file; do
  session_files+=("$file")
done < <(
  find "$SESSIONS_DIR" -type f \
    \( -name '*.jsonl' -o -name '*.jsonl.reset.*' -o -name '*.jsonl.deleted.*' \) \
    -mtime "-$DAYS" -print0 | sort -z
)

if [[ "${#session_files[@]}" -eq 0 ]]; then
  echo "no session files found in the last $DAYS day(s) under: $SESSIONS_DIR" >&2
  exit 1
fi

arg_max="$(getconf ARG_MAX 2>/dev/null || echo 262144)"
arg_bytes=0
for file in "${session_files[@]}"; do
  arg_bytes=$((arg_bytes + ${#file} + 1))
done

use_sessions_dir=0
if [[ "${#session_files[@]}" -gt 5000 || "$arg_bytes" -gt $((arg_max / 2)) ]]; then
  use_sessions_dir=1
  echo "warning: too many session paths for a safe argv payload (${#session_files[@]} files, ~${arg_bytes} bytes)." >&2
  echo "warning: falling back to --sessions $SESSIONS_DIR (replay may include files older than --days)." >&2
fi

cmd=(
  openclawbrain replay
  --state "$STATE"
  --sessions
)
if [[ "$use_sessions_dir" -eq 1 ]]; then
  cmd+=("$SESSIONS_DIR")
else
  cmd+=("${session_files[@]}")
fi
cmd+=(
  --full-learning  # alias: --full-pipeline
  --resume
  --checkpoint "$CHECKPOINT"
  --replay-workers "$REPLAY_WORKERS"
  --workers "$WORKERS"
  --progress-every "$PROGRESS_EVERY"
  --checkpoint-every-seconds "$CHECKPOINT_EVERY_SECONDS"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --persist-state-every-seconds "$PERSIST_STATE_EVERY_SECONDS"
)

echo "Running replay:"
echo "  state: $STATE"
echo "  sessions_dir: $SESSIONS_DIR"
echo "  days: $DAYS"
echo "  selected_files: ${#session_files[@]}"
echo "  checkpoint: $CHECKPOINT"

exec "${cmd[@]}"
