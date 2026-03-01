#!/usr/bin/env bash
set -euo pipefail

FAILS=0
WARNS=0
PASSES=0

if [[ -t 1 ]]; then
  C_RESET='\033[0m'
  C_BOLD='\033[1m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_RED='\033[31m'
else
  C_RESET=''
  C_BOLD=''
  C_GREEN=''
  C_YELLOW=''
  C_RED=''
fi

TRANSIENT_PATH_RE='/Users/[^[:space:]"]+/worktrees|/private/var/folders'

log_section() {
  printf '\n%s== %s ==%s\n' "$C_BOLD" "$1" "$C_RESET"
}

pass() {
  PASSES=$((PASSES + 1))
  printf '%bPASS%b %s\n' "$C_GREEN" "$C_RESET" "$1"
}

warn() {
  WARNS=$((WARNS + 1))
  printf '%bWARN%b %s\n' "$C_YELLOW" "$C_RESET" "$1"
}

fail() {
  FAILS=$((FAILS + 1))
  printf '%bFAIL%b %s\n' "$C_RED" "$C_RESET" "$1"
}

file_has_transient_path() {
  local file="$1"
  grep -E -q "$TRANSIENT_PATH_RE" "$file"
}

format_bytes() {
  local bytes="$1"
  local kib=$((1024))
  local mib=$((1024 * 1024))
  local gib=$((1024 * 1024 * 1024))

  if (( bytes >= gib )); then
    printf '%d GiB' $((bytes / gib))
  elif (( bytes >= mib )); then
    printf '%d MiB' $((bytes / mib))
  elif (( bytes >= kib )); then
    printf '%d KiB' $((bytes / kib))
  else
    printf '%d B' "$bytes"
  fi
}

backup_summary() {
  local brain_dir="$1"
  local count=0
  local total_bytes=0

  while IFS= read -r -d '' dir; do
    count=$((count + 1))
    local size
    size=$(du -sk "$dir" 2>/dev/null | awk '{print $1}' || true)
    if [[ "$size" =~ ^[0-9]+$ ]]; then
      total_bytes=$((total_bytes + size * 1024))
    fi
  done < <(find "$brain_dir" -maxdepth 1 -type d -name 'backup-*' -print0 2>/dev/null)

  printf 'backup dirs=%d total=%s' "$count" "$(format_bytes "$total_bytes")"
}

check_transient_path_surface() {
  local file="$1"
  local label="$2"

  if [[ ! -f "$file" ]]; then
    pass "$label: not present"
    return
  fi

  if file_has_transient_path "$file"; then
    fail "$label: transient path leak pattern detected"
  else
    pass "$label: no transient path leak pattern found"
  fi
}

check_launchagent_env_keys() {
  local file="$1"
  local label="$2"

  if [[ ! -f "$file" ]]; then
    warn "$label: missing (cannot verify EnvironmentVariables keys)"
    return
  fi

  if grep -q 'OPENAI_API_KEY' "$file"; then
    pass "$label: EnvironmentVariables references OPENAI_API_KEY key name"
  else
    warn "$label: OPENAI_API_KEY key name not found in plist"
  fi
}

check_launchagent_workspace_hint() {
  local file="$1"
  local label="$2"

  if [[ ! -f "$file" ]]; then
    warn "$label: missing (cannot verify workspace root hints)"
    return
  fi

  if grep -E -q '~/.openclaw/workspace|/\.openclaw/workspace|OPENCLAW_WORKSPACE' "$file"; then
    pass "$label: workspace root hint found"
  else
    warn "$label: no workspace root hint found (possible config drift)"
  fi
}

check_brain_dir() {
  local name="$1"
  local brain_dir="$HOME/.openclawbrain/$name"
  local state="$brain_dir/state.json"
  local sock="$brain_dir/daemon.sock"

  if [[ ! -d "$brain_dir" ]]; then
    pass "$name: brain directory not present"
    return
  fi

  if [[ -f "$state" ]]; then
    pass "$name: state.json exists"
  else
    fail "$name: missing state.json"
  fi

  if [[ -S "$sock" ]]; then
    pass "$name: daemon.sock exists"
  else
    warn "$name: daemon.sock missing"
  fi

  pass "$name: $(backup_summary "$brain_dir")"
}

log_section 'Transient Path Leak Sweep'
check_transient_path_surface "$HOME/Library/LaunchAgents/com.openclawbrain.main.plist" 'LaunchAgent main plist'
check_transient_path_surface "$HOME/Library/LaunchAgents/com.openclawbrain.pelican.plist" 'LaunchAgent pelican plist'
check_transient_path_surface "$HOME/Library/LaunchAgents/com.openclawbrain.bountiful.plist" 'LaunchAgent bountiful plist'
check_transient_path_surface "$HOME/.openclaw/cron/jobs.json" 'Cron jobs.json'
check_transient_path_surface "$HOME/.openclaw/config.yaml" 'OpenClaw config.yaml'

log_section 'LaunchAgent Config Drift'
check_launchagent_env_keys "$HOME/Library/LaunchAgents/com.openclawbrain.main.plist" 'LaunchAgent main plist'
check_launchagent_env_keys "$HOME/Library/LaunchAgents/com.openclawbrain.pelican.plist" 'LaunchAgent pelican plist'
check_launchagent_env_keys "$HOME/Library/LaunchAgents/com.openclawbrain.bountiful.plist" 'LaunchAgent bountiful plist'
check_launchagent_workspace_hint "$HOME/Library/LaunchAgents/com.openclawbrain.main.plist" 'LaunchAgent main plist'
check_launchagent_workspace_hint "$HOME/Library/LaunchAgents/com.openclawbrain.pelican.plist" 'LaunchAgent pelican plist'
check_launchagent_workspace_hint "$HOME/Library/LaunchAgents/com.openclawbrain.bountiful.plist" 'LaunchAgent bountiful plist'

log_section 'Brain On Sanity'
check_brain_dir 'main'
check_brain_dir 'pelican'
check_brain_dir 'bountiful'

printf '\n%sSummary%s PASS=%d WARN=%d FAIL=%d\n' "$C_BOLD" "$C_RESET" "$PASSES" "$WARNS" "$FAILS"
exit "$FAILS"
