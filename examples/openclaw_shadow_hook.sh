#!/usr/bin/env bash
set +e

CRABPATH_GRAPH="${CRABPATH_GRAPH:-$HOME/.crabpath/graph.json}"
CRABPATH_INDEX="${CRABPATH_INDEX:-$HOME/.crabpath/embeddings.json}"
KILL_SWITCH="$HOME/.crabpath-shadow-disabled"
SHADOW_LOG="$HOME/.crabpath/shadow.log"
QUERY="$*"

if [ -z "$QUERY" ]; then
  exit 0
fi

if [ -f "$KILL_SWITCH" ]; then
  exit 0
fi

mkdir -p "$HOME/.crabpath"

if command -v timeout >/dev/null 2>&1; then
  OUTPUT="$(timeout 15s crabpath query --graph "$CRABPATH_GRAPH" --index "$CRABPATH_INDEX" "$QUERY" 2>&1 || true)"
else
  OUTPUT="$(crabpath query --graph "$CRABPATH_GRAPH" --index "$CRABPATH_INDEX" "$QUERY" 2>&1 || true)"
fi

{
  printf '%s query=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$QUERY"
  printf '%s\n' "$OUTPUT"
  printf '---\n'
} >> "$SHADOW_LOG"
