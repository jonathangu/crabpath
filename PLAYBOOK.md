> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# CrabPath Playbook: from install to active mode

## Quick Start (5 minutes)

1. Install
   - `pip install crabpath`
2. Bootstrap once
   - `crabpath migrate --workspace ~/.openclaw/workspace --output-graph graph.json`
3. Health check
   - `crabpath health --graph graph.json`
4. Start shadow mode
   - See the **Shadow Mode** section below, then run your hook with:
     - `tail -f ~/.crabpath/shadow.jsonl`

## Shadow Mode

- What it does
  - Runs in parallel with your normal agent behavior.
  - Only suggests context and telemetry.
  - Never modifies responses.
- How to enable it
  - Add this block to `AGENTS.md` (or agent bootstrap prompt):
    - `Before answering, run CrabPath query in shadow and compare suggested context before responding.`
    - `Never apply CrabPath output directly to the user response unless explicitly approved.`
  - Keep your normal static context loaders in place.
- Shadow hook script pattern
  - Save as `~/.bin/zsh.001/query` (executable) and source it where your shell reads hook scripts.
  - Pattern:
    - Read the current user message into `$CRABPATH_QUERY`.
    - If `~/.crabpath-shadow-disabled` exists, skip logging.
    - Otherwise call:
      - `crabpath query --graph "$CP_GRAPH" --index "$CP_INDEX" --top 12 --query "$CRABPATH_QUERY" --json`
    - Persist a tiny trail in `~/.crabpath/shadow.jsonl` for analysis.
- Logging
  - Default log: `~/.crabpath/shadow.jsonl`
  - Read latest entries:
    - `python - <<'PY'`
    - `from crabpath import ShadowLog`
    - `print(ShadowLog().tail(10))`
    - `PY`
  - Useful fields:
    - `selected_node_ids`, `selected_node_snippets`
    - `retrieval_scores`, `reward`, `reward_source`
    - `trajectory_edges`, `tier_snapshot`, `proto_edge_count`
- Cost
  - Scoring path: `~/bin/zsh.001/query` (LLM-backed retrieval scoring path).
  - Free path: no score flag / no API call variant.

## Graduating to Active Mode

- Switch when:
  - You see stable picks across N recent queries.
  - Health checks trend green and no major regressions.
  - Shadow log shows useful retrieval overlap with agent success cases.
- Then:
  - Use CrabPath output as supplementary context.
  - Keep static loading as fallback until you confirm steady behavior.
  - Continue to call `crabpath health` daily and review shadow logs weekly.

## Monitoring

- `crabpath health --graph graph.json`
  - Inspect all 8 health metrics.
- `crabpath evolve --graph graph.json --snapshots evolution.jsonl --report`
  - Track structural drift, pruning, and tier shifts over time.
- Shadow log
  - Check what was retrieved, what was scored, and why each pick was followed.
  - Use `ShadowLog().summary(last_n=50)` to watch:
    - average selected nodes
    - average reward
    - tier trends

## Troubleshooting

- Brain death (too aggressive decay)
  - Autotuner handles it via `crabpath evolve` + config suggestions.
  - Watch for reflex/habitual collapse and high `dormant_pct`.
- No cross-file edges
  - Lower promotion threshold (or run more queries so co-firing can promote edges).
- Kill switch
  - `rm graph.json` to remove current memory graph.
  - `touch ~/.crabpath-shadow-disabled` to stop shadow scoring immediately.
