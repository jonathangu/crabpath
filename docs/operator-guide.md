# OpenClawBrain Principal Operator Guide

## 1) What OpenClawBrain is
OpenClawBrain is the long-term memory layer for OpenClaw agents: it builds a learned graph from your workspace (`~/.openclaw/workspace`) plus session feedback (`~/.openclaw/agents/<agent>/sessions`), then serves fast query/learn operations from a persistent daemon state file (`~/.openclawbrain/<agent>/state.json`).

## New agent recipe
Use the canonical bootstrap SOP when creating a brand-new OpenClaw profile-style setup (new workspace + dedicated brain + launchd + routing):
- [docs/new-agent-sop.md](new-agent-sop.md)
Use packaged adapter CLIs for agent hooks (no `~/openclawbrain` clone required):
- `python3 -m openclawbrain.openclaw_adapter.query_brain ...`
- `python3 -m openclawbrain.openclaw_adapter.capture_feedback ...`

## 2) Turn brain ON
Canonical run command:

```bash
openclawbrain serve --state ~/.openclawbrain/main/state.json
```

What `serve` does:
- Starts the long-lived `openclawbrain daemon` worker.
- Exposes a Unix socket at `~/.openclawbrain/main/daemon.sock` (for agent `main`).
- Keeps the daemon hot in memory and restarts it if needed.

## 3) Confirm it's ON

```bash
openclawbrain status --state ~/.openclawbrain/main/state.json
ls -l ~/.openclawbrain/main/daemon.sock
```

Healthy signal:
- `status` shows `Daemon: running ~/.openclawbrain/main/daemon.sock`
- `daemon.sock` exists as a Unix socket (`-S` / `srw...`).

## 4) Safe rebuild paths

### quick cutover
Use when you need improvements now and can defer full replay/harvest.

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions ~/.openclaw/agents/main/sessions \
  --fast-learning \
  --stop-after-fast-learning \
  --resume \
  --checkpoint ~/.openclawbrain/main/replay_checkpoint.json
```

Then keep serving from the daemon state.

### no-drama rebuild_then_cutover
Use when you need single-writer safety during rebuild and minimal cutover downtime.

```bash
examples/ops/rebuild_then_cutover.sh main ~/.openclaw/workspace \
  ~/.openclaw/agents/main/sessions
```

This rebuilds into a fresh directory, verifies it, stops daemon briefly at cutover, swaps dirs atomically, then restarts.

### full-learning guidance + when to schedule it
Use full-learning (`--full-learning`) off-peak (nightly/low traffic, and after large session growth or major workspace changes).  
Single-writer rule still applies: either run full-learning on a NEW state before cutover, or stop daemon writes before replaying LIVE.

## 5) Checkpoints & resume
Default checkpoint path is next to state: `~/.openclawbrain/main/replay_checkpoint.json`.

Inspect checkpoint state (recommended):

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --show-checkpoint \
  --resume
```

Machine-readable checkpoint status:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --show-checkpoint \
  --resume \
  --json
```

Direct file inspection is still useful for debugging:

```bash
jq . ~/.openclawbrain/main/replay_checkpoint.json
```

Flag meanings:
- `--checkpoint <path>`: checkpoint file location.
- `--resume`: continue from saved per-session line offsets.
- `--ignore-checkpoint`: start from scratch even if checkpoint exists.
- `--checkpoint-every-seconds N`: periodic time-based checkpointing.
- `--checkpoint-every N`: checkpoint every N replay windows/merge batches.
- `--stop-after-fast-learning`: end after fast-learning phase (for quick cutover).

## 6) Progress: expected output by phase
- Fast-learning prints progress and completes with `Completed fast-learning; stopped before replay/harvest.` when `--stop-after-fast-learning` is set.
- Replay emits a progress heartbeat every 30 seconds by default; add `--progress-every N` for per-item cadence.
- Use `--quiet` to suppress replay banners/progress when scripting.
- Replay phase: stderr shows `Loaded <N> interactions from session files`; with `--progress-every`, shows `[replay] <done>/<total> (<pct>%)`.
- Replay completion: `Replayed <n>/<total> queries, <m> cross-file edges created`
- Full-learning completion (`--full-learning`): final line includes harvest summary, e.g. `harvest: tasks=<k>, damped_edges=<x>`.

## 7) Concurrency rules (single writer)
Only one process should write a given `state.json` at a time.

Writers include:
- daemon learning/injection flows
- `openclawbrain replay`
- maintenance/harvest operations that persist state

If violated, writes can clobber each other (lost updates, split/corrupted operational state, stale checkpoints).  
OpenClawBrain enforces this with `state.json.lock` next to your state file.

If a lock is active and you still need to proceed, override only when you are certain there is no conflicting writer:
- `--force`
- `OPENCLAWBRAIN_STATE_LOCK_FORCE=1`

Recommended production path: use `examples/ops/rebuild_then_cutover.sh` so rebuild/replay happens on a fresh state and cutover is atomic.

## 8) Optional async teacher routing pass

Run this off the hot path (cron/manual). Query serving remains LLM-free.

```bash
openclawbrain async-route-pg \
  --state ~/.openclawbrain/main/state.json \
  --since-hours 24 \
  --max-queries 200 \
  --sample-rate 0.1 \
  --teacher openai \
  --teacher-model gpt-5-mini \
  --apply \
  --json
```

Operational notes:
- Default is dry-run. Add `--apply` to persist updates.
- If `OPENAI_API_KEY` is missing (or `--teacher none`), it reports teacher unavailable and writes no updates.
- Keep this under single-writer discipline like replay/maintain.

## 9) Troubleshooting

| Symptom | Likely cause | Commands |
|---|---|---|
| `status` says `Daemon: not running` | service not started, crashed, or wrong state path | `openclawbrain serve --state ~/.openclawbrain/main/state.json` then `openclawbrain status --state ~/.openclawbrain/main/state.json` |
| `daemon.sock` missing | service never started or wrong agent/state directory | `ls -la ~/.openclawbrain/main` and restart `openclawbrain serve` with the same `--state` |
| Replay fails with lock/single-writer message | another process holds `state.json.lock` | stop the other writer, use `examples/ops/rebuild_then_cutover.sh ...`, or expert override with `--force` / `OPENCLAWBRAIN_STATE_LOCK_FORCE=1` |
| Replay restarts from old work | checkpoint not used, wrong path, or intentionally ignored | run with `--resume --checkpoint ~/.openclawbrain/main/replay_checkpoint.json`; inspect with `openclawbrain replay --state ~/.openclawbrain/main/state.json --show-checkpoint --resume` |
| `LLM required for fast-learning` | no OpenAI client/key configured for fast-learning mining | set `OPENAI_API_KEY` or run `--edges-only` replay path |
| CLI says invalid sessions path | wrong sessions directory/file path | `ls -la ~/.openclaw/agents/main/sessions` and pass existing dir/files to `--sessions` |

## 10) Prompt-context trim eval (offline)
Use the lightweight harness to measure trim rate and dropped-authority distribution at common caps (`20k`/`30k`) directly from `state.json`:

```bash
python examples/eval/prompt_context_eval.py \
  --state ~/.openclawbrain/main/state.json \
  --queries-file /path/to/queries.txt
```

If no `--queries-file` is provided, a small built-in sample query set is used.

## 11) Defaults that matter (v12.2.5+)
- `max_prompt_context_chars` default: **30000** (daemon)
- `max_fired_nodes` default: **30** (daemon)
- prompt context is ordered deterministically but importance-first: **authority → score → stable source order**.

Useful telemetry fields (daemon `query` response + journal metadata):
- `prompt_context_len`, `prompt_context_max_chars`, `prompt_context_trimmed`
- `prompt_context_included_node_ids`
- `prompt_context_dropped_node_ids` (capped) + `prompt_context_dropped_count`
- `prompt_context_dropped_authority_counts`

## 12) OpenClaw adapter defaults for context efficiency
For OpenClaw integration, keep prompts token-tight and avoid re-sending files OpenClaw already loads:

- Use `python3 -m openclawbrain.openclaw_adapter.query_brain ... --format prompt`.
- Learn/inject in one canonical call with `python3 -m openclawbrain.openclaw_adapter.capture_feedback --state ... --chat-id ... --kind ... --content ... [--outcome ...] [--dedup-key ...]`.
- Keep `--exclude-bootstrap` enabled (adapter default) so `AGENTS.md`, `SOUL.md`, `USER.md`, `MEMORY.md`, and `active-tasks.md` are not duplicated in `prompt_context`.
- Start with `--max-prompt-context-chars 8000` to `12000`; increase only when retrieval quality requires it.
- Use `--exclude-recent-memory <today> <yesterday>` only when those daily notes are already loaded elsewhere in the same turn.

This preserves deterministic `prompt_context` while cutting duplicate tokens, matching the project’s “context efficiency/compression” operating model.

### Recipe: always-on same-turn self-learning
For the canonical policy text, use:
- `docs/openclaw-integration.md` → `Always-on self-learning (default)`
- `docs/new-agent-sop.md` → `Always-on self-learning policy (recommended)` (SOUL.md snippet)

Why this matters:
- Same-turn `capture_feedback` injects correction/teaching/directive immediately and can reinforce/penalize the just-fired route for that `chat_id`.
- `--dedup-key` (or `--message-id`) makes harvest/replay retries idempotent so feedback is not double-injected.

## 13) Bootstrap files + memory notes are always indexed (v12.2.5+)
Even if your OpenClaw workspace `.gitignore` excludes local operator files (common), OpenClawBrain will still index:
- `SOUL.md`, `AGENTS.md`, `USER.md`, `TOOLS.md`, `MEMORY.md`, `IDENTITY.md`, `HEARTBEAT.md`
- `active-tasks.md`, `WORKFLOW_AUTO.md`
- everything under `memory/`

This is intentional: these files are the “constitution + history” of your agent.

## 14) OpenClaw media understanding (audio/image) → better memory
OpenClaw has a built-in **media-understanding** pipeline that can:
- transcribe audio/voice notes
- describe images
- extract text from files

When enabled in OpenClaw config, it will set `ctx.Transcript` and/or append extracted blocks to `ctx.Body`, so the resulting session logs contain the actual text (not only `[media attached: ...]` stubs). OpenClawBrain replay/full-learning can then learn from that text.

If you rely on toolResult-only transcripts/OCR, keep `openclawbrain replay --include-tool-results` enabled (default).

## 15) Correction wiring: what exists vs what you still need
OpenClawBrain supports `correction(chat_id, lookback=N)` (it remembers recent fired paths per `chat_id`). To get *automatic* corrections in live chat, OpenClaw must:
1) pass a stable `chat_id` into each brain query
2) detect correction messages
3) call `correction(...)`

If you don't have that OpenClaw integration yet, you can still apply corrections manually via `openclawbrain self-learn` (offline) or daemon `correction` calls.

## 16) Operator audit: detect path leaks & config drift
Run this first (safe: does not print env var values or full file contents):

```bash
if [ -x examples/ops/audit_openclawbrain.sh ]; then
  examples/ops/audit_openclawbrain.sh
else
  files=(
    "$HOME/Library/LaunchAgents/com.openclawbrain.main.plist"
    "$HOME/Library/LaunchAgents/com.openclawbrain.pelican.plist"
    "$HOME/Library/LaunchAgents/com.openclawbrain.bountiful.plist"
    "$HOME/.openclaw/cron/jobs.json"
    "$HOME/.openclaw/config.yaml"
  )
  existing=()
  for f in "${files[@]}"; do [ -f "$f" ] && existing+=("$f"); done
  if [ "${#existing[@]}" -eq 0 ]; then
    echo "PASS no key files present to scan"
  elif command -v rg >/dev/null 2>&1; then
    rg -l -e '/Users/[^[:space:]"]+/worktrees|/private/var/folders' "${existing[@]}" \
      && echo "FAIL transient path leak pattern detected" \
      || echo "PASS no transient path leak pattern found"
  else
    grep -E -l '/Users/[^[:space:]"]+/worktrees|/private/var/folders' "${existing[@]}" \
      && echo "FAIL transient path leak pattern detected" \
      || echo "PASS no transient path leak pattern found"
  fi
fi
```

Full audit script:

```bash
examples/ops/audit_openclawbrain.sh
echo "exit_code=$?"  # number of FAIL checks; WARNs do not fail
```

What it checks:
- transient path leak patterns in LaunchAgents, cron jobs, and OpenClaw config
- launchd drift hints (missing plist files, missing env key names, missing workspace-root hints)
- per-brain sanity for `~/.openclawbrain/{main,pelican,bountiful}`: `state.json`, `daemon.sock`, and a backup directory summary (count + total size)

How to respond when it flags:
- `FAIL transient path leak pattern detected`: replace transient paths with stable operator-managed roots, then reload launchd/cron as needed
- `FAIL missing state.json`: restore from known-good backup or rebuild state, then restart daemon
- `WARN daemon.sock missing`: start or restart `openclawbrain serve` for that brain and re-check status
- `WARN missing env/workspace hints`: align LaunchAgent plists with your standard template, then `launchctl unload/load` and rerun the audit

## 16) Secret pointers recipe: harvest + audit
Use this to track capabilities and key pointers without ever storing values:

```bash
python3 -m openclawbrain.ops.harvest_secret_pointers \
  --workspace ~/.openclaw/workspace \
  --out ~/.openclaw/workspace/docs/secret-pointers.md
```

To include centralized env files (for example `~/.openclaw/credentials/env/*.env`), repeat `--extra-env-file`:

```bash
python3 -m openclawbrain.ops.harvest_secret_pointers \
  --workspace ~/.openclaw/workspace \
  --extra-env-file ~/.openclaw/credentials/env/backyard-ripe.env \
  --extra-env-file ~/.openclaw/credentials/env/quant-research.env \
  --out ~/.openclaw/workspace/docs/secret-pointers.md
```

By default, harvest also inventories local credential files from `~/.openclaw/credentials` (if present) and records only filename/path/mode pointers. It never reads or prints file contents. Override or disable as needed:

```bash
python3 -m openclawbrain.ops.harvest_secret_pointers \
  --workspace ~/.openclaw/workspace \
  --credentials-dir ~/.openclaw/credentials \
  --no-include-credentials
```

FULL accounting example (workspace + centralized env files + default credentials inventory):

```bash
python3 -m openclawbrain.ops.harvest_secret_pointers \
  --workspace ~/.openclaw/workspace \
  --extra-env-file ~/.openclaw/credentials/env/backyard-ripe.env \
  --extra-env-file ~/.openclaw/credentials/env/quant-research.env \
  --out ~/.openclaw/workspace/docs/secret-pointers.md
```

Optional JSON output for automation:

```bash
python3 -m openclawbrain.ops.harvest_secret_pointers \
  --workspace ~/.openclaw/workspace \
  --json
```

Run leak audit (path + line only, no secret echo):

```bash
python3 -m openclawbrain.ops.audit_secret_leaks \
  --workspace ~/.openclaw/workspace \
  --state ~/.openclawbrain/main/state.json \
  --strict
```

Host-level global registry (recommended for multi-workspace operators):

```bash
python3 -m openclawbrain.ops.sync_registry
```

Why this avoids drift:

- `sync_registry` writes one canonical registry under `~/.openclaw/credentials/registry`.
- Each workspace `docs/secret-pointers.md` and `docs/capabilities.md` becomes a symlink to that canonical file.
- Any refresh updates all workspaces at once because they all reference the same target.
