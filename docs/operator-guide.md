# OpenClawBrain Principal Operator Guide

## 1) What OpenClawBrain is
OpenClawBrain is the long-term memory layer for OpenClaw agents: it builds a learned graph from your workspace (`~/.openclaw/workspace`) plus session feedback (`~/.openclaw/agents/<agent>/sessions`), then serves fast query/learn operations from a persistent daemon state file (`~/.openclawbrain/<agent>/state.json`).

## 2) Turn brain ON
Canonical run command (this repo has no `openclawbrain serve` subcommand):

```bash
python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json
```

What `socket_server` does:
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

Inspect:

```bash
cat ~/.openclawbrain/main/replay_checkpoint.json
# or:
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
- Fast-learning only: ends with `Completed fast-learning; stopped before replay/harvest.`
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
Use rebuild-then-cutover for safe parallel operations.

## 8) Troubleshooting

| Symptom | Likely cause | Commands |
|---|---|---|
| `status` says `Daemon: not running` | socket server not started, crashed, or wrong state path | `python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json` then `openclawbrain status --state ~/.openclawbrain/main/state.json` |
| `daemon.sock` missing | server never started or wrong agent/state directory | `ls -la ~/.openclawbrain/main` and restart socket server with the same `--state` |
| Replay refuses/unsafe on LIVE | active daemon socket detected; single-writer guard | stop daemon first, or run `examples/ops/rebuild_then_cutover.sh main ~/.openclaw/workspace ~/.openclaw/agents/main/sessions` |
| Replay restarts from old work | checkpoint not used, wrong path, or intentionally ignored | run with `--resume --checkpoint ~/.openclawbrain/main/replay_checkpoint.json`; inspect file with `jq . ~/.openclawbrain/main/replay_checkpoint.json` |
| `LLM required for fast-learning` | no OpenAI client/key configured for fast-learning mining | set `OPENAI_API_KEY` or run `--edges-only` replay path |
| CLI says invalid sessions path | wrong sessions directory/file path | `ls -la ~/.openclaw/agents/main/sessions` and pass existing dir/files to `--sessions` |
