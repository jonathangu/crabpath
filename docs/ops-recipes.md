# OpenClawBrain Ops Recipes

Practical operator runbooks for cutovers and large replays.

Canonical day-0/day-2 runbook: **[docs/operator-guide.md](operator-guide.md)**

**OpenClaw path note:** OpenClaw agent session logs typically live at `~/.openclaw/agents/<agent>/sessions` (e.g., `~/.openclaw/agents/main/sessions`). You can pass that directory directly via `--sessions <dir>`.

## Cutover fast

Use this when you need improved retrieval quickly and can defer full replay/harvest.

1. Run fast-learning only:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions ~/.openclaw/agents/main/sessions \
  --fast-learning \
  --stop-after-fast-learning \
  --progress-every 25 \
  --resume \
  --checkpoint ~/.openclawbrain/main/replay_checkpoint.json
```

Example progress output during fast-learning:

```text
[fast_learning] 250/1800 (13.9%) elapsed=41.7s rate=6.00/s eta=258.3s
```

2. Start the daemon so serving traffic uses the latest state:

```bash
python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json
```

3. Run full replay/harvest later (off-peak):

⚠️ Single-writer rule: `replay` writes `state.json`. If your daemon is running and also writing/learning, do **not** replay against the LIVE state. Either stop the daemon during full-learning, or use the no-drama rebuild flow below.

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions ~/.openclaw/agents/main/sessions \
  --full-learning \
  --resume \
  --checkpoint ~/.openclawbrain/main/replay_checkpoint.json
```

`examples/ops/cutover_then_background_full_learning.sh` automates this sequence.

## No-drama rebuild + cutover (single-writer safe)

Use this when you need a full rebuild/replay without stopping the currently serving daemon during the rebuild itself.

Why:
- Rebuild/replay writes `state.json`.
- If rebuild and daemon both write the same LIVE state, writes can clobber/split state.

What this flow does:
1. Builds and replays into a brand-new directory: `~/.openclawbrain/<agent>.rebuild.<timestamp>`.
2. Verifies `state.json` in that new directory.
3. Stops the agent daemon briefly only at cutover time.
4. Atomically swaps directories (`LIVE -> .bak`, `NEW -> LIVE`) and restarts.

Run:

```bash
examples/ops/rebuild_then_cutover.sh <agent> <workspace_dir> <sessions_path...>
```

Example:

```bash
examples/ops/rebuild_then_cutover.sh main ~/.openclaw/workspace \
  ~/.openclaw/agents/main/sessions \
  ~/.openclaw/agents/main/sessions/session-2026-03-01.jsonl
```

Notes:
- `replay --sessions` accepts one or more paths, and each path may be a sessions directory or an individual `.jsonl` file.
- The helper script uses fast-learning (`--fast-learning --stop-after-fast-learning`) for fast, safe cutover.
- For full-learning, either run it into NEW before cutover, or rebuild again into a new directory and cut over again. Avoid replaying directly against LIVE while the daemon is running.
- Tradeoff: events learned while rebuild is running are not in the NEW snapshot unless you pause learning traffic or run a small delta replay before/after cutover.
- On systems without `launchctl`, the script skips stop/start and tells you what to do manually.

## Parallel replay

For large histories, run replay in parallel workers and checkpoint frequently:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions ~/.openclaw/agents/main/sessions \
  --full-learning \
  --replay-workers 4 \
  --workers 4 \
  --checkpoint ~/.openclawbrain/main/replay_checkpoint.json \
  --checkpoint-every-seconds 60 \
  --checkpoint-every 50 \
  --persist-state-every-seconds 300 \
  --resume
```

Notes:
- `--replay-workers` controls edge-replay parallelism.
- `merge_batches` in replay output indicates how many merge windows were applied.
- Use both `--checkpoint-every-seconds` and `--checkpoint-every` for long runs so restarts resume from recent progress.
- Replay startup now prints a banner (unless `--json`) with checkpoint path, `resume`, `ignore_checkpoint`, and planned phases.

Inspect checkpoint progress without opening JSON:

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

Resume semantics:
- Resume takes effect only when `--resume` is set and `--ignore-checkpoint` is not set.
- If phase-scoped offsets are missing, replay falls back to legacy top-level `sessions` offsets and emits a warning.

Use `examples/ops/replay_last_days.sh` when you want a bounded replay window.

## Prompt caching

For stable prompt caching, build the appendix from fired nodes and append it late:

```bash
python3 examples/openclaw_adapter/query_brain.py \
  ~/.openclawbrain/main/state.json \
  "user query summary" \
  --format prompt
```

Guidelines:
- Append the `[BRAIN_CONTEXT v1]...[/BRAIN_CONTEXT]` block near the end of the final prompt.
- Keep stable node ordering (do not reshuffle lines between retries).
- Avoid echoing/paraphrasing this block earlier in the prompt; duplicated context reduces cache stability.

## Media memory

OpenClaw session logs often store uploads as `[media attached: ...]` stubs. Those stubs are low-signal by themselves, so replay should ingest allowlisted `toolResult` text (OCR/transcripts/captions) when present.

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions ~/.openclaw/agents/main/sessions \
  --include-tool-results \
  --tool-result-allowlist image,openai-whisper,openai-whisper-api,openai-whisper-local,summarize \
  --tool-result-max-chars 20000
```

Flags:
- `--include-tool-results` enables attachment of tool outputs to media-stub user turns.
- `--tool-result-allowlist` limits ingestion to trusted tool names.
- `--tool-result-max-chars` bounds appended text size per interaction.
