# Framework-agnostic operations examples

`examples/ops/` contains generic, minimal scripts that demonstrate both loops.

- `query_and_learn.py`: fast-loop interaction (`query -> traverse -> outcome`)
- `run_maintenance.py`: slow-loop maintenance run (`health,decay,merge,prune`)
- `compact_notes.py`: compact old daily notes into teaching summaries before graph updates
- `replay_last_days.sh`: full-learning replay over session files modified in the last N days, with safe checkpoint defaults for long runs
- `cutover_then_background_full_learning.sh`: fast-learning cutover first, then background full-learning replay via `nohup`
- `rebuild_then_cutover.sh`: rebuild into a new brain directory, verify, then atomic cutover with launchd-aware stop/start
- Replay aliases: `--extract-learning-events` = `--fast-learning`, `--full-pipeline` = `--full-learning`

By default, examples use OpenAI callbacks as the production path:

- `OPENAI_API_KEY` is expected in the environment.
- Embeddings use `text-embedding-3-small`.
- LLM decisions use `gpt-5-mini`.
- If `OPENAI_API_KEY` is missing, examples automatically fall back to `--embedder hash`
  via `HashEmbedder` for zero-dependency testing.

## Callback construction

OpenClawBrain core is callback-only and does not import `openai`:

```python
from examples.ops.callbacks import make_embed_fn, make_llm_fn

embedder = make_embed_fn("openai")  # or "hash"
llm = make_llm_fn("gpt-5-mini")

run_maintenance(..., embed_fn=embedder, llm_fn=llm)
```

Use `--embedder hash` explicitly when you want deterministic, zero-API testing.

## Shell helper usage

Replay the last 14 days (defaults shown):

```bash
examples/ops/replay_last_days.sh \
  --state ~/.openclawbrain/main/state.json \
  --sessions-dir ~/.openclaw/agents/main/sessions \
  --days 14 \
  --replay-workers 4 \
  --workers 4 \
  --progress-every 2000 \
  --checkpoint-every-seconds 60 \
  --checkpoint-every 50 \
  --persist-state-every-seconds 300
```

Fast cutover, then background full-learning replay:

```bash
examples/ops/cutover_then_background_full_learning.sh \
  --state ~/.openclawbrain/main/state.json \
  --sessions-dir ~/.openclaw/agents/main/sessions \
  --replay-workers 4 \
  --workers 4 \
  --progress-every 2000 \
  --checkpoint-every-seconds 60 \
  --checkpoint-every 50 \
  --persist-state-every-seconds 300
```

No-drama rebuild then cutover:

```bash
examples/ops/rebuild_then_cutover.sh main ~/.openclaw/workspace \
  ~/.openclaw/agents/main/sessions
```

All shell scripts run with `set -euo pipefail` and do not print secrets.
