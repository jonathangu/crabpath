# BrainProfile

`BrainProfile` is a strict JSON configuration scaffold for daemon/socket defaults.

Load order:

1. Built-in defaults
2. Profile file (`--profile PATH`)
3. Environment overrides (`OPENCLAWBRAIN_*`)
4. CLI flags

CLI flags are overrides, not source of truth.

## Supported Environment Overrides

- `OPENCLAWBRAIN_STATE_PATH`
- `OPENCLAWBRAIN_JOURNAL_PATH`
- `OPENCLAWBRAIN_EMBED_MODEL`
- `OPENCLAWBRAIN_MAX_PROMPT_CONTEXT_CHARS`
- `OPENCLAWBRAIN_MAX_FIRED_NODES`
- `OPENCLAWBRAIN_ROUTE_MODE`
- `OPENCLAWBRAIN_ROUTE_TOP_K`
- `OPENCLAWBRAIN_ROUTE_ALPHA_SIM`
- `OPENCLAWBRAIN_ROUTE_USE_RELEVANCE`
- `OPENCLAWBRAIN_REWARD_SOURCE`
- `OPENCLAWBRAIN_REWARD_WEIGHT_CORRECTION`
- `OPENCLAWBRAIN_REWARD_WEIGHT_TEACHING`
- `OPENCLAWBRAIN_REWARD_WEIGHT_DIRECTIVE`
- `OPENCLAWBRAIN_REWARD_WEIGHT_REINFORCEMENT`

## Example `brainprofile.json`

```json
{
  "paths": {
    "state_path": "/Users/example/.openclawbrain/main/state.json",
    "journal_path": "/Users/example/.openclawbrain/main/journal.jsonl"
  },
  "policy": {
    "max_prompt_context_chars": 24000,
    "max_fired_nodes": 40,
    "route_mode": "edge+sim",
    "route_top_k": 8,
    "route_alpha_sim": 0.65,
    "route_use_relevance": true
  },
  "reward": {
    "source": "explicit",
    "weight_correction": -1.0,
    "weight_teaching": 0.5,
    "weight_directive": 0.75,
    "weight_reinforcement": 1.0
  },
  "embedder": {
    "embed_model": "text-embedding-3-small"
  }
}
```
