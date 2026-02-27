# CrabPath Architecture: two-timescale operation

CrabPath runs on two timescales.

## 1) Overview

CrabPath stores a learned retrieval policy as a graph of **nodes** and directed **edges** in `state.json`.
It separates:

- **Online learning (fast loop)**: cheap updates while serving queries.
- **Maintenance (slow loop)**: periodic structural operations to keep the graph healthy.

The core library is pure and does not assume any scheduler.

## 2) Online Learning (fast loop)

### What happens

```text
query -> traverse -> log trace -> feedback -> learn
```

1. The query is embedded and used as seeds for `traverse()`.
2. `traverse()` walks the graph, producing fired nodes and context.
3. A query trace is logged with `log_query()`.
4. Feedback arrives as `outcome` and is applied with:
   - `apply_outcome()` for local credit assignment, or
   - `apply_outcome_pg()` for full-policy updates.
5. New material can be added with `inject()`.

### Vocabulary

- **Brain**: persisted `state.json` plus associated artifacts.
- **Query trace**: ordered fired nodes + metadata from one query.
- **Feedback signal**: scalar outcome (`+1`, `-1`, or intermediate).
- **Online learning**: edge-weight updates plus injection in response to each interaction.

## 3) Maintenance (slow loop)

### What happens

```text
health -> decay -> merge -> prune -> connect
```

1. **Health snapshot**: measure graph structure and quality with `measure_health()`.
2. **Decay**: weaken dormant edges with `apply_decay()`.
3. **Merge**: propose and apply structural merges with `suggest_merges()` and `apply_merge()`.
4. **Prune**: remove weak edges and orphaned nodes.
5. **Connect**: connect learning nodes to workspace neighborhoods.

### Functions used

- `run_maintenance()` (new)
- or directly: `measure_health()`, `apply_decay()`, `suggest_merges()`, `apply_merge()`, `prune_edges()`, `prune_orphan_nodes()`, `connect_learning_nodes()`

### When it runs

Maintenance is **not** part of request path.
Run it periodically (cron, timer, CI job, custom workflow) or after meaningful events (re-index, large workspace changes, etc.).

## 4) Integration Contract

### Framework provides

- Paths:
  - `state_path`
  - optional `journal_path`
- Callbacks:
  - `embed_fn(text) -> vector` (optional)
  - `llm_fn(system, user) -> str` (optional)
- Scheduling and lock strategy.

### CrabPath provides

- `state.json` (graph + index + meta)
- `journal.jsonl` (append-only telemetry)
- `fired_log.jsonl` (per-chat fired IDs when using adapters)
- `injected_corrections.jsonl` (dedup for injected correction nodes)

### In one sentence

CrabPath handles query-time scoring and maintenance operators;
you own orchestration, retries, and scheduler.

## 5) Brain directory layout

A brain directory is the operational unit.

- `state.json` — graph, index, and metadata.
- `journal.jsonl` — query, learn, health events and maintenance traces.
- `fired_log.jsonl` — optional fired-node history (chat/session based).
- `injected_corrections.jsonl` — optional dedup ledger for correction nodes.

## 6) Vocabulary table

| Term | Meaning |
|---|---|
| Brain | Persisted state containing graph, index, and metadata. |
| Query trace | Ordered fired nodes and query context for one interaction. |
| Feedback signal | Scalar outcome for later reinforcement (`+1` good, `-1` bad). |
| Online learning | Per-query weight updates using outcomes. |
| Maintenance | Periodic structural updates over the stored graph. |
| Prune | Remove weak edges and orphaned nodes. |
| Merge | Combine redundant/co-firing nodes. |
| Connect | Attach learning nodes back into workspace neighborhoods. |
| Dormant edge | Low-significance edge from health perspective. |
| Orphan | Node with no incoming and no outgoing edges. |

## 7) Scheduling

Run maintenance with your scheduler of choice. CrabPath has no scheduler dependency.

### cron

```cron
# Nightly graph health + decay + merge + prune
0 3 * * * /usr/bin/python3 -m crabpath.cli maintain --state ~/.crabpath/main/state.json

# Weekly deep sweep
0 4 * * 0 /usr/bin/python3 -m crabpath.cli maintain --state ~/.crabpath/main/state.json --tasks health,decay,prune,merge,connect
```

### systemd timer

- `crabpath-maintenance.service`: run `/usr/bin/python3 -m crabpath.cli maintain --state ...`
- `crabpath-maintenance.timer`: `OnCalendar=daily` (or `OnUnitActiveSec=24h`)

### GitHub Actions

```yaml
name: maintenance
on:
  schedule:
    - cron: '0 3 * * *'
jobs:
  maintain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install crabpath
      - run: python3 examples/ops/run_maintenance.py --state ./brains/main/state.json
```

## 8) Fast vs slow loop (concise)

### Fast loop

- **Query path**: `query -> traverse -> log_query`
- **Learning**: `apply_outcome()` / `apply_outcome_pg()`
- **Insertions**: `inject_*`

### Slow loop

- **Maintenance command**: `crabpath maintain ...`
- **Health metrics**: `measure_health()` and `log_health()`
- **Topology updates**: decay, merge, prune, connect

## 9) ASCII diagram

```text
            (request path)                              (scheduled / batch)
  user ──▶ query ──▶ fired nodes ──▶ response     ──▶   maintenance runner
                │         │                               │
                │         └─▶ log trace (journal/fired)    ├─ health + decay
                │                                         ├─ merge
        feedback (+/-)                                    ├─ prune
                │                                         ├─ connect
                └─▶ learn/inject                           └─ publish new state.json
```

## 10) Practical notes

- Use `run_maintenance()` for a complete slow-loop pass.
- Keep `--tasks` tight when experimenting (`health,decay` first, then add `merge`/`prune`/`connect`).
- Keep telemetry consistent so regressions are diagnosable.
