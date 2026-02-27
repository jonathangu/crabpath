# OpenClawBrain Architecture: two-timescale operation

OpenClawBrain runs on two timescales.

For production setup and runbooks, see [OpenClawBrain Operator Setup Guide](setup-guide.md).

## 1) Overview

OpenClawBrain stores a learned retrieval policy as a graph of **nodes** and directed **edges** in `state.json`.
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
   - `apply_outcome_pg()` for the default full-policy updates, or
   - `apply_outcome()` for local credit assignment with a simpler sparse update.
5. New material can be added with `inject()`.

## 2-b) Persistent Worker (`openclawbrain daemon`)

OpenClawBrain now supports a persistent worker process (`openclawbrain daemon`) that keeps
`Graph` and `VectorIndex` in memory for the lifetime of the process.

- Starts once and loads `state.json` at startup.
- Accepts NDJSON requests over stdin and writes one-line JSON responses to stdout.
- Avoids per-call reload overhead in the fast path, often saving ~100–800ms per query-heavy call path.
- Supports periodic and explicit save behavior.

Command:

```bash
python3 -m openclawbrain daemon --state PATH [--embed-model text-embedding-3-small] [--auto-save-interval 10]
```

Example request/response:

```json
{"id":"req-1","method":"query","params":{"query":"how to deploy","top_k":4,"chat_id":"telegram:123"}}
```

```json
{"id":"req-1","result":{"fired_nodes":["a"],"context":"...","seeds":[["a",0.96]],"embed_query_ms":1.1,"traverse_ms":2.4,"total_ms":3.5}}
```

Supported methods:

- `query`: returns `fired_nodes`, `context`, `seeds`, timing (`embed_query_ms`, `traverse_ms`, `total_ms`)
- `learn`: returns `edges_updated`
- `inject`: writes a new node and returns `{injected, node_id}`
- `correction`: penalizes recent query history for a chat and optionally injects a correction node
- `maintain`: returns `health_before`, `health_after`, `merges_applied`
- `health`: returns health metrics
- `info`: returns node/edge counts and `embedder`
- `save`: persists state immediately
- `reload`: reloads `state.json` from disk
- `shutdown`: persists pending writes and exits

Auto-save behavior:

- `write_count` increments after write operations (`learn`, `inject`, `correction`, `maintain`).
- On every `N` writes, where `N` is `--auto-save-interval`, the daemon saves state.
- `shutdown` persists pending state before exit.

### Socket Transport

`openclawbrain.socket_server` is an asyncio Unix socket bridge around the same NDJSON daemon worker.

- The socket listener creates `daemon.sock` under the agent directory.
- Each request is routed as NDJSON through the same request parser.
- Each response preserves the same JSON-RPC `id`.
- Concurrent clients are supported with a serialized lock to keep daemon execution single-threaded and deterministic.

Missing vs adapter scripts (current production reality):

- `inject`/`correction`: now supported by the daemon as NDJSON methods.
- fired log tracking: adapters can still use `fired_log.jsonl`, while the daemon also writes append-only compatibility entries from query calls.
- socket interface: now provided by `openclawbrain.socket_server` with per-agent Unix socket.

### Vocabulary

- **Brain**: persisted `state.json` plus associated artifacts.
- **Query trace**: ordered fired nodes + metadata from one query.
- **Feedback signal**: scalar outcome (`+1`, `-1`, or intermediate).
- **Online learning**: edge-weight updates plus injection in response to each interaction.

## 3) Maintenance (slow loop)

### What happens

```text
health -> decay -> scale -> split -> merge -> prune -> connect
```

1. **Health snapshot**: measure graph structure and quality with `measure_health()`.
2. **Decay**: weaken edges with adaptive half-life via `apply_decay()`.
3. **Scale**: apply `apply_synaptic_scaling()` to keep per-node positive outgoing weight mass in a mild budget.
4. **Split**: identify bloated multi-topic nodes with `suggest_splits()` and apply `split_node()`.
5. **Merge**: propose and apply structural merges with `suggest_merges()` and `apply_merge()`.
6. **Prune**: remove weak edges and orphaned nodes.
7. **Connect**: connect learning nodes to workspace neighborhoods.

### Functions used

- `run_maintenance()` (new)
- or directly: `measure_health()`, `apply_decay()`, `apply_synaptic_scaling()`, `suggest_splits()`, `split_node()`, `suggest_merges()`, `apply_merge()`, `prune_edges()`, `prune_orphan_nodes()`, `connect_learning_nodes()`

Split and merge form a balancing pair: split raises topic granularity when nodes become overloaded, while merge later recombines co-firing fragments when redundancy and similarity make them better represented as one node.

## Self-Regulation

Maintenance now includes three biologically inspired mechanisms:

- **Homeostatic decay (firing-rate homeostasis).** `run_maintenance()` counts reflex-tier edges and adapts decay half-life toward a target ratio of 5-15%, speeding up decay when too many edges stay reflex and slowing down when reflexivity is too low.
- **Synaptic scaling (activity-dependent normalization).** `apply_synaptic_scaling()` computes each node's outgoing positive mass and gently rescales above-budget source nodes, preserving relative edge structure while preventing hub lock-in.
- **Tier hysteresis (stateful transition smoothing).** Habitual and reflex boundaries are widened to reduce edge flapping between tiers (`reflex >= 0.6`, `habitual 0.15-0.6`).

These map to standard biological motifs: firing-rate homeostasis keeps overall excitability in range, synaptic scaling preserves relative patterns under constrained total synaptic weight, and hysteresis reduces chatter near decision thresholds.

## 4) Constitutional Anchors

OpenClawBrain uses node metadata to mark maintenance authority:

- `authority="constitutional"`: immutable anchors (`SOUL.md`, `AGENTS.md`, `USER.md`-style hard rules).
  - never decay
  - never prune nodes or edges
  - never merge
- `authority="canonical"`: stable long-lived anchors (`MEMORY.md`, `TOOLS.md`-style references).
  - decay at half speed (2x half-life)
  - never prune if the node is still weighted
  - merge only with LLM confirmation
- `authority="overlay"` (default): normal maintenance policy for session and working nodes (`TEACHING`, `CORRECTION`, derived nodes).

Maintenance behavior:

- `prune_edges()` skips any edge touching a constitutional node.
- `prune_orphan_nodes()` keeps constitutional and canonical nodes even if disconnected.
- `run_maintenance()` split candidates skip constitutional nodes, mark canonical nodes as confirmation-only, and merge candidates are filtered to exclude constitutional nodes; canonical merges require LLM confirmation.
- `run_maintenance()` decay skips constitutional sources and applies 2x half-life to canonical sources.

### When it runs

Maintenance is **not** part of request path.
Run it periodically (cron, timer, CI job, custom workflow) or after meaningful events (re-index, large workspace changes, etc.).

## 5) Integration with OpenClaw

OpenClawBrain is designed to plug into OpenClaw’s existing contract:
OpenClaw agents read `AGENTS.md` and follow the runbook.

### End-to-end data flow

```text
User message → OpenClaw agent → reads AGENTS.md → queries OpenClawBrain daemon → gets context → agent responds → learns from outcome
```

### Practical shape (today)

- OpenClaw agent runs a shell command from `AGENTS.md`.
- The command calls the OpenClawBrain adapter scripts (Python) to query/learn.
- Fired node IDs are logged per chat (`--chat-id`) so same-turn corrections can apply to the right retrieval.

This is optimized for production through the socket transport wrapper.

### Practical shape (production)

Run `openclawbrain daemon` as a long-lived worker and talk to it from a small integration layer:

- daemon keeps `state.json` hot in memory
- OpenClaw integration sends NDJSON requests (query/learn/maintain)
- you avoid per-call Python startup + state reload overhead

(Keeping the daemon as stdio is intentional: it’s easy to supervise and very hard to accidentally expose on the network.)

## 5-b) Integration Contract

### Framework provides

- Paths:
  - `state_path`
  - optional `journal_path`
- Callbacks:
  - `embed_fn(text) -> vector` (optional)
  - `llm_fn(system, user) -> str` (optional)
- Scheduling and lock strategy.

#### Callback Construction

OpenClawBrain requires callbacks only. Core code does not import provider SDKs.

```python
from examples.ops.callbacks import make_embed_fn, make_llm_fn

embed_fn = make_embed_fn("openai")  # defaults to text-embedding-3-small
llm_fn = make_llm_fn("gpt-5-mini")

run_maintenance(
    state_path="...",
    embed_fn=embed_fn,
    llm_fn=llm_fn,
)

apply_outcome_pg(...)

```

#### Cost and defaults

- Recommended: OpenAI `text-embedding-3-small` + `gpt-5-mini` for routing/scoring in production.
- Hash embeddings are supported as a testing/offline fallback and should not be treated as the default production path.

### OpenClawBrain provides

- `state.json` (graph + index + meta)
- `journal.jsonl` (append-only telemetry)
- `fired_log.jsonl` (per-chat fired IDs when using adapters)
- `injected_corrections.jsonl` (dedup for injected correction nodes)

### In one sentence

OpenClawBrain handles query-time scoring and maintenance operators;
you own orchestration, retries, and scheduler.

## 6) Brain directory layout

A brain directory is the operational unit.

- `state.json` — graph, index, and metadata.
- `journal.jsonl` — query, learn, health events and maintenance traces.
- `fired_log.jsonl` — optional fired-node history (chat/session based).
- `injected_corrections.jsonl` — optional dedup ledger for correction nodes.

## 7) Vocabulary table

| Term | Meaning |
|---|---|
| Brain | Persisted state containing graph, index, and metadata. |
| Query trace | Ordered fired nodes and query context for one interaction. |
| Feedback signal | Scalar outcome for later reinforcement (`+1` good, `-1` bad). |
| Online learning | Per-query weight updates using outcomes. |
| Maintenance | Periodic structural updates over the stored graph. |
| Split | Break overloaded, broad nodes into focused children. |
| Prune | Remove weak edges and orphaned nodes. |
| Merge | Combine redundant/co-firing nodes. |
| Connect | Attach learning nodes back into workspace neighborhoods. |
| Dormant edge | Low-significance edge from health perspective. |
| Orphan | Node with no incoming and no outgoing edges. |

## 8) Scheduling

Run maintenance with your scheduler of choice. OpenClawBrain has no scheduler dependency.

### cron

```cron
# Nightly graph health + decay + split + prune
0 3 * * * /usr/bin/python3 -m openclawbrain.cli maintain --state ~/.openclawbrain/main/state.json

# Weekly deep sweep
0 4 * * 0 /usr/bin/python3 -m openclawbrain.cli maintain --state ~/.openclawbrain/main/state.json --tasks health,decay,split,merge,prune,connect
```

### systemd timer

- `openclawbrain-maintenance.service`: run `/usr/bin/python3 -m openclawbrain.cli maintain --state ...`
- `openclawbrain-maintenance.timer`: `OnCalendar=daily` (or `OnUnitActiveSec=24h`)

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
      - run: pip install openclawbrain
      - run: python3 examples/ops/run_maintenance.py --state ./brains/main/state.json
```

## 9) Fast vs slow loop (concise)

### Fast loop

- **Query path**: `query -> traverse -> log_query`
- **Learning**: `apply_outcome_pg()` (default) / `apply_outcome()` (simpler alternative)
- **Insertions**: `inject_*`

### Slow loop

- **Maintenance command**: `openclawbrain maintain ...`
- **Health metrics**: `measure_health()` and `log_health()`
- **Topology updates**: decay, split, merge, prune, connect

## 9) ASCII diagram

```text
            (request path)                              (scheduled / batch)
  user ──▶ query ──▶ fired nodes ──▶ response     ──▶   maintenance runner
                │         │                               │
                │         └─▶ log trace (journal/fired)    ├─ health + decay
                │                                         ├─ split
                │                                         ├─ merge
                feedback (+/-)                                    ├─ prune
                │                                         ├─ connect
                └─▶ learn/inject                           └─ publish new state.json
```

## 10) Practical notes

- Use `run_maintenance()` for a complete slow-loop pass.
- Keep `--tasks` tight when experimenting (`health,decay` first, then add `split`/`merge`/`prune`/`connect`).
- Keep telemetry consistent so regressions are diagnosable.

## 11) Context Lifecycle

OpenClawBrain is designed to work with both file-based canonical state and graph-based recall.

The full cycle is:

- Agent edits files (canonical state) → normal OpenClaw behavior
- Harvester runs periodically → ingests file changes + session corrections
- OpenClawBrain sync → re-embeds changed chunks, sets authority levels
- OpenClawBrain learns → online edge updates from outcomes
- OpenClawBrain maintains → structural ops (decay/split/merge/prune/connect)
- Files get compacted → old daily notes shrink, content migrates to graph

```text
Files (small, current) ──edit──→ Files
  │                                │
  │ harvest + sync                 │ compact
  ▼                                ▼
OpenClawBrain Graph (learned) ◄─────── Harvester
  │
  └── single retrieval engine
```

## 12) Authority Levels

Authority is metadata used by maintenance policy:

- `constitutional`: never decay/prune/merge (`SOUL.md`, `AGENTS.md`)
- `canonical`: slow decay, protected prune (`USER.md`, `TOOLS.md`, `MEMORY.md`)
- `overlay`: normal rules (e.g. `TEACHING`, `CORRECTION`, session-derived signals)

## 13) Write Policy

When should the agent edit a file vs inject into OpenClawBrain?

| Situation | Action |
|---|---|
| New durable fact | Edit file → sync re-embeds it |
| Correction received | Edit file + learn_correction.py |
| Soft teaching | `openclawbrain inject --type TEACHING` only |
| Wrong retrieval | `learn_correction.py` (graph-only) |
| New rule/contract | Edit `AGENTS.md` or `SOUL.md` |
| Monitoring item | Edit `HEARTBEAT.md` |

## 14) File Compaction

- Daily notes older than 7 days get compacted.
- Key facts are extracted and injected as `TEACHING` nodes.
- Files shrink to ~15 lines (headers + key decisions).
- Full historical content lives in the graph for retrieval and learning.
