# OpenClawBrain


> Your retrieval routes become the prompt — assembled by learned routing, not top-k similarity.

**Current release: v12.2.1**
**Website:** https://openclawbrain.ai

**Setup:** [Setup Guide](docs/setup-guide.md)

## OpenClaw Integration (start here if you run OpenClaw)

OpenClawBrain is designed to be the memory layer for **OpenClaw agents**.

- Guide: **[docs/openclaw-integration.md](docs/openclaw-integration.md)**

Quickstart (OpenClaw users):

```bash
pip install openclawbrain
openclawbrain init --workspace ~/.openclaw/workspace --output ~/.openclawbrain/main --embedder openai --llm openai
python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json
```

Production Deployment (socket):

Use LaunchAgent/systemd to keep the socket server running:

```bash
python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json
```

macOS (`~/Library/LaunchAgents/com.openclawbrain.daemon.plist`):

```xml
<key>ProgramArguments</key>
<array>
  <string>/usr/bin/python3</string>
  <string>-m</string>
  <string>openclawbrain.socket_server</string>
  <string>--state</string>
  <string>/Users/YOU/.openclawbrain/main/state.json</string>
</array>
```

Linux (`/etc/systemd/system/openclawbrain-daemon.service`):

```ini
[Service]
ExecStart=/usr/bin/python3 -m openclawbrain.socket_server --state /home/YOUR_USER/.openclawbrain/main/state.json
```

```bash
python3 -m openclawbrain.socket_client --socket ~/.openclawbrain/main/daemon.sock --method health --params "{}"
```

**OpenClawBrain learns from your agent feedback, so wrong answers get suppressed instead of resurfacing.** It builds a memory graph over your workspace, remembers what worked, and routes future answers through learned paths.

- Zero dependencies. Pure Python 3.10+.
- Built-in hash embeddings for offline/testing; OpenAI embeddings are recommended for production.
- Builds a **`state.json`** brain from your workspace.
- Queries follow learned routes instead of only similarity matches.
- Positive feedback (`+1`) uses the default policy-gradient learner `apply_outcome_pg()` (conserving probability mass across traversed nodes), while negative (`-1`) creates inhibitory edges.
- Over time, less noise appears and recurring mistakes are less likely.

- OpenClawBrain integrates with your agent's file-based workspace through incremental sync, constitutional anchors, and automatic compaction.
- See the [context lifecycle](docs/architecture.md) for details.

## Install

```bash
pip install openclawbrain
```

See also: [Setup Guide](docs/setup-guide.md) for a complete local configuration walkthrough.

## Why OpenClawBrain

- Static retrieval vs learned routing: OpenClawBrain continuously updates node-to-node edges so good routes strengthen and bad routes decay.
- No correction propagation vs inhibitory edges: incorrect context can be actively suppressed and forgotten less often than in similarity-only systems.
- Bulk context load vs targeted traversal: context windows stay focused (roughly 52KB → 3-13KB in typical sessions) by following likely retrieval routes.
- No structural maintenance vs prune/merge/compact: OpenClawBrain includes scheduled maintenance commands to keep the graph healthy and compact.
- No protection vs constitutional anchors: anchor critical nodes with authority so operational instructions do not drift.

## 5-minute quickstart (A→B learning story)

```bash
# 1. Build a brain from the sample workspace
openclawbrain init --workspace examples/sample_workspace --output /tmp/brain
Large texts are automatically rechunked to stay under embedding model limits (12K chars). No content is skipped or truncated.

# 2. Check state health
openclawbrain doctor --state /tmp/brain/state.json
# output
# PASS: python_version
# PASS: state_file_exists
# PASS: state_json_valid
# Summary: 8/9 checks passed

# 3. Query (text output includes node IDs)
openclawbrain query "how do I deploy" --state /tmp/brain/state.json --top 3 --json
# output (abbrev.)
# {"fired": ["deploy.md::0", "deploy.md::1", "deploy.md::2"], ...}

# 4. Teach it (good path)
openclawbrain learn --state /tmp/brain/state.json --outcome 1.0 --fired-ids "deploy.md::0,deploy.md::1"
# output
# {"edges_updated": 2, "max_weight_delta": 0.155}
#
# `learn` defaults to `apply_outcome_pg()` for full-policy updates.
# `apply_outcome()` remains available for simpler sparse updates.

# 5. Inject a correction
openclawbrain inject --state /tmp/brain/state.json \
  --id "fix::1" --content "Never skip CI for hotfixes" --type CORRECTION

# 5b. Add new knowledge (no correction needed, just a new fact)
openclawbrain inject --state /tmp/brain/state.json \
  --id "teaching::monitoring-tip" \
  --content "Check Grafana dashboards before every deploy" \
  --type TEACHING

# 6. Query again and see the route change
openclawbrain query "can I skip CI" --state /tmp/brain/state.json --top 3
# output
# fix::1
# ~~~~~~
# Never skip CI for hotfixes
# ...

# 7. Re-check health for a quick signal
openclawbrain health --state /tmp/brain/state.json
```

## Correcting mistakes (the main workflow)

When your agent retrieves wrong context, teach OpenClawBrain in one command:

```bash
openclawbrain inject --state brain/state.json \
  --id "correction::42" \
  --content "Never show API keys in chat messages" \
  --type CORRECTION
```

What happens:
1. OpenClawBrain creates a new node with your correction text
2. It connects that node to the most related workspace chunks
3. It adds **inhibitory edges** — negative-weight links that suppress those chunks
4. Next query touching that topic: the correction appears, the bad route is dampened

To add knowledge without suppressing anything, use `--type TEACHING` instead.

### Structural corrections with split

Alongside inhibitory edges and periodic merge, maintenance now supports runtime splitting:

- `openclawbrain maintain` runs `scale` after `decay`, then `split` before `merge`.
- `suggest_splits()` finds bloated multi-topic nodes (including merged-byline nodes).
- `split_node()` rewires outgoing and incoming edges into focused child nodes, then removes the parent.
- Inhibitory edges are always copied to every child, so suppressions are not lost.
- `openclawbrain maintain` now also includes optional homeostatic controls: decay half-life auto-adjusts to keep reflex-edge ratio in range, and synaptic scaling uses a soft per-node weight budget (5.0) with fourth-root scaling.
- `Tier hysteresis: habitual band 0.15-0.6` prevents threshold thrashing.

## Self-learning (autonomous agent learning)

Agents can learn from their own observations — both mistakes and successes — without human feedback (self-correct available as CLI/API alias).

```bash
# Agent detected a failure — penalize the bad path and inject a correction
openclawbrain self-learn --state brain/state.json \
  --content 'Always download model artifacts before terminating training instances' \
  --fired-ids 'infra.md::3,cleanup.md::1' \
  --outcome -1.0 --type CORRECTION

# Agent succeeded — reinforce the good path and record what worked
openclawbrain self-learn --state brain/state.json \
  --content 'Download-then-terminate sequence works reliably for model training' \
  --fired-ids 'infra.md::3,download.md::1' \
  --outcome 1.0 --type TEACHING

# Agent learned something new (neutral — just adding knowledge)
openclawbrain self-learn --state brain/state.json \
  --content 'GBM training takes ~40 min on g5.xlarge' \
  --outcome 0 --type TEACHING
```

The full spectrum:

| Situation | outcome | type | Effect |
|-----------|---------|------|--------|
| Agent made a mistake | -1.0 | CORRECTION | Penalize fired path + inject with inhibitory edges |
| Agent learned a fact | 0.0 | TEACHING | Inject knowledge only, no weight changes |
| Agent succeeded | +1.0 | TEACHING | Reinforce fired path + inject positive knowledge |

Via socket (Python):

```python
from openclawbrain.socket_client import OCBClient

with OCBClient('~/.openclawbrain/main/daemon.sock') as client:
    # Agent detected its own mistake
    client.self_learn(
        content='Always download artifacts before terminating instances',
        fired_ids=['infra.md::3', 'cleanup.md::1'],
        outcome=-1.0,
        node_type='CORRECTION',
    )

    # Agent observed a success — reinforce
    client.self_learn(
        content='Chunked download with checksum verification works reliably',
        fired_ids=['download.md::0', 'validate.md::1'],
        outcome=1.0,
        node_type='TEACHING',
    )
```

This enables autonomous learning loops: agents observe outcomes, detect failures and successes, and teach themselves — no human in the loop. `self-correct` is available as CLI/API alias.

## Adding new knowledge (no rebuild needed)

When you learn something that isn't in any workspace file, inject it directly:

```bash
openclawbrain inject --state brain/state.json \
  --id "teaching::codex-spark" \
  --content "Use Codex CLI with gpt-5.3-codex-spark for coding tasks — free on Pro plan" \
  --type TEACHING
```

TEACHING nodes connect to related workspace chunks just like CORRECTION nodes,
but without inhibitory edges — they add knowledge instead of suppressing it.

Three injection types:
- **CORRECTION** — creates inhibitory edges that suppress related wrong paths
- **TEACHING** — adds knowledge with normal positive connections
- **DIRECTIVE** — same as TEACHING (use for standing instructions)

For agent frameworks that need to correlate corrections with earlier queries,
see `examples/correction_flow/` for the fired-node logging pattern.

You can also reinforce good retrievals:

```bash
# After a query returns helpful context, strengthen those paths
openclawbrain learn --state brain/state.json --outcome 1.0 \
  --fired-ids "deploy.md::0,deploy.md::1"
```

Or weaken bad ones:

```bash
openclawbrain learn --state brain/state.json --outcome -1.0 \
  --fired-ids "monitoring.md::2"
```

## What it looks like in practice

```bash
# Before learning
openclawbrain query "how do we handle incidents" --state /tmp/brain/state.json --top 3

# After one good learn on the best route
openclawbrain learn --state /tmp/brain/state.json --outcome 1.0 --fired-ids "incidents.md::0,deploy.md::1"

# After one negative learn on a bad route
openclawbrain learn --state /tmp/brain/state.json --outcome -1.0 --fired-ids "monitoring.md::2,incidents.md::0"

# Query again to observe new routing
openclawbrain query "incident runbook for deploy failures" --state /tmp/brain/state.json --top 4
```

## How it compares

| | Plain RAG | OpenClawBrain |
|---|-----------|----------|
| Retrieval | Similarity search | Learned graph traversal |
| Feedback | None | `learn +1/-1` updates edge weights |
| Wrong answers | Can keep resurfacing | Inhibitory edges suppress them |
| Adding knowledge | Re-index/re-embed | `inject --type TEACHING` (no rebuild) |
| Over time | Same results for same query | Routes become habitual behavior |
| Dependencies | Vector DB or service | Zero dependencies |

## How OpenClawBrain differs from related tools

| | OpenClawBrain | Plain RAG | Reflexion | MemGPT |
|---|----------|-----------|-----------|--------|
| What it learns | Retrieval routes | Nothing | Reasoning via self-reflection text | Memory read/write policies |
| Negative feedback | Inhibitory edges suppress bad paths | None | None (additive only) | None |
| New knowledge | `inject` node (no rebuild) | Re-embed corpus | Add to reflection prompt | Update tier config |
| Integration | Standalone library, any agent | Vector DB required | Tied to agent loop | Tied to agent architecture |
| Cold start | Hash embeddings, no API key | Needs embedding service | Needs prior episodes | Needs configured tiers |
| State | Single `state.json` file | External DB | Prompt history | Multi-tier storage |

## Real embeddings + LLM routing (OpenAI)

Production deployments use:

- **Embeddings:** `text-embedding-3-small` (1536-dim)
- **LLM routing/scoring:** `gpt-5-mini`
- **Offline/testing fallback:** `hash` embeddings (lower quality, no API key required).

```python
from openai import OpenAI
from openclawbrain import split_workspace, VectorIndex

client = OpenAI()


def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small", input=[text]
    ).data[0].embedding


def llm(system, user):
    return client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    ).choices[0].message.content

graph, texts = split_workspace("./workspace", llm_fn=llm)
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, embed(content))
```

See `examples/openai_embedder/` for a complete example.

## CLI Reference

| Command | Description |
|---------|-------------|
| `init` | Build a brain from workspace files |
| `query` | Traverse graph and return context |
| `learn` | Apply outcome feedback to fired edges |
| `self-learn` | Add outcome-aware lesson entries from agent observations |
| `self-correct` | Alias for `self-learn` |
| `merge` | Suggest and apply node merges |
| `anchor` | Set/list/remove constitutional authority on nodes |
| `connect` | Connect learning nodes to workspace neighborhoods |
| `maintain` | Run structural maintenance (health, decay, scale, split, merge, prune, connect) |
| `compact` | Compact old daily notes into graph nodes |
| `sync` | Incremental re-embed after file changes |
| `inject` | Add CORRECTION/TEACHING/DIRECTIVE nodes |
| `replay` | Replay session queries, optionally with `--fast-learning` or `--full-learning` |
| `harvest` | Apply slow-learning pass from `learning_events.jsonl` to current graph |
| `health` | Show graph health metrics |
| `status` | `openclawbrain status --state brain/state.json [--json]` returns a one-command health overview: version, nodes, edges, tier distribution, daemon status, embedder, decay half-life |
| `journal` | Show event journal |
| `doctor` | Run diagnostic checks |
| `info` | Show brain info (nodes, edges, embedder) |
| `daemon` | Start persistent worker (JSON-RPC over stdio, state loaded once) |

## State persistence

State writes are atomic (`temp` + `fsync` + `rename`) with `.bak` backup. Crash-safe.

## Persistent Worker (`openclawbrain daemon`)

For production use, run OpenClawBrain as a long-lived daemon so the graph stays hot in memory and query paths avoid repeated startup+reload overhead.

Why this matters:

- First load initializes `state.json` once, then keeps the process and index warm.
- Saves about 100-800ms per call versus shelling out per query (production measure: ~504ms per warm query path on Mac Mini M4 Pro).
- Reduces memory churn and tail latency under steady traffic.

Start it with:

```bash
openclawbrain daemon --state ~/.openclawbrain/main/state.json
```

Protocol:

- Transport: `stdin`/`stdout` with newline-delimited JSON (NDJSON).
- Each request is a single JSON object with `id`, `method`, and `params`.
- Each response is a single JSON object with the same `id` and either `result` or `error`.

Example request/response:

```bash
echo '{"id":"req-1","method":"query","params":{"query":"how to deploy","top_k":4,"chat_id":"telegram:123"}}' | openclawbrain daemon --state ~/.openclawbrain/main/state.json
```

```json
{"id":"req-1","result":{"fired_nodes":["a"],"context":"...","seeds":[["a",0.96]],"embed_query_ms":1.1,"traverse_ms":2.4,"total_ms":3.5}}
```

Supported methods (all 10):

- `query`: run route traversal and return `fired_nodes`, `context`, timing, and seeds.
- `learn`: apply outcomes (`+1/-1`) with default `apply_outcome_pg()` updates and return `edges_updated`.
- `inject`: add TEACHING/CORRECTION/DIRECTIVE nodes and connect them to related workspace chunks.
- `correction`: atomically apply negative feedback to last-fired nodes and inject a CORRECTION node.
- `maintain`: run maintenance ops and return health/merge summary fields.
- `health`: return current health metrics for the loaded graph.
- `info`: return state metadata and object counts.
- `save`: persist current in-memory state to disk immediately.
- `reload`: reload `state.json` without restarting.
- `shutdown`: persist pending writes and exit cleanly.
- `query`/`learn`/`maintain`/`health`/`info` responses include `embed_query_ms`, `traverse_ms`, and `total_ms` timing fields where applicable.
- `query`/`learn`/`inject`/`correction` are the only mutation-capable methods; the daemon is the single source of truth for those changes while state is hot in memory.

Current limitations:

- Per-chat mutation APIs remain scoped through request payloads (`chat_id`) and adapter-layer bookkeeping.
- Concurrent writers are serialized by the socket transport lock and one active request at a time.

Production timing (Mac Mini M4 Pro, OpenAI embeddings):
- MAIN (1,158 nodes): 397ms embed + 107ms traverse = **504ms total**
- PELICAN (582 nodes): 634ms embed + 51ms traverse = **685ms total**
- BOUNTIFUL (285 nodes): 404ms embed + 27ms traverse = **431ms total**

See `examples/ops/client_example.py` for a Python client and `docs/architecture.md` for protocol details.

## True Policy Gradient (apply_outcome_pg)

`apply_outcome_pg` implements a full REINFORCE policy-gradient update and is now the default learning rule used by daemon/CLI correction and learn paths.

- It updates **all outgoing edges** for each visited node on the fired trajectory, not only traversed edges.
- It uses the update:
  `Δw = (η(z-b)γ^ℓ)/τ · (𝟙[j=a] - π(j|i))`
  where:
  - `η` = learning rate
  - `z` = outcome reward
  - `b` = baseline
  - `γ` = discount
  - `ℓ` = trajectory depth
  - `τ` = temperature
  - `π(j|i)` = action probability from softmax (including STOP)
  - `𝟙[j=a]` = 1 for the taken action, else 0
- Conservation property: for each source node `i`, outgoing updates sum to zero, so total outgoing mass is preserved.
- Use `apply_outcome_pg` when you want smoother, probability-based updates across alternatives; use `apply_outcome` for a simpler sparse update that only touches traversed edges.

```python
from openclawbrain import apply_outcome_pg, LearningConfig

config = LearningConfig(learning_rate=0.1, temperature=1.0, baseline=0.0)
updates = apply_outcome_pg(graph, fired_nodes=["a", "b", "c"], outcome=1.0, config=config)
```

Full derivation: https://jonathangu.com/openclawbrain/gu2016/

## Write policy summary

| Situation | Action |
|-----------|--------|
| Durable fact | Edit file → sync re-embeds |
| Correction | Edit file + daemon `correction` method |
| Soft teaching | openclawbrain inject --type TEACHING |
| Wrong retrieval | daemon `correction` (graph-only, no rebuild) |
| New rule | Edit AGENTS.md or SOUL.md |

## Production stats (current)

- MAIN: 1,160 nodes, 2,551 edges, 43 learnings
- PELICAN: 555 nodes, 2,211 edges, 181 learnings
- BOUNTIFUL: 289 nodes, 1,101 edges, 35 learnings
- CORMORANT: 1,672 nodes, ~7,100 edges, 22 learnings (first external user!)

## Traversal defaults

| Setting | Default | Purpose |
|---------|---------|---------|
| `beam_width` | `8` | Frontier size per hop (wider = reaches farther routes) |
| `max_hops` | `30` | Safety ceiling; damping controls convergence |
| `fire_threshold` | `0.01` | Minimum score required to fire a candidate node |
| `reflex_threshold` | `0.6` | Edges with weight `>= 0.6` auto-follow (no route function) |
| `habitual_range` | `0.15 - 0.6` | Edges in this band run through route function |
| `inhibitory_threshold` | `-0.01` | Edges at or below suppress targets |
| `max_fired_nodes` | `None` | Hard stop on fired node count |
| `max_context_chars` | `None` | Hard stop on rendered traversal context |
| `edge_damping` | `0.3` | Per-reuse decay (`weight × 0.3^k`) |

```python
from openclawbrain import traverse, TraversalConfig

result = traverse(
    graph,
    seeds,
    config=TraversalConfig(max_context_chars=20000, max_fired_nodes=30),
)
```

`query` and `query_brain.py` honor these budgets and stop as soon as any termination condition is met.

## External benchmarks

External retrieval benchmarks are optional and use separately downloaded datasets.
OpenClawBrain ships a quick-start workflow for MultiHop-RAG and HotPotQA in
`benchmarks/external/README.md`, but the datasets are not in the repository.

Quick run (from project root):

```bash
mkdir -p benchmarks/external
curl -L https://huggingface.co/datasets/yixuantt/MultiHopRAG/raw/main/MultiHopRAG.json -o benchmarks/external/multihop_rag.json
curl -L https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -o benchmarks/external/hotpotqa_dev_distractor.json
python3 benchmarks/external/run_multihop_rag.py --limit 50
python3 benchmarks/external/run_hotpotqa.py --limit 50
```

## Python API

```python
from openclawbrain import (
    split_workspace,
    traverse,
    apply_outcome_pg,
    apply_outcome,
    inject_node,
    inject_correction,
    inject_batch,
    VectorIndex,
    HashEmbedder,
    TraversalConfig,
    save_state,
    load_state,
    ManagedState,
    measure_health,
    suggest_splits,
    split_node,
    replay_queries,
    score_retrieval,
)
```

## State lifecycle

- **Where it lives:** a single `state.json` file (portable, version-controllable)
- **How big:** ~180KB for 20 nodes (hash), ~60MB for 1,600 nodes (OpenAI embeddings)
- **When to rebuild:** after major workspace restructuring or embedder changes
- **Embedder changes:** OpenClawBrain stores the embedder name + dimension in state metadata and hard-fails on mismatch — no silent corruption
- **Maintenance:** use `openclawbrain maintain` (`decay` + `scale` + `split` + `merge` + `prune` + `connect`) to rebalance structure as the graph evolves

## Cost control

- **Recommended:** OpenAI `text-embedding-3-small` (~$0.02/MB) + `gpt-5-mini` for routing/scoring. Embeddings are generated at init and cached in `state.json`; `gpt-5-mini` runs on query only.
- **Offline/testing fallback:** hash embeddings with no API calls. Fast for local tryout, but lower retrieval quality than OpenAI.
- **Batch init:** `openclawbrain init` embeds all workspace files in one batch call. Subsequent queries reuse cached vectors.
- **Upgrade path:** when moving to production, rebuild with OpenAI embeddings via `openclawbrain init` and stop relying on the hash fallback.

## Optional: warm start from sessions

If you have prior conversation logs, replay them:

```bash
openclawbrain replay --state /tmp/brain/state.json --sessions ./sessions/
```

For transcript-backed fast-learning:

```bash
openclawbrain replay \
  --state /tmp/brain/state.json \
  --sessions ./sessions/ \
  --fast-learning \
  --resume \
  --workers 4 \
  --checkpoint /tmp/brain/replay_checkpoint.json
```

For the full learning pipeline (replay + fast-learning + harvest):

```bash
openclawbrain replay \
  --state /tmp/brain/state.json \
  --sessions ./sessions/ \
  --full-learning
```

The fast-learning and harvest pipeline is sidecar-only to the core files:
`learning_events.jsonl` is append-only, and `replay` updates `state.json` via the same graph mutation model as existing injection commands.

Skip this if you are just getting started.

## Production experience

Three brains run in production on a Mac Mini M4 Pro:

| Brain | Nodes | Edges | Learning Corrections | Sessions Replayed |
|-------|-------|-------|---------------------|-------------------|
| MAIN | 1,142 | 2,814 | 43 | 215 |
| PELICAN | 512 | 1,984 | 181 | 183 |
| BOUNTIFUL | 273 | 1,073 | 35 | 300 |

## Design Tenets

- No network calls in core.
- No secret discovery (no dotfiles, no keychain lookup).
- Embedder identity stored in state metadata; hard-fail on dimension mismatch.
- One canonical state format (`state.json`).
- Traversal defaults are budget-first for safety: `beam_width=8`, `max_hops=30`, `fire_threshold=0.01`.

## Paper + links

[jonathangu.com/openclawbrain](https://jonathangu.com/openclawbrain/) — 8 deterministic simulations + production deployment data.

- PyPI: `pip install openclawbrain`
- GitHub: [jonathangu/openclawbrain](https://github.com/jonathangu/openclawbrain)
- ClawHub: `clawhub install openclawbrain`
- Benchmarks: `python3 benchmarks/run_benchmark.py` (deterministic per commit; timings vary by machine)
