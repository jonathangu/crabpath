# 🦀 CrabPath: The Graph is the Prompt

**LLM-guided memory traversal with learned pointer weights and corrected policy gradients.**

📄 **[Read the paper →](https://jonathangu.com/crabpath/)**

CrabPath is a memory architecture for AI agents where documents are nodes, weighted pointers are edges, and an LLM is the activation function. The graph learns which paths lead to good outcomes using trajectory-aware credit assignment (Gu, 2016), compiling expensive LLM reasoning into cheap reflexive routing over time.

## Requirements

- Python 3.10+
- Zero dependencies (stdlib only)
- Optional: semantic embeddings via providers (`auto_embed()` by default)
- Supported providers:
  - `openai_embed`: `OPENAI_API_KEY` (`text-embedding-3-small`)
  - `gemini_embed`: `GEMINI_API_KEY` or `GOOGLE_API_KEY` (`text-embedding-004`)
  - `cohere_embed`: `COHERE_API_KEY` (`embed-v4`)
  - `ollama_embed`: no API key required (`nomic-embed-text`, local Ollama)

`auto_embed()` is the default recommendation. It tries providers in order:
OpenAI → Gemini → Ollama.

`ollama_embed` is free/local and requires no API key.

## Install

```bash
pip install crabpath   # or: pip install .
```

`auto_embed()` is optional but recommended. Without any configured provider,
CrabPath falls back to keyword-based routing. With one configured, semantic
embeddings use the selected provider for substantially better retrieval quality.

## The Loop

Every agent integration follows the same pattern:

```
1. Load the graph         g = Graph.load("memory.json")
2. Seed relevant nodes    seeds = {"deploy": 1.0, "error": 0.5}
3. Activate               result = activate(g, seeds)
4. Use fired nodes        context = [n.content for n, _ in result.fired]
5. Learn from outcome     learn(g, result, outcome=1.0)
6. Save                   g.save("memory.json")
```

That's the whole integration. See [`examples/agent_memory.py`](examples/agent_memory.py) for a working version.

## CLI

CrabPath ships with a JSON-only CLI for agent runtimes.
All commands emit one-line JSON on `stdout` and JSON errors on `stderr`.
Commands that use embeddings (`query`, `migrate`, `split`) use keyword fallback
when no provider can be selected.

```bash
python -m crabpath.cli query "deploy broke after config change" --graph crabpath_graph.json --index crabpath_embeddings.json --top 12
```

```json
{"fired":[{"id":"...","content":"...","energy":0.85}], "inhibited":["..."], "guardrails":["..."]}
```

```bash
python -m crabpath.cli learn --graph crabpath_graph.json --outcome 1.0 --fired-ids node1,node2,node3
```

```json
{"ok":true,"edges_updated":5}
```

```bash
python -m crabpath.cli snapshot --graph crabpath_graph.json --session sess-123 --turn 42 --fired-ids node1,node2
```

```json
{"ok":true,"snapshot_path":"crabpath_graph.events.db"}
```

```bash
python -m crabpath.cli feedback --session sess-123 --turn-window 5
```

```json
{"turn_id":42,"fired_ids":["..."],"turns_since_fire":3,"suggested_outcome":-1.0}
```

```bash
python -m crabpath.cli stats --graph crabpath_graph.json
```

```json
{"nodes":153,"edges":143,"avg_weight":0.72,"top_hubs":["..."]}
```

```bash
python -m crabpath.cli consolidate --graph crabpath_graph.json --min-weight 0.05
```

```json
{"ok":true,"pruned_edges":12,"pruned_nodes":3}
```

```bash
crabpath migrate --workspace ~/.openclaw/workspace --session-logs session.jsonl --output-graph graph.json --output-embeddings embed.json
```

```json
{"ok":true,"graph_path":"graph.json","embeddings_path":"embed.json","info":{...}}
```

```bash
crabpath split --graph graph.json --node-id tools --save
```

```json
{"ok":true,"action":"split","node_id":"tools","chunk_ids":["tools:chunk-1","tools:chunk-2"],"chunk_count":2}
```

```bash
crabpath sim --queries 200 --output results.json
```

```json
{"ok":true,"queries":200,"result":{"final":{"nodes":1234}}}
```

## MCP Server

`crabpath/mcp_server.py` runs an stdio MCP server:

```bash
python -m crabpath.mcp_server
```

Supported tools:
- `query`
- `migrate`
- `learn`
- `stats`
- `split`
- `add`
- `remove`
- `consolidate`

## OpenAI Tools

`tools/openai-tools.json` contains OpenAI function-calling definitions for agent integrations.

```python
import json
from pathlib import Path

openai_tools = json.loads(Path("tools/openai-tools.json").read_text())["tools"]
print([tool["function"]["name"] for tool in openai_tools])
```

## OpenAPI

`tools/openapi.yaml` documents the REST API.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"deploy","graph":"graph.json","top":8}'
```

```bash
curl -X POST http://localhost:8000/sim \
  -H "Content-Type: application/json" \
  -d '{"queries":200,"output":"results.json"}'
```

## OpenClaw Integration

Phase 1 adds `OpenClawCrabPathAdapter`, a small wrapper for
OpenClaw sessions with delayed outcome attribution.

```python
from crabpath import (
    Graph,
    EmbeddingIndex,
    OpenClawCrabPathAdapter,
    map_correction_to_snapshot,
    auto_outcome,
)

adapter = OpenClawCrabPathAdapter(
    graph_path="crabpath_graph.json",
    index_path="crabpath_embeddings.json",
    embed_fn=lambda texts: [[0.1] for _ in texts],  # your embedding function
)
graph, index = adapter.load()

seeds = adapter.seed(
    query_text="deploy broke after config change",
    memory_search_ids=["memory-hit-12", "memory-hit-19"],
)
firing = adapter.activate(seeds, max_steps=3, decay=0.1, top_k=12)
ctx = adapter.context(firing)
llm_context = ctx["contents"]
guardrails = ctx["guardrails"]

adapter.learn(firing, outcome=1.0)   # or outcome=-1.0
adapter.snapshot(session_id="session-1", turn_id=42, firing_result=firing)

adapter.save()

snapshot = map_correction_to_snapshot(session_id="session-1", turn_window=5)
if snapshot is not None:
    turns_since_fire = snapshot["turns_since_fire"]
    inferred = auto_outcome(corrections_count=0, turns_since_fire=turns_since_fire)
    # feed inferred into adapter.learn(firing...) in your worker loop
```

## Getting Started

Quick start for existing AI agent workspaces with workspace files:

```bash
python3 scripts/bootstrap_from_workspace.py /path/to/workspace --output /path/to/crabpath_graph.json
```

After bootstrap, load and run the same graph workflow you already use:

```python
from crabpath import Graph, Node, Edge

graph = Graph.load("/path/to/crabpath_graph.json")
print(f"nodes={graph.node_count}, edges={graph.edge_count}")
```

## Migration & Replay

### Migration command example

```bash
export OPENAI_API_KEY=sk-...
crabpath migrate --workspace ~/.openclaw/workspace --session-logs session.jsonl --output-graph graph.json --output-embeddings embed.json
```

Developer flow for pre-warming:

1. Install `crabpath`.
2. Point at workspace files (`~/.openclaw/workspace`).
3. Optionally pass session logs.
4. Receive a pre-warmed graph immediately.

```python
from pathlib import Path
from crabpath.migrate import migrate

graph, info = migrate(
    workspace_dir=Path("~/.openclaw/workspace").expanduser(),
    session_logs=["session.jsonl"],  # optional
)
graph.save("graph.json")
print(info["nodes"], info["edges"])
```

```bash
export OPENAI_API_KEY=sk-...
crabpath migrate --workspace ~/.openclaw/workspace --session-logs session.jsonl --output-graph graph.json --output-embeddings embed.json
```

## Quick Start

```python
from crabpath import Graph, Node, Edge, activate, learn

g = Graph()

# Nodes are neurons: content + threshold
g.add_node(Node(id="check-config", content="git diff HEAD~1 -- config/"))
g.add_node(Node(id="check-logs", content="tail -n 200 /var/log/svc.log"))
g.add_node(Node(id="no-untested", content="Never claim fixed without testing", threshold=0.5))
g.add_node(Node(id="claim-fixed", content="Tell user it's fixed", threshold=2.0))

# Edges: positive = excitatory, negative = inhibitory
g.add_edge(Edge(source="check-config", target="check-logs", weight=1.5))
g.add_edge(Edge(source="no-untested", target="claim-fixed", weight=-1.0))

# Activate
result = activate(g, seeds={"check-config": 1.0, "no-untested": 1.0})

for node, energy in result.fired:
    print(f"  [{energy:.2f}] {node.content}")
print(f"  Inhibited: {result.inhibited}")  # claim-fixed blocked
print(f"  Timing: {result.fired_at}")      # which step each node fired

# Learn: STDP strengthens causal edges (src fired before tgt)
learn(g, result, outcome=1.0)

# Save/load
g.save("memory.json")
g = Graph.load("memory.json")
```

## Synaptogenesis

Synaptogenesis is the edge-formation layer in CrabPath:

- **Proto-edges** are provisional links created when nodes co-fire before promotion.
- **Promotion** turns qualified proto-edges into real edges, with initial causal or co-fire weight.
- **Hebbian reinforcement** strengthens existing co-fired edges (`+`).
- **Skip penalty** (`×0.9`) weakens edges that are repeatedly bypassed.
- **Tiers** split routing pressure:
  - `dormant` (`< 0.3`, invisible routing)
  - `habitual` (`0.3`–`0.8`, model-assisted routing)
  - `reflex` (`> 0.8`, auto-follow)
- Prolonged inactivity/low credit decays and removes weak structure.

## Autotuner

The autotuner is outcome-driven: it tunes toward good behavior metrics instead of arbitrary knobs.

`HEALTH_TARGETS` defines the target operating range for each metric:

| Metric | Target range |
| --- | --- |
| `avg_nodes_fired_per_query` | 3.0–8.0 |
| `cross_file_edge_pct` | 5.0%–20.0% |
| `dormant_pct` | 60.0%–90.0% |
| `reflex_pct` | 1.0%–5.0% |
| `context_compression` | ≤20.0% |
| `proto_promotion_rate` | 5.0%–15.0% |
| `reconvergence_rate` | ≤10.0% |
| `orphan_nodes` | 0 |

`suggest_config(workspace_files)` warm-starts mitosis/synaptogenesis parameters from workspace size (small / medium / large) so new projects get sensible defaults.

`measure_health(graph, state, query_stats)` computes a `GraphHealth` snapshot;  
`autotune(graph, health)` returns `Adjustment` suggestions (for example, `decay_half_life` or `promotion_threshold`) when a metric drifts out of range.

```python
from pathlib import Path
from crabpath import Graph, suggest_config, measure_health, autotune
from crabpath.mitosis import bootstrap_workspace, MitosisState, MitosisConfig

workspace_files = {
    p.name: p.read_text() for p in Path("~/.openclaw/workspace").expanduser().glob("**/*.py")
}
config = suggest_config(workspace_files)

graph = Graph()
state = MitosisState()
bootstrap_workspace(
    graph,
    workspace_files,
    llm_call=my_llm,
    state=state,
    config=MitosisConfig(**config),
)

query_stats = {
    "avg_nodes_fired_per_query": 6.2,
    "promoted": 9,
    "created": 120,
    "reconvergence_events": 1,
    "context_chars": 2100,
    "queries": 150,
}

health = measure_health(graph, state, query_stats)
for adjustment in autotune(graph, health):
    print(adjustment.metric, adjustment.current, adjustment.target_range, adjustment.suggested_change)
```

## What Emerges

CrabPath’s behavior changes as it interacts. The graph does not only store facts about code; it forms habits, boundaries, and regulatory loops that feel more like cognitive behavior than indexing.

1. **Procedural Memory** — The graph learns sequences, not just facts. `Codex task → worktree reset → safety check` becomes a reflex chain. Multi-hop paths that repeatedly solve problems become auto-follow edges. Like muscle memory.
2. **Domain Separation with Cross-Domain Bridges** — Coding and safety knowledge separate into distinct subgraphs. The split is not isolation: when needed, safety-adjacent links emerge (for example, cleanup can route to safety checks). The graph finds domain boundaries and the bridges that should exist.
3. **Selective Forgetting** — ~95% of sibling edges decay to dormant after about 100 queries. This is deliberate pruning. The graph learns what NOT to load. A model that remembers everything becomes as bad as one that remembers nothing.
4. **Self-Regulation** — The autotuner keeps the graph in a stable operating range. Too much decay makes it back off; promotion that stalls makes it speed up. The graph protects itself from drift and misconfiguration through feedback.
5. **Individuation** — Two agents with identical workspace files but different query patterns converge on different graph shapes. Same genome, different phenotype. Structure is the experience record.

Memory retrieval is a learned skill. CrabPath is where that skill develops automatically from use.

### Health Metrics

Use these empirically calibrated ranges as a compact health check:

| Metric | Target | Why this matters |
| --- | --- | --- |
| `avg_nodes_fired_per_query` | `3-8` | Sparse activation: load only what is needed per query. |
| `cross_file_edge_pct` | `5-20%` | Cross-file discovery: graph finds useful cross-file connections. |
| `dormant_pct` | `60-90%` | Tier differentiation: most edges stay dormant; only useful ones stay active. |
| `reflex_pct` | `1-5%` | Compiled action paths: critical sequences become reflex edges. |
| `context_compression` | `<20%` | Context compression: only a small graph fraction is loaded per query. |
| `proto_promotion_rate` | `5-15%` | Promotion filtering: avoid over/under-learning. |
| `reconvergence_rate` | `<10%` | Split quality: low reconvergence means split clusters remain meaningful. |
| `orphan_nodes` | `0` | Full connectivity: every retained node should stay reachable. |

Self-tuning loop:

`measure` → `diagnose` → `apply` → `evaluate` → `learn`

```python
from crabpath import measure_health
from crabpath.autotune import self_tune, TuneMemory

tune_memory = TuneMemory.load("crabpath_tune_memory.json")
health, adjustments, changes = self_tune(
    graph=graph,
    state=state,
    query_stats=query_stats,
    syn_config=syn_config,
    decay_config=decay_config,
    mitosis_config=mitosis_config,
    cycle_number=cycle,
    tune_memory=tune_memory,
)
```

`autotune` does more than suggest: `self_tune()` applies adjustments directly through `apply_adjustments()`.

`TuneMemory` adds meta-learning:
- It tracks metric/knob/direction triplets over time.
- Successful triples are preferred.
- Failing triples are blocked.
- The graph gradually learns what knobs matter for your workload.

`TuneMemory` output (from `TuneMemory.report()`):

```text
TuneMemory report
Working triples:
- avg_nodes_fired_per_query:decay_half_life:decrease => 3
Preferred triples:
- reflex_pct:hebbian_increment:increase => 5
Blocked triples:
- orphan_nodes:investigation:none => -3
```

Example `crabpath health` CLI output (JSON mode):

```bash
python -m crabpath.cli health \
  --graph crabpath_graph.json \
  --query-stats stats/last_200_queries.json \
  --json
```

```json
{
  "ok": true,
  "graph": "crabpath_graph.json",
  "query_stats_provided": true,
  "metrics": [
    {"metric":"avg_nodes_fired_per_query","value":6.2,"target_range":[3.0,8.0],"status":"✅"},
    {"metric":"cross_file_edge_pct","value":11.8,"target_range":[5.0,20.0],"status":"✅"},
    {"metric":"reconvergence_rate","value":4.3,"target_range":[null,10.0],"status":"✅"},
    {"metric":"orphan_nodes","value":0,"target_range":[0.0,0.0],"status":"✅"}
  ]
}
```

## How Activation Works

Each node has a **potential** (energy) and a **threshold**. When potential ≥ threshold, the node **fires**.

1. Seed nodes receive energy
2. Fired nodes send `weight × energy` to outgoing neighbors
   - Positive weight → excitatory (adds energy)
   - Negative weight → inhibitory (removes energy)
3. Fired nodes reset to 0 (refractory — can't fire twice)
4. Unfired potentials decay each step (leak)
5. Repeat until nothing fires

### Persistent warmth

By default, each `activate()` call starts fresh. With `reset=False`, energy carries over — related queries build on each other:

```python
activate(g, seeds={"deploy": 1.0})                 # deploy nodes warm up
activate(g, seeds={"error": 0.5}, reset=False)      # deploy context still warm
```

### Traces

Nodes remember when they last fired. Check what's warm:

```python
for node, trace in g.warm_nodes():
    print(f"  {node.id}: {trace:.2f}")
```

## How Learning Works

**STDP** (spike-timing-dependent plasticity): edges in the causal direction get more credit.

If A fires at step 0 and B fires at step 1 (A caused B), and the task succeeds:
- Edge A→B: **strengthened** (causal)
- Edge B→A: **weakened** (anti-causal)

This encodes *sequences*, not just co-occurrence. Over many episodes, the graph learns which procedures work in which order.

## Recursive Cell Division (Mitosis)

**The graph finds its own resolution.**

Every file starts as one monolithic node. A cheap LLM splits it into 4 coherent chunks. All chunks get sibling edges at weight 1.0 — they behave as one unit initially. Over time, as the graph activates and learns, edges between chunks that don't co-fire decay. When all sibling edges reconverge to 1.0 (always co-fire), the chunks are functionally monolithic again — the LLM re-splits them, potentially finding different boundaries this time.

**Lifecycle:**
```
workspace file
    ↓
[LLM split] → 4 chunks (sibling edges = 1.0)
    ↓
[activation + decay] → edges diverge (0.3, 0.8, 1.0, 0.5...)
    ↓
[some edges reconverge to 1.0] → always co-fire
    ↓
[merge + re-split] → LLM splits differently
    ↓
repeat...
```

### API

#### Bootstrap from workspace files

```python
from crabpath import Graph
from crabpath.mitosis import bootstrap_workspace, MitosisState, MitosisConfig

def llm_call(system_prompt: str, user_prompt: str) -> str:
    # Your cheap LLM call (e.g., GPT-4o-mini, Claude Haiku)
    return your_llm_client.call(system_prompt, user_prompt)

graph = Graph()
state = MitosisState()
workspace_files = {
    "AGENTS.md": open("AGENTS.md").read(),
    "TOOLS.md": open("TOOLS.md").read(),
    # ... more files
}

results = bootstrap_workspace(
    graph=graph,
    workspace_files=workspace_files,
    llm_call=llm_call,
    state=state,
    config=MitosisConfig(num_chunks=4, sibling_weight=1.0),
)

print(f"Split {len(results)} files into {sum(len(r.chunk_ids) for r in results)} chunks")
graph.save("crabpath_graph.json")
state.save("mitosis_state.json")
```

#### Maintenance (run periodically)

```python
from crabpath.mitosis import mitosis_maintenance

# After every N queries (or every activate() call)
maintenance_result = mitosis_maintenance(
    graph=graph,
    llm_call=llm_call,
    state=state,
)

print(f"Reconverged families: {maintenance_result['reconverged_families']}")
print(f"Re-split: {maintenance_result['resplit_count']}")
```

### Key Functions

- **`bootstrap_workspace(graph, files, llm_call, state)`** — Initialize graph from workspace files. Each file becomes a node, immediately split into 4 chunks by the LLM.
- **`split_node(graph, node_id, llm_call, state)`** — Split a single node into N chunks. Creates sibling edges at weight 1.0.
- **`check_reconvergence(graph, state)`** — Find families where all sibling edges have reconverged (weight ≈ 1.0). Returns parent IDs ready for re-split.
- **`merge_and_resplit(graph, parent_id, llm_call, state)`** — Merge reconverged chunks, then re-split with the LLM.
- **`mitosis_maintenance(graph, llm_call, state)`** — Run reconvergence check + re-split. Call this every N activations.

### Quick Example

```python
from crabpath import Graph, activate, learn
from crabpath.mitosis import bootstrap_workspace, mitosis_maintenance, MitosisState

def my_llm_call(sys, usr):
    # Your cheap LLM
    return llm_client.call(sys, usr)

# Bootstrap
g = Graph()
state = MitosisState()
workspace = {"notes.md": open("notes.md").read()}
bootstrap_workspace(g, workspace, my_llm_call, state)

# Query loop
for query in queries:
    seeds = {"notes.md::chunk-0-abc123": 1.0}  # seed from embedding search
    result = activate(g, seeds, max_steps=3, decay=0.1)
    
    # Use fired nodes as context
    context = [n.content for n, _ in result.fired]
    
    # Learn
    outcome = 1.0 if task_succeeded else -1.0
    learn(g, result, outcome)
    
    # Maintenance (every 10 queries)
    if query_count % 10 == 0:
        mitosis_maintenance(g, my_llm_call, state)

g.save("graph.json")
state.save("mitosis_state.json")
```

## API

### Node

```python
Node(
    id="...",           # unique identifier
    content="...",      # what this neuron knows
    threshold=1.0,      # fires when potential >= threshold
    potential=0.0,      # current energy (transient)
    trace=0.0,          # how recently this node fired
    metadata={},        # yours — types, tags, timestamps, whatever
)
```

### Edge

```python
Edge(source="a", target="b", weight=1.0)
# weight > 0: excitatory, weight < 0: inhibitory
```

### Graph

```python
g = Graph()
g.add_node(node)              # add a neuron
g.get_node("id")              # Node or None
g.remove_node("id")           # removes node + edges
g.add_edge(edge)              # add a connection
g.get_edge("a", "b")          # Edge or None
g.outgoing("a")               # [(node, edge), ...]
g.incoming("b")               # [(node, edge), ...]
g.warm_nodes()                # recently active, by trace
g.reset_potentials()          # zero all potentials
g.save("path.json")           # persist
Graph.load("path.json")       # restore
```

### Activation

```python
result = activate(
    graph, seeds,
    max_steps=3,       # propagation rounds
    decay=0.1,         # leak per step
    top_k=10,          # max returned
    reset=True,        # False for persistent warmth
    trace_decay=0.1,   # trace fade between calls
)
result.fired       # [(Node, energy), ...] sorted descending
result.inhibited   # [node_id, ...] blocked by negative edges
result.steps       # rounds completed
result.fired_at    # {node_id: step} timing info
```

### Learning

```python
learn(graph, result, outcome=1.0, rate=0.1)
# STDP: causal edges strengthened, anti-causal weakened
# weights clamped to [-10, 10]
```

## The Name

[Carcinisation](https://en.wikipedia.org/wiki/Carcinisation): nature keeps independently evolving crabs. We think agent memory keeps converging on this — weighted graphs with learned activation. CrabPath makes it explicit.

## Paper

📄 [jonathangu.com/crabpath](https://jonathangu.com/crabpath/) — motivation, math, experimental plan.

## License

Apache 2.0

---

*[Jonathan Gu](https://jonathangu.com)* 🦀
