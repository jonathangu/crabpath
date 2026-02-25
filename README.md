# 🦀 CrabPath: The Graph is the Prompt

**LLM-guided memory traversal with learned pointer weights and corrected policy gradients.**

📄 **[Read the paper →](https://jonathangu.com/crabpath/)**

CrabPath is a memory architecture for AI agents where documents are nodes, weighted pointers are edges, and an LLM is the activation function. The graph learns which paths lead to good outcomes using trajectory-aware credit assignment (Gu, 2016), compiling expensive LLM reasoning into cheap reflexive routing over time.

## Install

```bash
pip install crabpath   # or: pip install .
```

Zero dependencies. Pure Python.

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
