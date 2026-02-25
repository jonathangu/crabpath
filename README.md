# 🦀 CrabPath

**Agent memory that learns what to load and what to skip.**

Your agent reloads the same files every session regardless of the task. CrabPath replaces that with a graph that learns from outcomes: nodes that fire during success get stronger connections, nodes that fire during failure get weaker ones. Over time, the right context loads automatically.

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
