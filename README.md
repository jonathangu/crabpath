# 🦀 CrabPath

**Neuron-inspired memory graphs for AI agents. Everything evolves into this.**

---

In biology, [carcinisation](https://en.wikipedia.org/wiki/Carcinisation) is the phenomenon where crustaceans independently evolve into crab-like forms — over and over. Nature keeps reinventing the crab because it *works*.

CrabPath is a bet that agent memory will converge the same way.

## The Model

A node is a neuron. That's the whole idea.

Each node has **content** (what it knows), a **potential** (accumulated energy), a **threshold** (when to fire), and a **trace** (how recently it was active). Edges are weighted pointers — positive for excitation, negative for inhibition.

When you query the graph:
1. **Traces decay** (time has passed since last query)
2. **Seed nodes** receive energy
3. Nodes whose potential crosses their **threshold** → **fire**
4. Firing sends **weighted energy** along outgoing edges
   - Positive weight → adds energy to target (excitatory)
   - Negative weight → removes energy from target (inhibitory)
5. Fired nodes **reset** (refractory) and refresh their **trace**
6. Unfired potentials **decay** each step (leak)
7. Repeat until nothing fires or max steps reached

Learning is **STDP-aware** (spike-timing-dependent plasticity): edges in the causal direction (source fired *before* target) get more credit than anti-causal edges. This is how sequences get encoded into the graph — not just co-occurrence, but *order*.

**Zero dependencies.** Pure Python. The whole library is two files.

## Install

```bash
pip install crabpath
```

## Quick Start

```python
from crabpath import Graph, Node, Edge, activate, learn

g = Graph()

# Nodes are neurons: content + threshold
g.add_node(Node(id="check-logs", content="tail -n 200 /var/log/service.log"))
g.add_node(Node(id="check-config", content="git diff HEAD~1 -- config/"))
g.add_node(Node(id="no-untested-fix", content="Never claim fixed without testing"))
g.add_node(Node(id="claim-fixed", content="Tell user it's fixed", threshold=2.0))

# Positive edges: these go together
g.add_edge(Edge(source="check-config", target="check-logs", weight=1.5))

# Negative edges: this blocks that
g.add_edge(Edge(source="no-untested-fix", target="claim-fixed", weight=-1.0))

# Fire
result = activate(g, seeds={"check-config": 1.0, "no-untested-fix": 1.0})

for node, energy in result.fired:
    print(f"  [{energy:.2f}] {node.content}")

# Timing info: which step each node fired at
print(f"  Timing: {result.fired_at}")
print(f"  Inhibited: {result.inhibited}")

# Learn from outcome (STDP: causal edges get more credit)
learn(g, result, outcome=1.0)   # success → strengthen causal paths
learn(g, result, outcome=-1.0)  # failure → weaken them

# Check what's warm
for node, trace in g.warm_nodes():
    print(f"  Warm: {node.id} (trace={trace:.2f})")

# Persist
g.save("memory.json")
g2 = Graph.load("memory.json")
```

## Persistent State (Energy Lingers)

By default, each `activate()` call starts fresh. But with `reset=False`, energy carries over between calls — related queries build on each other:

```python
# First query: "deployment" context
r1 = activate(g, seeds={"deploy": 1.0}, reset=True)

# Second query (related): energy from first call lingers
r2 = activate(g, seeds={"error": 0.5}, reset=False)
# deploy-related nodes are still warm → lower bar to fire
```

## API

### Node

```python
Node(
    id="...",           # unique identifier
    content="...",      # what this neuron knows
    threshold=1.0,      # fires when potential >= threshold
    potential=0.0,      # current energy (transient)
    trace=0.0,          # decaying record of recent firing
    metadata={},        # your bag of whatever (types, tags, timestamps — your call)
)
```

### Edge

```python
Edge(
    source="a",         # from node
    target="b",         # to node
    weight=1.0,         # positive = excitatory, negative = inhibitory
)
```

### Graph

```python
g = Graph()

g.add_node(node)
g.get_node("id")            # Node or None
g.remove_node("id")         # removes node + connected edges
g.nodes()                   # all nodes

g.add_edge(edge)
g.get_edge("a", "b")        # Edge or None
g.outgoing("a")             # [(target_node, edge), ...]
g.incoming("b")             # [(source_node, edge), ...]
g.edges()                   # all edges

g.reset_potentials()         # set all potentials to 0
g.warm_nodes()               # recently-fired nodes, sorted by trace
g.save("path.json")
Graph.load("path.json")
```

### Activation

```python
result = activate(
    graph,
    seeds={"node-a": 1.0, "node-b": 0.5},
    max_steps=3,       # propagation rounds
    decay=0.1,         # potential leak per step
    top_k=10,          # max nodes to return
    reset=True,        # False to keep energy between calls
    trace_decay=0.1,   # how fast traces fade between calls
)

result.fired       # [(Node, energy_at_firing), ...] sorted descending
result.inhibited   # [node_id, ...] driven below 0
result.steps       # propagation rounds completed
result.fired_at    # {node_id: step} — when each node fired
```

### Learning

```python
learn(graph, result, outcome=1.0, rate=0.1)
# STDP-aware:
#   causal edges (src fired before tgt) → strengthen on success
#   anti-causal edges (tgt fired before src) → weaken on success
#   credit decays with temporal distance
# weights clamped to [-10, 10]
```

## Design Principles

1. **Minimum assumptions.** Nodes have `id`, `content`, `threshold`, `potential`, `trace`, `metadata`. No type system, no tags, no timestamps — put what you want in metadata.
2. **Zero dependencies.** Pure Python. Plain dicts internally. No NetworkX, no numpy, no nothing.
3. **Neuron-faithful.** Leaky integrate-and-fire: accumulate, threshold, fire, propagate, refractory, decay, trace. STDP-aware learning encodes sequences, not just co-occurrence.
4. **Persistence is JSON.** Save/load snapshots. Defaults are omitted for compact files.

## Why "CrabPath"?

🦀 Everything evolves into a crab. We think everything in agent memory evolves into this: weighted graphs, neuron-style activation, inhibition, outcome learning. CrabPath is the path everything walks.

## The Paper

📄 **[jonathangu.com/crabpath](https://jonathangu.com/crabpath/)** — full architecture, biological inspiration, math, and experimental plan.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{gu2026crabpath,
  title={CrabPath: Neuron-Inspired Memory Graphs for AI Agents},
  author={Gu, Jonathan},
  year={2026},
  url={https://github.com/jonathangu/crabpath}
}
```

---

*Built by [Jonathan Gu](https://jonathangu.com)* 🦀
