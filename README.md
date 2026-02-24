# 🦀 CrabPath

**Activation-driven memory graphs for AI agents. Everything evolves into this.**

---

In biology, [carcinisation](https://en.wikipedia.org/wiki/Carcinisation) is the phenomenon where crustaceans independently evolve into crab-like forms — over and over again. Nature keeps reinventing the crab because the crab body plan *works*.

CrabPath is a bet that agent memory systems will undergo the same convergence. Every serious agent memory system will eventually arrive at: **a weighted directed graph where activation spreads based on relevance, connections learn from outcomes, and frequently-used paths compile into cached reflexes.**

This is that system.

## What CrabPath Does

Instead of loading the same pile of files every session (or doing naive vector search), CrabPath:

1. **Propagates activation** through a learned memory graph when a task arrives
2. **Loads only what lights up** — the handful of facts, procedures, and constraints that matter for *this* task
3. **Learns from outcomes** — successful paths strengthen, failed paths weaken, bad patterns get inhibited
4. **Compiles hot paths** — frequently-successful routes become cached reflexes that cost nearly nothing to run

The result: agent memory that gets faster, cheaper, and more accurate over time.

## Key Concepts

### The Graph

Nodes aren't just text chunks. They're typed:

| Type | What it represents | Example |
|------|-------------------|---------|
| **Fact** | Declarative knowledge | "prod uses env var X" |
| **Rule** | Behavioral constraint | "never claim fixed without testing" |
| **Tool** | Tool affordance + local best practice | "use pty:true for Codex" |
| **Action** | Atomic executable step | `git diff HEAD~1 -- config/` |
| **Sequence** | Multi-step playbook | "debug deploy: config → logs → rollback" |
| **Episode** | Interaction trace with outcome | session log + success/failure label |
| **Error Class** | Reusable failure pattern | "context bloat", "stale cache" |

Edges are typed too: **Association**, **Sequence** (then), **Causation**, **Contingency** (if/else), **Inhibition** (blocks), **Preference**, **Abstraction**.

### Three-Tier Compute

Like biological nervous systems, CrabPath routes queries through three layers:

| Layer | Mechanism | Cost | When |
|-------|-----------|------|------|
| **Reflex** | Cached compiled path | ~$0 | Routine tasks (target: 70%) |
| **Habitual** | Cheap model activation | ~$0.001/query | Pattern matching |
| **Deliberative** | Full model reasoning | ~$0.01-0.80/query | Novel tasks (target: 5%) |

As the graph learns, more queries drop from deliberative → habitual → reflex.

### Myelination (Compiled Reflexes)

When a path fires often enough with consistent success, CrabPath compiles it into a deterministic macro — like how practiced skills become automatic in the brain. These are formalized as **options** (semi-MDP macro-actions) with:

- Initiation conditions (when to trigger)
- Internal policy (what to do)  
- Termination conditions (when to stop)
- Verification checks (did it work?)
- Fallback (degrade gracefully if environment changed)

### Learning Rules

- **Hebbian strengthening**: co-activated nodes during success → strengthen edges
- **Inhibition learning**: actions that caused failures → strengthen blocking edges
- **Decay**: unused connections fade over time
- **Pruning**: dead nodes/edges get archived
- **Consolidation**: nightly replay merges duplicates, promotes patterns to sequences
- **Immune system**: quarantine nodes correlated with repeated failures

## Status

🚧 **Early development** — we're building this in the open.

CrabPath grew out of running three persistent LLM agents for 20+ days ($13K+ in API costs, 1,482 classified interactions, 252 corrections). The paper documents the architecture, the math, and the experimental plan.

### What exists today
- [Research paper](https://jonathangu.com/crabpath/) with full architecture spec
- Core graph + activation engine (`pip install crabpath`)
- CLI for bootstrapping, querying, and learning
- 21 passing tests

### What we're building
- [ ] Persistence (save/load graphs to SQLite)
- [ ] Warm-start pipeline (bootstrap from agent logs — any format)
- [ ] Myelination compiler (option extraction)
- [ ] Embedding-based seed node selection
- [ ] Offline replay evaluation framework
- [ ] Framework integrations (OpenClaw, LangChain, etc. — optional extras)

## Architecture

```
┌─────────────────────────────────────────────┐
│                  CrabPath                    │
│                                              │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │  Reflex   │   │ Habitual │   │Delibera-│ │
│  │  Cache    │──▶│  Router  │──▶│  tive    │ │
│  │  (free)   │   │ (cheap)  │   │ (smart) │ │
│  └──────────┘   └──────────┘   └─────────┘ │
│       ▲              │              │        │
│       │              ▼              ▼        │
│  ┌──────────────────────────────────────┐   │
│  │         Memory Graph (G)             │   │
│  │  nodes: facts, rules, tools, actions │   │
│  │  edges: sequence, inhibition, cause  │   │
│  │  weights: learned from outcomes      │   │
│  └──────────────────────────────────────┘   │
│       ▲              │                       │
│       │              ▼                       │
│  ┌──────────┐   ┌──────────┐                │
│  │ Learning │   │  Nightly │                │
│  │   Loop   │   │ Cleanup  │                │
│  │(Hebbian) │   │(consolid)│                │
│  └──────────┘   └──────────┘                │
└─────────────────────────────────────────────┘
         ▲                    │
         │    Agent Tasks     ▼
    ┌─────────────────────────────┐
    │   LLM Agent (OpenClaw etc)  │
    └─────────────────────────────┘
```

## The Paper

📄 **Read the full paper**: [jonathangu.com/crabpath](https://jonathangu.com/crabpath/)

The paper covers:
- Why static context loading fails and what to do about it
- The full CrabPath architecture (typed graph, activation dynamics, learning rules)
- Mathematical framework (spectral graph theory, PageRank connection, MDP formulation)
- 10 biological mechanisms as testable engineering hypotheses
- Cost analysis and myelination economics
- Warm-start pipeline from existing agent data
- Experimental plan with ablation matrix
- Honest accounting of limitations and failure modes

## Quick Start

### Install

```bash
pip install crabpath
```

### Use as a library

```python
from crabpath.graph import MemoryGraph, MemoryNode, MemoryEdge, NodeType, EdgeType
from crabpath.activation import ActivationEngine

# Create a graph
graph = MemoryGraph()

# Add memories
graph.add_node(MemoryNode(
    id="rule-1",
    node_type=NodeType.RULE,
    content="Never claim fixed without testing on prod",
    summary="test before claiming fixed",
    tags=["deploy", "verification"],
    prior=0.9,
))

graph.add_node(MemoryNode(
    id="action-check-logs",
    node_type=NodeType.ACTION,
    content="tail -n 200 /var/log/service.log",
    summary="check service logs",
    tags=["deploy", "debug"],
    prior=0.6,
))

# Connect them
graph.add_edge(MemoryEdge(
    source="rule-1",
    target="action-check-logs",
    edge_type=EdgeType.SEQUENCE,
    weight=0.8,
))

# Activate the graph for a query
engine = ActivationEngine(graph)
result = engine.activate("deployment failed")

for node, score in result.activated_nodes:
    print(f"[{score:.3f}] {node.node_type.value}: {node.summary}")

# Learn from outcome
engine.learn(result, outcome="success")
```

### CLI

```bash
pip install crabpath

crabpath init                                    # Create a new graph
crabpath activate "deploy broken" --json         # Query the graph
crabpath stats                                   # Show graph stats
```

### Optional: Bootstrap from OpenClaw

```bash
# If you use OpenClaw, import your workspace into CrabPath
crabpath import-openclaw ~/.openclaw/workspace/
```

## Why "CrabPath"?

🦀 **Carcinisation** — nature's most repeated evolutionary experiment. Unrelated crustaceans keep independently evolving into crabs because the crab form is a local optimum that *works*.

We believe activation-driven memory graphs are the "crab form" of AI agent memory. Every system that gets serious about persistent agents will converge here: weighted graphs, spreading activation, inhibition, compiled reflexes.

CrabPath is the path everything walks. 🦀

## Contributing

We welcome contributions! This is early-stage research turning into a real system. If you're interested in:

- Graph algorithms and activation dynamics
- LLM agent memory systems
- Reinforcement learning / online learning
- Biological inspiration in AI systems
- Building developer tools

...come help us build the crab. 🦀

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{gu2026crabpath,
  title={CrabPath: Activation-Driven Memory Graphs for AI Agents},
  author={Gu, Jonathan},
  year={2026},
  url={https://github.com/jonathangu/crabpath}
}
```

---

*Built by [Jonathan Gu](https://jonathangu.com) • Powered by too many late nights and $13K in API costs* 🦀
