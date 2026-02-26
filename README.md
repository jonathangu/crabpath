# 🦀 CrabPath: The Graph is the Prompt

**LLM-guided memory traversal with learned pointer weights and corrected policy gradients.**

📄 **[Read the paper →](https://jonathangu.com/crabpath/)**

CrabPath is a memory architecture for AI agents where documents are nodes, weighted pointers are edges, and an LLM is the activation function. The graph learns which paths lead to good outcomes — and which to suppress — using trajectory-aware credit assignment ([Gu, 2016](https://jonathangu.com/crabpath/)). Over time, expensive LLM reasoning compiles into cheap reflexive routing.

**The key result:** CrabPath matches BM25 on retrieval accuracy (0.742 vs 0.737) but achieves **perfect negation accuracy** where BM25 and standard RAG score zero. The system learns what *not* to retrieve — inhibitory edges actively suppress known-wrong paths.

## Headline Numbers

| Metric | Value |
|---|---|
| Context reduction vs static | 90–99% |
| Negation accuracy (CrabPath vs BM25) | 1.000 vs 0.000 |
| Corrected PG vs myopic REINFORCE | +11 pp (non-overlapping 95% CIs) |
| Cost per turn (two-tier routing) | $0.004 vs $0.091 static |
| External benchmark (HotpotQA) | BM25 dominates cold-start IR; CrabPath learns (1948 edge updates) but topic diversity limits transfer |
| Recurring-topic benchmark (200 queries) | CrabPath R@2: 0.02→0.07 vs BM25 flat 0.27 |
| Sparsity crossover benchmark | Synthetic clusters: Phase2 Recall@3 crossover at ~50 nodes in sparse regimes (edge_ratio 0.1/0.05) |
| Tests | 327, 0 lint errors |
| Dependencies | Zero (stdlib only) |

## Install

```bash
pip install crabpath   # or: pip install .
pip install crabpath[embeddings]  # optional: OpenAI/Gemini/Cohere embedding providers
```

## Quick Start

```python
from crabpath import Graph, Node, Edge, activate, learn

g = Graph()
g.add_node(Node(id="check-tests", content="Run test suite before deploy"))
g.add_node(Node(id="skip-tests", content="Skip tests, deploy directly"))
g.add_node(Node(id="deploy", content="Deploy to production"))

g.add_edge(Edge(source="check-tests", target="deploy", weight=0.5))
g.add_edge(Edge(source="skip-tests", target="deploy", weight=0.5))

result = activate(g, seeds={"check-tests": 1.0, "skip-tests": 1.0})
learn(g, result, outcome=1.0)  # STDP: causal edges strengthened

g.save("memory.json")
```

## Recommended Import Path

Use `MemoryController` as the primary API for new integrations; it wraps query and learn in one place.

### Query-only

```python
from crabpath import Graph, MemoryController

graph = Graph.load("graph.json")
controller = MemoryController(graph)
result = controller.query("How do I respond to a rollback?")
print(result.context)
```

### Query + learn

```python
from crabpath import Graph, MemoryController

graph = Graph.load("graph.json")
controller = MemoryController(graph)

result = controller.query("How do I respond to a rollback?")
controller.learn(result, reward=1.0)
graph.save("graph.json")
```

### Full shadow mode (query, log, learn)

```python
import json
from pathlib import Path
from time import time
from crabpath import Graph, MemoryController

graph = Graph.load("graph.json")
controller = MemoryController(graph)
result = controller.query("Incident failed after deploy")

log_path = Path.home() / ".crabpath" / "shadow.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"ts": time(), "query": "Incident failed after deploy", "nodes": result.selected_nodes}) + "\n")

controller.learn(result, reward=1.0)
graph.save("graph.json")
```

Lower-level modules like `synaptogenesis`, `mitosis`, and `autotune` are still available for advanced users who need direct control.

## Architecture

CrabPath has three components:

1. **Nodes** — documents with content, summary, and semantic type (`fact`, `procedure`, `action`, `tool_call`)
2. **Edges** — signed weighted pointers (excitatory or inhibitory) with kind labels (`support`, `inhibit`, `follows`, `tool`)
3. **LLM activation** — reads node content, inspects candidate edges, decides which to traverse

### Core Modules

| Module | Purpose |
|---|---|
| `controller.py` | **MemoryController** — orchestrates query → retrieve → learn cycles |
| `inhibition.py` | Negative-edge routing, correction signals, suppression scoring |
| `learning.py` | Gu-corrected policy gradient + **LearningPhaseManager** (Hebbian → RL transition) |
| `synaptogenesis.py` | Proto-edge formation → promotion → Hebbian reinforcement → competition |
| `mitosis.py` | Recursive cell division: LLM-driven splitting, merging, neurogenesis |
| `autotune.py` | Self-regulation with meta-learning, safety guardrails, emergency brake |
| `router.py` | Three-tier LLM routing (reflex/habitual/dormant) |
| `traversal.py` | Multi-hop traversal with damped edges (`max_hops=30`, `episode_edge_damping=0.3`) |
| `decay.py` | Exponential weight decay |
| `embeddings.py` | Multi-provider embeddings (OpenAI, Gemini, Cohere, Ollama) |
| `migrate.py` | Bootstrap graphs from workspace files + session log replay |
| `adapter.py` | **OpenClawCrabPathAdapter** — drop-in integration for OpenClaw agents |
| `feedback.py` | Correction detection, LLM scoring, auto-feedback |
| `shadow_logger.py` | Shadow mode logging for safe evaluation |

### MemoryController (recommended entry point)

```python
from crabpath import MemoryController, ControllerConfig
from crabpath.learning import LearningConfig
from crabpath.synaptogenesis import SynaptogenesisConfig
from crabpath.inhibition import InhibitionConfig
from crabpath.decay import DecayConfig

controller = MemoryController(
    graph_path="graph.json",
    config=ControllerConfig(
        learning=LearningConfig(),
        synaptogenesis=SynaptogenesisConfig(),
        inhibition=InhibitionConfig(),
        decay=DecayConfig(),
    ),
    embed_fn=my_embed_fn,  # or None for keyword fallback
    router_fn=my_llm_call,  # or None for weight-only routing
)

# Query
result = controller.query("deploy broke after config change")
context = result.context       # rendered context string
context_chars = result.context_chars

# Learn from outcome
controller.learn(result, reward=1.0)  # or -1.0 for correction

# Save
controller.save()
```

### Three-Tier Routing

| Tier | Weight | Behavior | Cost |
|---|---|---|---|
| Reflex | > 0.9 | Auto-follow, no LLM call | Very low |
| Habitual | 0.1–0.9 | Presented as candidates to LLM | Moderate |
| Dormant | < 0.1 | Skipped unless explicit override | Near zero |

### Learning: Two-Phase Dynamics

1. **Phase 1 — Hebbian** (~20–100 queries): Co-firing differentiates edge weights. Policy gradient signals wash out because softmax is near-uniform.
2. **Phase 2 — RL refinement** (post-transition): Weights are differentiated enough for sharp softmax. Corrected policy gradient propagates reward across full traversal path.

The `LearningPhaseManager` detects the transition adaptively via weight-entropy and gradient-magnitude thresholds.

### Inhibition

The differentiator. When the user says "the codeword is elephant, NOT giraffe":
- BM25 returns both (semantically similar) → **score: 0.0**
- CrabPath's inhibitory edge suppresses giraffe → **score: 1.0**

```python
from crabpath.inhibition import apply_correction, is_inhibited
from crabpath.inhibition import InhibitionConfig

# After user correction
apply_correction(graph, trajectory=["giraffe", "elephant"], reward=-0.8, config=InhibitionConfig())

# During routing
if is_inhibited(graph, "giraffe", "elephant"):
    # skip this node
```

## Embeddings

```python
from crabpath import auto_embed, openai_embed, gemini_embed, ollama_embed

# Automatic (tries OpenAI → Gemini → Ollama)
vectors = auto_embed(["deploy broke", "check CI"])

# Or specific provider
vectors = openai_embed(["deploy broke"])      # needs OPENAI_API_KEY
vectors = gemini_embed(["deploy broke"])      # needs GEMINI_API_KEY
vectors = ollama_embed(["deploy broke"])      # free, local
```

## CLI

All commands emit JSON on stdout.

```bash
# Query the graph
crabpath query "deploy broke" --graph graph.json --index embed.json --top 8

# Learn from outcome
crabpath learn --graph graph.json --outcome 1.0 --fired-ids node1,node2

# Check graph health (8 metrics)
crabpath health --graph graph.json

# Track evolution over time
crabpath evolve --graph graph.json --snapshots evolution.jsonl --report

# Run lifecycle simulation
crabpath sim --queries 200 --output results.json

# Bootstrap from workspace
crabpath migrate --workspace ~/.openclaw/workspace --output-graph graph.json --output-embeddings embed.json

# Split a node via LLM
crabpath split --graph graph.json --node-id tools --save

# Stats
crabpath stats --graph graph.json

# Snapshot + feedback
crabpath snapshot --graph graph.json --session sess-1 --turn 42 --fired-ids n1,n2
crabpath feedback --session sess-1 --turn-window 5
```

## MCP Server

```bash
python -m crabpath.mcp_server
```

Tools: `query`, `migrate`, `learn`, `stats`, `split`, `add`, `remove`, `consolidate`

## OpenClaw Integration

```python
from crabpath import OpenClawCrabPathAdapter

adapter = OpenClawCrabPathAdapter(
    graph_path="graph.json",
    index_path="embed.json",
    embed_fn=auto_embed,
)

graph, index = adapter.load()
seeds = adapter.seed("deploy broke", memory_search_ids=["hit-1"])
firing = adapter.activate(seeds)
context = adapter.context(firing)

adapter.learn(firing, outcome=1.0)
adapter.save()
```

## Self-Regulation (Autotuner)

The autotuner maintains 8 health metrics in target ranges:

| Metric | Target |
|---|---|
| Avg nodes fired/query | 3–8 |
| Cross-file edge % | 5–20% |
| Dormant % | 60–90% |
| Reflex % | 1–5% |
| Context compression | ≤ 20% |
| Proto promotion rate | 5–15% |
| Reconvergence rate | ≤ 10% |
| Orphan nodes | 0 |

```python
from crabpath import measure_health, autotune
from crabpath.autotune import self_tune, TuneMemory

health = measure_health(graph, state, query_stats)
health, adjustments, changes = self_tune(
    graph, state, query_stats,
    syn_config, decay_config, mitosis_config,
    cycle_number=1,
    tune_memory=TuneMemory(),
)
```

## Recursive Cell Division (Mitosis)

Files start as monolithic nodes → LLM splits into coherent chunks → sibling edges at weight 1.0 → activation/decay differentiates → reconverged chunks get re-split differently. The graph finds its own resolution.

```python
from crabpath.mitosis import bootstrap_workspace, mitosis_maintenance, MitosisState

graph = Graph()
state = MitosisState()
bootstrap_workspace(graph, workspace_files, llm_call, state)

# Run periodically
mitosis_maintenance(graph, llm_call, state)
```

## What Emerges

Five phenomena observed in production:

1. **Procedural Memory** — multi-hop reflex chains (muscle memory for common workflows)
2. **Domain Separation with Bridges** — knowledge clusters + cross-domain connections
3. **Selective Forgetting** — 95% of sibling edges decay to dormant (the graph learns what NOT to load)
4. **Self-Regulation** — autotuner homeostasis prevents drift
5. **Individuation** — same files, different graph shapes based on usage patterns

## Getting Started (Safe Mode)

Zero risk, zero behavior changes:

```bash
# 1. Bootstrap (read-only — does NOT modify your workspace)
crabpath migrate --workspace ~/.openclaw/workspace \
  --output-graph graph.json --output-embeddings embed.json

# 2. Check health
crabpath health --graph graph.json

# 3. Shadow mode (run alongside, never modifies responses)
crabpath query "your query" --graph graph.json --index embed.json

# 4. Track evolution
crabpath evolve --graph graph.json --snapshots evolution.jsonl --report
```

Kill switch: delete `graph.json`. All state is in that one file.

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for a complete mapping of every paper claim to a script, seed, and expected output.

```bash
# Run all benchmarks
cd experiments && python run_all.py

# Run ablation study (seed=2026, 10K bootstrap resamples)
python scripts/ablation_study.py

# Run phase transition diagnostic
python scripts/phase_transition_plot.py
```

## Requirements

- Python 3.10+
- Zero dependencies (stdlib only)
- Optional: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or local Ollama for semantic embeddings
- Without any provider, CrabPath falls back to keyword-based routing

## The Name

[Carcinisation](https://en.wikipedia.org/wiki/Carcinisation): nature keeps independently evolving crabs. Agent memory keeps converging on index + similarity retrieval. CrabPath evolves past that local optimum.

## License

Apache 2.0

---

*[Jonathan Gu](https://jonathangu.com)* 🦀
