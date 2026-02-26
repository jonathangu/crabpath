# 🦀 CrabPath

Memory graph for AI agents that learns what to retrieve — and what to suppress.

## Why CrabPath?

The problem in 3 bullets:
- Static context loading wastes tokens (loads everything every turn)
- RAG retrieves by similarity, not usefulness (can't learn from feedback)
- CrabPath builds learned routes: positive outcomes strengthen paths, negative outcomes create inhibitory edges

The result: in simulation, active context drops from 30 nodes to 2.7 per query (91% reduction). Deploy procedure compiles from habitual to reflex in 50 queries. Bad paths get suppressed to -0.94 weight.

## Drawbacks (honest)

- **Cold start is slow**: 100% habitual at bootstrap, everything requires deliberation until enough queries accumulate
- **Embedding quality matters**: CrabPath is only as good as the vectors you give it
- **Without embeddings, retrieval is keyword-based and noisy** — CrabPath is a pure graph engine and doesn't compute embeddings itself. Use `crabpath embed` or `--embed-provider` to add them.
- **No production deployment yet**: all results are from simulation
- **Small workspaces (< 10 files) see limited benefit**
- **Not for one-shot queries**: CrabPath shines with recurring patterns over time

## Install

```bash
pip install crabpath
```

Python 3.10+. Zero dependencies. Pure stdlib.

## Getting Started

### As a Python library

```python
from crabpath import Graph, Node, Edge
from crabpath.split import split_workspace
from crabpath.traverse import traverse, TraversalConfig
from crabpath.learn import apply_outcome

# 1. Build graph from workspace
graph, texts = split_workspace('~/.openclaw/workspace')

# 2. Caller embeds texts (CrabPath never does this)
# vectors = {node_id: your_embed_fn(text) for node_id, text in texts.items()}

# 3. Traverse from seeds
result = traverse(graph, seeds=[('deploy-node', 1.0)], config=TraversalConfig())
print(result.fired)  # nodes that were visited
print(result.context)  # assembled context string

# 4. Learn from outcome
apply_outcome(graph, fired_nodes=result.fired, outcome=1.0)  # +1 good, -1 bad

# 5. Save
graph.save('graph.json')
```

### As a CLI (for AI agents)

```bash
# Split workspace into graph + node texts for embedding
crabpath init --workspace ./my-workspace --output ./crabpath-data

# Adding embeddings (strongly recommended)
#
# Option 1: OpenAI (requires OPENAI_API_KEY)
crabpath embed --texts ./crabpath-data/texts.json --output ./crabpath-data/index.json --provider openai

# Option 2: Ollama (free, local)
crabpath embed --texts ./crabpath-data/texts.json --output ./crabpath-data/index.json --provider ollama

# Option 3: Custom embedding script
crabpath embed --texts ./crabpath-data/texts.json --output ./crabpath-data/index.json --command 'python3 my_embed.py'

# Or combine everything in one shot:
crabpath init --workspace ./my-workspace --output ./crabpath-data --embed-provider openai

# Query (provide seeds from your own index/embedding)
# CLI also supports --query-vector for vector inputs in production use
crabpath query 'how do I deploy' --graph graph.json --index index.json --top 8 --json

### LLM Routing (for full mode)

With an LLM router, CrabPath makes smart traversal decisions instead of following edges by weight:

```bash
# Query with LLM routing
crabpath query 'how do I deploy' \
  --graph ~/.crabpath/graph.json \
  --index ~/.crabpath/index.json \
  --route-provider openai \
  --json

# Or one-shot with everything:
crabpath init --workspace ~/.openclaw/workspace \
  --output ~/.crabpath \
  --sessions ~/.openclaw/agents/main/sessions/ \
  --embed-provider openai \
  --route-provider openai
```

# Learn from feedback
crabpath learn --graph graph.json --outcome 1.0 --fired-ids node1,node2,node3

# Check health
crabpath health --graph graph.json
```

### Session Replay (warm up the graph)

Replay historical session logs to bootstrap edge weights from real usage patterns:

```bash
crabpath replay --graph ~/.crabpath/graph.json --sessions ~/.openclaw/agents/main/sessions/
```

Or combine with init:

```bash
crabpath init --workspace ~/.openclaw/workspace --output ~/.crabpath --sessions ~/.openclaw/agents/main/sessions/
```

### Logging

CrabPath logs every query, learn, and health check to `~/.crabpath/journal.jsonl`.

```bash
# View recent activity
crabpath journal --last 5

# Summary stats
crabpath journal --stats
```

### The key insight: CrabPath is a pure graph engine

CrabPath NEVER makes network calls. It doesn't compute embeddings or call LLMs. You provide:
- **Embeddings**: embed node texts with whatever you have (OpenAI, Gemini, Ollama, local)
- **Routing decisions** (optional): a callback that picks which edges to follow

This means CrabPath works everywhere — air-gapped, CI, local dev. Zero API keys required.

## How It Works

- Workspace files split into nodes, connected by weighted edges
- Three tiers: **reflex** (>0.8, auto-follow), **habitual** (0.3-0.8, deliberate), **dormant** (<0.3, suppressed)
- **Edge damping**: w' = w × 0.3^k prevents traversal loops without hard limits
- **Learning**: +1 outcomes strengthen paths (policy gradient). -1 outcomes create inhibitory edges
- **Decay**: unused edges fade over time (configurable half-life)
- **Autotune**: self-regulating health metrics keep the graph bounded

## Key Results (from simulation)

| Claim | Result |
|-------|--------|
| Context reduction | 30 → 2.7 nodes (91%) |
| Procedural compilation | 0.27 → 1.0 weight in 50 queries |
| Negation suppression | bad path → -0.94 weight |
| Selective forgetting | 93.3% dormant after 100 queries |
| Loop prevention | edge damping discovers branches |
| Domain separation | 5 cross-file bridges emerge |
| Brain death recovery | autotune detects + recovers |
| Individuation | 27 edges diverge between twin agents |

## Paper

Full technical details, math, and methodology: [jonathangu.com/crabpath/](https://jonathangu.com/crabpath/)

## Links

- Paper: [jonathangu.com/crabpath/](https://jonathangu.com/crabpath/)
- PyPI: [pypi.org/project/crabpath/](https://pypi.org/project/crabpath/)
- ClawHub: [clawhub.ai](https://clawhub.ai/) (search 'crabpath')

## License

Apache 2.0

## Notes on reproducibility

This project intentionally keeps the README focused on behavior and interfaces.

- Benchmarks and tuning constants live in code and tests.
- Simulation claims come from deterministic runs in the repository test suite.
- If you are building an automation loop, persist `graph.json` after learning so weight updates remain stateful.
