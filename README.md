# 🦀 CrabPath

CrabPath is a memory store for AI agents that learns what to retrieve — and what to suppress — from experience.

## Why?

- Static context loading wastes tokens because you load too much every turn.
- Classic RAG can’t learn from feedback, so retrieval stays similarity-only.
- CrabPath tracks which retrieval paths worked, and builds a graph of learned routes.

## Install

```bash
python3 -m venv ~/.crabpath-env && source ~/.crabpath-env/bin/activate
pip install crabpath            # PyPI
clawhub install crabpath        # or ClawHub (OpenClaw agents)

# For embeddings (strongly recommended):
pip install crabpath[openai]     # or: pip install crabpath[google]
```

Zero required dependencies. Python 3.10+. macOS Homebrew Python needs a venv (PEP 668).

### LLM Access (recommended)

CrabPath's smart routing needs a real LLM. Without one, it falls back to basic retrieval (like RAG). Provide access via any of:

```bash
# Option 1: Direct API key (easiest)
export OPENAI_API_KEY=sk-...          # also checks ~/.env automatically

# Option 2: Any OpenAI-compatible endpoint (most flexible)
export CRABPATH_LLM_URL=http://localhost:8080/v1/chat/completions
export CRABPATH_EMBEDDINGS_URL=http://localhost:8080/v1/embeddings
export CRABPATH_LLM_TOKEN=your-token  # if endpoint needs auth

# Option 3: Local Ollama (free, private)
ollama pull nomic-embed-text          # embeddings
ollama pull llama3                     # routing
```

Works with: OpenAI, Gemini, Ollama, LiteLLM, vLLM, OpenRouter — anything OpenAI-compatible.

**Without LLM access**, CrabPath still works using local TF-IDF embeddings and heuristic routing — but you're missing the learned routing that makes it better than RAG.

## Quick Start (60 seconds)

```python
from crabpath import Node, Edge, Graph, activate, learn

g = Graph()
g.add_node(Node("timeout", "Deployment timed out"))
g.add_node(Node("rollback", "Rollback and restore"))
g.add_node(Node("debug", "Inspect logs"))
g.add_edge(Edge("timeout", "rollback", 0.6))
g.add_edge(Edge("timeout", "debug", 0.4))

result = activate(g, seeds={"timeout": 1.0})
learn(g, result, outcome=1.0)  # reinforces paths that fired
print([node.id for node, energy in result.fired])
```

## For AI Agents (3 commands)

```bash
crabpath init --workspace ~/.openclaw/workspace --sessions ~/.openclaw/agents/main/sessions/
crabpath install-hook --agent-workspace ~/.openclaw/workspace
crabpath query 'how do I deploy' --graph ~/.crabpath/graph.json --index ~/.crabpath/embed.json --top 8 --json
```

## Which Interface?

| Interface | Status / Use |
|---|---|
| CLI (agents) | Primary agent-facing interface; JSON I/O for shell workflows. |
| MemoryController (Python) | Recommended direct integration for Python apps. |
| Adapter | Deprecated legacy bridge; prefer CLI or MemoryController. |

## How It Works

- Documents are split into nodes and edges become weighted pointers.
- Reflex edges (`>0.8`) auto-follow with near-zero overhead.
- Habitual edges (`0.3-0.8`) go through normal routing policy.
- Dormant edges (`<0.3`) are suppressed by default.
- Positive outcomes (`+1`) strengthen paths; negative outcomes (`-1`) create inhibitory edges.
- Decay drops unused connections, while the autotuner keeps graph routing healthy.

## Key Results

| Metric | Result |
|---|---|
| Context reduction | 90-99% |
| Negation accuracy | 1.0 vs 0.0 (BM25) |
| Internal tests | 360 |
| Required deps | Zero |

Full benchmark details: [docs/research/](docs/research/)

## When NOT to Use CrabPath

- Simple static-document RAG without feedback loops (use a vector DB).
- Very small codebases (`< 10` files) with insufficient structure to learn recurring routes.
- One-off questions with no recurring retrieval patterns.

## Links

- Paper: [jonathangu.com/crabpath/](https://jonathangu.com/crabpath/)
- ClawHub: [clawhub.ai](https://clawhub.ai/)
- PyPI: [pypi.org/project/crabpath/](https://pypi.org/project/crabpath/)

## License

Apache 2.0
