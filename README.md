# CrabPath

**Your agent workspace as a learning graph.** CrabPath builds a brain from your files that gets smarter from feedback. Good outcomes strengthen routes. Corrections create inhibitory edges that suppress wrong paths. Over time, queries return less noise and more signal.

- Zero dependencies. Pure Python 3.10+.
- Works offline with built-in hash embeddings. Plug in OpenAI for semantic search.
- `pip install crabpath`

## What You Get

- A `state.json` brain built from your workspace files
- Queries that traverse learned routes (not just similarity search)  
- `learn +1` strengthens good paths, `learn -1` suppresses bad ones
- `inject` adds corrections that create inhibitory edges in real time
- Session replay to warm-start from conversation history

## 5-Minute Quickstart (no API key needed)

```bash
# 1. Build a brain from your workspace
crabpath init --workspace ./my-workspace --output ./brain

# 2. Check it works
crabpath doctor --state ./brain/state.json

# 3. Query it  
crabpath query "how do I deploy" --state ./brain/state.json --top 5

# 4. Teach it (use node IDs from the query output)
crabpath learn --state ./brain/state.json --outcome 1.0 --fired-ids "deploy.md::0,deploy.md::1"

# 5. Inject a correction
crabpath inject --state ./brain/state.json --id "fix::1" \
  --content "Never skip CI for hotfixes" --type CORRECTION

# 6. Query again — the correction now appears in results
crabpath query "hotfix process" --state ./brain/state.json --top 5
```

## Teach Your Graph from Corrections

When your agent makes a mistake, inject it:

```bash
# CLI
crabpath inject --state brain/state.json \
  --id "correction::42" \
  --content "Never show API keys in chat messages" \
  --type CORRECTION
```

```python
# Python API
from crabpath import inject_correction, load_state, save_state

graph, index, meta = load_state("brain/state.json")
inject_correction(graph, index, "correction::42",
    "Never show API keys in chat messages",
    embed_fn=my_embed_fn)  # or vector=[...] 
save_state(graph, index, "brain/state.json", meta=meta)
```

Corrections create **inhibitory edges** — they actively suppress wrong routes during traversal.

## Real Embeddings (OpenAI)

The hash embedder works offline but semantic search needs real embeddings:

```python
from openai import OpenAI
from crabpath import split_workspace, VectorIndex

client = OpenAI()

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small", input=[text]
    ).data[0].embedding

graph, texts = split_workspace("./workspace")
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, embed(content))
```

See `examples/openai_embedder/` for a complete example.

## Session Replay (optional — warm start from history)

If you have conversation logs (OpenClaw `.jsonl` sessions), replay them to warm-start the graph:

```bash
crabpath replay --state brain/state.json --sessions ./sessions/
```

Skip this if you are just getting started — the graph learns from live queries too.

## CLI Reference

| Command | What it does |
|---------|-------------|
| `init` | Build brain from workspace files |
| `query` | Search the graph |
| `learn` | Reinforce (+) or suppress (-) fired paths |
| `inject` | Add a node with connections (CORRECTION creates inhibitory edges) |
| `replay` | Warm-start from session history |
| `health` | Graph health metrics |
| `doctor` | Validate state integrity |
| `info` | Graph statistics |
| `merge` | Suggest/apply node merges |
| `connect` | Suggest/apply cross-file connections |
| `journal` | Query event journal |

## Python API

```python
from crabpath import (
    split_workspace, traverse, apply_outcome,
    inject_node, inject_correction, inject_batch,
    VectorIndex, HashEmbedder, TraversalConfig,
    save_state, load_state, ManagedState,
    measure_health, replay_queries, score_retrieval
)
```

## Production Experience

Three brains run in production on a Mac Mini M4 Pro:

| Brain | Nodes | Edges | Learning Corrections | Sessions Replayed |
|-------|-------|-------|---------------------|-------------------|
| MAIN | 1,142 | 2,814 | 43 | 215 |
| PELICAN | 512 | 1,984 | 181 | 183 |
| BOUNTIFUL | 273 | 1,073 | 35 | 300 |

Corrections auto-sync every 2 hours from a learning harness (SQLite → inject_batch → inhibitory edges). Context injection was cut from 66KB to 49KB (-25%) by moving bulk knowledge into CrabPath.

## How It Compares

| | Plain RAG | CrabPath |
|---|-----------|----------|
| Retrieval | Similarity search | Learned graph traversal |
| Feedback | None | +1/-1 outcomes update edge weights |
| Wrong answers | Keep resurfacing | Inhibitory edges suppress them |
| Over time | Same results | Routes compile to reflex behavior |
| Dependencies | Vector DB | Zero (pure Python) |

## Design Tenets

- No network calls in core
- No secret discovery (no dotfiles, no keychain)
- Embedder identity stored in state metadata; hard-fail on dimension mismatch
- One canonical state format (`state.json`)

## Paper

[jonathangu.com/crabpath](https://jonathangu.com/crabpath/) — 8 deterministic simulations + production deployment data.

## Benchmarks

```bash
python3 benchmarks/run_benchmark.py
```

Results are deterministic per commit; timings vary by machine.

## Links

- PyPI: `pip install crabpath`
- GitHub: [jonathangu/crabpath](https://github.com/jonathangu/crabpath)
- Paper: [jonathangu.com/crabpath](https://jonathangu.com/crabpath/)
- ClawHub: `clawhub install crabpath`

---
