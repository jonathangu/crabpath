# CrabPath

CrabPath is a pure in-memory graph engine for retrieval routing that can learn from feedback and prioritize what to execute next, while staying independent of any model provider.

## Install

```bash
pip install crabpath
```

## Python API (pure callbacks)

```python
from crabpath import Graph, Node, Edge, VectorIndex, split_workspace, traverse, apply_outcome

# 1) Build graph + texts from workspace
graph, texts = split_workspace("./workspace")

# 2) Caller owns embeddings
embed = lambda text: [1.0, 2.0]  # your vector model, local or remote
index = VectorIndex()
for node_id, text in texts.items():
    index.upsert(node_id, embed(text))

# 3) Caller owns LLM callbacks (optional)
def route_fn(query: str, candidate_ids: list[str]) -> list[str]:
    return candidate_ids[:3]

def score_fn(system_prompt: str, user_prompt: str) -> str:
    return '{"scores": {"node-id": 1.0}}'

# 4) Query and learn
seeds = index.search([0.1, 0.2], top_k=8)
result = traverse(graph=graph, seeds=seeds, route_fn=route_fn)
apply_outcome(graph=graph, fired_nodes=result.fired, outcome=1.0)
```

## CLI (no providers, pure stdin/stdout)

```bash
# Build graph and index (auto local embeddings when installed)
crabpath init --workspace ./workspace --output ./crabpath-data --embed-command 'python3 embed_cb.py'
crabpath init --workspace ./workspace --output ./crabpath-data

# Build index only
crabpath embed --texts ./crabpath-data/texts.json --output ./crabpath-data/index.json --command 'python3 embed_cb.py'

# Query by keyword
crabpath query "how do i deploy" --graph ./crabpath-data/graph.json --top 5

# Query by vector payload file
crabpath query "noop" --graph ./crabpath-data/graph.json --index ./crabpath-data/index.json --query-vector-stdin < vec.json
# Query by local embedding
crabpath query "how do i deploy" --graph ./crabpath-data/graph.json --index ./crabpath-data/index.json

# Optional route callback and query scoring callback wiring
cat /tmp/query.vec | crabpath query "deploy" --graph ./crabpath-data/graph.json --index ./crabpath-data/index.json --route-command 'python3 route_cb.py' --embed-command 'python3 embed_cb.py'
```

## Embeddings

Included. `pip install crabpath` brings `all-MiniLM-L6-v2` (local, 80MB, CPU). No API key.

```bash
crabpath init --workspace ./ws --output ./data
# → embeds automatically
```

To use your own embeddings instead, pass `--embed-command`.

## Batch callbacks

Batching is available for both embedding and callback APIs using the same CLI entry point:

```bash
crabpath init --workspace ./workspace --output ./crabpath-data --embed-command 'python3 embed_batch.py'
crabpath init --workspace ./workspace --output ./crabpath-data
crabpath query "deploy" --graph ./crabpath-data/graph.json --route-command 'python3 route_batch.py' --json
```

Internally, `ThreadPoolExecutor` is used to parallelize single-item fallback while preserving batch callbacks when provided.

## Other pure graph commands

```bash
crabpath learn --graph graph.json --outcome 1.0 --fired-ids a,b,c
crabpath replay --graph graph.json --sessions ./sessions/*.jsonl
crabpath health --graph graph.json --json
crabpath merge --graph graph.json --json
crabpath connect --graph graph.json --json
crabpath journal --stats
```

## What this project is

CrabPath is a **library**, not a hosted service. It does no network calls and persists no credentials. Callers pass in embedding and routing callbacks; CrabPath only manages graph logic, scoring plumbing, and state updates.

## Paper

Technical details: https://jonathangu.com/crabpath/
