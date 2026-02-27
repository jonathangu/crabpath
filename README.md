# CrabPath

Pure routing graph engine for context-aware retrieval. Core is zero dependencies and zero network; the caller supplies semantic and LLM callbacks.

## 1. CrabPath

CrabPath is a deterministic graph engine that builds traversable context graphs and improves routing from feedback, without any external service requirement by default.

## 2. Design Tenets

- No network calls in core (not even for model downloads)
- No secret discovery (no dotfiles, no keychain, no env probing)
- No subprocess provider wrappers
- Embedder identity stored in state metadata (`name`, `dim`, `schema_version`); hard-fail on dimension mismatch
- One canonical state format (`state.json`)

## 3. Install

```bash
pip install crabpath
clawhub install crabpath         # ClawHub (for OpenClaw agents)
```

## 4. Quick Start (out of the box)

```python
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex, HashEmbedder

embedder = HashEmbedder()  # default hash-v1 (1024-dim)
graph, texts = split_workspace("./workspace")
index = VectorIndex()

for nid, content in texts.items():
    index.upsert(nid, embedder.embed(content))

query_vec = embedder.embed("how do I deploy to production?")
seeds = index.search(query_vec, top_k=8)
result = traverse(graph, seeds)
apply_outcome(graph, result.fired, outcome=1.0)
```

## 5. Real Embeddings via Caller Callbacks

```python
from openai import OpenAI
from crabpath import split_workspace, VectorIndex
from crabpath._batch import batch_or_single_embed

client = OpenAI()

def embed_batch_fn(items):
    ids, contents = zip(*items)
    resp = client.embeddings.create(model="text-embedding-3-small", input=list(contents))
    vectors = {ids[i]: resp.data[i].embedding for i in range(len(ids))}
    if any(len(v) != 1536 for v in vectors.values()):
        raise ValueError("dimension mismatch: expected 1536")
    return vectors

graph, texts = split_workspace("./workspace")
index = VectorIndex()
vecs = batch_or_single_embed(list(texts.items()), embed_batch_fn=embed_batch_fn)
for nid, vec in vecs.items():
    index.upsert(nid, vec)
```

## 6. LLM Callbacks

```python
from openai import OpenAI
from crabpath import split_workspace

client = OpenAI()

def llm_fn(system, user):
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content

# Pass to split for LLM chunking, or to query routes as needed
graph, texts = split_workspace("./workspace", llm_fn=llm_fn)
```

## 7. Session Replay (prominent)

Session replay is the fastest way to warm-start a new brain from history.

```python
from crabpath import replay_queries, split_workspace
from crabpath.replay import extract_queries_from_dir

graph, texts = split_workspace("./workspace")
queries = extract_queries_from_dir("./sessions/")
replay_queries(graph=graph, queries=queries)
```

Or via CLI:

```bash
crabpath init --workspace ./ws --output ./data --sessions ./sessions/
crabpath replay --state ./data/state.json --sessions ./sessions/
```

## Injecting External Knowledge

CrabPath nodes are not limited to file chunks. You can inject arbitrary knowledge as graph nodes:

```python
from crabpath.graph import Node
from crabpath import Graph

graph, texts = split_workspace("./workspace")

# Inject a learning/correction
node = Node(
    id="learning::42",
    content="Never show in-sample backtest results. Always use out-of-sample data.",
    summary="Out-of-sample backtesting rule",
    metadata={"source": "learning_db", "type": "CORRECTION"}
)
graph.add_node(node)
texts["learning::42"] = node.content  # embed this too
```

The OpenClaw adapter (`examples/openclaw_adapter/`) demonstrates injecting corrections from a learning harness SQLite database as first-class retrievable nodes.

## 8. Three Embedding Tiers

| Tier | Capability | Network | Notes |
| --- | --- | --- | --- |
| 1 | Default hash-v1 | none | Built into core, zero deps, deterministic baseline |
| 2 | Semantic callback | optional | Caller-provided `embed_fn` / `embed_batch_fn` (OpenAI, Gemini, local service, etc.) |
| 3 | Replay | none | No new embedding calls; uses historical query signals and graph traversal history |

## 9. Benchmarks

Benchmark results are commit-specific and should be regenerated locally:

Run `python3 benchmarks/run_benchmark.py` to see current deterministic results for this commit.

## 10. CLI

`--state` is the preferred API; `--graph`/`--index` are legacy compatibility flags.

```bash
crabpath init --workspace W --output O [--sessions S]
crabpath query TEXT --state S [--top N] [--json]
crabpath learn --state S --outcome N --fired-ids a,b,c
crabpath health --state S
crabpath doctor --state S
crabpath info --state S|--graph G
crabpath replay --state S --sessions S
crabpath merge --state S
crabpath connect --state S
crabpath journal [--stats]

# Legacy
crabpath query TEXT --graph G [--index I] [--query-vector-stdin] [--top N] [--json]
crabpath learn --graph G --outcome N --fired-ids a,b,c [--json]
crabpath replay --graph G --sessions S
crabpath health --graph G
crabpath merge --graph G
crabpath connect --graph G
```

## 11. Reproduce Results

See [REPRODUCE.md](REPRODUCE.md).

## 12. Paper

https://jonathangu.com/crabpath/

## 13. Status

- 8 deterministic simulations: all PASS
- 150 tests
- Production: 3 agent brains using real OpenAI embeddings
