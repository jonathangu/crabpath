# CrabPath

Pure graph engine for retrieval routing. Zero deps. Zero network calls. Caller provides embeddings and LLM callbacks.

## Install

```bash
pip install crabpath             # pure graph engine
pip install crabpath[embeddings] # + local embeddings (no API key)
```

## Python API

```python
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex

# 1. Split workspace into graph + texts
graph, texts = split_workspace("./workspace")

# 2. Caller embeds (use whatever you have)
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, your_embed_fn(content))

# 3. Query
seeds = index.search(your_embed_fn("how do I deploy"), top_k=8)
result = traverse(graph, seeds)

# 4. Learn
apply_outcome(graph, result.fired, outcome=1.0)
```

## Batch Callbacks

```python
from crabpath._batch import batch_or_single, batch_or_single_embed

# One call for all embeddings
vecs = batch_or_single_embed(
    list(texts.items()),
    embed_batch_fn=your_batch_embed
)

# One call for all LLM work
results = batch_or_single(
    [{"id": "n0", "system": "summarize", "user": content}],
    llm_batch_fn=your_batch_llm
)
```

## Local Embeddings

```bash
pip install crabpath[embeddings]
```

```python
from crabpath.embeddings import local_embed_fn, local_embed_batch_fn
# all-MiniLM-L6-v2, 80MB, CPU, no API key
```

## CLI (pure graph ops)

```
crabpath init --workspace W --output O [--sessions S]
crabpath query TEXT --graph G [--index I] [--query-vector-stdin] [--top N] [--json]
crabpath learn --graph G --outcome N --fired-ids a,b,c
crabpath replay --graph G --sessions S
crabpath health --graph G
crabpath merge --graph G
crabpath connect --graph G
crabpath journal [--stats]
```

## Paper

https://jonathangu.com/crabpath/
