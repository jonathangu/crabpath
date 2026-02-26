---
name: crabpath
description: Memory graph engine with learned routing. Caller provides embed/LLM callbacks.
metadata:
  openclaw:
    emoji: "🦀"
    requires:
      python: ">=3.10"
---

# CrabPath — Memory Graph Engine

Pure graph engine for retrieval routing. Zero deps. Zero network calls. Caller provides everything.

## Install

```bash
pip install crabpath
```

## Quick Start

```python
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex

graph, texts = split_workspace("~/.openclaw/workspace")

# Caller provides embed — use whatever you have
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, your_embed_fn(content))

seeds = index.search(your_embed_fn("deploy to prod"), top_k=8)
result = traverse(graph, seeds)
apply_outcome(graph, result.fired, outcome=1.0)
```

## Batch Callbacks

```python
from crabpath._batch import batch_or_single, batch_or_single_embed

# One API call for all embeddings
vecs = batch_or_single_embed(
    list(texts.items()),
    embed_batch_fn=lambda texts: {nid: your_embed(t) for nid, t in texts}
)

# One API call for all LLM work (splitting, summaries, scoring)
results = batch_or_single(
    [{"id": "n0", "system": "summarize", "user": content}],
    llm_batch_fn=lambda reqs: [{"id": r["id"], "response": your_llm(r)} for r in reqs]
)
```

If batch fn not provided, CrabPath parallelizes single calls via ThreadPoolExecutor.

## CLI

```
crabpath init --workspace W --output O [--embed-command CMD]
crabpath query TEXT --graph G [--index I] [--query-vector-stdin] [--route-command CMD] [--embed-command CMD]
crabpath learn --graph G --outcome N --fired-ids a,b,c
crabpath replay --graph G --sessions S
crabpath health --graph G
crabpath merge --graph G
crabpath connect --graph G
crabpath journal [--stats]
```

## Local embeddings option

Install `crabpath[embeddings]` to enable local embeddings (`all-MiniLM-L6-v2`) by default.
If installed, `init` and `query` automatically use local embeddings when `--embed-command` is not supplied.

```bash
pip install crabpath
crabpath init --workspace ./ws --output ./data
crabpath query "how do i deploy" --graph ./data/graph.json --index ./data/index.json
```

## Paper

https://jonathangu.com/crabpath/
