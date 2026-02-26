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

Pure graph engine. Zero deps. Zero network calls. Caller provides everything via callbacks.

## Install

```bash
pip install crabpath
```

## Python API

```python
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex

graph, texts = split_workspace("~/.openclaw/workspace")

index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, your_embed_fn(content))

seeds = index.search(your_embed_fn("deploy"), top_k=8)
result = traverse(graph, seeds)
apply_outcome(graph, result.fired, outcome=1.0)
```

## Batch Callbacks

```python
from crabpath._batch import batch_or_single_embed

vecs = batch_or_single_embed(
    list(texts.items()),
    embed_batch_fn=lambda texts: {nid: your_embed(t) for nid, t in texts}
)
```

## CLI (pure graph ops)

```
crabpath init --workspace W --output O [--sessions S]
crabpath query TEXT --graph G [--index I] [--query-vector-stdin]
crabpath learn --graph G --outcome N --fired-ids a,b,c
crabpath replay --graph G --sessions S
crabpath health --graph G
crabpath merge --graph G
crabpath connect --graph G
crabpath journal [--stats]
```

## Paper

https://jonathangu.com/crabpath/
