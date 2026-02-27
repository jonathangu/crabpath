---
name: crabpath
description: Memory graph engine with caller-provided embed and LLM callbacks; core is pure.
metadata:
  openclaw:
    emoji: "🦀"
    requires:
      python: ">=3.10"
---

# CrabPath

Pure graph core: zero deps, zero network calls. Caller provides callbacks.

## Design Tenets

- No network calls in core
- No secret discovery (no dotfiles, keychain, or env probing)
- No subprocess provider wrappers
- Embedder identity in state metadata; dimension mismatches are errors
- One canonical state format (`state.json`)

## Quick Start

```python
from crabpath import split_workspace, HashEmbedder, VectorIndex

graph, texts = split_workspace("./workspace")
embedder = HashEmbedder()
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, embedder.embed(content))
```

## Embeddings and LLM callbacks

- Default: `HashEmbedder` (hash-v1, 1024-dim)
- Real: callback `embed_fn` / `embed_batch_fn` (e.g., `text-embedding-3-small`)
- LLM routing: callback `llm_fn` using `gpt-5-mini` (example)

## Session Replay

`replay_queries(graph, queries)` can warm-start from historical turns.

## CLI

`--state` is preferred:

`crabpath query TEXT --state S [--top N] [--json]`

`--graph`/`--index` flags still supported for backward compatibility.

## Paper

https://jonathangu.com/crabpath/
