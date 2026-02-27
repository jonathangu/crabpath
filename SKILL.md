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

`crabpath doctor --state S`
`crabpath info --state S|--graph G`

## Quick Reference

## API Reference

- Core lifecycle:
  - `split_workspace`
  - `load_state`
  - `save_state`
  - `ManagedState`
  - `VectorIndex`
- Traversal and learning:
  - `traverse`
  - `TraversalConfig`
  - `TraversalResult`
  - `apply_outcome`
- Runtime injection APIs:
  - `inject_node`
  - `inject_correction`
  - `inject_batch`
- Maintenance helpers:
  - `suggest_connections`, `apply_connections`
  - `suggest_merges`, `apply_merge`
  - `measure_health`, `autotune`, `replay_queries`
- Embedding utilities:
  - `HashEmbedder`
  - `default_embed`
  - `default_embed_batch`
- Graph primitives:
  - `Node`
  - `Edge`
  - `Graph`
  - `split_workspace`
  - `generate_summaries`

## CLI Commands

- `crabpath init --workspace W --output O [--sessions S]`
- `crabpath query TEXT --state S [--top N] [--json]`
- `crabpath learn --state S --outcome N --fired-ids a,b,c [--json]`
- `crabpath inject --state S --id NODE_ID --content TEXT [--type CORRECTION|TEACHING|DIRECTIVE] [--json]`
- `crabpath health --state S`
- `crabpath doctor --state S`
- `crabpath info --state S|--graph G`
- `crabpath replay --state S --sessions S`
- `crabpath merge --state S`
- `crabpath connect --state S`
- `crabpath journal [--stats]`

## Paper

https://jonathangu.com/crabpath/
