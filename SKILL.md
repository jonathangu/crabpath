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

## Install

```bash
pip install crabpath
```

## Design Tenets

- No network calls in core (not even for model downloads)
- No secret discovery (no dotfiles, no keychain, no env probing)
- No subprocess provider wrappers
- Embedder identity stored in graph metadata; dimension mismatches are errors
- One canonical state format (state.json)

## Quick Start

```python
from crabpath import split_workspace, HashEmbedder, VectorIndex

graph, texts = split_workspace("./workspace")
embedder = HashEmbedder()
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, embedder.embed(content))
```

## Real embeddings callback

```python
from openai import OpenAI
from crabpath import split_workspace
from crabpath._batch import batch_or_single_embed

client = OpenAI()

def embed_batch(texts):
    ids, contents = zip(*texts)
    resp = client.embeddings.create(model="text-embedding-3-small", input=list(contents))
    return {ids[i]: resp.data[i].embedding for i in range(len(ids))}

graph, texts = split_workspace("./workspace")
vecs = batch_or_single_embed(list(texts.items()), embed_batch_fn=embed_batch)
```

## LLM callback

```python
def llm_fn(system, user):
    return client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
    ).choices[0].message.content

graph, texts = split_workspace("./workspace", llm_fn=llm_fn)
```

## Session Replay

`replay_queries(graph, queries)` can warm-start from historical turns.

## CLI

`crabpath init --workspace W --output O`
`crabpath query TEXT --state S [--top N] [--json]`
`crabpath learn --state S --outcome N --fired-ids a,b,c`
`crabpath health --state S`
`crabpath replay --state S --sessions S`
`crabpath merge --state S`
`crabpath connect --state S`
`crabpath journal [--stats]`

Legacy graph/index flags remain available:

`crabpath query TEXT --graph G [--index I] [--query-vector-stdin] [--top N] [--json]`
`crabpath learn --graph G --outcome N --fired-ids a,b,c`
`crabpath replay --graph G --sessions S`
`crabpath health --graph G`
`crabpath merge --graph G`
`crabpath connect --graph G`

## Paper

https://jonathangu.com/crabpath/
