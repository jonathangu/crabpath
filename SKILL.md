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

`crabpath init|query|learn|replay|health|merge|connect|journal`

## Paper

https://jonathangu.com/crabpath/
