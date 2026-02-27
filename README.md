# CrabPath

Pure routing graph engine for context-aware retrieval. Core is zero deps and zero network; the caller supplies semantic and LLM callbacks.

## 1. CrabPath

CrabPath is a deterministic graph engine that builds traversable context graphs and improves routing from feedback, without any external service requirement by default.

## 2. Install

```bash
pip install crabpath
```

## Design Tenets

- No network calls in core (not even for model downloads)
- No secret discovery (no dotfiles, no keychain, no env probing)
- No subprocess provider wrappers
- Embedder identity stored in graph metadata; dimension mismatches are errors
- One canonical state format (state.json)

## 3. Quick Start

```python
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex
from crabpath import HashEmbedder

graph, texts = split_workspace("./workspace")
embedder = HashEmbedder()  # default hash-v1
index = VectorIndex()

for nid, content in texts.items():
    index.upsert(nid, embedder.embed(content))

query_vec = embedder.embed("how do I deploy to production?")
seeds = index.search(query_vec, top_k=8)
result = traverse(graph, seeds)
apply_outcome(graph, result.fired, outcome=1.0)
```

## 4. Real Embeddings

```python
from openai import OpenAI
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex
from crabpath._batch import batch_or_single_embed

client = OpenAI()

def embed_batch(texts):
    ids, contents = zip(*texts)
    resp = client.embeddings.create(model="text-embedding-3-small", input=list(contents))
    return {ids[i]: resp.data[i].embedding for i in range(len(ids))}

graph, texts = split_workspace("./workspace")
index = VectorIndex()
vecs = batch_or_single_embed(list(texts.items()), embed_batch_fn=embed_batch)
for nid, vec in vecs.items():
    index.upsert(nid, vec)
```

## 5. LLM Callbacks

```python
def llm_fn(system, user):
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    return resp.choices[0].message.content

# Pass to split for LLM splitting, or traverse for routing
graph, texts = split_workspace("./workspace", llm_fn=llm_fn)
```

## 6. Session Replay

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
crabpath replay --graph ./data/graph.json --sessions ./sessions/
```

## 7. Three Embedding Tiers

Capability | How to enable | Network | Dependencies
---|---|---|---
Default (hash-v1) | `HashEmbedder` shipped in core | no | none
Local semantic | `pip install crabpath[embeddings]` | optional (local model) | local embedding extras
Remote semantic | callback `embed_fn` / `embed_batch_fn` (OpenAI, Gemini, etc.) | caller-provided | caller-provided

## 8. CLI

```bash
crabpath init --workspace W --output O [--sessions S]
crabpath query TEXT --state S [--top N] [--json]
crabpath learn --state S --outcome N --fired-ids a,b,c
crabpath health --state S
crabpath replay --state S --sessions S
crabpath merge --state S
crabpath connect --state S
crabpath query TEXT --graph G [--index I] [--query-vector-stdin] [--top N] [--json] (legacy)
crabpath learn --graph G --outcome N --fired-ids a,b,c [--json] (legacy)
crabpath replay --graph G --sessions S (legacy)
crabpath health --graph G (legacy)
crabpath merge --graph G (legacy)
crabpath connect --graph G (legacy)
crabpath journal [--stats]
```

## 9. Reproduce Results

See [REPRODUCE.md](REPRODUCE.md).

## 10. Paper

https://jonathangu.com/crabpath/
