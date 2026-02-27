# CrabPath

**CrabPath learns from your agent feedback, so wrong answers get suppressed instead of resurfacing.** It builds a memory graph over your workspace, remembers what worked, and routes future answers through learned paths.

- Zero dependencies. Pure Python 3.10+.
- Works offline with built-in hash embeddings.
- Builds a **`state.json`** brain from your workspace.
- Queries follow learned routes instead of only similarity matches.
- Positive feedback (`+1`) strengthens routes, negative (`-1`) creates inhibitory edges.
- Over time, less noise appears and recurring mistakes are less likely.

## Install

```bash
pip install crabpath
```

## 5-minute quickstart (A→B learning story)

```bash
# 1. Build a brain from the sample workspace
crabpath init --workspace examples/sample_workspace --output /tmp/brain

# 2. Check state health
crabpath doctor --state /tmp/brain/state.json
# output
# PASS: python_version
# PASS: state_file_exists
# PASS: state_json_valid
# Summary: 7/7 checks passed

# 3. Query (use --json once so you see node IDs)
crabpath query "how do I deploy" --state /tmp/brain/state.json --top 3 --json
# output (abbrev.)
# {"fired": ["deploy.md::0", "deploy.md::1", "deploy.md::2"], ...}

# 4. Teach it (good path)
crabpath learn --state /tmp/brain/state.json --outcome 1.0 --fired-ids "deploy.md::0,deploy.md::1"
# output
# {"edges_updated": 2, "max_weight_delta": 0.155}

# 5. Inject a correction
crabpath inject --state /tmp/brain/state.json \
  --id "fix::1" --content "Never skip CI for hotfixes" --type CORRECTION

# 6. Query again and see the route change
crabpath query "can I skip CI" --state /tmp/brain/state.json --top 3
# output
# fix::1
# ~~~~~~
# Never skip CI for hotfixes
# ...

# 7. Re-check health for a quick signal
crabpath health --state /tmp/brain/state.json
```

## What it looks like in practice

```bash
# Before learning
crabpath query "how do we handle incidents" --state /tmp/brain/state.json --top 3

# After one good learn on the best route
crabpath learn --state /tmp/brain/state.json --outcome 1.0 --fired-ids "incidents.md::0,deploy.md::1"

# After one negative learn on a bad route
crabpath learn --state /tmp/brain/state.json --outcome -1.0 --fired-ids "monitoring.md::2,incidents.md::0"

# Query again to observe new routing
crabpath query "incident runbook for deploy failures" --state /tmp/brain/state.json --top 4
```

## How it compares

| | Plain RAG | CrabPath |
|---|-----------|----------|
| Retrieval | Similarity search | Learned graph traversal |
| Feedback | None | `learn +1/-1` updates edge weights |
| Wrong answers | Can keep resurfacing | Inhibitory edges suppress them |
| Over time | Same results for same query | Routes become habitual behavior |
| Dependencies | Vector DB or service | Zero dependencies |

## Real embeddings + LLM routing (OpenAI)

Offline hash embeddings work for trying the product, but real deployments generally use:

- **Embeddings:** `text-embedding-3-small` (1536-dim)
- **LLM routing/scoring:** `gpt-5-mini`

```python
from openai import OpenAI
from crabpath import split_workspace, VectorIndex

client = OpenAI()


def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small", input=[text]
    ).data[0].embedding


def llm(system, user):
    return client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    ).choices[0].message.content

graph, texts = split_workspace("./workspace", llm_fn=llm)
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, embed(content))
```

See `examples/openai_embedder/` for a complete example.

## CLI Reference

| Command | What it does |
|---------|-------------|
| `init` | Build brain from workspace files |
| `query` | Search the graph |
| `learn` | Reinforce (+) or suppress (-) fired paths |
| `inject` | Add a node with connections (CORRECTION creates inhibitory edges) |
| `replay` | Warm-start from session history |
| `health` | Graph health metrics |
| `doctor` | Validate state integrity |
| `info` | Graph statistics |
| `merge` | Suggest/apply node merges |
| `connect` | Suggest/apply cross-file connections |
| `journal` | Query event journal |

## Teaching Your Graph

Use feedback and injections together:

- Positive learning: `learn --outcome 1.0 --fired-ids "A,B,C"` strengthens A→B and B→C.
- Negative learning: `learn --outcome -1.0 --fired-ids "A,B"` weakens bad routes and can create inhibitory edges.
- Inject correction: `inject --type CORRECTION` adds a correction node and connects it to related workspace nodes so it can actively suppress unsafe routes.
- Inject teaching: `inject --type TEACHING` adds a node with the same normal reinforcement behavior for safe routing additions.
- Inhibitory edges are simply negative-weight edges: they subtract activation from paths downstream, making them less likely to fire next time.

## Python API

```python
from crabpath import (
    split_workspace,
    traverse,
    apply_outcome,
    inject_node,
    inject_correction,
    inject_batch,
    VectorIndex,
    HashEmbedder,
    TraversalConfig,
    save_state,
    load_state,
    ManagedState,
    measure_health,
    replay_queries,
    score_retrieval,
)
```

## Optional: warm start from sessions

If you have prior conversation logs, replay them:

```bash
crabpath replay --state /tmp/brain/state.json --sessions ./sessions/
```

Skip this if you are just getting started.

## Production experience

Three brains run in production on a Mac Mini M4 Pro:

| Brain | Nodes | Edges | Learning Corrections | Sessions Replayed |
|-------|-------|-------|---------------------|-------------------|
| MAIN | 1,142 | 2,814 | 43 | 215 |
| PELICAN | 512 | 1,984 | 181 | 183 |
| BOUNTIFUL | 273 | 1,073 | 35 | 300 |

## Design Tenets

- No network calls in core.
- No secret discovery (no dotfiles, no keychain lookup).
- Embedder identity stored in state metadata; hard-fail on dimension mismatch.
- One canonical state format (`state.json`).

## Paper + links

[jonathangu.com/crabpath](https://jonathangu.com/crabpath/) — 8 deterministic simulations + production deployment data.

- PyPI: `pip install crabpath`
- GitHub: [jonathangu/crabpath](https://github.com/jonathangu/crabpath)
- ClawHub: `clawhub install crabpath`
- Benchmarks: `python3 benchmarks/run_benchmark.py` (deterministic per commit; timings vary by machine)
