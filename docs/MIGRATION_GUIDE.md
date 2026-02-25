# Migration Guide: workspace files → CrabPath graph

## The problem

Current agent prompts load workspace instruction files (`AGENTS.md`, `MEMORY.md`, `TOOLS.md`, `USER.md`, etc.) into every turn.

A typical workspace has 30K+ chars of static context. You pay the full token cost even when only one section is relevant. Every gate, every rule, every fact — loaded every turn.

## Phase 1: Shadow mode

Run CrabPath in parallel with your existing static files. Change nothing. Add one instruction to your agent:

> Before answering non-trivial questions, also query CrabPath and read the fired nodes as supplementary context.

The agent still has its normal files. CrabPath runs alongside. You compare what CrabPath surfaces vs what the agent actually needed. Log everything.

No risk. If CrabPath fires irrelevant nodes, ignore them. If it fires useful nodes the agent would have missed, that is the value.

```bash
# Bootstrap your workspace into a CrabPath graph
python3 scripts/bootstrap_from_workspace.py /path/to/workspace --output graph.json

# Build embeddings (requires OPENAI_API_KEY)
python3 -c "
from crabpath import Graph, EmbeddingIndex, openai_embed
g = Graph.load('graph.json')
idx = EmbeddingIndex()
idx.build(g, openai_embed())
idx.save('embeddings.json')
"
```

Run this once. The graph and embeddings persist on disk.

## Phase 2: Reduce static files

Once CrabPath consistently fires the right nodes, start trimming your workspace files:

| File | What moves to CrabPath | What stays |
|------|----------------------|------------|
| TOOLS.md | Codex workflow, model strategy, scripts | Nothing — all becomes nodes |
| MEMORY.md | Projects, people, cron jobs, learnings | Nothing — all becomes nodes |
| AGENTS.md | Procedures, gates, hygiene rules | Nothing — all becomes nodes |
| USER.md | Facts about the user | Nothing — all becomes nodes |
| SOUL.md | — | **Stays static** |
| Safety rules | — | **Stays static** |

Each section heading becomes a node. Cross-references become edges. The bootstrap script handles this automatically.

## Phase 3: Keep only the soul

What stays static forever:

- **SOUL.md** — identity, values, voice. This is who the agent is, not what it knows. Always needed.
- **Hard safety rules** — the "never" rules (no credentials on remote, no destructive commands). These must fire every turn.
- **Session context** — inbound metadata, channel info. Changes every message.

Everything else is in the graph.

```
Before:  SOUL (5.7K) + AGENTS (8.9K) + TOOLS (9.8K) + USER (2.6K) + MEMORY (3.7K) = 30.7K chars/turn
After:   SOUL (5.7K) + safety (1K) + CrabPath (~2K) = 8.7K chars/turn
Savings: 72% fewer tokens every turn
```

## The bootstrap script

```bash
python3 scripts/bootstrap_from_workspace.py /path/to/workspace --output graph.json
```

The script:

- reads all `.md` files under the workspace
- splits by `##`/`###` headings into nodes
- classifies each node type by text heuristics:
  - contains `Run:` or tool commands → `tool_call`
  - contains `never`, `always`, `must`, `do not` → `guardrail`
  - contains numbered steps → `procedure`
  - everything else → `fact`
- creates same-file edges at weight `0.6`
- creates cross-file reference edges at weight `0.4`
- writes a CrabPath graph JSON
- prints conversion stats

## Warm start weights

Every new node starts at `0.5`. The bootstrap script increases weight for sections that are frequently referenced by other sections. Over time, CrabPath's learning adjusts these weights from feedback.

## Expected results

On a real production workspace (181 files, 3,667 nodes):

- Static context: 1,656,670 chars loaded every turn
- CrabPath: ~1,500-2,000 chars loaded per turn (4-8 nodes)
- Reduction: 99.9%

Corrections propagate by edge weight updates. No manual file editing needed.
