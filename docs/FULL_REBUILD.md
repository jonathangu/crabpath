# Full OpenClawBrain Rebuild — Complete Learning Pipeline

A comprehensive brain rebuild that applies **all** learning mechanisms to historical conversation data.

## What This Does

Most brain operations are incremental — they learn from new queries as they come in. A **full rebuild** replays your entire conversation history through the graph with every learning mechanism active:

1. **Init** — GPT-5-mini smart splits + OpenAI embeddings
2. **Replay** — fire all historical queries through the graph to warm edges
3. **Auto-scoring** — `apply_outcome` on every query/response pair with heuristic outcome detection
4. **Corrections** — inject CORRECTION nodes from detected corrections in transcripts
5. **Teachings** — inject TEACHING nodes from lessons learned
6. **Maintenance** — full pipeline: health→decay→scale→split→merge→prune→connect (multiple passes)

## When to Use This

- **After major changes** to workspace structure or memory organization
- **Periodic refresh** (monthly?) to consolidate learning across all history
- **Migration** when switching embedders or adding new context files
- **Debugging** when the brain feels "stale" or missing obvious connections

## The Pipeline

### Phase 1: Init with LLM Splitting

```bash
openclawbrain init \
  --workspace ~/.openclaw/workspace \
  --output ~/.openclawbrain/main/state.json \
  --embedder openai \
  --llm openai \
  --json
```

**What it does:**
- Uses GPT-5-mini to intelligently split workspace files into semantic chunks
- Embeds all chunks with OpenAI `text-embedding-3-small` (dim=1536)
- Creates initial graph structure from workspace files

**Time:** ~30 min for 200 workspace files (326 LLM calls @ ~5 sec each)

### Phase 2: Replay All Sessions

Collect all session files:

```bash
# For main agent
SESSIONS=$(find ~/.openclaw/agents/main/sessions \
  -maxdepth 1 \( -name "*.jsonl" -o -name "*.jsonl.reset.*" -o -name "*.jsonl.deleted.*" \) \
  | grep -v ".lock$" | grep -v "sessions.json" | sort)

# For cross-agent context (e.g., bountiful in main sessions)
BOUNTIFUL_MAIN=$(grep -rl "5151316478" ~/.openclaw/agents/main/sessions/ | grep -v ".lock$")
BOUNTIFUL_OWN=$(find ~/.openclaw/agents/bountiful/sessions -maxdepth 1 -name "*.jsonl*" | grep -v ".lock$")
SESSIONS="$BOUNTIFUL_MAIN $BOUNTIFUL_OWN"
```

Clear the replay checkpoint:

```bash
python3 -c "
import json
with open('~/.openclawbrain/main/state.json') as f: d = json.load(f)
d['meta'].pop('last_replayed_ts', None)
with open('~/.openclawbrain/main/state.json', 'w') as f: json.dump(d, f)
"
```

Replay:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions $SESSIONS \
  --json
```

**What it does:**
- Extracts user queries and assistant responses from each session
- Fires each query through the graph (keyword seeding → traversal)
- Applies outcome weighting based on response quality heuristics
- Reinforces edges for co-activated nodes (Hebbian learning)
- Creates cross-file connections

**Output:**
```json
{
  "queries_replayed": 1899,
  "edges_reinforced": 57163,
  "cross_file_edges_created": 2617,
  "last_replayed_ts": 1772261254.121
}
```

### Phase 3: Extract & Inject Corrections/Teachings (LLM-Powered)

**This is the key difference from basic replay.**

Use GPT-5-mini to analyze each conversation and extract learning signals through
the built-in full-learning replay path:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions ~/.openclaw/agents/main/sessions \
  --full-learning \
  --json
```

**What it does:**

For each session transcript:
1. **LLM analyzes the conversation** (not regex patterns!)
2. **Detects corrections** — user says "no that's wrong", "actually X", "don't do Y"
   - Extracts the specific error and the correct answer
   - Severity-weighted: high/medium/low → -1.0/-0.5/-0.2 outcome
3. **Detects teachings** — user provides new facts, preferences, rules
   - Extracts the knowledge being taught
   - Severity-weighted: high/medium/low → +1.0/+0.5/+0.2 outcome
4. **Detects reinforcements** — user confirms good output
   - Identifies what was done right
5. **Injects CORRECTION/TEACHING nodes** with reinforcement signals

**Example LLM output:**
```json
{
  "corrections": [
    {
      "content": "ANAB is not Adage — they are different entities",
      "context": "User corrected entity confusion in financial data",
      "severity": "high"
    }
  ],
  "teachings": [
    {
      "content": "Don't include market cap for private companies in financial reports",
      "context": "User established a rule for report generation",
      "severity": "high"
    }
  ]
}
```

**Typical output:**
```
Analyzing 115 session files with LLM...
[1/115] session_abc.jsonl...
  Found: 3 corrections, 2 teachings, 1 reinforcements
[2/115] session_def.jsonl...
  Found: 1 corrections, 0 teachings, 2 reinforcements
...
{
  "sessions_analyzed": 115,
  "corrections_found": 342,
  "teachings_found": 187,
  "reinforcements_found": 94,
  "total_injected": 623
}
```

**Why this matters:**

The gap in basic replay is that it only sees **query patterns** (what files were accessed together). It never reads the **actual content** of the conversations. So when you said "no, ANAB is not Adage" or "don't include market cap for private companies," those corrections just flowed through as edge formations — the graph learned you talked about ANAB and Adage in the same session, but it didn't learn that they're different entities.

LLM-powered extraction captures the **semantic knowledge** from corrections and teachings, not just co-occurrence patterns.

### Phase 4: Maintenance (Multiple Passes)

Run 3 full maintenance cycles to let the graph stabilize:

```bash
for pass in 1 2 3; do
  openclawbrain maintain \
    --state ~/.openclawbrain/main/state.json \
    --tasks health,decay,scale,split,merge,prune,connect
done
```

**Task breakdown:**

| Task | What It Does | When to Run |
|------|-------------|-------------|
| `health` | Check orphans, distribution stats | Always (diagnostic) |
| `decay` | Apply temporal decay to edge weights | After each replay |
| `scale` | Synaptic scaling (normalize activations) | After large replays |
| `split` | Runtime node splitting (high-activation nodes) | After major growth |
| `merge` | Consolidate duplicate/similar nodes | Periodic cleanup |
| `prune` | Remove weak edges, dormant nodes | After decay cycles |
| `connect` | Create edges from TEACHING/CORRECTION nodes | After injections |

**Typical output (pass 1):**
```
Maintenance report:
  tasks: health, decay, scale, split, prune, merge, connect
  nodes: 4834 -> 5124
  edges: 9106 -> 11438
  merges: 23/50 candidates
  pruned: edges=2891 nodes=12
  decay_applied: True
  notes: split created 290 new nodes
```

### Phase 5: Slow Harvesting (Cross-Conversation Patterns)

Now implemented with `openclawbrain harvest`:

- Edge-density damping for correction-heavy nodes
- Connect/merge/split/prune/scale maintenance runs against the current graph

```bash
openclawbrain harvest \
  --state STATE \
  --events STATE_DIR/learning_events.jsonl \
  --tasks split,merge,prune,connect,scale \
  --json
```
This consumes `learning_events.jsonl` as a durable sidecar artifact and can run after `replay --fast-learning`.

### Phase 6: Final Health Check

```bash
openclawbrain health --state ~/.openclawbrain/main/state.json
```

**Healthy brain indicators:**

```
Brain health:
  Nodes: 5124
  Edges: 11438
  Reflex: 8.2%  Habitual: 61.3%  Dormant: 30.5%
  Orphans: 0
  Cross-file edges: 34.7%
```

- **No orphans** — every node is connected
- **30-40% cross-file edges** — good inter-context connections
- **Reflex 5-10%** — frequently activated core knowledge
- **Habitual 50-70%** — well-practiced patterns
- **Dormant < 40%** — not too much dead weight

## Full Script

```bash
#!/bin/bash
# Full learning suite: init + replay + LLM injections + 3x maintenance
set -e

STATE="~/.openclawbrain/main/state.json"
WORKSPACE="~/.openclaw/workspace"

echo "=== PHASE 1: INIT ==="
openclawbrain init \
  --workspace "$WORKSPACE" \
  --output "$STATE" \
  --embedder openai \
  --llm openai \
  --json

echo "=== PHASE 2: REPLAY (Edge Formation) ==="
python3 -c "
import json
with open('$STATE') as f: d = json.load(f)
d['meta'].pop('last_replayed_ts', None)
with open('$STATE', 'w') as f: json.dump(d, f)
"

SESSIONS=$(find ~/.openclaw/agents/main/sessions -maxdepth 1 \
  \( -name "*.jsonl" -o -name "*.jsonl.reset.*" \) | grep -v ".lock$" | sort)

echo "=== PHASE 3: LLM-POWERED FULL LEARNING ==="
openclawbrain replay --state "$STATE" --sessions $SESSIONS --full-learning --json

echo "=== PHASE 4: MAINTENANCE (3 passes) ==="
for pass in 1 2 3; do
  echo "Pass $pass/3..."
  openclawbrain harvest --state "$STATE" \
    --events "$(dirname "$STATE")/learning_events.jsonl" \
    --tasks split,merge,prune,connect,scale
done

echo "=== PHASE 5: FINAL HEALTH ==="
openclawbrain health --state "$STATE"
ls -lh "$STATE"
```

## Performance Notes

**Typical times** (Mac Mini M4 Pro, 64GB RAM):

| Brain | Workspace Files | Sessions | Queries | Total Time |
|-------|----------------|----------|---------|------------|
| Main (GUCLAW) | 186 | 115 | 1,899 | ~90 min |
| Pelican | 70 | 192 | 1,427 | ~45 min |
| Bountiful | 73 | 31 | 1,313 | ~35 min |

**Bottlenecks:**
- Init LLM splitting: ~5 sec per workspace file (sequential GPT-5-mini calls)
- Replay: ~0.5 sec per query (graph traversal + outcome application)
- Injections: ~0.2 sec per batched `inject_batch` insertion step
- Maintenance: ~10-30 sec per pass (depends on graph size)

**Parallelization:**
- Run multiple agents in parallel (main + pelican + bountiful) via `nohup`
- Each agent is independent — no shared state

## Troubleshooting

### "state file not found" after init

Init didn't finish. Check:
```bash
tail -50 /tmp/brain_init.log
ps aux | grep "openclawbrain init"
```

LLM splitting can be slow (~30 min for 200 files). Run with `nohup` to survive disconnects.

### Replay loads 0 interactions

Session files might be in a different format. Check:
```bash
head -1 ~/.openclaw/agents/main/sessions/SESSION_ID.jsonl
```

Expected: `{"type": "message", "message": {"role": "user", "content": "..."}}`

### Injections fail silently

If only transcript mining is needed first, run:

```bash
openclawbrain replay \
  --state STATE \
  --sessions SESSIONS \
  --fast-learning \
  --json
```

### Graph size explodes after multiple passes

Too many split nodes. Try:
```bash
openclawbrain maintain --state STATE --tasks merge,prune
```

Then verify:
```bash
openclawbrain health --state STATE
```

If dormant > 50%, prune more aggressively.

## Next Steps

After a full rebuild:

1. **Verify query quality** — test a few queries and check fired nodes
2. **Monitor performance** — watch `health` output over next few days
3. **Schedule periodic rebuilds** — monthly or after major workspace changes
4. **Backup** — `cp state.json state.json.backup` before experiments

## Related

- [Learning Pipeline](./LEARNING.md) — incremental learning from live queries
- [Maintenance Tasks](./MAINTENANCE.md) — task-by-task breakdown
- [Query Adapter](../examples/openclaw_adapter/) — how OpenClaw agents query the brain
