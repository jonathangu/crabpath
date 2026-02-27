```md
# OpenClaw / CrabPath adapter hook

OPENCLAW_WORKSPACE=/path/to/workspace
OPENCLAW_SESSIONS=/path/to/sessions
OPENCLAW_OUTPUT=/path/to/output
OPENCLAW_STATE=${OPENCLAW_OUTPUT}/state.json

## Rebuild brain
python3 examples/openclaw_adapter/init_agent_brain.py "$OPENCLAW_WORKSPACE" "$OPENCLAW_SESSIONS" "$OPENCLAW_OUTPUT"

## Query (human-readable)
python3 examples/openclaw_adapter/query_brain.py "$OPENCLAW_STATE" "Where should I put deployment notes?"

## Query (JSON)
python3 examples/openclaw_adapter/query_brain.py "$OPENCLAW_STATE" "Where should I put deployment notes?" --json

## Learn from fired nodes
python3 - <<'PY'
from pathlib import Path
from crabpath import apply_outcome, load_state, save_state

state_path = Path("$OPENCLAW_STATE")
graph, index, meta = load_state(str(state_path))
apply_outcome(graph=graph, fired_nodes=[ "PASTE_NODE_ID", "PASTE_NODE_ID" ], outcome=1.0)
save_state(
    graph=graph,
    index=index,
    path=str(state_path),
    embedder_name=str(meta.get("embedder_name", "text-embedding-3-small")),
    embedder_dim=int(meta.get("embedder_dim", 1536)),
)
PY

## Health
python3 - <<'PY'
import json
from pathlib import Path

from crabpath import load_state
from crabpath.autotune import measure_health

state_path = Path("$OPENCLAW_STATE")
graph, _, _ = load_state(str(state_path))
health = measure_health(graph)
print(json.dumps({"nodes": graph.node_count(), "edges": graph.edge_count(), **health.__dict__}, indent=2))
PY
```
