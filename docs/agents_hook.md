# OpenClawBrain Agents Hook

## When to QUERY

Before answering user-facing questions about prior work, context, incidents, preferences, or recurring procedures, query the brain first and include chat/session IDs so fired nodes are logged. Query results should be treated as retrieval context, not as a final truth source.

## When to LEARN

After every response, use the fired node IDs from the most recent query:
- `learn --outcome 1.0` for successful responses.
- `learn --outcome -1.0` for incorrect responses.
- `learn --outcome 0.0` for neutral interactions where neither reinforcement nor penalty is desired.

Always pass the same fired IDs you used to generate the answer so the graph receives a grounded credit signal.

## When to SELF-LEARN

Run self-learn when the agent detects its own success/failure during runtime:
- Negative outcome: inject a correction so the bad route is immediately suppressed.
- Positive outcome: reinforce the good route and add supportive memory.

Use `self-learn` when the agent itself is the evaluator (autonomy mode) and fired IDs are available.

## When to INJECT

Inject when you need to add knowledge that is not already in workspace files, including:
- new runbook rules discovered from incidents
- one-off vendor behaviors
- cross-session fixes that should be available globally

Use `DIRECTIVE` for persistent operating instructions and `TEACHING` for factual updates.  
Use `CORRECTION` only for known wrong paths you want to suppress.

## Edit Files vs INJECT

- Edit workspace files when information should be durable, discoverable, and versioned with the repository or agent docs.
- Inject when knowledge is episodic, high-cardinality, experiment-specific, or not appropriate for permanent file changes.
- If a lesson becomes stable over time, promote it into source workspace docs and run `sync`.
