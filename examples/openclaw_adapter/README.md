# OpenClaw adapter example

OpenClaw agents often inject API keys in-process (for example through their framework),
while subprocesses may not inherit those keys reliably.

Use the Python API with callback-based callbacks for both embeddings and
LLM-powered chunking/summarization. In this pattern, you do **not** shell out to
`crabpath` CLI for embedding or LLM operations.

Run:

```bash
OPENAI_API_KEY=... python3 examples/openclaw_adapter/init_agent_brain.py <workspace> <sessions> <output>
python3 examples/openclaw_adapter/query_brain.py <state.json> "How does this work?"
```

`init_agent_brain.py` replays historical sessions into the graph, embeds workspace
chunks with OpenAI (`text-embedding-3-small`), stores `state.json` metadata
(`embedder_name`, `embedder_dim`), and prints a health summary.

`query_brain.py` loads `state.json`, embeds a live query with
`text-embedding-3-small`, traverses the graph, and prints fired nodes/context.
