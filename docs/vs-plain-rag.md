# Why OpenClawBrain over plain RAG

| Feature | Plain RAG (BM25/vector) | OpenClawBrain |
|---------|------------------------|---------------|
| Retrieval method | Top-k similarity | Learned graph traversal |
| Learning | None | Policy gradient from outcomes |
| Corrections | Manual re-embedding | Inhibitory edges, instant |
| Context efficiency | Fixed top-k | Adapts: 30→2.7 nodes over time |
| Self-learning | No | self_learn API |
| Self-regulation | No | Homeostatic decay + synaptic scaling |
| Node splitting | No | Runtime split/merge lifecycle |
| Cold start | Strong (BM25 wins) | Needs warm-up queries |
| Best for | One-shot diverse queries | Repeated procedural tasks with feedback |

OpenClawBrain and plain RAG both start from lexical and vector similarity, but they diverge after the first retrieval pass. Plain RAG stays nearest-neighbor centered: for each query it scores candidates, selects a static top-k, and returns similar chunks with little memory of what happened before. That behavior is reliable early, especially during cold start, where there is not yet enough interaction history to trust learned structure. In that phase, a plain BM25/vector baseline can outperform a brand-new adaptive system because it does not depend on prior outcomes.

OpenClawBrain adds a learned graph layer that changes traversal choices over time. Successful paths become slightly more likely through policy-gradient updates, and failed paths decay or are actively suppressed through inhibitory edges and maintenance. This lets it compress useful context and avoid repetitive noise, especially for tasks repeated across sessions. The adaptive policy also improves long-run context efficiency: it can shrink retrieval breadth as confidence increases, while preserving a rich set of source nodes behind the scenes.

The practical consequence is simple: choose plain RAG when you need immediate broad retrieval performance with minimal warm-up. Choose OpenClawBrain when you have recurring query patterns, enough outcome signals, and a feedback loop (inject, learn, self-learn) that can continuously tune behavior. In production terms, plain RAG is a fast start; OpenClawBrain is better with feedback-rich repeated workflows.
