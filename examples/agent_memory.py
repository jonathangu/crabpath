"""
CrabPath agent integration — the basic loop.

This shows how to wire CrabPath into a real agent:

  1. Load (or bootstrap) the graph
  2. Seed relevant nodes from a task
  3. Activate → get context
  4. Learn from outcome
  5. Save

Run: python examples/agent_memory.py
"""

from pathlib import Path
from crabpath import Graph, Node, Edge, activate, learn

GRAPH_PATH = "agent_memory.json"


# ── 1. Load or bootstrap ──────────────────────────────────────

def bootstrap() -> Graph:
    """Create a starter graph. In practice, you'd build this from
    your agent's corrections, tool traces, and known procedures."""
    g = Graph()

    # Knowledge nodes
    g.add_node(Node(id="check-config", content="git diff HEAD~1 -- config/"))
    g.add_node(Node(id="check-logs", content="tail -n 200 /var/log/svc.log"))
    g.add_node(Node(id="verify-prod", content="Test on prod before reporting fixed"))
    g.add_node(Node(id="restart-svc", content="systemctl restart app"))

    # Inhibitory nodes (things NOT to do)
    g.add_node(Node(id="no-untested", content="Never claim fixed without testing", threshold=0.5))
    g.add_node(Node(id="no-cache-excuse", content="Never tell user to clear cache", threshold=0.5))

    # Procedure: config change → check logs → verify on prod
    g.add_edge(Edge(source="check-config", target="check-logs", weight=1.5))
    g.add_edge(Edge(source="check-logs", target="verify-prod", weight=1.2))
    g.add_edge(Edge(source="check-logs", target="restart-svc", weight=0.8))

    # Inhibition: "no untested fix" blocks premature victory
    g.add_node(Node(id="claim-fixed", content="Tell user: it's fixed", threshold=2.0))
    g.add_edge(Edge(source="no-untested", target="claim-fixed", weight=-1.0))

    return g


if Path(GRAPH_PATH).exists():
    g = Graph.load(GRAPH_PATH)
    print(f"Loaded graph: {g}")
else:
    g = bootstrap()
    print(f"Bootstrapped new graph: {g}")


# ── 2. Seed from a task ───────────────────────────────────────

def seed_from_task(graph: Graph, task: str) -> dict[str, float]:
    """Turn a task description into seed node IDs.

    This is the simplest approach: keyword matching.
    Replace with embeddings, an LLM classifier, or whatever
    fits your agent. CrabPath doesn't care how you seed.
    """
    seeds = {}
    words = task.lower().split()
    for node in graph.nodes():
        text = f"{node.id} {node.content}".lower()
        score = sum(1.0 for w in words if w in text)
        if score > 0:
            seeds[node.id] = score
    return seeds


# ── 3. Activate → get context ─────────────────────────────────

task = "The deployment is broken, config was changed"
seeds = seed_from_task(g, task)
print(f"\nTask: {task}")
print(f"Seeds: {seeds}")

result = activate(g, seeds=seeds, reset=False)  # persistent warmth

print(f"\nFired ({result.steps} steps):")
for node, energy in result.fired:
    print(f"  [{energy:.2f}] {node.content}")

if result.inhibited:
    print(f"Inhibited: {result.inhibited}")

print(f"Timing: {result.fired_at}")

# In your agent, you'd inject result.fired contents into context:
context_for_llm = [node.content for node, _ in result.fired]


# ── 4. Learn from outcome ─────────────────────────────────────

# After the agent completes the task, signal the outcome:
learn(g, result, outcome=1.0)   # success: causal edges strengthen
# learn(g, result, outcome=-1.0)  # failure: causal edges weaken

print(f"\nAfter learning, check-config→check-logs weight: "
      f"{g.get_edge('check-config', 'check-logs').weight:.2f}")


# ── 5. Save ───────────────────────────────────────────────────

g.save(GRAPH_PATH)
print(f"\nSaved to {GRAPH_PATH}")


# ── Bonus: check what's warm ──────────────────────────────────

print("\nWarm nodes:")
for node, trace in g.warm_nodes():
    print(f"  {node.id}: trace={trace:.2f}")
