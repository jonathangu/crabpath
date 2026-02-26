#!/usr/bin/env python3
"""
CrabPath Edge Damping Simulation
=================================
Demonstrates that edge damping (synaptic fatigue) prevents traversal
from getting trapped in high-weight loops, producing broader context
retrieval compared to undamped traversal.

Scenario: An agent workspace memory graph with a natural loop.
Query: "How do I deploy the new feature?"

Without damping → traversal cycles through the same 2-3 hub nodes.
With damping (0.3) → traversal explores outward after each revisit.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from crabpath.graph import Graph, Node, Edge
from crabpath.traversal import TraversalConfig, traverse
from crabpath.router import Router


def build_workspace_graph() -> Graph:
    """Build a realistic agent memory graph with a natural hub-and-spoke
    structure plus a high-weight loop (deploy ↔ CI ↔ staging)."""

    g = Graph()

    # === Nodes: a small workspace memory ===
    nodes = [
        Node(id="deploy",    content="Deployment procedure: merge to main, CI runs, push to staging, then promote to prod.", summary="Deploy procedure"),
        Node(id="ci",        content="CI pipeline: lint → test → build → docker push. Runs on every push to main.", summary="CI pipeline"),
        Node(id="staging",   content="Staging environment: mirrors prod. Run smoke tests before promoting.", summary="Staging env"),
        Node(id="prod",      content="Production: blue-green deploy behind load balancer. Rollback within 5 min.", summary="Production"),
        Node(id="rollback",  content="Rollback: revert to previous Docker tag. Alert on-call if metrics degrade.", summary="Rollback procedure"),
        Node(id="feature",   content="Feature branching: create feature/xyz branch, PR review required, squash merge.", summary="Feature branches"),
        Node(id="testing",   content="Testing strategy: unit (pytest), integration (docker-compose), E2E (playwright).", summary="Testing strategy"),
        Node(id="monitoring",content="Monitoring: Datadog dashboards, PagerDuty alerts, SLO 99.9% uptime.", summary="Monitoring"),
        Node(id="secrets",   content="Secrets management: 1Password CLI for dev, Vault for prod. Never hardcode.", summary="Secrets mgmt"),
        Node(id="database",  content="Database migrations: Alembic for schema changes. Always backward-compatible.", summary="DB migrations"),
        Node(id="docker",    content="Docker: multi-stage builds, slim base images, layer caching for fast CI.", summary="Docker setup"),
        Node(id="k8s",       content="Kubernetes: 3-node cluster, HPA on CPU/memory, PodDisruptionBudgets.", summary="K8s config"),
    ]
    for n in nodes:
        g.add_node(n)

    # === Edges: natural workspace connections ===
    # The deploy→ci→staging→deploy loop is the trap — all high weight
    edges = [
        # The loop (high weight — agents follow these constantly)
        Edge(source="deploy",  target="ci",       weight=0.95),
        Edge(source="ci",      target="staging",  weight=0.90),
        Edge(source="staging", target="deploy",   weight=0.85),  # back-edge creates cycle
        Edge(source="staging", target="prod",     weight=0.80),

        # Spokes off the loop (moderate weight — useful but less traveled)
        Edge(source="deploy",  target="rollback", weight=0.60),
        Edge(source="deploy",  target="feature",  weight=0.55),
        Edge(source="ci",      target="testing",  weight=0.65),
        Edge(source="ci",      target="docker",   weight=0.50),
        Edge(source="prod",    target="monitoring",weight=0.70),
        Edge(source="prod",    target="rollback", weight=0.65),
        Edge(source="staging", target="monitoring",weight=0.45),

        # Deeper connections
        Edge(source="docker",  target="k8s",      weight=0.55),
        Edge(source="k8s",     target="monitoring",weight=0.50),
        Edge(source="testing", target="feature",  weight=0.40),
        Edge(source="secrets", target="deploy",   weight=0.35),
        Edge(source="database",target="staging",  weight=0.40),
        Edge(source="rollback",target="monitoring",weight=0.50),
    ]
    for e in edges:
        g.add_edge(e)

    return g


def run_sim(damping: float, label: str, graph: Graph) -> list[str]:
    """Run traversal with given damping and return visit order."""
    cfg = TraversalConfig(
        max_hops=15,
        episode_edge_damping=damping,
        episode_visit_penalty=0.0,
        branch_beam=3,
    )
    router = Router()  # heuristic fallback (no LLM needed)

    trajectory = traverse(
        query="How do I deploy the new feature?",
        graph=graph,
        router=router,
        config=cfg,
        seed_nodes=[("deploy", 1.0)],
    )

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  edge_damping = {damping}")
    print(f"{'='*60}")
    print(f"\n  Visit order ({len(trajectory.visit_order)} nodes):")
    for i, node_id in enumerate(trajectory.visit_order):
        marker = " ←loop" if node_id in ("deploy", "ci", "staging") and i > 2 else ""
        print(f"    {i+1:2d}. {node_id}{marker}")

    print(f"\n  Steps detail:")
    for step in trajectory.steps:
        damped = ""
        if step.effective_weight != step.edge_weight:
            damped = f"  (base {step.edge_weight:.2f} → damped {step.effective_weight:.2f})"
        print(f"    {step.from_node:12s} → {step.to_node:12s}  "
              f"w={step.effective_weight:.3f}  tier={step.tier}{damped}")

    unique = set(trajectory.visit_order)
    loop_nodes = {"deploy", "ci", "staging"}
    explored = unique - loop_nodes
    print(f"\n  Summary:")
    print(f"    Total hops:      {len(trajectory.steps)}")
    print(f"    Unique nodes:    {len(unique)}")
    print(f"    Loop revisits:   {sum(1 for n in trajectory.visit_order[3:] if n in loop_nodes)}")
    print(f"    Nodes outside loop: {sorted(explored)}")

    return trajectory.visit_order


def main():
    graph = build_workspace_graph()
    print(f"Graph: {graph.node_count} nodes, {graph.edge_count} edges")

    # Run WITHOUT damping (damping=1.0 means no decay)
    no_damp = run_sim(1.0, "NO DAMPING (control)", graph)

    # Run WITH damping (0.3 = production default)
    damped = run_sim(0.3, "EDGE DAMPING = 0.3 (production)", graph)

    # === Comparison ===
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")

    no_damp_unique = set(no_damp)
    damped_unique = set(damped)
    loop_nodes = {"deploy", "ci", "staging"}

    no_damp_revisits = sum(1 for n in no_damp[3:] if n in loop_nodes)
    damped_revisits = sum(1 for n in damped[3:] if n in loop_nodes)

    print(f"\n  {'Metric':<25s} {'No Damping':>12s} {'Damped (0.3)':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Total hops':<25s} {len(no_damp)-1:>12d} {len(damped)-1:>12d}")
    print(f"  {'Unique nodes visited':<25s} {len(no_damp_unique):>12d} {len(damped_unique):>12d}")
    print(f"  {'Loop revisits (after 3)':<25s} {no_damp_revisits:>12d} {damped_revisits:>12d}")
    print(f"  {'Nodes outside loop':<25s} {len(no_damp_unique - loop_nodes):>12d} {len(damped_unique - loop_nodes):>12d}")

    only_damped = damped_unique - no_damp_unique
    if only_damped:
        print(f"\n  Nodes discovered ONLY with damping: {sorted(only_damped)}")

    # Verdict
    print(f"\n  VERDICT:", end=" ")
    if len(damped_unique) > len(no_damp_unique) and damped_revisits < no_damp_revisits:
        print("✅ Edge damping explores more nodes with fewer loop revisits.")
    elif len(damped_unique) > len(no_damp_unique):
        print("✅ Edge damping discovers more unique nodes.")
    elif damped_revisits < no_damp_revisits:
        print("✅ Edge damping reduces loop revisits.")
    else:
        print("⚠️  No significant difference in this scenario.")

    print()


if __name__ == "__main__":
    main()
