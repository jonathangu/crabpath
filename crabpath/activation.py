"""
CrabPath Activation — Neuron-style firing over a memory graph.

The model (Leaky Integrate-and-Fire):
  1. Traces decay (time has passed since last activation)
  2. Seed nodes receive energy (potential increases)
  3. Each step: nodes whose potential >= threshold FIRE
  4. Firing sends energy along outgoing edges: weight × signal
     - Positive weight → excitatory (adds energy to target)
     - Negative weight → inhibitory (removes energy from target)
  5. Fired nodes: potential resets to 0 (refractory), trace refreshes
  6. Non-fired potentials decay each step (leak)
  7. Return fired nodes ranked by energy, with timing info

Learning is STDP-aware: edges in the causal direction (source fired
before target) get more credit than anti-causal edges.

Zero dependencies. Pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import Edge, Graph, Node


@dataclass
class Firing:
    """Result of an activation pass."""

    fired: list[tuple[Node, float]]  # (node, energy_at_firing), descending
    inhibited: list[str]  # node IDs driven below 0 by inhibition
    steps: int = 0
    fired_at: dict[str, int] = field(default_factory=dict)  # node_id -> step when it fired


def activate(
    graph: Graph,
    seeds: dict[str, float],
    *,
    max_steps: int = 3,
    decay: float = 0.1,
    top_k: int = 10,
    reset: bool = True,
    trace_decay: float = 0.1,
) -> Firing:
    """
    Fire neurons in the graph.

    Args:
        graph: The memory graph.
        seeds: {node_id: energy} — initial energy injection.
        max_steps: Maximum propagation rounds.
        decay: Fraction of potential that leaks each step (0 = no leak, 1 = full reset).
        top_k: Number of fired nodes to return.
        reset: If True (default), zero all potentials before starting.
            Set False to let energy linger across calls.
        trace_decay: Fraction of trace that decays each call (models time passing).

    Returns:
        Firing with ranked fired nodes, inhibited IDs, and timing info.
    """
    # Decay traces (time has passed since last activation)
    for node in graph.nodes():
        node.trace *= 1.0 - trace_decay

    # Optionally start clean
    if reset:
        graph.reset_potentials()

    # Inject seed energy
    for node_id, energy in seeds.items():
        node = graph.get_node(node_id)
        if node:
            node.potential += energy

    fired_record: dict[str, float] = {}  # node_id -> energy at firing
    fired_at: dict[str, int] = {}  # node_id -> step number
    inhibited: set[str] = set()
    steps_taken = 0

    for step in range(max_steps):
        # Find neurons ready to fire
        to_fire: list[Node] = []
        for node in graph.nodes():
            if node.id not in fired_record and node.potential >= node.threshold:
                to_fire.append(node)

        if not to_fire:
            break

        steps_taken = step + 1

        # Fire each neuron
        for node in to_fire:
            signal = node.potential
            fired_record[node.id] = signal
            fired_at[node.id] = step

            # Refresh trace on firing
            node.trace = signal

            # Send energy along outgoing edges
            for target, edge in graph.outgoing(node.id):
                target.potential += edge.weight * signal
                if target.potential < 0:
                    inhibited.add(target.id)

            # Refractory: reset potential
            node.potential = 0.0

        # Leak: decay unfired potentials
        for node in graph.nodes():
            if node.id not in fired_record and node.potential > 0:
                node.potential *= 1.0 - decay

    # Remove inhibited nodes from fired results
    for nid in inhibited:
        fired_record.pop(nid, None)
        fired_at.pop(nid, None)

    # Rank by energy at firing, take top_k
    ranked = sorted(fired_record.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for nid, score in ranked:
        node = graph.get_node(nid)
        if node:
            result.append((node, score))

    return Firing(
        fired=result,
        inhibited=sorted(inhibited),
        steps=steps_taken,
        fired_at=fired_at,
    )


def learn(
    graph: Graph,
    result: Firing,
    outcome: float,
    rate: float = 0.1,
    create_edges: bool = True,
    max_new_edges: int = 10,
) -> None:
    """
    STDP-aware Hebbian learning: adjust weights between co-fired nodes.

    Timing matters:
    - Causal edges (source fired before target) get full credit.
    - Anti-causal edges (target fired before source) get reverse, weaker effect.
    - Simultaneous firing: partial credit.

    Credit decays with temporal distance: closer in time = more credit.

    Positive outcome → strengthen causal edges, weaken anti-causal.
    Negative outcome → weaken causal edges, strengthen anti-causal.
    Weights clamped to [-10, 10].

    If create_edges is True, missing edges between co-fired nodes are added
    using a half-strength STDP update:

    initial_weight = rate * outcome * timing_factor * 0.5

    We only create new edges when abs(initial_weight) > 0.05, and we create
    at most max_new_edges of them, choosing the strongest first.

    Args:
        graph: The memory graph.
        result: A previous Firing result (with timing info).
        outcome: Positive = good, negative = bad.
        rate: Learning rate.
        create_edges: If True, create new edges for co-fired node pairs.
        max_new_edges: Maximum number of new edges to create per call.
    """
    fired_ids = [n.id for n, _ in result.fired]
    fa = result.fired_at
    new_edges: list[tuple[float, str, str, float]] = []

    for src_id in fired_ids:
        for tgt_id in fired_ids:
            if src_id == tgt_id:
                continue
            edge = graph.get_edge(src_id, tgt_id)

            # STDP timing factor
            if fa and src_id in fa and tgt_id in fa:
                dt = fa[tgt_id] - fa[src_id]
                if dt > 0:
                    # Causal: source fired before target
                    timing = 1.0 / (1.0 + dt)
                elif dt < 0:
                    # Anti-causal: target fired before source (weaker, reversed)
                    timing = -0.5 / (1.0 + abs(dt))
                else:
                    # Simultaneous: partial credit
                    timing = 0.5
            else:
                # No timing info: fall back to symmetric
                timing = 1.0

            if edge:
                edge.weight += rate * outcome * timing
                edge.weight = max(-10.0, min(10.0, edge.weight))
            elif create_edges:
                initial_weight = rate * outcome * timing * 0.5
                if abs(initial_weight) > 0.05:
                    new_edges.append((abs(initial_weight), src_id, tgt_id, initial_weight))

    if create_edges and new_edges:
        new_edges.sort(key=lambda x: x[0], reverse=True)
        for _, src_id, tgt_id, initial_weight in new_edges[:max_new_edges]:
            graph.add_edge(
                Edge(
                    source=src_id,
                    target=tgt_id,
                    weight=max(-10.0, min(10.0, initial_weight)),
                )
            )
