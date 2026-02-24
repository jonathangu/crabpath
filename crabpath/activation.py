"""
CrabPath Activation Engine — Spreading activation over the memory graph.

Two-phase retrieval:
  Phase 1 (Propagation): Spread activation using summaries only (cheap)
  Phase 2 (Loading): Load full content for top-K activated nodes (expensive)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .graph import EdgeType, MemoryGraph, MemoryNode


@dataclass
class ActivationResult:
    """Result of an activation query."""
    query: str
    activated_nodes: list[tuple[MemoryNode, float]]  # (node, activation_score)
    inhibited_nodes: list[str]  # node IDs that were suppressed
    traversal_path: list[str]   # ordered node IDs visited
    tier: str  # "reflex", "habitual", or "deliberative"
    hops: int
    cost_estimate: float = 0.0


@dataclass 
class ActivationConfig:
    """Configuration for activation propagation."""
    alpha: float = 0.85          # damping factor (< 1 for convergence)
    max_hops: int = 4            # maximum propagation depth
    top_k: int = 10              # number of nodes to return
    threshold: float = 0.01      # minimum activation to propagate
    convergence_eps: float = 1e-4  # convergence criterion
    refractory_decay: float = 0.5  # suppression factor for recently-used nodes
    inhibition_strength: float = 0.8  # how strongly inhibitory edges suppress


class ActivationEngine:
    """
    Spreading activation over a CrabPath MemoryGraph.
    
    Implements the core CrabPath activation dynamics:
      x_{t+1} = σ(α · W⁺ᵀ · x_t + b(q)) ⊙ (1 - σ(W⁻ᵀ · x_t))
    
    Where:
      - W⁺ are excitatory edge weights
      - W⁻ are inhibitory edge weights  
      - b(q) is the query-dependent seed activation
      - α is the damping factor
      - σ is a nonlinearity (sigmoid or ReLU)
    """

    def __init__(self, graph: MemoryGraph, config: Optional[ActivationConfig] = None):
        self.graph = graph
        self.config = config or ActivationConfig()
        self._refractory: dict[str, float] = {}  # node_id → suppression level

    def activate(
        self,
        query: str,
        seed_nodes: Optional[list[str]] = None,
        context: Optional[dict] = None,
    ) -> ActivationResult:
        """
        Propagate activation through the graph for a given query.
        
        Args:
            query: The task/query string
            seed_nodes: Optional explicit seed nodes (if None, uses heuristics)
            context: Optional context dict (repo, time, active tasks, etc.)
            
        Returns:
            ActivationResult with top-K activated nodes and metadata
        """
        # Phase 0: Check reflex cache
        # TODO: implement myelinated reflex lookup
        
        # Phase 1: Seed activation
        activations: dict[str, float] = {}
        
        if seed_nodes:
            for nid in seed_nodes:
                node = self.graph.get_node(nid)
                if node and not node.quarantined:
                    activations[nid] = 1.0
        else:
            # Heuristic seeding: high-prior nodes + tag match
            # TODO: replace with embedding similarity + intent classifier
            for nid, node in self.graph._nodes.items():
                if node.quarantined:
                    continue
                # Simple keyword match for now
                query_lower = query.lower()
                tag_match = any(t.lower() in query_lower for t in node.tags)
                content_match = any(
                    word in node.summary.lower()
                    for word in query_lower.split()
                    if len(word) > 3
                )
                if tag_match or content_match:
                    activations[nid] = node.prior + (0.5 if tag_match else 0.2)
        
        if not activations:
            return ActivationResult(
                query=query,
                activated_nodes=[],
                inhibited_nodes=[],
                traversal_path=[],
                tier="deliberative",
                hops=0,
            )

        # Phase 2: Propagation
        inhibited = set()
        traversal = list(activations.keys())
        
        for hop in range(self.config.max_hops):
            new_activations: dict[str, float] = {}
            
            for nid, act in activations.items():
                if act < self.config.threshold:
                    continue
                
                # Get outgoing edges
                edges = self.graph.get_edges_from(nid)
                
                for edge in edges:
                    target = edge.target
                    target_node = self.graph.get_node(target)
                    
                    if not target_node or target_node.quarantined:
                        continue
                    
                    if edge.edge_type == EdgeType.INHIBITION:
                        # Inhibitory edge: suppress target
                        inhibited.add(target)
                        continue
                    
                    # Excitatory propagation
                    propagated = (
                        self.config.alpha
                        * act
                        * max(edge.weight, 0)
                    )
                    
                    # Apply refractory suppression
                    if target in self._refractory:
                        propagated *= (1 - self._refractory[target])
                    
                    current = new_activations.get(target, 0.0)
                    new_activations[target] = current + propagated
                    
                    if target not in traversal:
                        traversal.append(target)
            
            # Check convergence
            delta = sum(
                abs(new_activations.get(k, 0) - activations.get(k, 0))
                for k in set(list(new_activations.keys()) + list(activations.keys()))
            )
            
            # Merge activations
            for nid, act in new_activations.items():
                activations[nid] = activations.get(nid, 0) + act
            
            if delta < self.config.convergence_eps:
                break
        
        # Phase 3: Apply inhibition and select top-K
        for nid in inhibited:
            if nid in activations:
                activations[nid] *= (1 - self.config.inhibition_strength)
        
        # Sort by activation, take top-K
        sorted_nodes = sorted(
            activations.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self.config.top_k]
        
        result_nodes = []
        for nid, score in sorted_nodes:
            node = self.graph.get_node(nid)
            if node:
                result_nodes.append((node, score))
                # Update access stats
                node.access_count += 1
                import time
                node.last_accessed = time.time()
        
        # Update refractory periods
        for nid, _ in sorted_nodes:
            self._refractory[nid] = self.config.refractory_decay
        
        # Decay existing refractory periods
        to_remove = []
        for nid in self._refractory:
            self._refractory[nid] *= 0.9
            if self._refractory[nid] < 0.01:
                to_remove.append(nid)
        for nid in to_remove:
            del self._refractory[nid]
        
        return ActivationResult(
            query=query,
            activated_nodes=result_nodes,
            inhibited_nodes=list(inhibited),
            traversal_path=traversal,
            tier="habitual",  # TODO: determine based on reflex cache hit
            hops=hop + 1 if activations else 0,
        )

    def learn(
        self,
        result: ActivationResult,
        outcome: str,  # "success" or "failure"
        reward: float = 1.0,
        learning_rate: float = 0.1,
    ) -> None:
        """
        Update edge weights based on activation result and outcome.
        
        Hebbian learning: strengthen edges between co-activated nodes on success,
        weaken on failure. Strengthen inhibitory edges on failure.
        """
        is_success = outcome == "success"
        sign = 1.0 if is_success else -1.0
        
        # Update edges along the traversal path
        for i, nid in enumerate(result.traversal_path[:-1]):
            next_nid = result.traversal_path[i + 1]
            
            # Find connecting edges
            edges = self.graph.get_edges_from(nid)
            for edge in edges:
                if edge.target == next_nid:
                    # Hebbian update
                    delta = learning_rate * sign * reward
                    edge.weight = max(0.0, min(10.0, edge.weight + delta))
                    
                    if is_success:
                        edge.success_count += 1
                    else:
                        edge.failure_count += 1
                    
                    import time
                    edge.last_traversed = time.time()
        
        # On failure: strengthen inhibitory edges to activated nodes
        if not is_success:
            for node, score in result.activated_nodes:
                inhibitors = self.graph.get_inhibitors(node.id)
                for edge in inhibitors:
                    edge.weight -= learning_rate * reward  # more negative = stronger inhibition
