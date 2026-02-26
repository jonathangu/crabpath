"""Backward-compatible consolidation API shim."""

from __future__ import annotations

from .graph import (
    ConsolidationConfig,
    ConsolidationResult,
    consolidate,
    prune_orphan_nodes,
    prune_probationary,
    prune_weak_edges,
    should_merge,
    should_split,
    split_node,
)

__all__ = [
    "ConsolidationConfig",
    "ConsolidationResult",
    "consolidate",
    "prune_orphan_nodes",
    "prune_probationary",
    "prune_weak_edges",
    "should_merge",
    "should_split",
    "split_node",
]
