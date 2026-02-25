"""
🦀 CrabPath — Neuron-inspired memory graphs for AI agents.
Everything evolves into this.

CLI:
  python -m crabpath.cli
  crabpath  # via console_scripts entry point
"""

__version__ = "0.4.0"

from .graph import Graph, Node, Edge
from .activation import activate, learn, Firing
from .embeddings import EmbeddingIndex, openai_embed
from .adapter import OpenClawCrabPathAdapter
from .feedback import auto_outcome, map_correction_to_snapshot
from .neurogenesis import (
    BLOCKED_QUERIES,
    NeurogenesisConfig,
    NoveltyResult,
    assess_novelty,
    connect_new_node,
    deterministic_auto_id,
)

__all__ = [
    "Graph", "Node", "Edge",
    "activate", "learn", "Firing",
    "EmbeddingIndex", "openai_embed",
    "OpenClawCrabPathAdapter",
    "BLOCKED_QUERIES", "NeurogenesisConfig", "NoveltyResult",
    "assess_novelty", "connect_new_node", "deterministic_auto_id",
    "auto_outcome", "map_correction_to_snapshot",
]
