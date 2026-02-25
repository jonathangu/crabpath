"""
🦀 CrabPath: The Graph is the Prompt.

LLM-guided memory traversal with learned pointer weights
and corrected policy gradients.

CLI:
  python -m crabpath.cli
  crabpath  # via console_scripts entry point

Paper: https://jonathangu.com/crabpath/
"""

__version__ = "0.4.0"

from .graph import Graph, Node, Edge
from .activation import activate, learn, Firing
from .embeddings import EmbeddingIndex, openai_embed
from .adapter import CrabPathAgent, OpenClawCrabPathAdapter
from .feedback import auto_outcome, map_correction_to_snapshot
from .neurogenesis import (
    BLOCKED_QUERIES,
    NeurogenesisConfig,
    NoveltyResult,
    assess_novelty,
    connect_new_node,
    deterministic_auto_id,
)
from .mitosis import (
    MitosisConfig,
    MitosisState,
    SplitResult,
    split_node,
    split_with_llm,
    check_reconvergence,
    merge_and_resplit,
    bootstrap_workspace,
    mitosis_maintenance,
)

__all__ = [
    "Graph", "Node", "Edge",
    "activate", "learn", "Firing",
    "EmbeddingIndex", "openai_embed",
    "CrabPathAgent",
    "OpenClawCrabPathAdapter",
    "BLOCKED_QUERIES", "NeurogenesisConfig", "NoveltyResult",
    "assess_novelty", "connect_new_node", "deterministic_auto_id",
    "auto_outcome", "map_correction_to_snapshot",
    "MitosisConfig", "MitosisState", "SplitResult",
    "split_node", "split_with_llm", "check_reconvergence",
    "merge_and_resplit", "bootstrap_workspace", "mitosis_maintenance",
]
