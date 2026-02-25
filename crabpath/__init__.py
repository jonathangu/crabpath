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
from .migrate import migrate, MigrateConfig, gather_files, parse_session_logs
from .synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    ProtoEdge,
    record_cofiring,
    record_skips,
    decay_proto_edges,
    classify_tier,
    edge_tier_stats,
)
from .autotune import (
    DEFAULTS,
    Adjustment,
    GraphHealth,
    HEALTH_TARGETS,
    autotune,
    measure_health,
    suggest_config,
)
from .mitosis import (
    MitosisConfig,
    MitosisState,
    SplitResult,
    MergeResult,
    NeurogenesisResult,
    split_node,
    split_with_llm,
    should_merge,
    should_create_node,
    create_node,
    find_co_firing_families,
    merge_nodes,
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
    "migrate", "MigrateConfig", "gather_files", "parse_session_logs",
    "SynaptogenesisConfig", "SynaptogenesisState", "ProtoEdge",
    "record_cofiring", "record_skips", "decay_proto_edges",
    "classify_tier", "edge_tier_stats",
    "MitosisConfig", "MitosisState", "SplitResult", "MergeResult", "NeurogenesisResult",
    "split_node", "split_with_llm", "should_merge", "should_create_node",
    "create_node", "find_co_firing_families", "merge_nodes",
    "bootstrap_workspace", "mitosis_maintenance",
    "DEFAULTS", "Adjustment", "GraphHealth", "HEALTH_TARGETS",
    "autotune", "measure_health", "suggest_config",
]
