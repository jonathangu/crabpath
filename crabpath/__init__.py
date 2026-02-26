"""
🦀 CrabPath: The Graph is the Prompt.

LLM-guided memory traversal with learned pointer weights
and corrected policy gradients.

CLI:
  python -m crabpath.cli
  crabpath  # via console_scripts entry point

Paper: https://jonathangu.com/crabpath/
"""

__version__ = "1.0.0"

from . import inhibition
from .adapter import CrabPathAgent, OpenClawCrabPathAdapter
from .autotune import (
    DEFAULTS,
    HEALTH_TARGETS,
    Adjustment,
    GraphHealth,
    autotune,
    measure_health,
    suggest_config,
)
from .embeddings import (
    EmbeddingIndex,
    auto_embed,
    cohere_embed,
    gemini_embed,
    ollama_embed,
    openai_embed,
)
from .feedback import (
    auto_feedback,
    auto_outcome,
    detect_correction,
    map_correction_to_snapshot,
    score_retrieval,
)
from .graph import Edge, Graph, Node
from .inhibition import (
    InhibitionConfig,
    apply_correction,
    get_inhibitory_edges,
    inhibition_stats,
    is_inhibited,
    score_with_inhibition,
)
from .legacy.activation import Firing, activate, learn
from .migrate import MigrateConfig, gather_files, migrate, parse_session_logs
from .mitosis import (
    BLOCKED_QUERIES,
    MergeResult,
    MitosisConfig,
    MitosisState,
    NeurogenesisConfig,
    NeurogenesisResult,
    NoveltyResult,
    SplitResult,
    assess_novelty,
    bootstrap_workspace,
    connect_new_node,
    create_node,
    deterministic_auto_id,
    find_co_firing_families,
    merge_nodes,
    mitosis_maintenance,
    should_create_node,
    should_merge,
    split_node,
    split_with_llm,
)
from .controller import ControllerConfig, MemoryController, QueryResult
from .shadow_logger import ShadowLog
from .synaptogenesis import (
    ProtoEdge,
    SynaptogenesisConfig,
    SynaptogenesisState,
    classify_tier,
    decay_proto_edges,
    edge_tier_stats,
    record_cofiring,
    record_correction,
    record_skips,
)

__all__ = [
    "Graph",
    "Node",
    "Edge",
    "activate",
    "learn",
    "Firing",
    "EmbeddingIndex",
    "auto_embed",
    "gemini_embed",
    "cohere_embed",
    "ollama_embed",
    "openai_embed",
    "CrabPathAgent",
    "OpenClawCrabPathAdapter",
    "BLOCKED_QUERIES",
    "NeurogenesisConfig",
    "NoveltyResult",
    "assess_novelty",
    "connect_new_node",
    "deterministic_auto_id",
    "auto_outcome",
    "auto_feedback",
    "detect_correction",
    "score_retrieval",
    "map_correction_to_snapshot",
    "ShadowLog",
    "migrate",
    "MigrateConfig",
    "gather_files",
    "parse_session_logs",
    "SynaptogenesisConfig",
    "SynaptogenesisState",
    "ProtoEdge",
    "record_cofiring",
    "record_skips",
    "record_correction",
    "decay_proto_edges",
    "classify_tier",
    "edge_tier_stats",
    "InhibitionConfig",
    "apply_correction",
    "score_with_inhibition",
    "is_inhibited",
    "get_inhibitory_edges",
    "inhibition_stats",
    "inhibition",
    "MitosisConfig",
    "MitosisState",
    "SplitResult",
    "MergeResult",
    "NeurogenesisResult",
    "split_node",
    "split_with_llm",
    "should_merge",
    "ControllerConfig",
    "MemoryController",
    "QueryResult",
    "should_create_node",
    "create_node",
    "find_co_firing_families",
    "merge_nodes",
    "bootstrap_workspace",
    "mitosis_maintenance",
    "DEFAULTS",
    "Adjustment",
    "GraphHealth",
    "HEALTH_TARGETS",
    "autotune",
    "measure_health",
    "suggest_config",
]
