"""
Most users only need: from crabpath import Graph, MemoryController

🦀 CrabPath: The Graph is the Prompt.

LLM-guided memory traversal with learned pointer weights
and corrected policy gradients.

CLI:
  python -m crabpath
  crabpath  # via console_scripts entry point

Paper: https://jonathangu.com/crabpath/
"""

__version__ = "2.3.1"

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
from .controller import ControllerConfig, MemoryController, QueryResult
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
from .graph import (
    ConsolidationConfig,
    ConsolidationResult,
    Edge,
    Graph,
    Node,
    consolidate,
    prune_orphan_nodes,
    prune_probationary,
    prune_weak_edges,
    should_split,
)
from .inhibition import (
    InhibitionConfig,
    apply_correction,
    get_inhibitory_edges,
    inhibition_stats,
    is_inhibited,
    score_with_inhibition,
)
from .learning import LearningConfig, LearningResult, RewardSignal, make_learning_step
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
    # --- Core (start here) ---
    "Graph",
    "Node",
    "Edge",
    "MemoryController",
    "QueryResult",
    "ControllerConfig",

    # --- Embeddings ---
    "EmbeddingIndex",
    "auto_embed",
    "openai_embed",
    "gemini_embed",
    "cohere_embed",
    "ollama_embed",

    # --- Learning ---
    "activate",
    "learn",
    "Firing",
    "LearningConfig",
    "LearningResult",
    "RewardSignal",
    "make_learning_step",

    # --- Advanced (internals) ---
    "ConsolidationConfig",
    "ConsolidationResult",
    "prune_orphan_nodes",
    "prune_weak_edges",
    "prune_probationary",
    "should_split",
    "consolidate",
    "should_merge",
    "CrabPathAgent",
    "OpenClawCrabPathAdapter",
    "migrate",
    "MigrateConfig",
    "gather_files",
    "parse_session_logs",
    "auto_feedback",
    "auto_outcome",
    "detect_correction",
    "score_retrieval",
    "map_correction_to_snapshot",
    "DEFAULTS",
    "Adjustment",
    "GraphHealth",
    "HEALTH_TARGETS",
    "autotune",
    "measure_health",
    "suggest_config",
    "InhibitionConfig",
    "apply_correction",
    "score_with_inhibition",
    "is_inhibited",
    "get_inhibitory_edges",
    "inhibition_stats",
    "MitosisConfig",
    "MitosisState",
    "SplitResult",
    "MergeResult",
    "NeurogenesisConfig",
    "NeurogenesisResult",
    "NoveltyResult",
    "assess_novelty",
    "connect_new_node",
    "deterministic_auto_id",
    "split_node",
    "split_with_llm",
    "should_merge",
    "create_node",
    "should_create_node",
    "find_co_firing_families",
    "merge_nodes",
    "bootstrap_workspace",
    "mitosis_maintenance",
    "SynaptogenesisConfig",
    "SynaptogenesisState",
    "ProtoEdge",
    "record_cofiring",
    "record_skips",
    "record_correction",
    "classify_tier",
    "decay_proto_edges",
    "edge_tier_stats",
    "ShadowLog",
    "BLOCKED_QUERIES",
]
