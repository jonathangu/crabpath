"""CrabPath public API."""

from .graph import Edge, Graph, Node
from .index import VectorIndex
from ._batch import batch_or_single, batch_or_single_embed
from .autotune import GraphHealth, autotune, measure_health
from .decay import DecayConfig, apply_decay
from .learn import LearningConfig, apply_outcome
from .hasher import HashEmbedder, default_embed, default_embed_batch
from .score import score_retrieval
from .connect import apply_connections, suggest_connections
from .merge import apply_merge, suggest_merges
from .replay import replay_queries
from .split import generate_summaries, split_workspace
from .store import load_state, save_state
from .traverse import TraversalResult, TraversalConfig, traverse

__all__ = [
    "Node",
    "Edge",
    "Graph",
    "VectorIndex",
    "GraphHealth",
    "TraversalConfig",
    "TraversalResult",
    "HashEmbedder",
    "default_embed",
    "default_embed_batch",
    "DecayConfig",
    "LearningConfig",
    "measure_health",
    "autotune",
    "traverse",
    "split_workspace",
    "generate_summaries",
    "apply_outcome",
    "apply_decay",
    "batch_or_single",
    "batch_or_single_embed",
    "save_state",
    "load_state",
    "score_retrieval",
    "suggest_merges",
    "apply_merge",
    "suggest_connections",
    "apply_connections",
    "replay_queries",
]

__version__ = "8.0.0"
