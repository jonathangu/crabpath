"""CrabPath v3 public API."""

from .graph import Edge, Graph, Node
from .index import VectorIndex
from .split import split_workspace
from .traverse import TraversalConfig, TraversalResult, TraversalStep, traverse
from .learn import LearningConfig, apply_outcome, hebbian_update
from ._batch import batch_or_single, batch_or_single_embed
from .decay import DecayConfig, apply_decay
from .autotune import GraphHealth, autotune, measure_health
from .store import load_state, save_state

__all__ = [
    "Node",
    "Edge",
    "Graph",
    "VectorIndex",
    "TraversalConfig",
    "TraversalStep",
    "TraversalResult",
    "LearningConfig",
    "batch_or_single",
    "batch_or_single_embed",
    "DecayConfig",
    "GraphHealth",
    "split_workspace",
    "traverse",
    "apply_outcome",
    "hebbian_update",
    "apply_decay",
    "measure_health",
    "autotune",
    "save_state",
    "load_state",
]

__version__ = "3.0.0"
