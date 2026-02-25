"""
🦀 CrabPath — Neuron-inspired memory graphs for AI agents.
Everything evolves into this.
"""

__version__ = "0.2.0"

from .graph import Graph, Node, Edge
from .activation import activate, learn, Firing

__all__ = ["Graph", "Node", "Edge", "activate", "learn", "Firing"]
