"""Backward-compatible CLI entry point."""
import warnings
warnings.warn(
    "The 'crabpath' CLI is deprecated. Use 'openclawbrain' or 'ocb' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from openclawbrain.cli import main  # noqa: F401
