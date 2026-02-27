"""CrabPath is now OpenClawBrain. This shim provides backward compatibility."""
import warnings
warnings.warn(
    "The 'crabpath' package has been renamed to 'openclawbrain'. "
    "Please update your imports: 'import openclawbrain' instead of 'import crabpath'. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
from openclawbrain import *  # noqa: F401, F403
from openclawbrain import __all__, __version__
