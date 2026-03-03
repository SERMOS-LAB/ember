"""
EMBER: Evacuation Mobility Behavior Inference.

A modular toolkit for inferring wildfire evacuation behaviors from GPS data.
"""

__version__ = "0.1.0"

from . import io
from . import zones
from . import behavior
from . import departure
from . import metrics

__all__ = [
    "io",
    "zones",
    "behavior",
    "departure",
    "metrics",
]
