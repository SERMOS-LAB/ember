"""
EMBER: Evacuation Modeling from Behavioral Evidence in Records.

A modular toolkit for inferring wildfire evacuation behaviors from GPS data.
"""

__version__ = "0.2.0"

from . import io
from . import zones
from . import behavior
from . import departure
from . import activities
from . import destination
from . import metrics
from . import pipeline

__all__ = [
    "io",
    "zones",
    "behavior",
    "departure",
    "activities",
    "destination",
    "metrics",
    "pipeline",
]
