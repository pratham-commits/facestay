"""facestay: your face stays — smooth digital PTZ auto-framing.

Public API:
    AutoFramer      frame-in / frame-out processor (the main entry point)
    FramingConfig   tuning knobs (smoothing speeds, zoom targets, ...)
    FramingEngine   the pure-math crop controller (bring your own detector)
    Subject         detection input type for FramingEngine
"""

from .autoframer import AutoFramer
from .framer import FramingConfig, FramingEngine, FramingState, Subject

__all__ = [
    "AutoFramer",
    "FramingConfig",
    "FramingEngine",
    "FramingState",
    "Subject",
]

__version__ = "0.1.0"
