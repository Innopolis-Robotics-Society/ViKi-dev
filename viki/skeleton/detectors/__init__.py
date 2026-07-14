"""
viki.skeleton.detectors
-----------------------
Modular skeleton detection: a list of PartialLandmarkDetector instances
assembled by CompositeLandmarkDetector into one HandDetection per frame.
"""

from viki.skeleton.detectors.base import (
    FusionMode,
    PartialDetection2D,
    PartialLandmarkDetector,
)
from viki.skeleton.detectors.composite import CompositeLandmarkDetector
from viki.skeleton.detectors.rtmpose_wholebody import RTMPoseWholeBody

__all__ = [
    "CompositeLandmarkDetector",
    "FusionMode",
    "PartialDetection2D",
    "PartialLandmarkDetector",
    "RTMPoseWholeBody",
]
