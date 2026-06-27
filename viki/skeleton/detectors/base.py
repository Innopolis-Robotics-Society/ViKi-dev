"""
viki.skeleton.detectors.base
----------------------------
Interfaces and shared value types for the modular skeleton detection.

A skeleton frame is built from N independent partial detectors:
each declares the slots it writes into (`indices`) and a `priority`
used by CompositeLandmarkDetector to resolve slot conflicts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from viki.skeleton.models import LM, PreparedFrame


class FusionMode(str, Enum):
    ANY = "any"  # at least one partial detector must succeed
    ALL = "all"  # every partial detector must succeed


@dataclass
class PartialDetection2D:
    """A single partial detector's pixel-space contribution."""

    indices: tuple[int, ...]  # global layout slots this detector writes (length k).
    px: np.ndarray  # (k, 2) float32 pixel coords (NaN allowed).
    lm_z_rel: np.ndarray  # (k,) float32 MediaPipe-style relative z.
    per_index_confidence: np.ndarray
    device_id: str
    timestamp_us: int


class PartialLandmarkDetector(ABC):
    """
    Abstract partial detector. Each implementation owns a fixed subset of
    the global skeleton layout described by class-level attributes.
    """

    name: str
    indices: tuple[int, ...]
    priority: int  # lower wins on slot conflicts

    @abstractmethod
    def detect(self, frame: PreparedFrame) -> PartialDetection2D | None:
        """
        Run detection on one frame.

        parameters
        ----------
        frame : prepared camera frame (RGB + depth + K).

        returns
        -------
        PartialDetection2D on success, None when this detector failed
        on this frame.
        """
        ...

    def close(self) -> None:
        """Release detector-owned resources. No-op by default."""
        return None
