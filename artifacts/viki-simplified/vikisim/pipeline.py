"""Simplified SkeletonPipeline (mirrors viki/skeleton/pipeline.py)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .calibration import CalibrationManager
from .manager import CameraManager
from .models import (
    HandDetection,
    Landmarks3D,
    PipelineResult,
    PreparedFrame,
    SkeletonFrame,
)
from .sync import SyncedFrameGroup


class SkeletonPipeline:
    """End-to-end: SyncedFrameGroup -> per-camera 3D -> fused SkeletonFrame."""

    def __init__(
        self,
        calibrator: CalibrationManager,
        manager: CameraManager,
        hand: str = "right",
        depth_debug: bool = False,
    ) -> None:
        self.calibrator = calibrator
        self.manager = manager
        self._hand = hand
        self._depth_debug = depth_debug
        self._detectors: Dict[str, Any] = {}
        self._prev_hand_pos: Dict[str, Any] = {}

    def process(self, group: SyncedFrameGroup) -> PipelineResult:
        ...  # prepare -> detect -> lift_to_3d -> fuse

    def set_depth_debug(self, enabled: bool) -> None:
        self._depth_debug = enabled
