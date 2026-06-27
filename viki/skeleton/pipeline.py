"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.a
"""

from __future__ import annotations

from typing import Dict, Optional, Literal
import logging

from viki.calibration.models import CalibrationExtrinsics, CalibrationIntrinsics

logger = logging.getLogger(__name__)

import numpy as np

from viki.capture.base import SyncedFrameGroup
from viki.calibration.manager import CalibrationManager
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.fusion import fuse
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.hand_detector import HandDetector
from viki.skeleton.models import (
    Landmarks3D,
    LM,
    SkeletonFrame,
    PipelineResult,
    HandDetection,
    PreparedFrame,
)


class SkeletonPipeline:
    """
    End-to-end skeleton detection from SyncedFrameGroup to SkeletonFrame.

    Parameters
    ----------
    calibrator : CalibrationManager
        Running calibrator. Used to read intrinsics per camera.
    calib_path : str
        Path to calibration_results.npz
    master_id : str
        Device ID of the master camera (world frame origin). Default: "kinect_0".
    subordinate_id : str
        Device ID of the subordinate camera. Default: "kinect_1".
    hand : {"right", "left"}
        Which hand to track.
    """

    def __init__(
        self,
        calibrator: CalibrationManager,
        hand: Literal[
            "right", "left"
        ] = "right",  # TODO move hand and mirrored configuration from multiple files (defined in multiple files (hand_detector.py and pipeline.py))
    ) -> None:
        self._calibrator = calibrator
        self._cache = UndistortCache()
        self._detector = HandDetector(hand=hand, mode="live")

    def process(self, group: SyncedFrameGroup) -> PipelineResult:
        """
        Run the full pipeline on one SyncedFrameGroup.

        Returns a PipelineResult containing the fused 3D frame and per-camera 2D detections.

        Parameters
        ----------
        group : SyncedFrameGroup
            Output of MultiCameraSync.get_synced_frame().

        Returns
        -------
        PipelineResult
        """
        detections: dict[str, HandDetection | None] = {}
        lms_3d: dict[str, Landmarks3D | None] = {}

        # Process all frames in the group
        for dev_id, frame in group.frames.items():
            # logger.debug(f"got frame from {dev_id}")
            prepared = self._prepare_camera(dev_id, group)
            if prepared is None:
                detections[dev_id] = None
                lms_3d[dev_id] = None
                continue

            det = self._detector.detect(prepared)
            detections[dev_id] = det
            if det is None:
                lms_3d[dev_id] = None
            else:
                lms_3d[dev_id] = lift_to_3d(det, prepared)
            # logger.debug(f"result frame of {dev_id}: prepared: {prepared is not None}, detection: {det is not None}, lifted to 3D: {lms_3d[dev_id] is not None}")

        dev_ids = group.device_ids
        if not dev_ids:
            return PipelineResult(fused_frame=None, detections={})

        extrinsics: Dict[str, CalibrationExtrinsics] = {}
        for dev_id in dev_ids:
            extr = self._calibrator.get_extrinsics(dev_id)
            if not extr:
                extrinsics[dev_id] = CalibrationExtrinsics()
            else:
                extrinsics[dev_id] = extr

        fused = fuse(dev_ids, lms_3d, extrinsics, group.sync_timestamp_us)

        return PipelineResult(fused_frame=fused, detections=detections)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    def __enter__(self) -> "SkeletonPipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _prepare_camera(
        self, device_id: str, group: SyncedFrameGroup
    ) -> Optional[PreparedFrame]:
        """Stage 1: prepare frame for detection."""
        frame = group.frames.get(device_id)
        if frame is None:
            logger.debug("SkeletonPipeline: no synced frames from SyncFrameGroup")
            return None

        intrinsics = self._calibrator.get_intrinsics(device_id)
        if intrinsics is None:
            # Fallback to identity-like intrinsics so we can still get 2D detections
            # This will result in slightly inaccurate 3D lifting but allows 2D viz
            K = np.eye(3, dtype=np.float32)
            dist = np.zeros(5, dtype=np.float32)
        else:
            K = intrinsics.camera_matrix
            dist = intrinsics.dist_coeffs

        return prepare_frame(frame, K, dist, self._cache)

    def _lift_camera(
        self, device_id: str, group: SyncedFrameGroup, detection: HandDetection
    ) -> Optional[Landmarks3D]:
        """Stage 3: lift 2D detection to 3D."""

        # Use the same preparation logic as _prepare_camera to ensure we have
        # a fallback K matrix if calibration is missing.
        prepared = self._prepare_camera(device_id, group)
        if prepared is None:
            return None

        return lift_to_3d(detection, prepared)
