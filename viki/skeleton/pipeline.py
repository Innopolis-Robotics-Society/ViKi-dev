"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.a
"""

from __future__ import annotations

from typing import Dict, Optional, Literal
import logging

from viki.calibration.models import CalibrationExtrinsics

logger = logging.getLogger(__name__)

import numpy as np

from viki.capture.base import SyncedFrameGroup
from viki.calibration.manager import CalibrationManager
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.fusion import fuse
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.detectors import (
    CompositeLandmarkDetector,
    FusionMode,
    MediaPipeArm,
)
from viki.skeleton.models import (
    Landmarks3D,
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
        Provides per-device intrinsics and extrinsics for prep, lift, fusion.
    detector : optional CompositeLandmarkDetector.
        If None, a default composite is created with only MediaPipeArm in
        FusionMode.ANY (arm-pose only configuration). To enable additional
        detectors later, build the composite explicitly and pass
        it here.
    hand : {"right", "left"}
        Which arm to track for the default detector. Ignored when `detector`
        is supplied explicitly.
    """

    def __init__(
        self,
        calibrator: CalibrationManager,
        detector: Optional[CompositeLandmarkDetector] = None,
        hand: Literal["right", "left"] = "right",
    ) -> None:
        self._calibrator = calibrator
        self._cache = UndistortCache()
        if detector is None:
            detector = CompositeLandmarkDetector(
                detectors=[MediaPipeArm(hand=hand, mode="live")],
                mode=FusionMode.ANY,
            )
        self._detector = detector
        self._ext_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def process(self, group: SyncedFrameGroup) -> PipelineResult:
        """
        Run the full pipeline on one SyncedFrameGroup.

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

    def _get_relative_extrinsics(
        self, master_id: str, subordinate_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the relative transform.

        Returns identity R and zero T if either device's extrinsics are missing.
        """
        key = (master_id, subordinate_id)
        cached = self._ext_cache.get(key)
        if cached is not None:
            return cached

        # I assume that extrinsics should be already uploaded in main module
        # However, keep this as fallback
        self._calibrator.load_extrinsics(master_id)
        self._calibrator.load_extrinsics(subordinate_id)

        ext_master = self._calibrator.get_extrinsics(master_id)
        ext_sub = self._calibrator.get_extrinsics(subordinate_id)
        if ext_master is None or ext_sub is None:
            logger.debug(
                "SkeletonPipeline: missing extrinsics for master=%s or sub=%s; "
                "falling back to identity R and zero T",
                master_id,
                subordinate_id,
            )
            R = np.eye(3, dtype=np.float64)
            T = np.zeros((3, 1), dtype=np.float64)
            # Do not cache the fallback so a later calibration write is picked up.
            return R, T

        R_master = np.asarray(ext_master.rotation_matrix, dtype=np.float64)
        R_sub = np.asarray(ext_sub.rotation_matrix, dtype=np.float64)
        t_master = np.asarray(ext_master.tvec, dtype=np.float64).reshape(3, 1)
        t_sub = np.asarray(ext_sub.tvec, dtype=np.float64).reshape(3, 1)

        R = R_master @ R_sub.T
        T = t_master - R @ t_sub
        self._ext_cache[key] = (R, T)
        return R, T

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
