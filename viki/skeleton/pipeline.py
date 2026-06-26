"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.a
"""

from __future__ import annotations

from time import sleep
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

import numpy as np

from viki.capture.base import SyncedFrameGroup
from viki.calibration.manager import CalibrationManager
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.fusion import fuse, load_extrinsics
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.hand_detector import HandDetector
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame, PipelineResult, HandDetection, PreparedFrame


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
        calib_path: str = "viki/capture/calibration_results.npz", #TODO move to loading calibration from data/intrinsics_calibration.json
        hand: Literal["right", "left"] = "right", #TODO move hand and mirrored configuration from multiple files (defined in multiple files (hand_detector.py and pipeline.py))
    ) -> None:
        self._calibrator = calibrator
        self._cache = UndistortCache()
        self._detector = HandDetector(hand=hand, mode="live")
        self._R, self._T = load_extrinsics(calib_path)


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
            lms_3d[dev_id] = self._lift_camera(dev_id, group, det)
            # logger.debug(f"result frame of {dev_id}: prepared: {prepared is not None}, detection: {det is not None}, lifted to 3D: {lms_3d[dev_id] is not None}")
 
        # Fusion logic:

        # Master camera is the first device in the group.
        # Subordinate camera is the second device (if available).
        dev_ids = list(group.frames.keys())
        if not dev_ids:
            return PipelineResult(fused_frame=None, detections={})

        master_id = dev_ids[0]
        lm0 = lms_3d.get(master_id)
        
        if len(dev_ids) >= 2:
            sub_id = dev_ids[1]
            lm1 = lms_3d.get(sub_id)
            fused = fuse(lm0, lm1, self._R, self._T, group.sync_timestamp_us)
        else:
            # Single camera case: just use the first camera as origin
            fused = fuse(lm0, None, self._R, self._T, group.sync_timestamp_us)
        
        return PipelineResult(fused_frame=fused, detections=detections)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    def __enter__(self) -> "SkeletonPipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _prepare_camera(self, device_id: str, group: SyncedFrameGroup) -> Optional[PreparedFrame]:
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
        self, device_id: str, group: SyncedFrameGroup, detection: Optional[HandDetection]
    ) -> Optional[Landmarks3D]:
        """Stage 3: lift 2D detection to 3D."""
        if detection is None:
            return None

        # Use the same preparation logic as _prepare_camera to ensure we have
        # a fallback K matrix if calibration is missing.
        prepared = self._prepare_camera(device_id, group)
        if prepared is None:
            return None
            
        return lift_to_3d(detection, prepared)
