"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.a
"""

from __future__ import annotations

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
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame, PipelineResult


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
        calib_path: str = "viki/capture/calibration_results.npz",
        hand: Literal["right", "left"] = "right",
        arm_only: bool = False,
    ) -> None:
        self._calibrator = calibrator

        self._cache = UndistortCache()
        self._detector = HandDetector(hand=hand, arm_only=arm_only)
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
        print(f"DEBUG: Pipeline process called with {len(group.frames)} frames")
        detections: dict[str, HandDetection | None] = {}
        lms_3d: dict[str, Landmarks3D | None] = {}

        # Process all frames in the group
        for dev_id, frame in group.frames.items():
            print(f"DEBUG: Processing device {dev_id}")
            prepared = self._prepare_camera(dev_id, group)
            if prepared is None:
                print(f"DEBUG: Pipeline: Preparation failed for {dev_id}")
                detections[dev_id] = None
                lms_3d[dev_id] = None
                continue
            
            det = self._detector.detect(prepared)
            print(f"DEBUG: Detector result for {dev_id}: {det is not None}")
            detections[dev_id] = det
            lms_3d[dev_id] = self._lift_camera(dev_id, group, det)

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
        
        print(f"DEBUG: Fusion result: {fused is not None}")
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

        # Re-get intrinsics for deprojection
        intrinsics = self._calibrator.get_intrinsics(device_id)
        if intrinsics is None:
            return None

        # We need the prepared frame for depth data. 
        # To avoid re-preparing, we can just prepare it again here (it's cached).
        prepared = self._prepare_camera(device_id, group)
        if prepared is None:
            return None
            
        return lift_to_3d(detection, prepared)
