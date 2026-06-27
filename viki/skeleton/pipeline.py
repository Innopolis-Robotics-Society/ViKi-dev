"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.a
"""

from __future__ import annotations
 
from concurrent.futures import ThreadPoolExecutor
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
import viki.config


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
        manager: CameraManager,
        hand: Literal["right", "left"] = viki.config.HAND_TO_DETECT,
    ) -> None:
        self._calibrator = calibrator
        self._manager = manager
        self._cache = UndistortCache()
        self._detectors: dict[str, HandDetector] = {}
        self._hand_type = hand
        self._R, self._T = load_extrinsics()
        self._executor = ThreadPoolExecutor(max_workers=4)


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
 
        # 1. Run detections in parallel across all cameras
        futures = {
            self._executor.submit(self._detect_camera, dev_id, group): dev_id 
            for dev_id in group.frames.keys()
        }
 
        for future in futures:
            dev_id, det, prepared = future.result()
            detections[dev_id] = det
            # 2. Lift to 3D (sequential, but fast)
            lms_3d[dev_id] = self._lift_camera(dev_id, group, det, prepared)
 
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

    def _detect_camera(self, dev_id: str, group: SyncedFrameGroup) -> tuple[str, Optional[HandDetection], Optional[PreparedFrame]]:
        """Helper for parallel detection."""
        prepared = self._prepare_camera(dev_id, group)
        if prepared is None:
            return dev_id, None, None
        
        if dev_id not in self._detectors:
            self._detectors[dev_id] = HandDetector(hand=self._hand_type, mode="video")
        
        det = self._detectors[dev_id].detect(prepared)
        return dev_id, det, prepared

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._executor.shutdown(wait=False)
        for detector in self._detectors.values():
            detector.close()

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
        self, device_id: str, group: SyncedFrameGroup, detection: Optional[HandDetection], prepared: Optional[PreparedFrame] = None
    ) -> Optional[Landmarks3D]:
        """Stage 3: lift 2D detection to 3D."""
        if detection is None:
            return None
 
        # Use the provided prepared frame, or re-prepare if missing
        if prepared is None:
            prepared = self._prepare_camera(device_id, group)
        
        if prepared is None:
            return None
            
        backend = self._manager.get_backend(device_id)
        return lift_to_3d(detection, prepared, backend)
