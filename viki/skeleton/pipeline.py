"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.a
"""

from __future__ import annotations
 
from concurrent.futures import ThreadPoolExecutor
from time import sleep

from typing import Dict, Optional, Literal
import logging

from viki.calibration.models import CalibrationExtrinsics
from viki.capture.kinect import KinectBackend

logger = logging.getLogger(__name__)

import numpy as np

from viki.capture.base import SyncedFrameGroup
from viki.capture.manager import CameraManager
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
    LM,
)
import viki.config


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
        manager: CameraManager,
        hand: Literal["right", "left"] = viki.config.HAND_TO_DETECT,
    ) -> None:
        self._hand = hand
        self._calibrator = calibrator
        self._manager = manager
        self._cache = UndistortCache()
        self._detectors: dict[str, CompositeLandmarkDetector] = {}
        self._hand_type = hand
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Bone length EMA tracking for outlier rejection
        self._bone_emas: dict[tuple[LM, LM], float] = {}
        self._ema_alpha = 0.1
        self._tracked_bones = [
            (LM.SHOULDER, LM.ELBOW),
            (LM.ELBOW, LM.WRIST),
        ]


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

        # # Process all frames in the group
        # for dev_id, frame in group.frames.items():
        #     # logger.debug(f"got frame from {dev_id}")
        #     prepared = self._prepare_camera(dev_id, group)
        #     if prepared is None:
        #         detections[dev_id] = None
        #         lms_3d[dev_id] = None
        #         continue

        #     det = self._detector.detect(prepared)
        #     detections[dev_id] = det
        #     if det is None:
        #         lms_3d[dev_id] = None
        #     else:
        #         lms_3d[dev_id] = lift_to_3d(det, prepared)
        #     # logger.debug(f"result frame of {dev_id}: prepared: {prepared is not None}, detection: {det is not None}, lifted to 3D: {lms_3d[dev_id] is not None}")

        # dev_ids = group.device_ids
        if not dev_ids:
            return PipelineResult(fused_frame=None, detections={})

        extrinsics: Dict[str, CalibrationExtrinsics] = {}
        for dev_id in dev_ids:
            extr = self._calibrator.get_extrinsics(dev_id)
            if not extr:
                extrinsics[dev_id] = CalibrationExtrinsics()
            else:
                extrinsics[dev_id] = extr

        fused = fuse(dev_ids, lms_3d, extrinsics, group.sync_timestamp_us, bone_emas=self._bone_emas)

        if fused:
            # Update bone EMAs from the fused result
            for parent, child in self._tracked_bones:
                if parent in fused.points and child in fused.points:
                    dist = np.linalg.norm(fused.points[parent] - fused.points[child])
                    
                    # Outlier rejection: only update EMA if distance is plausible
                    # (e.g., within 30% of current EMA or first measurement)
                    if (parent, child) not in self._bone_emas:
                        self._bone_emas[(parent, child)] = float(dist)
                    else:
                        current_ema = self._bone_emas[(parent, child)]
                        if 0.7 * current_ema < dist < 1.3 * current_ema:
                            self._bone_emas[(parent, child)] = (
                                self._ema_alpha * dist + (1.0 - self._ema_alpha) * current_ema
                            )
                        else:
                            # logger.debug(f"Rejected bone length outlier: {dist:.3f}m (EMA: {current_ema:.3f}m)")
                            pass

        return PipelineResult(fused_frame=fused, detections=detections)

    def _detect_camera(self, dev_id: str, group: SyncedFrameGroup) -> tuple[str, Optional[HandDetection], Optional[PreparedFrame]]:
        """Helper for parallel detection."""
        prepared = self._prepare_camera(dev_id, group)
        if prepared is None:
            return dev_id, None, None
        
        if dev_id not in self._detectors:
            self._detectors[dev_id] = CompositeLandmarkDetector(
                detectors=[MediaPipeArm(hand=self._hand, mode="live")], # pyright: ignore
                mode=FusionMode.ANY,
            )


        
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
        if backend is None or not isinstance(backend, KinectBackend):
            return None
        return lift_to_3d(detection, prepared, backend)
