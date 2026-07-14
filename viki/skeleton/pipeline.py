"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import sleep

from typing import Dict, Optional, Literal
import logging

from viki.calibration.models import CalibrationExtrinsics

logger = logging.getLogger(__name__)

import numpy as np
import os
from viki.capture.base import SyncedFrameGroup

from viki.capture.manager import CameraManager
from viki.calibration.manager import CalibrationManager
from viki.skeleton.camera_prep import prepare_frame
from viki.skeleton.fusion import fuse
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.detectors import (
    CompositeLandmarkDetector,
    FusionMode,
    RTMPoseWholeBody,
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
    End‑to‑end skeleton detection from SyncedFrameGroup to SkeletonFrame.

    This pipeline:
        1. Prepares each camera frame (undistort, depth clean).
        2. Runs RTMPose whole-body detection on each camera in parallel;
           one RTMPoseWholeBody call fills all 23 slots (arm + hand + fingers).
        3. Lifts 2D detections to 3D using depth maps.
        4. Fuses per‑camera 3D landmarks into a single world‑frame skeleton.

    Parameters
    ----------
    calibrator : CalibrationManager
        Provides per‑device intrinsics and extrinsics.
    manager : CameraManager
        Provides access to camera backends (for depth projection).
    hand : Literal["right", "left"]
        Which hand to track. Default from config.
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
        self._detectors: dict[str, CompositeLandmarkDetector] = {}
        self._hand_type = hand
        self._executor = ThreadPoolExecutor(max_workers=4)

        self._bone_emas: dict[tuple[LM, LM], float] = {}
        self._ema_alpha = 0.1
        self._tracked_bones: list[tuple[LM, LM]] = []

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
            Contains fused SkeletonFrame and per‑camera detections.
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

        # Extract confidences for weighted fusion
        confidences: dict[str, dict[LM, float]] = {}
        for dev_id, det in detections.items():
            if det:
                # HandDetection currently exposes only overall confidence;
                # broadcast it as a per-landmark weight for the fuser.
                confidences[dev_id] = {LM(i): det.confidence for i in range(LM.N)}

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

        fused = fuse(
            dev_ids,
            lms_3d,
            extrinsics,
            group.sync_timestamp_us,
            confidences=confidences,
            bone_emas=self._bone_emas,
        )

        if not np.isnan([x for _, x in fused.points.items()]).any():
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
                                self._ema_alpha * dist
                                + (1.0 - self._ema_alpha) * current_ema
                            )
                        else:
                            # logger.debug(f"Rejected bone length outlier: {dist:.3f}m (EMA: {current_ema:.3f}m)")
                            pass

        return PipelineResult(fused_frame=fused, detections=detections)

    def _detect_camera(
        self, dev_id: str, group: SyncedFrameGroup
    ) -> tuple[str, Optional[HandDetection], Optional[PreparedFrame]]:
        """
        Helper for parallel detection.
        Returns a tuple (device_id, detection, prepared_frame).
        """
        prepared = self._prepare_camera(dev_id, group)
        if prepared is None:
            return dev_id, None, None

        if dev_id not in self._detectors:
            self._detectors[dev_id] = CompositeLandmarkDetector(
                detectors=[
                    RTMPoseWholeBody(
                        hand=self._hand,
                        model_mode="balanced",
                        device="cpu",
                    ),
                ],
                mode=FusionMode.ANY,
            )

        det = self._detectors[dev_id].detect(prepared)
        return dev_id, det, prepared

    def close(self) -> None:
        """Release detector resources (ONNXRuntime sessions, threadpool)."""
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
        """
        Stage 1: prepare frame for detection.

        Parameters
        ----------
        device_id : str
            Camera ID.
        group : SyncedFrameGroup
            The sync group containing the frame.

        Returns
        -------
        PreparedFrame or None
            Prepared frame, or None if the frame is missing.
        """
        frame = group.frames.get(device_id)
        if frame is None:
            logger.debug("SkeletonPipeline: no synced frames from SyncFrameGroup")
            return None

        prepared = prepare_frame(frame)

        # Load base depth map for this camera
        base_path = os.path.join(
            viki.config.SKELETON_DEPTH_BASE_DIR, f"{device_id}.npy"
        )
        if os.path.exists(base_path):
            try:
                prepared.base_depth_m = np.load(base_path)
            except Exception as e:
                logger.error(f"Failed to load base depth for {device_id}: {e}")

        return prepared

    def _lift_camera(
        self,
        device_id: str,
        group: SyncedFrameGroup,
        detection: Optional[HandDetection],
        prepared: Optional[PreparedFrame] = None,
    ) -> Optional[Landmarks3D]:
        """
        Stage 3: lift 2D detection to 3D.

        Parameters
        ----------
        device_id : str
            Camera ID.
        group : SyncedFrameGroup
            The sync group (used to re‑prepare if needed).
        detection : Optional[HandDetection]
            2D detection (None if no hand).
        prepared : Optional[PreparedFrame]
            Prepared frame (if already available).

        Returns
        -------
        Landmarks3D or None
            3D landmarks in camera coordinates, or None if detection absent or backend not Kinect.
        """
        if detection is None:
            return None

        # Use the provided prepared frame, or re-prepare if missing
        if prepared is None:
            prepared = self._prepare_camera(device_id, group)

        if prepared is None:
            return None

        backend = self._manager.get_backend(device_id)
        if backend is None:
            return None
        return lift_to_3d(detection, prepared, backend)
