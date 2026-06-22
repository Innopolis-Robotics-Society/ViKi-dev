"""
viki.skeleton.pipeline
----------------------
Public orchestrator for the skeleton detection pipeline.a
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from viki.capture.base import SyncedFrameGroup
from viki.calibration.manager import CalibrationManager
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.fusion import fuse, load_extrinsics
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.hand_detector import HandDetector
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame


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
        master_id: str = "kinect_0",
        subordinate_id: str = "kinect_1",
        hand: str = "right",
    ) -> None:
        self._calibrator = calibrator
        self._master_id = master_id
        self._subordinate_id = subordinate_id

        self._cache = UndistortCache()
        self._detector = HandDetector(hand=hand)
        self._R, self._T = load_extrinsics(calib_path)

    def process(self, group: SyncedFrameGroup) -> Optional[SkeletonFrame]:
        """
        Run the full pipeline on one SyncedFrameGroup.

        Returns None if both cameras fail to detect a hand.

        Parameters
        ----------
        group : SyncedFrameGroup
            Output of MultiCameraSync.get_synced_frame().

        Returns
        -------
        SkeletonFrame | None
        """
        lm0 = self._process_camera(self._master_id, group)
        lm1 = self._process_camera(self._subordinate_id, group)

        return fuse(lm0, lm1, self._R, self._T, group.sync_timestamp_us)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    def __enter__(self) -> "SkeletonPipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _process_camera(
        self, device_id: str, group: SyncedFrameGroup
    ) -> Optional[Landmarks3D]:
        """Run stages 1–3 for one camera. Returns None if any stage fails."""

        frame = group.frames.get(device_id)
        if frame is None:
            return None

        # Intrinsics required for undistort and deprojection
        intrinsics = self._calibrator.get_intrinsics(device_id)
        if intrinsics is None:
            return None
        K = intrinsics.camera_matrix
        dist = intrinsics.dist_coeffs

        # Stage 1: camera_prep
        prepared = prepare_frame(frame, K, dist, self._cache)

        # Stage 2: hand detection
        detection = self._detector.detect(prepared)
        if detection is None:
            return None

        # Stage 3: lift to 3D
        return lift_to_3d(detection, prepared)
