"""
viki.skeleton.models
--------------------
Dataclasses for data flowing between skeleton pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


#
# from Stage 1 to 2: synced frames to input of hand detector
# Used other class to encapsulate from capture sync logic
#
@dataclass
class PreparedFrame:
    """
    A single camera frame ready for model inference.

    Produced by camera_prep from a raw Frame:
      - color is undistorted and converted BGR to RGB
      - depth is float32 metres with 0 replaced by nan

    Carries K so that downstream geometry code doesn't need to reach
    back into CameraManager.
    """

    rgb: np.ndarray  # (H, W, 3) uint8, RGB, undistorted
    depth_m: np.ndarray  # (H, W)    float32, metres
    K: np.ndarray  # (3, 3)    intrinsic matrix for this frame
    device_id: str
    timestamp_us: int
    aligned_depth: Optional[np.ndarray] = None  # (H, W) uint16, SDK estimated
    base_depth_m: Optional[np.ndarray] = None  # (H, W)    float32, background depth


#
# from Stage 2 to 3: hand detector output to geometry input
#
@dataclass
class HandDetection:
    """
    Raw MediaPipe output for one camera, pixel-space.

    Parameters
    ----------

    px[i] : (u, v) pixel coordinates of landmark i (float, subpixel).
    lm_z_rel[i] : MediaPipe's relative z for landmark i.
                  NOT metric depth. Relative to wrist (landmark 0). Approximated by mediapipe as ratio of landmarks
                  Use only as fallback when depth_m is nan at that pixel.
    confidence : overall hand detection score from MediaPipe [0..1].

    None is returned by the detector when no hand is found; this dataclass
    is only instantiated on a successful detection.
    """

    points: dict[LM, np.ndarray]
    lm_z_rel: np.ndarray  # float32, MediaPipe relative z
    confidence: float
    device_id: str
    timestamp_us: int


@dataclass
class Landmarks3D:
    points: dict[LM, np.ndarray]
    device_id: str
    timestamp_us: int

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "points": {index.value: vec.tolist() for index, vec in self.points.items()},
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class EndEffectorPose:
    """
    World-frame pose of the end-effector (wrist).

    Fields
    ------
    position     : (3,) float32 — WRIST world XYZ in metres.
    R_world_palm : (3, 3) float32 — rotation from palm frame to world.
                   Palm frame:
                       x_palm = normalise(MIDDLE_MCP - WRIST)
                       z_palm = normalise((MIDDLE_MCP - WRIST) × (THUMB_CMC - WRIST))
                       y_palm = z_palm × x_palm
    rpy_deg      : (3,) float32 — roll/pitch/yaw in degrees, extrinsic XYZ
                   i.e. R = Rz(yaw) · Ry(pitch) · Rx(roll).
    valid        : True when every required landmark was present and the
                   palm frame could be resolved.
    timestamp_us : same as the containing SkeletonFrame.
    """

    position: np.ndarray
    R_world_palm: np.ndarray
    rpy_deg: np.ndarray
    valid: bool
    timestamp_us: int

    def as_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "R_world_palm": self.R_world_palm.tolist(),
            "rpy_deg": self.rpy_deg.tolist(),
            "valid": self.valid,
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class SkeletonFrame:
    """
    Final fused skeleton in world coordinates.

    Attributes
    ----------
    points : dict[LM, np.ndarray]
        Mapping from landmark enum to world‑frame (X, Y, Z) in metres.
    timestamp_us : int
        Sync timestamp of the fused frame.
    end_effector : Optional[EndEffectorPose]
        World‑frame wrist pose, if computable.
    """
    points: dict[LM, np.ndarray]
    timestamp_us: int
    end_effector: Optional[EndEffectorPose] = None

    def as_dict(self) -> dict:
        record = {
            "points": {index.value: vec.tolist() for index, vec in self.points.items()},
            "timestamp_us": self.timestamp_us,
            "end_effector": (
                self.end_effector.as_dict() if self.end_effector is not None else None
            ),
        }
        if self.end_effector is not None:
            record["end_effector"] = self.end_effector.as_dict()
        return record


@dataclass
class PipelineResult:
    """
    Result of a full pipeline run.

    Attributes
    ----------
    fused_frame : SkeletonFrame
        The world‑space 3D skeleton (or None if fusion failed).
    detections : dict[str, HandDetection | None]
        Per‑camera 2D detections (None if no hand found).
    """
    fused_frame: SkeletonFrame  # The world-space 3D skeleton
    detections: dict[str, HandDetection | None]  # Per-camera 2D landmarks


# MediaPipe Hands landmark indices
class LM(IntEnum):
    """
    Landmark indices for MediaPipe Hands (21 hand landmarks) plus two arm landmarks.

    Hand landmarks: 0 (WRIST) to 20 (PINKY_TIP).
    Arm landmarks (not detected): 21 (ELBOW), 22 (SHOULDER) – kept for schema compatibility.
    """
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # Arm landmarks — never detected (MediaPipeArm disabled), but kept for
    # backward compat and hand_angles schema.
    ELBOW = 21
    SHOULDER = 22

    N = 23

    @property
    def FINGERTIPS(self) -> tuple[LM, LM, LM, LM, LM]:
        return (
            self.THUMB_TIP,
            self.INDEX_TIP,
            self.MIDDLE_TIP,
            self.RING_TIP,
            self.PINKY_TIP,
        )

    @property
    def MCP_JOINTS(self) -> tuple[LM, LM, LM, LM]:
        return (self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP)

    @property
    def ARM_CHAIN(self) -> tuple[LM, LM, LM]:
        return (self.SHOULDER, self.ELBOW, self.WRIST)


HAND_LM_ORDER = [
    LM.WRIST,  # 0
    LM.THUMB_CMC,  # 1
    LM.THUMB_MCP,  # 2
    LM.THUMB_IP,  # 3
    LM.THUMB_TIP,  # 4
    LM.INDEX_MCP,  # 5
    LM.INDEX_PIP,  # 6
    LM.INDEX_DIP,  # 7
    LM.INDEX_TIP,  # 8
    LM.MIDDLE_MCP,  # 9
    LM.MIDDLE_PIP,  # 10
    LM.MIDDLE_DIP,  # 11
    LM.MIDDLE_TIP,  # 12
    LM.RING_MCP,  # 13
    LM.RING_PIP,  # 14
    LM.RING_DIP,  # 15
    LM.RING_TIP,  # 16
    LM.PINKY_MCP,  # 17
    LM.PINKY_PIP,  # 18
    LM.PINKY_DIP,  # 19
    LM.PINKY_TIP,  # 20
]
ARM_LM_ORDER = [LM.ELBOW, LM.SHOULDER]  # indices 21, 22
