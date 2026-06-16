"""
viki.skeleton.models
--------------------
Dataclasses for data flowing between skeleton pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
      - color is undistorted and converted BGR → RGB
      - depth is float32 metres with 0 replaced by nan

    Carries K so that downstream geometry code doesn't need to reach
    back into CameraManager.
    """
    rgb:          np.ndarray    # (H, W, 3) uint8, RGB, undistorted
    depth_m:      np.ndarray    # (H, W)    float32, metres
    K:            np.ndarray    # (3, 3)    intrinsic matrix for this frame
    device_id:    str
    timestamp_us: int


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
    px:           np.ndarray    # float32 — (u, v) pixel coords
    lm_z_rel:     np.ndarray    # float32 — MediaPipe relative z
    confidence:   float
    device_id:    str
    timestamp_us: int


#
# From stage 3 to 4: geometry output to fusion input
#
class LandmarkSource(str, Enum):
    """
    Provenance of a single 3-D landmark point.

    Used downstream by the smoother (One Euro Filter) to apply different
    trust levels, and by the dataset writer to annotate point quality.
    """
    DEPTH   = "depth"    # depth_m[v, u] was valid → fully metric
    MP_Z    = "mp_z"     # depth was nan; z estimated from MediaPipe relative z + wrist scale
    MISSING = "missing"  # hand not detected in this camera at all


@dataclass
class Landmarks3D:
    """
    23 landmarks lifted into 3-D camera space for one camera.

    Indices 0–20: hand landmarks (MediaPipe Hands convention).
    Index 21: elbow  (from MediaPipe Pose; overrides nothing, adds arm context).
    Index 22: shoulder (from MediaPipe Pose).
    WRIST (index 0) is overridden by the body model wrist for arm–hand continuity.

    points[i]  — (X, Y, Z) in metres, in the coordinate frame of device_id.
    source[i]  — how point i was obtained (see LandmarkSource).

    To transform into a common frame (kinect_0), apply:
        P_cam0 = R @ points[i].reshape(3,1) + T
    where R, T come from calibration_results.npz extrinsic_data.
    """
    points:       np.ndarray    # (23, 3) float32, metres, local camera frame
    source:       np.ndarray    # (23,)   LandmarkSource
    device_id:    str
    timestamp_us: int

    def valid_mask(self) -> np.ndarray:
        """(23,) bool — True for every point that is not MISSING."""
        return self.source != LandmarkSource.MISSING

    def depth_mask(self) -> np.ndarray:
        """(23,) bool — True only for points backed by real depth data."""
        return self.source == LandmarkSource.DEPTH


#
# Public api
#

# MediaPipe Hands landmark indices — avoids magic numbers downstream.
class LM:
    WRIST       = 0
    THUMB_CMC   = 1
    THUMB_MCP   = 2
    THUMB_IP    = 3
    THUMB_TIP   = 4
    INDEX_MCP   = 5
    INDEX_PIP   = 6
    INDEX_DIP   = 7
    INDEX_TIP   = 8
    MIDDLE_MCP  = 9
    MIDDLE_PIP  = 10
    MIDDLE_DIP  = 11
    MIDDLE_TIP  = 12
    RING_MCP    = 13
    RING_PIP    = 14
    RING_DIP    = 15
    RING_TIP    = 16
    PINKY_MCP   = 17
    PINKY_PIP   = 18
    PINKY_DIP   = 19
    PINKY_TIP   = 20

    # Arm landmarks (from MediaPipe Pose, appended after hand landmarks)
    ELBOW       = 21
    SHOULDER    = 22

    FINGERTIPS  = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    MCP_JOINTS  = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    ARM_CHAIN   = (SHOULDER, ELBOW, WRIST)   

    N = 23


@dataclass
class SkeletonFrame:
    """
    Fused 23-landmark arm+hand pose in the kinect_0 world frame.

    Indices 0–20 : hand landmarks (MediaPipe Hands convention).
                   WRIST (0) is taken from the body model for arm–hand continuity.
    Index 21     : elbow  (MediaPipe Pose).
    Index 22     : shoulder (MediaPipe Pose).

    This is the public output of SkeletonPipeline.process().
    All coordinates are in metres in the kinect_0 coordinate system.

    Fields
    ------
    landmarks   : (23, 3) float32 — 3-D positions, metres, kinect_0 frame.
    source      : (23,)   LandmarkSource — quality of each point after fusion.
    confidence  : (23,)   float32 — per-point confidence in [0, 1].
                  For DEPTH points: 1.0.
                  For MP_Z points: MediaPipe detection score.
                  For MISSING: 0.0.
    origin      : (23,)   str — which camera contributed each point
                  ("kinect_0", "kinect_1", "fallback").
                  Useful for debugging and dataset quality annotation.
    timestamp_us: int — sync_timestamp_us from the SyncedFrameGroup.

    Usage
    -----
    frame = pipeline.process(synced_group)
    if frame is None:
        # hand not visible in any camera — skip
        ...

    wrist = frame.landmarks[LM.WRIST]          # (3,) xyz metres
    tip   = frame.landmarks[LM.INDEX_TIP]
    good  = frame.source[LM.INDEX_TIP] == LandmarkSource.DEPTH
    """
    landmarks:    np.ndarray    # (23, 3) float32, metres, kinect_0 frame
    source:       np.ndarray    # (23,)   LandmarkSource
    confidence:   np.ndarray    # (23,)   float32
    origin:       np.ndarray    # (23,)   str
    timestamp_us: int

    def reliable_mask(self) -> np.ndarray:
        """(23,) bool — points backed by real depth from either camera."""
        return self.source == LandmarkSource.DEPTH

    def as_dict(self) -> dict:
        """Serialisable snapshot — useful for JSON logging and dataset writing."""
        return {
            "timestamp_us": self.timestamp_us,
            "landmarks":    self.landmarks.tolist(),
            "source":       [s.value for s in self.source],
            "confidence":   self.confidence.tolist(),
            "origin":       self.origin.tolist(),
        }
