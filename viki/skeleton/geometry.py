"""
viki.skeleton.geometry
----------------------
Mapping 2D points to 3D space using depth and camera intrinsics.

MediaPipe z fallback
--------------------
When depth_m[v, u] is nan (no depth data), we estimate Z using MediaPipe's
relative z coordinate.
"""

from __future__ import annotations

import numpy as np

from viki.skeleton.models import HandDetection, LandmarkSource, Landmarks3D, LM, PreparedFrame


def _pixel_to_3d(
    u: float, v: float, Z: float,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Deproject a single pixel into 3-D camera space. Returns (X, Y, Z) metres."""
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

# So this is only needed if we don't get anything from depth cameras
# in real case not needed
_FALLBACK_WRIST_Z_M = 0.7  # assumed wrist depth (metres) when no real depth sensor


def _wrist_scale(
    wrist_px: np.ndarray,   # (2,) [u, v]
    wrist_z_rel: float,
    depth_m: np.ndarray,    # (H, W)
) -> float | None:
    """
    Compute the scale factor to convert MediaPipe relative z to metres.
    This method is needed only as a fallback if we don't get valid depth.

    Returns None if the wrist depth pixel is nan or out of bounds.
    """
    # Ensure wrist coordinates are valid before rounding
    if np.isnan(wrist_px[0]) or np.isnan(wrist_px[1]):
        return None
    u, v = int(round(wrist_px[0])), int(round(wrist_px[1]))
    h, w = depth_m.shape[:2]
    if not (0 <= v < h and 0 <= u < w):
        return None
    Z_wrist = depth_m[v, u]
    if np.isnan(Z_wrist).any() or wrist_z_rel == 0.0:
        return None
    return float(Z_wrist / wrist_z_rel)


def lift_to_3d(detection: HandDetection, frame: PreparedFrame) -> Landmarks3D:
    """
    Deproject all 23 pixel landmarks into 3-D camera space.

    For each landmark, it's checkeed whether depth_m has a valid value.

    Parameters
    ----------
    detection : HandDetection
        23 pixel-space landmarks from hand_detector.
    frame : PreparedFrame
        Provides depth_m and intrinsic matrix K.

    Returns
    -------
    Landmarks3D
        23 points in metres in the coordinate frame of detection.device_id.
    """
    K = frame.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    depth_m = frame.depth_m
    h, w = depth_m.shape[:2]

    mp_z_scale = _wrist_scale(detection.px[LM.WRIST], float(detection.lm_z_rel[LM.WRIST]), depth_m)
    if mp_z_scale is None:
        z_rel_wrist = float(detection.lm_z_rel[LM.WRIST])
        if z_rel_wrist != 0.0:
            mp_z_scale = _FALLBACK_WRIST_Z_M / z_rel_wrist

    points = np.full((LM.N, 3), np.nan, dtype=np.float32)
    source = np.array([LandmarkSource.MISSING] * LM.N, dtype=object)

    for i in range(LM.N):
        u, v = detection.px[i, 0], detection.px[i, 1]
        if np.isnan(u) or np.isnan(v):
            continue
        ui, vi = int(round(u)), int(round(v))

        # Skip landmarks that projected outside the image boundary
        if not (0 <= vi < h and 0 <= ui < w):
            continue

        Z = depth_m[vi, ui]
 
        if not np.isnan(Z).any():
            # Valid depth
            points[i] = _pixel_to_3d(u, v, Z, fx, fy, cx, cy)
            source[i] = LandmarkSource.DEPTH


        elif mp_z_scale is not None:
            # No depth, we estimate Z
            Z_approx = float(detection.lm_z_rel[i]) * mp_z_scale
            if Z_approx > 0:
                points[i] = _pixel_to_3d(u, v, Z_approx, fx, fy, cx, cy)
                source[i] = LandmarkSource.MP_Z

    return Landmarks3D(
        points=points,
        source=source,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )
