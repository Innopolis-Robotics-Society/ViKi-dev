"""
viki.skeleton.geometry
----------------------
Pure-math lifting of 2-D pixel landmarks into 3-D camera space.

MediaPipe z fallback
--------------------
When depth_m[v, u] is nan (no depth data), we estimate Z using MediaPipe's
relative z coordinate.  MediaPipe z is relative to the wrist (index 0) and
is expressed in units proportional to hand size, not metres.

We scale it using the wrist anchor:
    scale = depth_m[v_wrist, u_wrist] / lm_z_rel[WRIST]
    Z_i   = lm_z_rel[i] * scale

This only works when the wrist itself has valid depth.  If the wrist depth is
also nan the point is marked MISSING.
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


def _wrist_scale(
    wrist_px: np.ndarray,   # (2,) [u, v]
    wrist_z_rel: float,
    depth_m: np.ndarray,    # (H, W)
) -> float | None:
    """
    Compute the scale factor to convert MediaPipe relative z to metres.

    Returns None if the wrist depth pixel is nan or out of bounds.
    """
    u, v = int(round(wrist_px[0])), int(round(wrist_px[1]))
    h, w = depth_m.shape
    if not (0 <= v < h and 0 <= u < w):
        return None
    Z_wrist = depth_m[v, u]
    if np.isnan(Z_wrist) or wrist_z_rel == 0.0:
        return None
    return float(Z_wrist / wrist_z_rel)


def lift_to_3d(detection: HandDetection, frame: PreparedFrame) -> Landmarks3D:
    """
    Deproject all 23 pixel landmarks into 3-D camera space.

    For each landmark:
      1. Sample depth_m at the landmark pixel.
      2. If depth is valid, it gets DEPTH source, full metric deprojection.
      3. If depth is nan and wrist has valid depth, it gets approximate MP_Z source, z scaled
         from MediaPipe relative z using wrist as anchor.
      4. Otherwise is MISSING, point set to (nan, nan, nan).

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
    h, w = depth_m.shape

    # Precompute wrist scale for MP_Z fallback
    mp_z_scale = _wrist_scale(
        detection.px[LM.WRIST],
        float(detection.lm_z_rel[LM.WRIST]),
        depth_m,
    )

    points = np.full((LM.N, 3), np.nan, dtype=np.float32)
    source = np.array([LandmarkSource.MISSING] * LM.N, dtype=object)

    for i in range(LM.N):
        u, v = detection.px[i, 0], detection.px[i, 1]
        ui, vi = int(round(u)), int(round(v))

        # Skip landmarks that projected outside the image boundary
        if not (0 <= vi < h and 0 <= ui < w):
            continue

        Z = depth_m[vi, ui]

        if not np.isnan(Z):
            # Valid depth — full metric deprojection
            points[i] = _pixel_to_3d(u, v, Z, fx, fy, cx, cy)
            source[i] = LandmarkSource.DEPTH

        elif mp_z_scale is not None:
            # No depth — estimate Z from MediaPipe relative z and wrist scale
            Z_approx = float(detection.lm_z_rel[i]) * mp_z_scale
            if Z_approx > 0:
                points[i] = _pixel_to_3d(u, v, Z_approx, fx, fy, cx, cy)
                source[i] = LandmarkSource.MP_Z
            # Negative or zero estimated depth → leave as MISSING

        # Wrist depth unknown → leave as MISSING

    return Landmarks3D(
        points=points,
        source=source,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )
