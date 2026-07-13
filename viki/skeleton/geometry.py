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
from typing import Any, Optional
from time import time

import numpy as np
import logging

from viki.config import (
    SKELETON_DEPTH_SAMP_RADIUS,
    SKELETON_DEPTH_SUBTRACT_THRESHOLD,
    DEPTH_PROJECTION_DEBUG,
)

logger = logging.getLogger(__name__)

from viki.skeleton.models import HandDetection, Landmarks3D, LM, PreparedFrame

_last_depth_viz_time = 0.0


def _pixel_to_3d(
    u: float,
    v: float,
    Z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Deproject a single pixel into 3‑D camera space.

    Parameters
    ----------
    u, v : float
        Pixel coordinates (depth image).
    Z : float
        Depth value in metres.
    fx, fy, cx, cy : float
        Intrinsic parameters (depth camera).

    Returns
    -------
    np.ndarray
        (X, Y, Z) in metres, shape (3,).
    """
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)


def lift_to_3d(
    detection: HandDetection, frame: PreparedFrame, backend: Any
) -> Landmarks3D:
    """
    Deproject all pixel landmarks into 3‑D camera space.

    For each landmark:
      1. Project the color pixel (u, v) to depth-camera space via the
         SDK's built-in calibration (handles parallax).
      2. Sample a circular ROI at the projected location in the depth map,
         with optional background‑subtraction filtering.
      3. Take the median of valid depths → Z.
      4. Deproject (u_depth, v_depth, Z) with *depth* intrinsics → (X, Y, Z).

    Parameters
    ----------
    detection : HandDetection
        2D landmark detections (pixel coordinates) from MediaPipe.
    frame : PreparedFrame
        Prepared frame with depth_m (metres), depth_K, and optional base_depth_m.
    backend : Any
        Camera backend providing ``project_color_to_depth``.

    Returns
    -------
    Landmarks3D
        3D landmarks in camera coordinates.
    """
    global _last_depth_viz_time

    # Depth intrinsics (fall back to colour if not available)
    K = frame.depth_K if frame.depth_K is not None else frame.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depth_m = frame.depth_m
    h, w = depth_m.shape[:2]

    # Viz throttle: one debug frame per second
    now = time()
    should_viz = DEPTH_PROJECTION_DEBUG and (now - _last_depth_viz_time > 1.0)
    if should_viz:
        _last_depth_viz_time = now

    points = {LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)}
    viz_data = []
    r = SKELETON_DEPTH_SAMP_RADIUS

    for i in range(LM.N):
        u, v = detection.points[LM(i)][0], detection.points[LM(i)][1]
        if np.isnan(u) or np.isnan(v):
            continue

        Z_proj = np.nan
        z_guess = 1.0

        # 1. Project color pixel to depth-camera space
        proj_res = backend.project_color_to_depth(u, v, z_guess)
        if proj_res is None:
            logger.debug(f"LM {i}: projection failed for ({u}, {v})")
            continue

        ud, vd = proj_res
        ru, rv = int(round(ud)), int(round(vd))

        # 2. Circular ROI around the projected depth pixel
        v_start, v_end = max(0, rv - r), min(h, rv + r + 1)
        u_start, u_end = max(0, ru - r), min(w, ru + r + 1)
        roi = depth_m[v_start:v_end, u_start:u_end]

        if roi.size == 0:
            continue

        # 3. Background subtraction (if a base depth map exists)
        if frame.base_depth_m is not None:
            base_roi = frame.base_depth_m[v_start:v_end, u_start:u_end]
            if base_roi.shape == roi.shape:
                diff = np.maximum(0, base_roi - roi)
            else:
                diff = roi
        else:
            diff = roi

        # Circular mask
        roi_h, roi_w = diff.shape
        vv, uu = np.meshgrid(np.arange(roi_h), np.arange(roi_w), indexing="ij")
        circ = (vv + v_start - rv) ** 2 + (uu + u_start - ru) ** 2 <= r ** 2

        # Valid: inside circle, not NaN, and above subtraction threshold
        valid = circ & ~np.isnan(roi) & (diff > SKELETON_DEPTH_SUBTRACT_THRESHOLD)
        vals = roi[valid]

        if vals.size > 0:
            # Filter to foreground — keep values above avg difference
            diffs = diff[valid]
            threshold = (np.max(diffs) + np.min(diffs)) / 2
            foreground = diffs >= threshold
            filtered = vals[foreground]

            if filtered.size > 0:
                Z_proj = float(np.median(filtered))
            else:
                Z_proj = float(np.median(vals))

        if not np.isnan(Z_proj):
            points[LM(i)] = _pixel_to_3d(ud, vd, Z_proj, fx, fy, cx, cy)

        # Debug viz (arm landmarks only, once per second)
        if should_viz and LM(i) in (LM.SHOULDER, LM.ELBOW, LM.WRIST):
            status = "OK" if vals.size > 0 else "NO_DEPTH"
            z_dot = Z_proj if not np.isnan(Z_proj) else 1.0
            final = backend.project_color_to_depth(u, v, z_dot)
            fud, fvd = final if final else (ud, vd)
            viz_data.append(
                {
                    "name": f"{LM(i).name}_{status}",
                    "u": u, "v": v, "ud": fud, "vd": fvd, "r": r,
                    "v_start": v_start, "v_end": v_end,
                    "u_start": u_start, "u_end": u_end,
                    "diff_roi": diff,
                    "z_proj": Z_proj,
                }
            )

    if viz_data:
        from viki.skeleton.viz import visualize_depth_subtraction
        visualize_depth_subtraction(
            base_depth=frame.base_depth_m,
            current_depth=depth_m,
            landmark_data=viz_data,
        )

    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )
