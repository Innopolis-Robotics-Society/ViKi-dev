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
from time import sleep

import numpy as np
import logging
import os

from viki.config import Z_CONVERGENCE_THRESHOLD
logger = logging.getLogger(__name__)

from viki.capture.kinect import KinectBackend
from viki.skeleton.models import HandDetection, Landmarks3D, LM, PreparedFrame


def _pixel_to_3d(
    u: float,
    v: float,
    Z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Deproject a single pixel into 3-D camera space. Returns (X, Y, Z) metres."""
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)


def color_to_depth_pixel(u: float, v: float, Z: float, K: np.ndarray, backend: KinectBackend, raw_depth: np.ndarray, aligned_depth: Optional[np.ndarray] = None) -> tuple[float, float, float] | None:
    """
    Maps a pixel from the color camera to the depth camera coordinate space,
    validated against the SDK's estimation if available.
    
    Returns:
        (u_depth, v_depth, final_z) or None if projection fails.
    """
    return backend.get_validated_depth(u, v, Z, raw_depth, aligned_depth)

# So this is only needed if we don't get anything from depth cameras
# in real case not needed
_FALLBACK_WRIST_Z_M = 0.7  # assumed wrist depth (metres) when no real depth sensor


def _wrist_scale(
    wrist_px: np.ndarray,  # (2,) [u, v]
    wrist_z_rel: float,
    depth_m: np.ndarray,  # (H, W)
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


def lift_to_3d(detection: HandDetection, frame: PreparedFrame, backend: KinectBackend) -> Landmarks3D:
    """
    Deproject all 23 pixel landmarks into 3-D camera space using converge/diverge priority.
    """
    K = frame.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    depth_m = frame.depth_m
    h, w = depth_m.shape[:2]

    # 1. Calculate MediaPipe Z scale (Z_est)
    mp_z_scale = _wrist_scale(detection.points[LM.WRIST], float(detection.lm_z_rel[LM.WRIST]), depth_m)
    if mp_z_scale is None:
        z_rel_wrist = float(detection.lm_z_rel[LM.WRIST])
        if z_rel_wrist != 0.0:
            mp_z_scale = _FALLBACK_WRIST_Z_M / z_rel_wrist

    points = {LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)}
    
    for i in range(LM.N):
        u, v = detection.points[LM(i)][0], detection.points[LM(i)][1]
        if np.isnan(u) or np.isnan(v):
            continue

        # --- Source 1: Deterministic Projection (Z_proj) ---
        Z_proj = np.nan
        Z_guess = 1.0 
        for _ in range(3):
            res = color_to_depth_pixel(u, v, Z_guess, K, backend, depth_m, frame.aligned_depth)
            if res is None:
                break
            
            ud, vd, z_val = res
            ui, vi = int(round(ud)), int(round(vd))
            
            if not (0 <= vi < h and 0 <= ui < w):
                break
                
            # Sample depth in a 3x3 window for refinement
            v_start, v_end = max(0, vi - 1), min(h, vi + 2)
            u_start, u_end = max(0, ui - 1), min(w, ui + 2)
            window = depth_m[v_start:v_end, u_start:u_end]
            valid_window = window[~np.isnan(window)]
            
            if valid_window.size > 0:
                Z_proj = np.median(valid_window)
                Z_guess = float(Z_proj)
            else:
                Z_proj = np.nan
                break
        
        # --- Source 2: MediaPipe Estimator (Z_est) ---
        Z_est = np.nan
        if mp_z_scale is not None:
            val = float(detection.lm_z_rel[i]) * mp_z_scale
            if val > 0:
                Z_est = val

        # --- Converge/Diverge Decision Logic ---
        Z_final = np.nan
        if not np.isnan(Z_proj) and not np.isnan(Z_est):
            conf = detection.confidence
            # Use the closer value as the sensor contribution to maintain background rejection,
            # then blend with the MediaPipe estimation based on confidence to smooth transitions.
            Z_final = (min(Z_proj, Z_est) + conf * Z_est) / (1.0 + conf)
        elif not np.isnan(Z_proj):
            Z_final = Z_proj
        elif not np.isnan(Z_est):
            Z_final = Z_est

        if not np.isnan(Z_final):
            points[LM(i)] = _pixel_to_3d(u, v, Z_final, fx, fy, cx, cy)

    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )

