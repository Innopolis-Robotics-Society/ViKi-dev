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
from typing import Optional
from time import sleep, time
import random

import numpy as np
import logging
import os

from viki.config import SKELETON_DEPTH_SAMP_RADIUS, SKELETON_ENABLE_DEPTH_VALIDATION, MEDIAPIPE_ESTIMATION_CHECK, DEPTH_PROJECTION_DEBUG
logger = logging.getLogger(__name__)


from viki.capture.kinect import KinectBackend
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
    return float(abs(Z_wrist / wrist_z_rel))


def lift_to_3d(detection: HandDetection, frame: PreparedFrame, backend: KinectBackend) -> Landmarks3D:
    """
    Deproject all 23 pixel landmarks into 3-D camera space using converge/diverge priority.
    """
    global _last_depth_viz_time
    K = frame.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    depth_m = frame.depth_m
    h, w = depth_m.shape[:2]

    # Determine if we should visualize this frame (once per second)
    now = time()
    should_viz_this_frame = False
    if DEPTH_PROJECTION_DEBUG and (now - _last_depth_viz_time > 1.0):
        should_viz_this_frame = True
        _last_depth_viz_time = now

    # 1. Calculate MediaPipe Z scale (Z_est)
    mp_z_scale = _wrist_scale(detection.points[LM.WRIST], float(detection.lm_z_rel[LM.WRIST]), depth_m)
    if mp_z_scale is None:
        z_rel_wrist = float(detection.lm_z_rel[LM.WRIST])
        if z_rel_wrist != 0.0:
            mp_z_scale = _FALLBACK_WRIST_Z_M / z_rel_wrist

    points = {LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)}
    
    # If visualizing, pick one random landmark to avoid flooding disk
    viz_target_lm = random.randint(0, LM.N - 1) if should_viz_this_frame else None

    for i in range(LM.N):
        u, v = detection.points[LM(i)][0], detection.points[LM(i)][1]
        if np.isnan(u) or np.isnan(v):
            continue

        # --- Source 1: Deterministic Projection (Z_proj) ---
        Z_proj = np.nan
        if SKELETON_ENABLE_DEPTH_VALIDATION:
            r = SKELETON_DEPTH_SAMP_RADIUS
            
            # 1. Project color (u, v) to depth (ud, vd) using an initial Z guess
            # We use the wrist scale as a baseline for the guess, ensuring it is positive
            z_guess = mp_z_scale if (mp_z_scale is not None and mp_z_scale > 0) else _FALLBACK_WRIST_Z_M
            proj_res = backend.project_color_to_depth(u, v, z_guess)
            
            if proj_res is None:
                logger.debug(f"LM {i}: Projection failed for ({u}, {v})")
                continue
                
            ud, vd = proj_res
            ru, rv = int(round(ud)), int(round(vd))
            v_start, v_end = max(0, rv - r), max(0, min(h, rv + r + 1))
            u_start, u_end = max(0, ru - r), max(0, min(w, ru + r + 1))
            
            current_roi = depth_m[v_start:v_end, u_start:u_end]
            if frame.base_depth_m is not None:
                base_roi = frame.base_depth_m[v_start:v_end, u_start:u_end]
                if base_roi.shape == current_roi.shape:
                    diff_roi = np.maximum(0, base_roi - current_roi)
                else:
                    logger.warning(f"LM {i}: base_roi shape {base_roi.shape} != current_roi shape {current_roi.shape}. Using current_roi.")
                    diff_roi = current_roi
            else:
                diff_roi = current_roi

            if diff_roi.size == 0:
                logger.debug(f"LM {i}: ROI empty at ({ru}, {rv}) - skipping")
                continue

            roi_h, roi_w = diff_roi.shape
            vv_rel, uu_rel = np.meshgrid(np.arange(roi_h), np.arange(roi_w), indexing='ij')
            
            # Calculate distances using absolute coordinates derived from relative grid + offsets
            mask = (vv_rel + v_start - rv)**2 + (uu_rel + u_start - ru)**2 <= r**2
            
            valid_vals = diff_roi[mask & ~np.isnan(diff_roi)]
            
            # Visualization logic
            if DEPTH_PROJECTION_DEBUG and i == viz_target_lm:
                status = "SUCCESS" if valid_vals.size > 0 else "NO_VALID_DEPTH"
                from viki.skeleton.viz import visualize_depth_subtraction
                logger.info(
                    f"LM {i} [{status}] [u={u:.1f}, v={v:.1f}] img_size={depth_m.shape} "
                    f"ROI({v_start}:{v_end}, {u_start}:{u_end}) "
                    f"Shape={diff_roi.shape}, ValidPx={valid_vals.size}"
                )
                visualize_depth_subtraction(
                    base_depth=frame.base_depth_m,
                    current_depth=depth_m,
                    u=u, v=v, r=r,
                    v_start=v_start, v_end=v_end,
                    u_start=u_start, u_end=u_end,
                    diff_roi=diff_roi,
                    z_proj=np.median(valid_vals) if valid_vals.size > 0 else np.nan,
                    landmark_name=f"LM_{i}_{status}"
                )

            if valid_vals.size > 0:
                # Robust mean: take values within 10% of the median to filter outliers
                med = np.median(valid_vals)
                mask_robust = (valid_vals >= 0.9 * med) & (valid_vals <= 1.1 * med)
                robust_vals = valid_vals[mask_robust]
                Z_proj = np.mean(robust_vals) if robust_vals.size > 0 else med
            else:
                logger.debug(f"LM {i}: No valid depth in masked ROI at ({ru}, {rv})")
                Z_proj = np.nan
        else:
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
        if MEDIAPIPE_ESTIMATION_CHECK and mp_z_scale is not None:
            val = float(detection.lm_z_rel[i]) * mp_z_scale
            if val > 0:
                Z_est = val

        # --- Converge/Diverge Decision Logic ---
        Z_final = np.nan
        if not np.isnan(Z_proj) and not np.isnan(Z_est):
            conf = detection.confidence
            dist = abs(Z_proj - Z_est)
            
            if dist < 0.10:
                # Close agreement: weighted average (favor sensor)
                Z_final = (Z_proj * 2.0 + Z_est * conf) / (2.0 + conf)
            elif dist < 0.25:
                # Moderate disagreement: blend more equally
                Z_final = (Z_proj + Z_est * conf) / (1.0 + conf)
            else:
                # Large disagreement: likely sensor hit background/noise
                # Favor MediaPipe's relative structure
                Z_final = Z_est * (1.0 + 0.2 * conf) # slight nudge
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

