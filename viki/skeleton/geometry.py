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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import logging
import os

from viki.config import SKELETON_DEPTH_SAMP_RADIUS, SKELETON_ENABLE_DEPTH_VALIDATION, DEPTH_PROJECTION_DEBUG, SKELETON_DEPTH_SUBTRACT_THRESHOLD # type: ignore
logger = logging.getLogger(__name__)


from viki.capture.kinect import KinectBackend
from viki.skeleton.models import HandDetection, Landmarks3D, LM, PreparedFrame


_last_depth_viz_time = 0.0
_last_known_z = {LM(i): 1.0 for i in range(LM.N)}
_viz_executor = ThreadPoolExecutor(max_workers=1)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of a 1D array."""
    if values.size == 0:
        return np.nan
    idx = np.argsort(values)
    sorted_vals = values[idx]
    sorted_weights = weights[idx]
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    if total_weight <= 0:
        return float(np.median(values))
    return float(sorted_vals[np.searchsorted(cum_weights, total_weight / 2)])


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
    points = {LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)}
    
    # If visualizing, we'll collect data for the arm chain
    viz_data = []
    arm_chain = (LM.SHOULDER, LM.ELBOW, LM.WRIST)

    for i in range(LM.N):
        u, v = detection.points[LM(i)][0], detection.points[LM(i)][1]
        if np.isnan(u) or np.isnan(v):
            continue

        # --- Source 1: Deterministic Projection (Z_proj) ---
        Z_proj = np.nan
        z_guess = 1.0 
        if SKELETON_ENABLE_DEPTH_VALIDATION:
            r = SKELETON_DEPTH_SAMP_RADIUS
            
            # 1. Project color (u, v) to depth (ud, vd)
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
            
            # Sample absolute depth from current_roi where diff_roi is positive (object present)
            # and it's within the circular mask.
            valid_mask = mask & ~np.isnan(current_roi) & (diff_roi > SKELETON_DEPTH_SUBTRACT_THRESHOLD)
            valid_vals = current_roi[valid_mask]
            
            if valid_vals.size > 0:
                # Use a hard threshold based on the difference intensity to isolate the object.
                # We take the max and min of the differences in the ROI and discard points below the average.
                valid_diffs = diff_roi[valid_mask]
                avg_diff = (np.max(valid_diffs) + np.min(valid_diffs)) / 2
                
                # Only keep points that are 'bright' enough (significantly closer than background)
                object_mask_flat = valid_diffs >= avg_diff
                filtered_depths = valid_vals[object_mask_flat]
                
                if filtered_depths.size > 0:
                    Z_proj = np.median(filtered_depths)
                    # Create a full-ROI mask for the filtered pixels
                    final_mask = np.zeros_like(valid_mask, dtype=bool)
                    # valid_mask is a 1D array of coordinates? No, it's the boolean mask.
                    # We need to map object_mask_flat back to the ROI shape.
                    # valid_mask is the mask used to get valid_vals.
                    # So we can just use it to index.
                    # But we need a boolean mask of the same shape as current_roi.
                    
                    # Reconstruct the mask for current_roi
                    full_object_mask = np.zeros_like(current_roi, dtype=bool)
                    # valid_mask.nonzero() gives indices of True values.
                    # object_mask_flat indices correspond to valid_mask's True values.
                    valid_indices = np.where(valid_mask)
                    object_indices = np.where(object_mask_flat)[0]
                    
                    # Map object_indices back to original ROI indices
                    target_v = valid_indices[0][object_indices]
                    target_u = valid_indices[1][object_indices]
                    full_object_mask[target_v, target_u] = True
                    search_mask = full_object_mask
                else:
                    # Fallback to original median if filtering removes everything
                    Z_proj = np.median(valid_vals)
                    search_mask = valid_mask
                
                _last_known_z[LM(i)] = float(Z_proj)

                # Find the pixel that provided the median for visualization
                median_pixel = None
                if not np.isnan(Z_proj):
                    diff_to_med = np.abs(current_roi - Z_proj)
                    diff_to_med[~search_mask] = np.inf
                    idx = np.argmin(diff_to_med)
                    median_pixel = np.unravel_index(idx, current_roi.shape)
            else:
                logger.debug(f"LM {i}: No valid depth in masked ROI at ({ru}, {rv})")
                Z_proj = np.nan
                median_pixel = None


            # Visualization logic
            if DEPTH_PROJECTION_DEBUG and should_viz_this_frame and LM(i) in arm_chain:
                status = "SUCCESS" if valid_vals.size > 0 else "NO_VALID_DEPTH"
                
                # Final guess projection for the yellow dot
                final_ud, final_vd = ud, vd
                z_for_dot = Z_proj if not np.isnan(Z_proj) else _last_known_z[LM(i)]
                res_final = backend.project_color_to_depth(u, v, float(z_for_dot))
                if res_final:
                    final_ud, final_vd = res_final
                
                viz_data.append({
                    "name": f"{LM(i).name}_{status}",
                    "u": u, "v": v, "ud": final_ud, "vd": final_vd, "r": r,
                    "v_start": v_start, "v_end": v_end,
                    "u_start": u_start, "u_end": u_end,
                    "diff_roi": diff_roi,
                    "z_proj": Z_proj,
                    "median_pixel": median_pixel,
                })

        else:
            for _ in range(3):
                res = color_to_depth_pixel(u, v, z_guess, K, backend, depth_m, frame.aligned_depth)
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
                    z_guess = float(Z_proj)
                else:
                    Z_proj = np.nan
                    break
        
        Z_final = Z_proj

        if not np.isnan(Z_final):
            points[LM(i)] = _pixel_to_3d(u, v, float(Z_final), fx, fy, cx, cy)

    # Save multi-joint visualization if data was collected
    if viz_data:
        from viki.skeleton.viz import visualize_depth_subtraction
        # Offload plotting to background thread to prevent pipeline freezes
        _viz_executor.submit(
            visualize_depth_subtraction,
            base_depth=frame.base_depth_m,
            current_depth=depth_m,
            landmark_data=viz_data
        )

    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )


def calculate_pitch(wrist_point, middle_finger_points):
    '''Here we use wrist_point (x, y, z) and middle_finger_points (list(x,y,z)) to get approximate
    vector of angle regarding to world-coordinates'''
    pass

def calculate_yaw(wrist_point, middle_finger_points):
    '''Here we use wrist_point (x, y, z) and middle_finger_points (list(x,y,z)) to get approximate
    vector of angle regarding to world-coordinates'''
    pass

def calculate_roll(wrist_point, thumbs_points):
    '''Here we use wrist_point (x, y, z) and thumb_points (list(x,y,z) to get approximate
    vector of angle regarding to world-coordinates'''
    pass

'''flexion/extension — сгибание/разгибание
radial/ulnar deviation — отведение в стороны
pronation/supination — ключевой поворот кисти'''

