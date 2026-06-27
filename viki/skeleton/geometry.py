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


def color_to_depth_pixel(u: float, v: float, Z: float, K: np.ndarray, backend: KinectBackend) -> tuple[float, float] | None:
    """
    Maps a pixel from the color camera to the depth camera coordinate space using SDK calibration.
    
    Args:
        u, v: Color pixel coordinates (undistorted)
        Z: Depth in metres
        K: Color intrinsic matrix (3, 3)
        backend: Kinect backend providing SDK calibration
    
    Returns:
        (u_depth, v_depth) or None if projection fails.
    """
    # 1. Color Pixel (undistorted) -> 3D Point in Color Space
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    p_color = _pixel_to_3d(u, v, Z, fx, fy, cx, cy)
    
    # 2. Color Space -> Depth Space
    from viki.capture.kinect import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH
    p_depth = backend.transform_3d_to_3d(
        p_color[0], p_color[1], p_color[2], 
        K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH
    )
    
    if p_depth is None:
        logger.debug(f"SDK transform_3d_to_3d failed for pixel ({u}, {v}) at Z={Z}")
        return None
    
    # 3. Depth 3D Point -> Depth Pixel
    res = backend.project_3d_to_2d(
        p_depth[0], p_depth[1], p_depth[2], 
        K4A_CALIBRATION_TYPE_DEPTH
    )
    if res is None:
        logger.debug(f"SDK project_3d_to_2d failed for P_depth={p_depth} (u={u}, v={v}, Z={Z})")
        
    return res

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
    Deproject all 23 pixel landmarks into 3-D camera space.
    
    For each landmark, it's checked whether depth_m has a valid value.
    
    Parameters
    ----------
    detection : HandDetection
        23 pixel-space landmarks from CompositeLandmarkDetector.
    frame : PreparedFrame
        Provides depth_m and intrinsic matrix K.
    backend : KinectBackend
        Backend used for SDK-based reprojection.
    
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

    mp_z_scale = _wrist_scale(detection.points[LM.WRIST], float(detection.lm_z_rel[LM.WRIST]), depth_m)
    if mp_z_scale is None:
        z_rel_wrist = float(detection.lm_z_rel[LM.WRIST])
        if z_rel_wrist != 0.0:
            mp_z_scale = _FALLBACK_WRIST_Z_M / z_rel_wrist
        # logger.debug("did not find the wrist, using fallback")
 
    points = {LM(idx): np.full(2, np.nan, dtype=np.float32) for idx in range(LM.N)}
    for i in range(LM.N):
        u, v = detection.points[LM(i)][0], detection.points[LM(i)][1]
        if np.isnan(u) or np.isnan(v):
            continue
        
        # Iterative depth sampling to find the correct depth pixel
        # Initial guess for depth
        Z_guess = 1.0 
        ui, vi = None, None
        
        for _ in range(3):
            res = color_to_depth_pixel(u, v, Z_guess, K, backend)
            if res is None:
                break
            
            u_depth, v_depth = res
            ui, vi = int(round(u_depth)), int(round(v_depth))
            
            if not (0 <= vi < h and 0 <= ui < w):
                break
                
            # Sample depth in a 3x3 window
            v_start, v_end = max(0, vi - 1), min(h, vi + 2)
            u_start, u_end = max(0, ui - 1), min(w, ui + 2)
            window = depth_m[v_start:v_end, u_start:u_end]
            valid_window = window[~np.isnan(window)]
            
            if valid_window.size > 0:
                Z_guess = np.median(valid_window)
            else:
                break
        
        if ui is None or vi is None:
            continue

        # Final robust sampling at the found location
        v_start, v_end = max(0, vi - 1), min(h, vi + 2)
        u_start, u_end = max(0, ui - 1), min(w, ui + 2)
        window = depth_m[v_start:v_end, u_start:u_end]
        valid_window = window[~np.isnan(window)]
        
        if valid_window.size > 0:
            Z = np.median(valid_window)
        else:
            Z = np.nan
        
        if not np.isnan(Z):
            # Valid depth
            points[LM(i)] = _pixel_to_3d(u, v, Z, fx, fy, cx, cy)
        elif mp_z_scale is not None:
            # No depth, we estimate Z
            Z_approx = float(detection.lm_z_rel[i]) * mp_z_scale
            if Z_approx > 0:
                points[LM(i)] = _pixel_to_3d(u, v, Z_approx, fx, fy, cx, cy)
    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )
