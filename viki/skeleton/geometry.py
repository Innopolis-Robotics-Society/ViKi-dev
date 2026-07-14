"""
viki.skeleton.geometry
----------------------
Mapping 2D points to 3D space using depth and camera intrinsics.

Z estimation uses MediaPipe z_rel for hand-internal depth structure
and depth-camera median for absolute hand position.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import logging

logger = logging.getLogger(__name__)

from viki.skeleton.models import HandDetection, Landmarks3D, LM, PreparedFrame

_Z_SCALE_FALLBACK = 0.1  # meters per z_rel unit, when least-squares fails


def lift_to_3d(
    detection: HandDetection, frame: PreparedFrame, backend: Any
) -> Landmarks3D:
    """
    Deproject pixel landmarks to 3‑D camera space.

    For each landmark:
      1. Project color pixel (u, v) to depth-camera space via SDK calibration.
      2. Collect valid depth Z values across ALL landmarks.
      3. ``hand_z = median(valid_Zs)`` — robust absolute hand position.
      4. Estimate ``z_scale`` converting MediaPipe z_rel to metres (least‑squares
         against depth samples, or fallback constant).
      5. ``Z_i = hand_z + z_scale * z_rel[i]`` — per‑landmark depth.
      6. Deproject (u_depth, v_depth, Z_i) to (X, Y, Z).

    No ROI sampling, no background subtraction, no per‑landmark depth
    variability.  The function never blinks when MediaPipe is stable
    and at least a few depth pixels are valid.

    Parameters
    ----------
    detection : HandDetection
        2D landmark detections (pixel coordinates) from MediaPipe.
    frame : PreparedFrame
        Prepared frame with depth_m (metres) and depth_K.
    backend : Any
        Camera backend providing ``project_color_to_depth`` and
        optionally ``deproject_2d_to_3d``.

    Returns
    -------
    Landmarks3D
        3D landmarks in camera coordinates.
    """
    depth_m = frame.depth_m
    h, w = depth_m.shape[:2]
    if h == 0 or w == 0:
        return Landmarks3D(
            points={LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)},
            device_id=detection.device_id,
            timestamp_us=detection.timestamp_us,
        )

    # Phase 1 — project all landmarks, collect valid Z + z_rel pairs
    proj_data: list[tuple[int, float, float, float] | None] = [None] * LM.N
    all_z: list[float] = []
    all_zrel: list[float] = []

    for i in range(LM.N):
        uv = detection.points[LM(i)]
        u, v = uv[0], uv[1]
        if np.isnan(u) or np.isnan(v):
            continue

        res = backend.project_color_to_depth(u, v, 1.0)
        if res is None:
            continue

        ud, vd = res
        ui, vi = int(round(ud)), int(round(vd))

        if not (0 <= vi < h and 0 <= ui < w):
            continue

        z = depth_m[vi, ui]
        proj_data[i] = (i, ud, vd, z)
        if not np.isnan(z) and z > 0.01:
            all_z.append(z)
            all_zrel.append(detection.lm_z_rel[i])

    # Phase 2 — robust hand Z
    if len(all_z) < 3:
        nan_count = sum(1 for p in proj_data if p is not None)
        logger.warning(
            "lift_to_3d: only %d/%d valid depth samples for %s frame %d",
            len(all_z), nan_count, detection.device_id, detection.timestamp_us,
        )
        return Landmarks3D(
            points={LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)},
            device_id=detection.device_id,
            timestamp_us=detection.timestamp_us,
        )

    hand_z = float(np.median(all_z))

    # Phase 3 — z_scale from least-squares or fallback
    if len(all_z) >= 3:
        A = np.column_stack([np.ones(len(all_z), dtype=np.float32), np.array(all_zrel, dtype=np.float32)])
        coeffs, _, _, _ = np.linalg.lstsq(A, np.array(all_z, dtype=np.float32), rcond=None)
        est_hand_z = float(coeffs[0])
        z_scale = float(coeffs[1])
        if 0.01 <= z_scale <= 1.0:
            hand_z = est_hand_z
        else:
            z_scale = _Z_SCALE_FALLBACK
    else:
        z_scale = _Z_SCALE_FALLBACK

    # Phase 4 — per-landmark reconstruction
    points: dict[LM, np.ndarray] = {LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)}

    for data in proj_data:
        if data is None:
            continue
        i, ud, vd, z_raw = data

        Z = hand_z + detection.lm_z_rel[i] * z_scale
        Z = max(Z, 0.05)

        # Deproject — prefer SDK, fall back to pinhole
        xyz = None
        if hasattr(backend, 'deproject_2d_to_3d'):
            xyz = backend.deproject_2d_to_3d(ud, vd, Z)

        if xyz is not None:
            points[LM(i)] = np.array(xyz, dtype=np.float32)
        else:
            K = frame.depth_K if frame.depth_K is not None else frame.K
            if K is not None and K[0, 0] > 0:
                X = (ud - K[0, 2]) * Z / K[0, 0]
                Y = (vd - K[1, 2]) * Z / K[1, 1]
                points[LM(i)] = np.array([X, Y, Z], dtype=np.float32)

    nan_count = sum(1 for p in points.values() if np.isnan(p).any())
    if nan_count > 0:
        logger.warning(
            "lift_to_3d: %d/%d landmarks NaN for %s frame %d",
            nan_count, LM.N, detection.device_id, detection.timestamp_us,
        )

    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )
