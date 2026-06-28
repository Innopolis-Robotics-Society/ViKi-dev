"""
viki.skeleton.fusion
"""

from __future__ import annotations

import numpy as np
import viki.config

from viki.calibration.models import CalibrationExtrinsics
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame


def fuse(
    dev_ids: list[str],
    lms: dict[str, Landmarks3D | None],
    extrinsics: dict[str, CalibrationExtrinsics],
    timestamp_us: int,
    bone_emas: dict[tuple[LM, LM], float] | None = None,
) -> SkeletonFrame | None:

    observations: dict[LM, dict[str, np.ndarray]] = {}

    for dev_id in dev_ids:
        lm = lms.get(dev_id)
        if lm is None:
            continue

        extr = extrinsics.get(dev_id)
        if not extr:
            continue
        T = extr.trasnform_matrix

        ps = lm.points
        world_points: dict[LM, np.ndarray] = {}
        for index, vec in ps.items():
            if len(vec.flatten()) != 3 or np.isnan(vec).any():
                continue
            
            pos_mtx = np.eye(4)
            pos_mtx[:3, 3] = vec
            world_vec = (T @ pos_mtx)[:3, 3].flatten()
            world_points[index] = world_vec

        for index, vec in world_points.items():
            if index not in observations:
                observations[index] = {}
            observations[index][dev_id] = vec

    if not observations:
        return None

    # 1. Compute initial means
    out_points: dict[LM, np.ndarray] = {}
    for index, points in observations.items():
        n = len(points)
        mean_vec = np.zeros(3)
        for dev_id, vec in points.items():
            mean_vec += vec
        out_points[index] = mean_vec / n

    # 2. Apply Kinematic Constraints (Arm Chain)
    # Shoulder (22) -> Elbow (21) -> Wrist (0)
    chain = [(LM.SHOULDER, LM.ELBOW), (LM.ELBOW, LM.WRIST)]
    tolerance = viki.config.BONE_TOLERANCE
    for parent, child in chain:
        if parent in out_points and child in out_points:
            # Priority: Manual Config > EMA
            target_len = viki.config.BONE_LENGTHS.get((parent, child))
            if target_len is None and bone_emas is not None:
                target_len = bone_emas.get((parent, child))
            
            if target_len is None:
                continue
            
            # Project child along the observed direction
            dir_vec = out_points[child] - out_points[parent]
            dist = np.linalg.norm(dir_vec)
            if dist > 1e-4:
                # Apply soft constraint: only clip if outside [target * (1-TOL), target * (1+TOL)]
                lower = target_len * (1.0 - tolerance)
                upper = target_len * (1.0 + tolerance)
                
                clipped_dist = np.clip(dist, lower, upper)
                out_points[child] = out_points[parent] + (dir_vec / dist) * clipped_dist

    return SkeletonFrame(
        out_points,
        timestamp_us,
    )
