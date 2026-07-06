"""
viki.skeleton.fusion
"""

from __future__ import annotations

import numpy as np
import viki.config

from viki.calibration.models import CalibrationExtrinsics
from viki.skeleton.hand_angles import compute_end_effector_pose
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame


def fuse(
    dev_ids: list[str],
    lms: dict[str, Landmarks3D | None],
    extrinsics: dict[str, CalibrationExtrinsics],
    timestamp_us: int,
    confidences: dict[str, dict[LM, float]] | None = None,
    bone_emas: dict[tuple[LM, LM], float] | None = None,
) -> SkeletonFrame:

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
                vec = np.full(3, np.nan, dtype=np.float32)

            pos_mtx = np.eye(4)
            pos_mtx[:3, 3] = vec
            world_vec = (T @ pos_mtx)[:3, 3].flatten()
            world_points[index] = world_vec

        for index, vec in world_points.items():
            if index not in observations:
                observations[index] = {}
            observations[index][dev_id] = vec

    if not observations:
        all_idxs = set(range(LM.N))
        points = {LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in all_idxs}
        return SkeletonFrame(
            points=points,
            timestamp_us=timestamp_us,
            end_effector=compute_end_effector_pose(points, timestamp_us),
        )

    # 1. Compute initial means
    out_points: dict[LM, np.ndarray] = {}
    for index, points in observations.items():
        weighted_sum = np.zeros(3)
        total_weight = 0.0

        for dev_id, vec in points.items():
            # Get confidence for this joint from this camera
            weight = 1.0
            if confidences and dev_id in confidences:
                weight = confidences[dev_id].get(index, 1.0)

            weighted_sum += vec * weight
            total_weight += weight

        if total_weight > 1e-6:
            out_points[index] = weighted_sum / total_weight
        else:
            # Fallback to simple mean if all weights are zero
            n = len(points)
            out_points[index] = np.zeros(3)
            for dev_id, vec in points.items():
                out_points[index] += vec
            out_points[index] /= n

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
        points=out_points,
        timestamp_us=timestamp_us,
        end_effector=compute_end_effector_pose(out_points, timestamp_us),
    )
