"""
viki.skeleton.fusion
--------------------
Fuse per‑camera 3D landmark observations into a single world‑frame skeleton.

The fusion step uses extrinsic calibration (transform matrices) to convert
each camera's 3D landmarks to world coordinates, then performs weighted
averaging across cameras (weights can be confidence-based).
"""

from __future__ import annotations

import numpy as np

from viki.calibration.models import CalibrationExtrinsics
from viki.skeleton.hand_angles import compute_end_effector_pose
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame


def fuse(
    dev_ids: list[str],
    lms: dict[str, Landmarks3D | None],
    extrinsics: dict[str, CalibrationExtrinsics],
    timestamp_us: int,
    confidences: dict[str, dict[LM, float]] | None = None,
) -> SkeletonFrame:
    """
    Fuse per‑camera 3D landmarks into a single world‑frame skeleton.

    For each camera with valid landmarks and extrinsics, the landmarks are
    transformed into the world coordinate system using the camera's
    `transform_matrix`. Landmarks from all cameras are then aggregated
    using a weighted average (by per‑landmark confidence, default 1.0).

    Parameters
    ----------
    dev_ids : list[str]
        List of camera device IDs (order does not matter).
    lms : dict[str, Landmarks3D | None]
        Per‑camera 3D landmarks (or None if detection failed).
    extrinsics : dict[str, CalibrationExtrinsics]
        Extrinsic parameters for each camera.
    timestamp_us : int
        Timestamp (µs) of the fused frame.
    confidences : dict[str, dict[LM, float]], optional
        Per‑camera, per‑landmark confidence values (0..1). If omitted, all weights are 1.

    Returns
    -------
    SkeletonFrame
        Fused skeleton with world‑frame landmarks and end‑effector pose.
        If no valid observations, all landmarks are set to NaN.
    """
    observations: dict[LM, dict[str, np.ndarray]] = {}

    for dev_id in dev_ids:
        lm = lms.get(dev_id)
        if lm is None:
            continue

        extr = extrinsics.get(dev_id)
        if not extr:
            continue
        T = extr.transform_matrix
        # Invert the World-to-Camera transform to get Camera-to-World
        # T_inv = np.linalg.inv(T)

        ps = lm.points
        world_points: dict[LM, np.ndarray] = {}
        for index, vec in ps.items():
            if len(vec.flatten()) != 3 or np.isnan(vec).any():
                # Skip landmarks with no valid 3D observation from this camera
                # rather than polluting the per-landmark set with NaN, which
                # would poison the weighted average below.
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
            # A camera with no valid observation for this landmark stored a NaN
            # in `points` (kept above for clarity). Skip it so a single missing
            # observation cannot poison the whole landmark via NaN propagation.
            if np.isnan(vec).any():
                continue

            # Get confidence for this joint from this camera
            weight = 1.0
            if confidences and dev_id in confidences:
                weight = confidences[dev_id].get(index, 1.0)

            weighted_sum += vec * weight
            total_weight += weight

        if total_weight > 1e-6:
            out_points[index] = weighted_sum / total_weight
        else:
            # All weights zero — leave as NaN rather than a fake world origin.
            out_points[index] = np.full(3, np.nan, dtype=np.float32)

    return SkeletonFrame(
        points=out_points,
        timestamp_us=timestamp_us,
        end_effector=compute_end_effector_pose(out_points, timestamp_us),
    )
