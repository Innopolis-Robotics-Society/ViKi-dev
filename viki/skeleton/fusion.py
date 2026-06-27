"""
viki.skeleton.fusion
"""

from __future__ import annotations

import numpy as np

from viki.calibration.models import CalibrationExtrinsics
from viki.skeleton.models import Landmarks3D, LM, SkeletonFrame


def fuse(
    dev_ids: list[str],
    lms: dict[str, Landmarks3D | None],
    extrinsics: dict[str, CalibrationExtrinsics],
    timestamp_us: int,
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
        for index, vec in ps.items():

            pos_mtx = np.eye(4)
            pos_mtx[:3, 3] = vec
            world_vec = (pos_mtx @ T)[:3, 3].flatten()

            observations[index][dev_id] = world_vec

    if not observations:
        return None

    out_points: dict[LM, np.ndarray] = {}
    for index, points in observations.items():

        n = len(points)
        mean_vec = np.zeros(3)
        for dev_id, vec in points.items():
            mean_vec += vec
        mean_vec /= n

        out_points[index] = mean_vec

    return SkeletonFrame(
        out_points,
        timestamp_us,
    )
