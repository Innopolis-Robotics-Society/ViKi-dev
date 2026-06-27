"""
viki.skeleton.fusion
<<<<<<< HEAD
--------------------

Currently it fused LandMarks3D from two cameras into a single SkeletonFrame
based on simple priority rules.
In the future we could experiment with complex fusion strategies (per mark stuff etc)
e.g. Kalman Filter, or just weighted sum, or confidence based approaches

Current Fusion strategy (per landmark)
-------------------------------
Priority 1 — kinect_0 DEPTH     : best quality, no transform needed.
Priority 2 — kinect_1 DEPTH     : transform P_cam0 = R @ P_cam1 + T.
Priority 3 — kinect_0 MP_Z      : metric but less accurate.
Priority 4 — kinect_1 MP_Z      : transform same as above.
Priority 5 — MISSING from both  : point stays nan, source = MISSING.

=======
>>>>>>> feat/fusion
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
