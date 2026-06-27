"""
viki.skeleton.fusion
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

"""

from __future__ import annotations

import numpy as np

from viki.skeleton.models import LandmarkSource, Landmarks3D, LM, SkeletonFrame

# Priority order used in fuse() — lower index wins.
_PRIORITY = [
    (LandmarkSource.DEPTH, "master"),
    (LandmarkSource.DEPTH, "subordinate"),
    (LandmarkSource.MP_Z,  "master"),
    (LandmarkSource.MP_Z,  "subordinate"),
]


def _transform_to_cam0(points: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform (N, 3) points from cam1 frame to cam0 frame.
    """
    return (R @ points.T + T).T.astype(np.float32)


def fuse(
    lm0: Landmarks3D | None,
    lm1: Landmarks3D | None,
    R: np.ndarray,
    T: np.ndarray,
    timestamp_us: int,
) -> SkeletonFrame | None:
    """
    Merge two per-camera Landmarks3D into one SkeletonFrame in kinect_0 space.

    Returns None only if both are None.

    Parameters
    ----------
    lm0 : Landmarks3D | None
        Landmarks in kinect_0 frame.
    lm1 : Landmarks3D | None
        Landmarks in kinect_1 frame (will be transformed via R, T).
    R, T : extrinsic matrices from load_extrinsics().
    timestamp_us : int
        sync_timestamp_us from SyncedFrameGroup.

    Returns
    -------
    SkeletonFrame | None
    """
    if lm0 is None and lm1 is None:
        return None

    # 23 points transformation
    lm1_in_cam0: np.ndarray | None = None
    if lm1 is not None:
        lm1_in_cam0 = _transform_to_cam0(lm1.points, R, T)  # (23, 3)

    # Output arrays
    out_points     = np.full((LM.N, 3), np.nan, dtype=np.float32)
    out_source     = np.array([LandmarkSource.MISSING] * LM.N, dtype=object)
    out_confidence = np.zeros(LM.N, dtype=np.float32)
    out_origin     = np.array(["missing"] * LM.N, dtype=object)

    for i in range(LM.N):
        for src, cam_id in _PRIORITY:
            if cam_id == "kinect_0":
                lm = lm0
                pts = lm0.points if lm0 is not None else None
            else:
                lm = lm1
                pts = lm1_in_cam0

            if lm is None or pts is None:
                continue
            if lm.source[i] != src:
                continue
            if np.isnan(pts[i]).any():
                continue

            out_points[i]     = pts[i]
            out_source[i]     = src
            out_confidence[i] = 1.0 if src == LandmarkSource.DEPTH else 0.5
            out_origin[i]     = cam_id
            break  # first match wins

    return SkeletonFrame(
        landmarks=out_points,
        source=out_source,
        confidence=out_confidence,
        origin=out_origin,
        timestamp_us=timestamp_us,
    )
