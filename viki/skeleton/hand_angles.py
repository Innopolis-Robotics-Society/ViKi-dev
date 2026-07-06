"""
viki.skeleton.hand_angles
-------------------------
Palm-frame orientation from 3-D hand landmarks.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from viki.skeleton.models import EndEffectorPose, LM


MIN_VECTOR_NORM = 1e-6
NAN_VEC3 = np.full(3, np.nan, dtype=np.float32)
NAN_ROT = np.full((3, 3), np.nan, dtype=np.float32)
END_EFFECTOR_REQUIRED_LM: tuple[LM, ...] = (LM.WRIST, LM.THUMB_CMC, LM.MIDDLE_MCP)


def _normalise(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < MIN_VECTOR_NORM:
        return None
    return np.asarray(vec, dtype=np.float64) / norm


def _rot_to_rpy_extrinsic_xyz(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw in radians for R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    sy = -float(rotation[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = float(np.arcsin(sy))
    if abs(sy) > 1.0 - 1e-6:
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    else:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def invalid_end_effector_pose(timestamp_us: int) -> EndEffectorPose:
    return EndEffectorPose(
        position=NAN_VEC3.copy(),
        R_world_palm=NAN_ROT.copy(),
        rpy_deg=NAN_VEC3.copy(),
        valid=False,
        timestamp_us=timestamp_us,
    )


def compute_palm_rotation(
    wrist: np.ndarray,
    thumb_cmc: np.ndarray,
    middle_mcp: np.ndarray,
) -> np.ndarray | None:
    """
    Compute R_world_palm from wrist, thumb CMC, and middle MCP landmarks.

    Columns are x_palm, y_palm, z_palm in world coordinates.
    """
    wrist = np.asarray(wrist, dtype=np.float64)
    thumb_cmc = np.asarray(thumb_cmc, dtype=np.float64)
    middle_mcp = np.asarray(middle_mcp, dtype=np.float64)
    if not (
        np.all(np.isfinite(wrist))
        and np.all(np.isfinite(thumb_cmc))
        and np.all(np.isfinite(middle_mcp))
    ):
        return None

    to_middle = middle_mcp - wrist
    to_thumb = thumb_cmc - wrist
    x_palm = _normalise(to_middle)
    z_palm = _normalise(np.cross(to_middle, to_thumb))
    if x_palm is None or z_palm is None:
        return None

    y_palm = np.cross(z_palm, x_palm)
    y_palm = _normalise(y_palm)
    if y_palm is None:
        return None

    rotation = np.column_stack([x_palm, y_palm, z_palm])
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0
    return rotation.astype(np.float32)


def compute_end_effector_pose(
    points: Mapping[LM, np.ndarray],
    timestamp_us: int,
) -> EndEffectorPose:
    coords: dict[LM, np.ndarray] = {}
    for landmark in END_EFFECTOR_REQUIRED_LM:
        point = points.get(landmark)
        if point is None or not np.all(np.isfinite(point)):
            return invalid_end_effector_pose(timestamp_us)
        coords[landmark] = np.asarray(point, dtype=np.float64)

    rotation = compute_palm_rotation(
        coords[LM.WRIST],
        coords[LM.THUMB_CMC],
        coords[LM.MIDDLE_MCP],
    )
    if rotation is None:
        return invalid_end_effector_pose(timestamp_us)

    rpy_rad = _rot_to_rpy_extrinsic_xyz(rotation.astype(np.float64))
    return EndEffectorPose(
        position=coords[LM.WRIST].astype(np.float32),
        R_world_palm=rotation,
        rpy_deg=np.degrees(rpy_rad).astype(np.float32),
        valid=True,
        timestamp_us=timestamp_us,
    )
