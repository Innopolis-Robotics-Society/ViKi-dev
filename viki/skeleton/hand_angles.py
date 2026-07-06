"""
viki.skeleton.hand_angles
-------------------------
<<<<<<< HEAD
Palm-frame orientation from 3-D hand landmarks.
=======
Hand orientation from a 3-D skeleton frame

1. compute_hand_angles is deprecated and used for live-demo usage only
Angles expressed in a *forearm-local* right-handed basis, so they
are invariant to overall arm/body rotation in the world


Required landmarks:
    WRIST, ELBOW, SHOULDER, THUMB_CMC, MIDDLE_MCP

Angles (all in degrees, sign consistent with the applied rotation):
    flexion_deg    : angle of ``to_middle`` in the (x, y) plane,
    deviation_deg  : angle of ``to_middle`` in the (x, z) plane,
    roll_deg       : angle of ``palm_normal`` in the (y, z) plane,


2. ``compute_end_effector_pose`` is world-frame

Returns the full world-frame pose of the wrist end-effector: 3-D position
plus a proper rotation matrix ``R_world_palm ∈ SO(3)`` from a palm-attached
frame to the world

Required landmarks:
    WRIST, THUMB_CMC, MIDDLE_MCP  (no elbow / shoulder needed).
>>>>>>> 5527db99f1e8525dbcca29c0e3ff6414454b266c
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from viki.skeleton.models import LM, EndEffectorPose

# Landmarks that must be present and finite for a valid computation.
REQUIRED_LM: tuple[LM, ...] = (
    LM.WRIST,
    LM.ELBOW,
    LM.SHOULDER,
    LM.THUMB_CMC,
    LM.MIDDLE_MCP,
)

_MIN_LEN = 1e-6  # zero-length vector threshold
_MIN_UP_REF_ORTHO = 0.05

_NAN_VEC3 = np.full(3, np.nan, dtype=np.float32)


@dataclass
class HandAngles:
    """
    Hand orientation summary in the forearm-local frame.

    fields
    ------
    flexion_deg   : palmar / dorsi flexion angle in degrees. (угол сгибания / разгибания кисти)
    deviation_deg : radial / ulnar deviation angle in degrees. (угол отведения в сторону большого пальца)
    roll_deg      : palm rotation around the forearm axis in degrees. (ротация вокруг оси предплечья)
    palm_normal   : (3,) float32 world-frame palm normal for visualisation.
    forearm_axis  : (3,) float32 world-frame forearm axis
    valid         : True when every required landmark was present and the
                    forearm-local frame could be resolved.
    """

    flexion_deg: float
    deviation_deg: float
    roll_deg: float
    palm_normal: np.ndarray = field(default_factory=lambda: _NAN_VEC3.copy())
    forearm_axis: np.ndarray = field(default_factory=lambda: _NAN_VEC3.copy())
    valid: bool = False


def _normalise(v: np.ndarray) -> np.ndarray | None:
    """
    parameters
    ----------
    v : (3,) vector.

    returns
    -------
    v / |v|, or None if |v| < _MIN_LEN.
    """
    n = float(np.linalg.norm(v))
    if n < _MIN_LEN:
        return None
    return v / n


def _invalid() -> HandAngles:
    """Return a fully-NaN HandAngles with valid=False."""
    return HandAngles(
        flexion_deg=float("nan"),
        deviation_deg=float("nan"),
        roll_deg=float("nan"),
        palm_normal=_NAN_VEC3.copy(),
        forearm_axis=_NAN_VEC3.copy(),
        valid=False,
    )


def compute_hand_angles(points: Mapping[LM, np.ndarray]) -> HandAngles:
    """
    Compute forearm-local flexion / deviation / roll from a landmark dict.

    parameters
    ----------
    points : mapping of LM enum to world-frame position in metres.
             ``SkeletonFrame.points`` or ``Landmarks3D.points``

    returns
    -------
    HandAngles. On success `.valid == True` and every scalar / vector is
    finite. On failure (missing / NaN landmarks, or arm ~straight) the
    result is fully NaN with `.valid == False`.
    """
    coords: dict[LM, np.ndarray] = {}
    for lm in REQUIRED_LM:
        p = points.get(lm)
        if p is None or not np.all(np.isfinite(p)):
            return _invalid()
        coords[lm] = np.asarray(p, dtype=np.float64)

    wrist = coords[LM.WRIST]
    elbow = coords[LM.ELBOW]
    shoulder = coords[LM.SHOULDER]
    thumb = coords[LM.THUMB_CMC]
    middle = coords[LM.MIDDLE_MCP]

    # 1. Local x: forearm direction, elbow to wrist.
    x = _normalise(wrist - elbow)
    if x is None:
        return _invalid()

    # 2. Local y: upper-arm direction, orthogonalised against x.
    up_ref = shoulder - elbow
    y_raw = up_ref - float(np.dot(up_ref, x)) * x
    y_norm = float(np.linalg.norm(y_raw))
    if y_norm < _MIN_UP_REF_ORTHO:
        return _invalid()
    y = y_raw / y_norm
    z = np.cross(x, y)

    # 3. Palm basis.
    to_middle = _normalise(middle - wrist)
    to_thumb = _normalise(thumb - wrist)
    if to_middle is None or to_thumb is None:
        return _invalid()
    palm_normal = _normalise(np.cross(to_middle, to_thumb))
    if palm_normal is None:
        return _invalid()

    # 4. Angles in the local frame.
    tm_x = float(np.dot(to_middle, x))
    tm_y = float(np.dot(to_middle, y))
    tm_z = float(np.dot(to_middle, z))
    pn_y = float(np.dot(palm_normal, y))
    pn_z = float(np.dot(palm_normal, z))

    flexion_deg = float(np.degrees(np.arctan2(tm_y, tm_x)))
    deviation_deg = float(np.degrees(np.arctan2(-tm_z, tm_x)))
    roll_deg = float(np.degrees(np.arctan2(pn_z, pn_y)))

    return HandAngles(
        flexion_deg=flexion_deg,
        deviation_deg=deviation_deg,
        roll_deg=roll_deg,
        palm_normal=palm_normal.astype(np.float32),
        forearm_axis=x.astype(np.float32),
        valid=True,
    )


# Landmarks required to build the palm frame in the world.
_EE_REQUIRED_LM: tuple[LM, ...] = (LM.WRIST, LM.THUMB_CMC, LM.MIDDLE_MCP)


def _rot_to_rpy_extrinsic_xyz(R: np.ndarray) -> np.ndarray:
    """
    Extract roll/pitch/yaw (radians) from a rotation
    R = Rz(yaw) · Ry(pitch) · Rx(roll)  (extrinsic XYZ).

    """
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = float(np.arcsin(sy))
    if abs(sy) > 1.0 - 1e-6:
        # Gimbal lock
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _invalid_pose(timestamp_us: int) -> EndEffectorPose:
    return EndEffectorPose(
        position=_NAN_VEC3.copy(),
        R_world_palm=np.full((3, 3), np.nan, dtype=np.float32),
        rpy_deg=_NAN_VEC3.copy(),
        valid=False,
        timestamp_us=timestamp_us,
    )


def compute_end_effector_pose(
    points: Mapping[LM, np.ndarray],
    timestamp_us: int,
) -> EndEffectorPose:
    """
    Compute the world-frame pose of the wrist end-effector from a fused
    skeleton.

    Parameters
    ----------
    points       : mapping LM to world-frame metres.
    timestamp_us : timestamp to embed in the returned pose.

    Returns
    -------
    EndEffectorPose
    """
    coords: dict[LM, np.ndarray] = {}
    for lm in _EE_REQUIRED_LM:
        p = points.get(lm)
        if p is None or not np.all(np.isfinite(p)):
            return _invalid_pose(timestamp_us)
        coords[lm] = np.asarray(p, dtype=np.float64)

    wrist = coords[LM.WRIST]
    to_middle = coords[LM.MIDDLE_MCP] - wrist
    to_thumb = coords[LM.THUMB_CMC] - wrist

    x_palm = _normalise(to_middle)
    z_palm = _normalise(np.cross(to_middle, to_thumb))
    if x_palm is None or z_palm is None:
        return _invalid_pose(timestamp_us)

    # Re-orthogonalise y against x and z, in case the three landmarks are not perfectly coplanar.
    y_palm = np.cross(z_palm, x_palm)
    y_norm = float(np.linalg.norm(y_palm))
    if y_norm < _MIN_LEN:
        return _invalid_pose(timestamp_us)
    y_palm = y_palm / y_norm

    R = np.column_stack([x_palm, y_palm, z_palm]).astype(np.float32)
    rpy_rad = _rot_to_rpy_extrinsic_xyz(R.astype(np.float64))
    rpy_deg = np.degrees(rpy_rad).astype(np.float32)

    return EndEffectorPose(
        position=wrist.astype(np.float32),
        R_world_palm=R,
        rpy_deg=rpy_deg,
        valid=True,
        timestamp_us=timestamp_us,
    )
