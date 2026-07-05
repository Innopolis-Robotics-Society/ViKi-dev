"""
viki.skeleton.hand_angles
-------------------------
Compute hand orientation angles from a 3-D skeleton frame.

The angles are expressed in a *forearm-local* right-handed basis, so they
are invariant to overall arm/body rotation in the world:

    x  =  normalize(WRIST - ELBOW)                    # forearm axis
    y  =  normalize(SHOULDER - ELBOW  ⟂ x)            # arm plane "up"
    z  =  cross(x, y)                                 # arm plane normal

Required landmarks
------------------
    WRIST, ELBOW, SHOULDER, THUMB_CMC, MIDDLE_MCP

If any of these is missing (absent from the mapping or contains NaN) or
if the arm is (nearly) straight — so that `y` cannot be resolved — the
result is returned with `.valid == False` and every scalar / vector set
to NaN.

Angles (all in degrees, sign consistent with the applied rotation)
------------------------------------------------------------------
flexion_deg    : angle of `to_middle` in the (x, y) plane, `arctan2(y, x)`.
                 A rotation of `to_middle` around the +z axis by +φ
                 produces `flexion_deg == +φ`.
deviation_deg  : angle of `to_middle` in the (x, z) plane, `arctan2(-z, x)`.
                 A rotation of `to_middle` around the +y axis by +φ
                 produces `deviation_deg == +φ`.
roll_deg       : angle of `palm_normal` in the (y, z) plane,
                 `arctan2(z, y)`. A rotation of the palm around the +x
                 axis (forearm) by +φ produces `roll_deg == +φ`.

The reference direction of the palm normal is
    palm_normal = normalize(cross(to_middle, to_thumb))
which points to the dorsal side for a right hand and to the palmar side
for a left hand (a consequence of the anchor point ordering). This flip
does not affect the roll change — only its offset from a chosen "zero".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from viki.skeleton.models import LM

# Landmarks that must be present and finite for a valid computation.
REQUIRED_LM: tuple[LM, ...] = (
    LM.WRIST,
    LM.ELBOW,
    LM.SHOULDER,
    LM.THUMB_CMC,
    LM.MIDDLE_MCP,
)

# Numerical guards.
_MIN_LEN = 1e-6           # zero-length vector threshold
_MIN_UP_REF_ORTHO = 0.05  # min |up_ref ⟂ forearm|; below → arm ≈ straight

_NAN_VEC3 = np.full(3, np.nan, dtype=np.float32)


@dataclass
class HandAngles:
    """
    Hand orientation summary in the forearm-local frame.

    fields
    ------
    flexion_deg   : palmar / dorsi flexion angle in degrees.
    deviation_deg : radial / ulnar deviation angle in degrees.
    roll_deg      : palm rotation around the forearm axis in degrees.
    palm_normal   : (3,) float32 world-frame palm normal for visualisation.
    forearm_axis  : (3,) float32 world-frame forearm axis (elbow→wrist).
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
    points : mapping of LM enum → (3,) world-frame position in metres.
             Typically ``SkeletonFrame.points`` or ``Landmarks3D.points``.

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

    # 1. Local x: forearm direction, elbow → wrist.
    x = _normalise(wrist - elbow)
    if x is None:
        return _invalid()

    # 2. Local y: upper-arm direction, orthogonalised against x.
    #    If the arm is straight, up_ref ∥ x and y is ill-defined.
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
