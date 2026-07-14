"""Palm-frame orientation helpers for optimisation inputs."""

from __future__ import annotations

import numpy as np


_MIN_LEN = 1e-6


def _normalize(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < _MIN_LEN:
        return None
    return v / n


def compute_palm_rotation(
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    pinky_mcp: np.ndarray,
) -> np.ndarray | None:
    """Build R_world_palm from wrist and MCP knuckle spread points."""
    coords = [np.asarray(p, dtype=np.float64) for p in (wrist, index_mcp, middle_mcp, pinky_mcp)]
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in coords):
        return None

    fwd = coords[2] - coords[0]               # MIDDLE_MCP - WRIST
    spread = coords[3] - coords[1]             # PINKY_MCP - INDEX_MCP
    x_palm = _normalize(fwd)
    z_palm = _normalize(np.cross(fwd, spread))
    if x_palm is None or z_palm is None:
        return None
    y_palm = _normalize(np.cross(z_palm, x_palm))
    if y_palm is None:
        return None

    rotation = np.column_stack([x_palm, y_palm, z_palm])
    if not np.isfinite(rotation).all():
        return None
    if float(np.linalg.det(rotation)) < 0.0:
        return None
    return rotation
