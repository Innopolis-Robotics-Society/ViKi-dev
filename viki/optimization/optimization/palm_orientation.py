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
    thumb_cmc: np.ndarray,
    middle_mcp: np.ndarray,
) -> np.ndarray | None:
    """Build R_world_palm from wrist, thumb CMC, and middle MCP points."""
    wrist = np.asarray(wrist, dtype=np.float64)
    thumb_cmc = np.asarray(thumb_cmc, dtype=np.float64)
    middle_mcp = np.asarray(middle_mcp, dtype=np.float64)
    if wrist.shape != (3,) or thumb_cmc.shape != (3,) or middle_mcp.shape != (3,):
        return None
    if not (np.isfinite(wrist).all() and np.isfinite(thumb_cmc).all() and np.isfinite(middle_mcp).all()):
        return None

    to_middle = middle_mcp - wrist
    to_thumb = thumb_cmc - wrist
    x_palm = _normalize(to_middle)
    z_palm = _normalize(np.cross(to_middle, to_thumb))
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
