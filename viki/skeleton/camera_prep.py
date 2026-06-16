"""
viki.skeleton.camera_prep
-------------------------
Converts a raw capture Frame into a PreparedFrame ready for model inference.

This module depends on viki.capture.base (Frame) and viki.skeleton.models
(PreparedFrame). It has no dependency on MediaPipe or CameraManager.
"""

from __future__ import annotations

import cv2
import numpy as np

from viki.capture.base import Frame
from viki.skeleton.models import PreparedFrame


class UndistortCache:
    """
    Caches cv2.initUndistortRectifyMap results per device_id.

    Computing the remap maps is expensive (~1ms). At 30fps across two cameras
    that is 60ms/s wasted if recomputed every frame. This cache computes each
    map exactly once and reuses it for the lifetime of the pipeline.

    Usage
    -----
    cache = UndistortCache()
    map1, map2 = cache.get(device_id, K, dist, (w, h))
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    """

    def __init__(self) -> None:
        # device_id → (map1, map2)
        self._maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(
        self,
        device_id: str,
        K: np.ndarray,
        dist: np.ndarray,
        shape: tuple[int, int],  # (width, height)
    ) -> tuple[np.ndarray, np.ndarray]:
        if device_id not in self._maps:
            w, h = shape
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist, None, K, (w, h), cv2.CV_32FC1
            )
            self._maps[device_id] = (map1, map2)
        return self._maps[device_id]

    def invalidate(self, device_id: str | None = None) -> None:
        """Clear cached maps. Pass device_id to clear one camera, None to clear all."""
        if device_id is None:
            self._maps.clear()
        else:
            self._maps.pop(device_id, None)


def prepare_frame(
    frame: Frame,
    K: np.ndarray,
    dist: np.ndarray,
    cache: UndistortCache,
) -> PreparedFrame:
    """
    Convert a raw Frame into a PreparedFrame.

    Parameters
    ----------
    frame : Frame
        Raw frame from CameraManager (colour BGR with distortion, depth uint16 mm).
    K : np.ndarray
        (3, 3) intrinsic matrix for this camera (from calibration).
    dist : np.ndarray
        Distortion coefficients [k1, k2, p1, p2, k3] (from calibration).
    cache : UndistortCache
        Shared cache — pass the same instance for all frames from the same session.

    Returns
    -------
    PreparedFrame
        Ready for hand detection and geometry lifting.
    """
    h, w = frame.color.shape[:2]

    # Undistort 
    map1, map2 = cache.get(frame.device_id, K, dist, (w, h))
    color_undist = cv2.remap(frame.color, map1, map2, cv2.INTER_LINEAR)

    # BGR - RGB
    rgb = cv2.cvtColor(color_undist, cv2.COLOR_BGR2RGB)

    # Depth clean. Setting 0 as nan
    depth = frame.depth.astype(np.float32)
    depth[depth == 0] = np.nan
    depth_m = depth / 1000.0

    return PreparedFrame(
        rgb=rgb,
        depth_m=depth_m,
        K=K,
        device_id=frame.device_id,
        timestamp_us=frame.timestamp_us,
    )
