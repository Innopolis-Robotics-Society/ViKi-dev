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

    Recomputing remap tables for every frame is expensive; this cache stores
    the tables once per camera and image size, so subsequent frames reuse them.

    Attributes
    ----------
    _maps : dict[str, tuple[np.ndarray, np.ndarray]]
        Maps device_id -> (map1, map2) remap tables.
    """

    def __init__(self) -> None:
        self._maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(
        self,
        device_id: str,
        K: np.ndarray,
        dist: np.ndarray,
        shape: tuple[int, int],  # (width, height)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve or compute the undistortion remap tables for a camera.

        Parameters
        ----------
        device_id : str
            Camera identifier.
        K : np.ndarray
            3x3 intrinsic matrix.
        dist : np.ndarray
            Distortion coefficients (length 4 or 5).
        shape : tuple[int, int]
            Image size as (width, height).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (map1, map2) as required by cv2.remap.
        """
        if device_id not in self._maps:
            w, h = shape
            # единожды вычисляем карту выпрямления и кэшируем её для повторного использования
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist, None, K, (w, h), cv2.CV_32FC1
            )
            self._maps[device_id] = (map1, map2)
        return self._maps[device_id]

    def invalidate(self, device_id: str | None = None) -> None:
        """
        Clear cached maps.

        Parameters
        ----------
        device_id : str, optional
            If provided, remove only the cache for that device.
            If None, clear all caches.
        """
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

    The colour image is undistorted and converted to RGB.
    The depth image is undistorted, converted to metres, and invalid zeros
    are set to NaN.

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
    depth_undist = cv2.remap(frame.depth.astype(np.float32), map1, map2, cv2.INTER_NEAREST)


    # BGR - RGB
    rgb = cv2.cvtColor(color_undist, cv2.COLOR_BGR2RGB)

    # Depth clean. Setting 0 as nan
    depth = depth_undist
    depth[depth == 0] = np.nan
    depth_m = depth / 1000.0


    return PreparedFrame(
        rgb=rgb,
        depth_m=depth_m,
        aligned_depth=frame.aligned_depth,
        K=K,
        device_id=frame.device_id,
        timestamp_us=frame.timestamp_us,
    )
