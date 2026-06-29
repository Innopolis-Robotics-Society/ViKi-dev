"""
viki.viz.depth
--------------
Image preparation for the camera streams:

- ``DepthColorizer`` — turns a uint16 depth frame into a colour-mapped BGR
  image, with an EMA-smoothed display range and last-good-frame hold.
- ``Undistorter`` — applies cached intrinsic undistortion to a colour image.

Both hold per-stream state, so create one instance per stream.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np



class DepthColorizer:
    """Stateful uint16-depth → BGR turbo colour map for a single stream."""

    def __init__(
        self,
        alpha: float = 0.05,
        min_valid_fraction: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.min_valid_fraction = min_valid_fraction
        self.d_min: float = 0.0
        self.d_max: float = 1.0
        self._ema_initialised = False
        self._last_good: Optional[np.ndarray] = None

    def colorize(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """
        Return a BGR colour-mapped depth image, or ``None`` if the frame should
        be skipped (mostly-empty depth with no prior good frame to hold).

        Mostly-empty frames (SDK dropped the depth capture) hold the last good
        image so the stream doesn't flash black.
        """
        valid = depth[depth > 0]
        valid_fraction = valid.size / max(depth.size, 1)

        if valid_fraction < self.min_valid_fraction:
            return self._last_good  
        
        # Update EMA range using 2nd/98th percentile to ignore outliers.
        p2 = float(np.percentile(valid, 2))
        p98 = float(np.percentile(valid, 98))
        if not self._ema_initialised:
            self.d_min, self.d_max = p2, p98
            self._ema_initialised = True
        else:
            self.d_min = self.alpha * p2 + (1 - self.alpha) * self.d_min
            self.d_max = self.alpha * p98 + (1 - self.alpha) * self.d_max

        norm = np.clip(
            (depth.astype(np.float32) - self.d_min) / (self.d_max - self.d_min + 1e-6), 0, 1
        )
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        self._last_good = img
        return img


# ... existing code ...
class Undistorter:
    """Apply intrinsic undistortion to colour images, caching the remap tables."""

    def __init__(self, mtx: np.ndarray, dist: np.ndarray) -> None:
        self.mtx = mtx
        self.dist = dist
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None

    def apply(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self._map1 is None:
            # Precompute the mapping once for performance.
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.mtx, (w, h), cv2.CV_32FC1
            )
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)


class DepthStabilizer:
    """Removes temporal jitter and noise from depth maps."""

    def __init__(
        self, 
        window_size: int = 5, 
        use_bilateral: bool = False
    ) -> None:
        self.window_size = window_size
        self.use_bilateral = use_bilateral
        self.buffer: list[np.ndarray] = []

    def stabilize(self, depth: np.ndarray) -> np.ndarray:
        """Apply temporal median and optional bilateral filtering."""
        if self.buffer and depth.shape != self.buffer[0].shape:
            self.buffer.clear()

        self.buffer.append(depth)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) < 2:
            return depth

        # Temporal median filter
        stack = np.stack(self.buffer, axis=0)
        median_depth = np.median(stack, axis=0).astype(np.uint16)

        if self.use_bilateral:
            # Bilateral filter expects float32 or uint8
            float_depth = median_depth.astype(np.float32)
            smoothed = cv2.bilateralFilter(float_depth, d=5, sigmaColor=50, sigmaSpace=5)
            median_depth = smoothed.astype(np.uint16)

        return median_depth

