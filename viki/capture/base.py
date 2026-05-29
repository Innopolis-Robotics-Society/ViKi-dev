"""
viki.capture.base
-----------------
Abstract camera interface.

All backends (RealSense, Azure Kinect, ...) implement CameraBackend.
Consumers work only with this interface — the concrete backend is
injected from the outside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Distortion coefficients (k1,k2,p1,p2[,k3]) — optional
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))


@dataclass
class Frame:
    """
    A single synchronised frame from one camera.

    Fields
    ------
    color         : np.ndarray  HxWx3, uint8, BGR (OpenCV convention)
    depth         : np.ndarray  HxW,   uint16, millimetres
    timestamp_us  : int         capture time, microseconds (device monotonic clock)
    device_id     : str         unique device identifier (serial number or alias)
    color_intrinsics : CameraIntrinsics | None
    depth_intrinsics : CameraIntrinsics | None
    """
    color: np.ndarray
    depth: np.ndarray
    timestamp_us: int
    device_id: str
    color_intrinsics: Optional[CameraIntrinsics] = None
    depth_intrinsics: Optional[CameraIntrinsics] = None

    def has_depth(self) -> bool:
        return self.depth is not None and self.depth.size > 0


class CameraBackend(ABC):
    """
    Abstract camera backend.

    Lifecycle
    ---------
    backend = SomeBackend(...)
    backend.start()
    try:
        while True:
            frame = backend.get_frame()
            ...
    finally:
        backend.stop()

    Or via context manager:
        with SomeBackend(...) as cam:
            frame = cam.get_frame()
    """

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the stream and release the device."""

    @abstractmethod
    def get_frame(self) -> Frame:
        """
        Fetch the next frame (blocking call).

        Raises
        ------
        RuntimeError  if the backend is not started or the device is lost
        TimeoutError  if no frame arrives within the configured timeout
        """

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Unique device identifier."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """True if the stream is active."""

    # ------------------------------------------------------------------
    # Context manager — implemented here, no need to override
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraBackend":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
