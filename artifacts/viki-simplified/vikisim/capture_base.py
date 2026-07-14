"""Simplified camera backend + frame abstractions (mirrors viki/capture/base.py).

Only the structural shape relevant to the UML is kept: the ABC contract and the
data containers that flow through the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CameraIntrinsics:
    """Pin-hole intrinsics for one camera stream."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Frame:
    """One synchronized color+depth sample from a single camera."""

    device_id: str
    timestamp_us: int
    color: Any  # HxWx3 uint8 BGR
    depth: Any  # HxW uint16 millimetres
    color_K: Optional[CameraIntrinsics] = None
    depth_K: Optional[CameraIntrinsics] = None


class CameraBackend(ABC):
    """Device-specific capture backend (RealSense / Kinect / ...)."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def get_frame(self) -> Optional[Frame]:
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...
