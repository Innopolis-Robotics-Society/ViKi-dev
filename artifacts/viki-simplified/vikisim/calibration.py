"""Simplified CalibrationManager + extrinsics (mirrors viki/calibration/manager.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .manager import CameraManager


@dataclass
class CalibrationExtrinsics:
    """Pose of the board relative to a camera; yields camera->world transform."""

    rvec: Any = None
    tvec: Any = None

    @property
    def transform_matrix(self) -> Any:
        ...  # 4x4 camera -> world homogeneous matrix


class CalibrationManager:
    """Stores per-device intrinsics and extrinsics; loads them from disk."""

    def __init__(self, manager: CameraManager) -> None:
        self.manager = manager
        self.extrinsics: Dict[str, CalibrationExtrinsics] = {}
        self.intrinsics: Dict[str, Any] = {}

    def load_all_extrinsics(self) -> None:
        ...

    def get_extrinsics(self, device_id: str) -> Optional[CalibrationExtrinsics]:
        return self.extrinsics.get(device_id)

    def get_intrinsics(self, device_id: str) -> Optional[Any]:
        return self.intrinsics.get(device_id)
