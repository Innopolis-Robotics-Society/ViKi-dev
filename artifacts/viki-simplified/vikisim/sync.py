"""Simplified MultiCameraSync (mirrors viki/capture/sync.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .capture_base import Frame


@dataclass
class SyncedFrameGroup:
    """A set of frames (one per camera) aligned to a common host timestamp."""

    frames: Dict[str, Frame] = field(default_factory=dict)
    sync_timestamp_us: int = 0


class MultiCameraSync:
    """Pulls the latest frame from every camera and groups them by host clock."""

    def __init__(self, manager: "CameraManager") -> None:
        self.manager = manager

    def get_synced_frame(self) -> Optional[SyncedFrameGroup]:
        ...  # pick latest frames near a common timestamp
