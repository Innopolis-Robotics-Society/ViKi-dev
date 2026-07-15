"""Simplified skeleton data model (mirrors viki/skeleton/models.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional


class LM(IntEnum):
    """MediaPipe Hands landmark indices (+ reserved arm landmarks)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_MCP = 13
    PINKY_MCP = 17
    PINKY_TIP = 20
    ELBOW = 21
    SHOULDER = 22
    N = 23


@dataclass
class PreparedFrame:
    """Color+depth ready for inference (depth in metres, 0 -> nan)."""

    rgb: Any
    depth_m: Any
    depth_K: Optional[Any]
    device_id: str
    timestamp_us: int


@dataclass
class HandDetection:
    """Raw MediaPipe output for one camera, pixel space."""

    points: Dict[LM, Any]
    lm_z_rel: Any
    confidence: float
    device_id: str
    timestamp_us: int


@dataclass
class Landmarks3D:
    """3D landmarks in camera coordinates."""

    points: Dict[LM, Any]
    device_id: str
    timestamp_us: int


@dataclass
class EndEffectorPose:
    """World-frame wrist pose."""

    position: Any
    R_world_palm: Any
    rpy_deg: Any
    valid: bool
    timestamp_us: int


@dataclass
class SkeletonFrame:
    """Final fused skeleton in world coordinates."""

    points: Dict[LM, Any]
    timestamp_us: int
    end_effector: Optional[EndEffectorPose] = None


@dataclass
class PipelineResult:
    """Output of one full pipeline run."""

    fused_frame: Optional[SkeletonFrame]
    detections: Dict[str, Optional[HandDetection]]
    debug_depth_marks: Optional[Dict[str, Dict[LM, Any]]] = None
