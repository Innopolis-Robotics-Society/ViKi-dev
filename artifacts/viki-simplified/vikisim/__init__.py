"""Simplified ViKi skeleton subsystem (for UML generation only)."""

from .calibration import CalibrationExtrinsics, CalibrationManager
from .capture_base import CameraBackend, CameraIntrinsics, Frame
from .manager import CameraManager, _CameraWorker
from .models import (
    EndEffectorPose,
    HandDetection,
    Landmarks3D,
    LM,
    PipelineResult,
    PreparedFrame,
    SkeletonFrame,
)
from .pipeline import SkeletonPipeline
from .sync import MultiCameraSync, SyncedFrameGroup
from .worker import SkeletonWorker

__all__ = [
    "CameraBackend",
    "CameraIntrinsics",
    "Frame",
    "CameraManager",
    "_CameraWorker",
    "SyncedFrameGroup",
    "MultiCameraSync",
    "CalibrationExtrinsics",
    "CalibrationManager",
    "PreparedFrame",
    "HandDetection",
    "Landmarks3D",
    "EndEffectorPose",
    "SkeletonFrame",
    "PipelineResult",
    "LM",
    "SkeletonPipeline",
    "SkeletonWorker",
]
