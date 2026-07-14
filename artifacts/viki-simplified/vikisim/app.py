"""Simplified app assembly (mirrors viki/server/app.py lifespan wiring)."""

from __future__ import annotations

from typing import Any

from .calibration import CalibrationManager
from .manager import CameraManager
from .pipeline import SkeletonPipeline
from .sync import MultiCameraSync
from .worker import SkeletonWorker


class AppState:
    """Container wiring all subsystems together (FastAPI app.state)."""

    def __init__(self) -> None:
        self.manager: CameraManager = CameraManager()
        self.calibrator: CalibrationManager = CalibrationManager(self.manager)
        self.sync: MultiCameraSync = MultiCameraSync(self.manager)
        self.skeleton_pipeline: SkeletonPipeline = SkeletonPipeline(
            self.calibrator, self.manager
        )
        self.skeleton_recorder: Any = None
        self.skeleton_worker: SkeletonWorker = SkeletonWorker(
            self.manager,
            self.sync,
            self.skeleton_pipeline,
            self.skeleton_recorder,
        )

    def startup(self) -> None:
        self.calibrator.load_all_extrinsics()
        self.skeleton_worker.start()
