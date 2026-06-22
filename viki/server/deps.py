"""
viki.server.deps
----------------
FastAPI dependencies. The manager and calibrator are created once in the
app lifespan and stored on ``app.state``; these resolve them so route
handlers receive them via ``Depends`` instead of reaching into app state.
"""
from __future__ import annotations

from fastapi import Request, WebSocket
from typing import Any

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.skeleton_worker import SkeletonWorker


def get_manager(request: Any) -> CameraManager:
    return request.app.state.manager


def get_calibrator(request: Any) -> CalibrationManager:
    return request.app.state.calibrator


def get_worker(request: Any) -> SkeletonWorker:
    return request.app.state.skeleton_worker
