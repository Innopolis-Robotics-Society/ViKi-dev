"""
viki.server.deps
----------------
FastAPI dependencies. The manager and calibrator are created once in the
app lifespan and stored on ``app.state``; these resolve them so route
handlers receive them via ``Depends`` instead of reaching into app state.
"""
from __future__ import annotations

from fastapi import Request

from viki.capture.calibration import CalibrationManager
from viki.capture.manager import CameraManager


def get_manager(request: Request) -> CameraManager:
    return request.app.state.manager


def get_calibrator(request: Request) -> CalibrationManager:
    return request.app.state.calibrator
