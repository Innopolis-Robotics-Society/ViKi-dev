"""
viki.server.app
---------------
Application assembly only: lifespan resources, static files, router wiring.
All request logic lives in ``viki.server.routes`` and ``viki.server.streams``.
"""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.capture.sync import MultiCameraSync
from viki.skeleton.pipeline import SkeletonPipeline
from viki.skeleton.recorder import SkeletonRecorder
from viki.server.skeleton_worker import SkeletonWorker
from importlib import import_module

from viki.server.routes import (
    calibration,
    cameras,
    skeleton,
    recording,
    system,
    optimization,
    dataset,
)

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Initialises and starts:
        - CameraManager (all cameras)
        - CalibrationManager (for intrinsics/extrinsics)
        - MultiCameraSync (for software synchronisation)
        - SkeletonPipeline, SkeletonRecorder, and SkeletonWorker (background thread)

    On shutdown, stops the skeleton worker and all cameras.
    """
    app.state.manager = CameraManager()
    app.state.calibrator = CalibrationManager(app.state.manager)
    app.state.calibrator.load_all_extrinsics()
    app.state.sync = MultiCameraSync(app.state.manager)
    app.state.skeleton_pipeline = SkeletonPipeline(
        app.state.calibrator, app.state.manager
    )
    from viki.skeleton.models import LM

    app.state.skeleton_recorder = SkeletonRecorder(
        filter_indices=[LM.WRIST, LM.MIDDLE_MCP, LM.THUMB_CMC]
    )

    app.state.skeleton_worker = SkeletonWorker(
        app.state.manager,
        app.state.sync,
        app.state.skeleton_pipeline,
        app.state.skeleton_recorder,
    )
    app.state.skeleton_worker.start()
    from viki.skeleton.processor import SkeletonProcessor

    app.state.skeleton_processor = SkeletonProcessor()

    yield
    app.state.skeleton_worker.stop()
    app.state.manager.stop_all()


app = FastAPI(title="ViKi Capture Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(cameras.router)
app.include_router(calibration.router)
app.include_router(skeleton.router)
app.include_router(system.router)
app.include_router(recording.router)
app.include_router(optimization.router)
app.include_router(dataset.router)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend HTML page."""
    return (STATIC_DIR / "index.html").read_text()
