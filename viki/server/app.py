"""
viki.server.app
---------------
Application assembly only: lifespan resources, router wiring, UI redirect.
All request logic lives in ``viki.server.routes`` and ``viki.server.streams``.
"""

from __future__ import annotations

import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.capture.sync import MultiCameraSync
from viki.skeleton.pipeline import SkeletonPipeline
from viki.skeleton.recorder import SkeletonRecorder
from viki.server.skeleton_worker import SkeletonWorker
from viki.server.routes import calibration, cameras, optimization, skeleton, recording, system

# The web UI is now a separate Streamlit process; `/` redirects to it.
STREAMLIT_URL = os.environ.get("VIKI_STREAMLIT_URL", "http://localhost:8501")


logging.basicConfig(level=logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = CameraManager()
    app.state.calibrator = CalibrationManager(app.state.manager)
    # for device in range(app.state.manager.active_device_ids):

    # app.state.calibrator.load_intrinsics(app.state.manager.active_device_ids)
    # app.state.calibrator.load_extrinsics()
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
app.include_router(cameras.router)
app.include_router(calibration.router)
app.include_router(skeleton.router)
app.include_router(system.router)
app.include_router(optimization.router)

app.include_router(recording.router)


@app.get("/")
async def index():
    """Redirect to the Streamlit web UI."""
    return RedirectResponse(STREAMLIT_URL)
