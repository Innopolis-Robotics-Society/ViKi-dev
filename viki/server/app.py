"""
viki.server.app
---------------
Application assembly only: lifespan resources, static files, router wiring.
All request logic lives in ``viki.server.routes`` and ``viki.server.streams``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.routes import calibration, cameras, skeleton

import logging

logging.basicConfig(level=logging.DEBUG)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = CameraManager()
    app.state.calibrator = CalibrationManager(app.state.manager)
    yield
    app.state.manager.stop_all()


app = FastAPI(title="ViKi Capture Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(cameras.router)
app.include_router(calibration.router)
app.include_router(skeleton.router)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()
