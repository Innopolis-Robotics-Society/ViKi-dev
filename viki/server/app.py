"""
viki.server.app
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from viki.capture.manager import CameraManager

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = CameraManager()
    yield
    app.state.manager.stop_all()


app = FastAPI(title="ViKi Capture Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class StartRequest(BaseModel):
    fps: int = 30
    color_width: int = 640
    color_height: int = 480
    depth_mode: str = "NFOV_UNBINNED"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/devices")
async def list_devices():
    return app.state.manager.list_devices()


@app.post("/api/cameras/{device_id}/start")
async def start_camera(device_id: str, req: StartRequest):
    try:
        app.state.manager.start(
            device_id,
            fps=req.fps,
            color_width=req.color_width,
            color_height=req.color_height,
            depth_mode=req.depth_mode,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "started", "device_id": device_id}


@app.post("/api/cameras/{device_id}/stop")
async def stop_camera(device_id: str):
    app.state.manager.stop(device_id)
    return {"status": "stopped", "device_id": device_id}


@app.get("/api/cameras/{device_id}/info")
async def camera_info(device_id: str):
    info = app.state.manager.get_info(device_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Camera not found or not started")
    return info


@app.get("/api/cameras/{device_id}/stream")
def colour_stream(device_id: str):
    return StreamingResponse(
        _mjpeg_gen(app.state.manager, device_id, "color"),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

@app.get("/api/cameras/{device_id}/depth")
def depth_stream(device_id: str):
    return StreamingResponse(
        _mjpeg_gen(app.state.manager, device_id, "depth"),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

def _mjpeg_gen(mgr: CameraManager, device_id: str, kind: str):
    while True:
        frame = mgr.latest_frame(device_id)
        if frame is None:
            img = _placeholder(640, 480, f"{device_id}: not started")
        elif kind == "color":
            img = frame.color
        else:
            depth = frame.depth
            print(f"[depth:{device_id}] shape={depth.shape} min={depth.min()} max={depth.max()} nonzero={(depth>0).sum()}", flush=True)
            img = _depth_colormap(depth)

        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = jpeg.tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(data)).encode()
            + b"\r\n\r\n" + data + b"\r\n"
        )
        time.sleep(1 / 30)


def _depth_colormap(depth: np.ndarray) -> np.ndarray:
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d_min, d_max = valid.min(), valid.max()
    norm = np.clip(
        (depth.astype(np.float32) - d_min) / (d_max - d_min + 1e-6), 0, 1
    )
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (80, 80, 80), 2, cv2.LINE_AA)
    return img