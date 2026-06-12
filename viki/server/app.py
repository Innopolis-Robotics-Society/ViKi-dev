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
from viki.capture.calibration import CalibrationManager

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = CameraManager()
    app.state.manager.load_calibration()
    app.state.calibrator = CalibrationManager()
    yield
    app.state.manager.stop_all()


app = FastAPI(title="ViKi Capture Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class StartRequest(BaseModel):
    fps: int = 30
    color_width: int = 640
    color_height: int = 480
    depth_mode: str = "NFOV_UNBINNED"
    # Kinect-only: hardware sync wiring (ignored for RealSense)
    # 0 = standalone, 1 = master, 2 = subordinate
    wired_sync_mode: int = 0
    # Subordinate capture delay relative to master trigger, microseconds.
    # A small positive value (e.g. 160) staggers the depth IR projectors.
    subordinate_delay_us: int = 0
    # Require color and depth to arrive in the same capture (recommended for sync recording).
    synchronized_images_only: bool = False


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
            wired_sync_mode=req.wired_sync_mode,
            subordinate_delay_us=req.subordinate_delay_us,
            synchronized_images_only=req.synchronized_images_only,
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


# ── Calibration Endpoints ───────────────────────────────────────────────────────

@app.get("/api/calibrate/stream")
def calibration_stream():
    return StreamingResponse(
        _calibration_gen(app.state.manager),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/api/calibrate/capture")
async def capture_calibration_sample():
    mgr = app.state.manager
    cal = app.state.calibrator
    
    active_ids = mgr.active_device_ids()
    if not active_ids:
        raise HTTPException(status_code=400, detail="No cameras active")
        
    frames = {}
    for dev_id in active_ids:
        frame = mgr.latest_frame(dev_id)
        if frame is None:
            raise HTTPException(status_code=500, detail=f"Failed to get frame from {dev_id}")
        frames[dev_id] = frame
        
    success_map = cal.add_sample(frames)
    return {"success_map": success_map, "sample_count": cal.sample_count}


@app.post("/api/calibrate/run")
async def run_calibration():
    try:
        result = app.state.calibrator.run_calibration()
        
        # Immediately apply the new calibration to the running manager
        app.state.manager.update_calibration(
            result["intrinsics"], 
            result["dist_coeffs"]
        )
        
        # Remove the large matrices from the response to keep it clean
        resp = result.copy()
        resp.pop("intrinsics", None)
        resp.pop("dist_coeffs", None)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calibrate/status")
async def calibration_status():
    return {"sample_count": app.state.calibrator.sample_count}


@app.post("/api/calibrate/clear")
async def clear_calibration():
    app.state.calibrator.clear()
    return {"status": "cleared"}


def _calibration_gen(mgr: CameraManager):
    last_ts = -1
    while True:
        active_ids = mgr.active_device_ids()
        if not active_ids:
            yield _mjpeg_frame(_placeholder(640, 480, "No cameras active"))
            time.sleep(0.1)
            continue

        frames = {}
        for dev_id in active_ids:
            frame = mgr.latest_frame(dev_id)
            if frame:
                frames[dev_id] = frame.color
        
        if not frames:
            time.sleep(0.01)
            continue

        # Create a mosaic
        imgs = list(frames.values())
        # Simple horizontal stack, resizing to 480p for the mosaic
        resized = [cv2.resize(img, (640, 480)) for img in imgs]
        combined = np.hstack(resized)
        
        yield _mjpeg_frame(combined)
        time.sleep(0.03)


def _mjpeg_frame(img: np.ndarray) -> bytes:
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = jpeg.tobytes()
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(data)).encode()
        + b"\r\n\r\n" + data + b"\r\n"
    )

# ── Helpers ────────────────────────────────────────────────────────────────────

_DEPTH_EMA_ALPHA = 0.05  # range settles over ~20 frames (~0.67 s at 30 fps)
# Minimum fraction of pixels that must be valid for a depth frame to be displayed.
# Frames below this threshold (blank / missing depth) are dropped and the last
# good image is held instead.
_DEPTH_MIN_VALID_FRACTION = 0.05


def _mjpeg_gen(mgr: CameraManager, device_id: str, kind: str):
    last_ts = -1
    # Depth-stream state — only used when kind == "depth"
    d_min: float = 0.0
    d_max: float = 1.0
    ema_initialised = False
    last_good_depth_img: np.ndarray | None = None
    
    # Cache for undistortion
    calib = mgr.get_calibration(device_id)
    map1, map2 = None, None

    while True:
        frame = mgr.latest_frame(device_id)


        if frame is None:
            if device_id not in mgr.active_device_ids():
                return
            img = _placeholder(640, 480, f"{device_id}: not started")
            last_ts = -1
        elif frame.host_timestamp_us == last_ts:
            time.sleep(0.005)
            continue
        else:
            last_ts = frame.host_timestamp_us
            if kind == "color":
                img = frame.color
                # Apply undistortion if calibration is available
                if calib is not None:
                    mtx, dist = calib["mtx"], calib["dist"]
                    h, w = img.shape[:2]
                    if map1 is None:
                        # Precompute mapping for performance
                        map1, map2 = cv2.initUndistortRectifyMap(
                            mtx, dist, None, mtx, (w, h), cv2.CV_32FC1
                        )
                    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            else:
                depth = frame.depth
                valid = depth[depth > 0]
                valid_fraction = valid.size / max(depth.size, 1)

                if valid_fraction < _DEPTH_MIN_VALID_FRACTION:
                    # Blank or mostly-zero frame (missing depth capture from SDK).
                    # Hold the last good image so the stream doesn't flash black.
                    if last_good_depth_img is None:
                        time.sleep(0.005)
                        continue
                    img = last_good_depth_img
                else:
                    # Update EMA range using 2nd/98th percentile to ignore outliers.
                    p2  = float(np.percentile(valid, 2))
                    p98 = float(np.percentile(valid, 98))
                    if not ema_initialised:
                        d_min, d_max = p2, p98
                        ema_initialised = True
                    else:
                        d_min = _DEPTH_EMA_ALPHA * p2  + (1 - _DEPTH_EMA_ALPHA) * d_min
                        d_max = _DEPTH_EMA_ALPHA * p98 + (1 - _DEPTH_EMA_ALPHA) * d_max

                    norm = np.clip(
                        (depth.astype(np.float32) - d_min) / (d_max - d_min + 1e-6), 0, 1
                    )
                    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    last_good_depth_img = img

        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = jpeg.tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(data)).encode()
            + b"\r\n\r\n" + data + b"\r\n"
        )


def _placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (80, 80, 80), 2, cv2.LINE_AA)
    return img