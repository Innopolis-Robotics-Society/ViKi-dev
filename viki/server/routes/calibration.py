"""
viki.server.routes.calibration
------------------------------
Calibration endpoints: live mosaic preview, sample capture, running the
calibration solve, status, and clearing collected samples.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from viki.capture.calibration import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.deps import get_calibrator, get_manager
from viki.server.streams import calibration_mosaic

router = APIRouter(prefix="/api/calibrate", tags=["calibration"])

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


@router.get("/stream")
def calibration_stream(mgr: CameraManager = Depends(get_manager)):
    return StreamingResponse(
        calibration_mosaic(mgr),
        media_type=_MJPEG_MEDIA,
        headers=_STREAM_HEADERS,
    )


@router.post("/capture")
async def capture_calibration_sample(
    mgr: CameraManager = Depends(get_manager),
    cal: CalibrationManager = Depends(get_calibrator),
):
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


@router.post("/run")
async def run_calibration(
    mgr: CameraManager = Depends(get_manager),
    cal: CalibrationManager = Depends(get_calibrator),
):
    try:
        result = cal.run_calibration()

        # Immediately apply the new calibration to the running manager.
        mgr.update_calibration(result["intrinsics"], result["dist_coeffs"])

        # Remove the large matrices from the response to keep it clean.
        resp = result.copy()
        resp.pop("intrinsics", None)
        resp.pop("dist_coeffs", None)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def calibration_status(cal: CalibrationManager = Depends(get_calibrator)):
    return {"sample_count": cal.sample_count}


@router.post("/clear")
async def clear_calibration(cal: CalibrationManager = Depends(get_calibrator)):
    cal.clear()
    return {"status": "cleared"}
