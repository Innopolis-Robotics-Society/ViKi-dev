"""
viki.server.routes.calibration
------------------------------
Calibration endpoints: live mosaic preview, sample capture, running the
calibration solve, status, and clearing collected samples.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.deps import get_calibrator, get_manager
from viki.server.routes.models import (
    ChessboardParams,
    IntrinsicsResponse,
    ExtrinsicsResponse,
)
from viki.server.streams import calibration_mosaic
from viki.config import INTRINSICS_FILENAME, EXTRINSICS_FILENAME

router = APIRouter(prefix="/api/calibrate", tags=["calibration"])

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


@router.get("/stream")
def stream(mgr: CameraManager = Depends(get_manager)):
    return StreamingResponse(
        calibration_mosaic(mgr),
        media_type=_MJPEG_MEDIA,
        headers=_STREAM_HEADERS,
    )


@router.post("/reset")
async def reset(cal: CalibrationManager = Depends(get_calibrator)):
    cal.stop_all()
    return {"status": "success"}


@router.post("/capture/{device_id}")
async def capture(
    device_id: str,
    cal: CalibrationManager = Depends(get_calibrator),
):
    cal.capture(device_id)


@router.post("/capture")
async def capture_all(
    cal: CalibrationManager = Depends(get_calibrator),
):
    res = cal.capture_all()
    return res if res is not None else {"success_map": {}, "sample_count": 0}


@router.post("/start/{device_id}")
async def start_worker(
    device_id: str,
    mode: str = "auto",
    params: Optional[ChessboardParams] = None,
    cal: CalibrationManager = Depends(get_calibrator),
):
    if not params:
        chessboard_size = (8, 6)
        square_size = 0.025
    else:
        chessboard_size = params.chessboard_size
        square_size = params.square_size
    cal.start(device_id, chessboard_size, square_size, mode)


@router.get("/status/{device_id}")
async def status(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    return {"samples_count": cal.status(device_id)}


@router.get("/samples_count/{device_id}")
async def samples_count(
    device_id: str, cal: CalibrationManager = Depends(get_calibrator)
):
    return {"samples_count": cal.samples_count(device_id)}


@router.get("/is_device_active/{device_id}")
async def is_device_active(
    device_id: str, cal: CalibrationManager = Depends(get_calibrator)
):
    return {"is_device_active": cal.is_device_active(device_id)}


@router.post("/clear/{device_id}")
async def clear(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    cal.clear(device_id)
    return {"status": "cleared"}


@router.post("/intrinsics/{device_id}", response_model=IntrinsicsResponse)
async def intrinsics_post(
    device_id: str, cal: CalibrationManager = Depends(get_calibrator)
):
    intrinsics = cal.intrinsics_calibration(device_id, INTRINSICS_FILENAME)
    return IntrinsicsResponse(
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        dist_coeffs=intrinsics.dist_coeffs.tolist(),
    )


@router.get("/intrinsics/{device_id}", response_model=IntrinsicsResponse)
async def intrinsics(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    intrinsics = cal.get_intrinsics(device_id)
    if not intrinsics:
        raise HTTPException(
            status_code=404, detail="Intrinsics not found for this device"
        )
    return IntrinsicsResponse(
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        dist_coeffs=intrinsics.dist_coeffs.tolist(),
    )


@router.get("/extrinsics/{device_id}", response_model=ExtrinsicsResponse)
async def extrinsics(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    extrinsics = cal.get_extrinsics(device_id)
    if not extrinsics:
        raise HTTPException(
            status_code=404, detail="Extrinsics not found for this device"
        )
    return ExtrinsicsResponse(
        rvec=extrinsics.rvec.tolist(),
        tvec=extrinsics.tvec.tolist(),
    )
