"""
viki.server.routes.calibration
------------------------------
Calibration endpoints: live mosaic preview, sample capture, running the
calibration solve, status, and clearing collected samples.
"""

from __future__ import annotations

import cv2

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import logging
logger = logging.getLogger(__name__)

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.server.deps import get_calibrator, get_manager
from viki.server.routes.models import (
    ArucoBoardParametersData,
    BoardParametersData,
    IntrinsicsResponse,
    ExtrinsicsResponse,
)
from viki.config import INTRINSICS_FILENAME, EXTRINSICS_FILENAME

router = APIRouter(prefix="/api/calibration", tags=["calibration"])

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


@router.post("/reset")
async def reset(cal: CalibrationManager = Depends(get_calibrator)):
    cal.stop_all()
    return {"status": "success"}


@router.post("/sync")
async def sync(
    params: ArucoBoardParametersData | BoardParametersData, 
    board_type: str,
    cal: CalibrationManager = Depends(get_calibrator)
):
    # Extract common params
    board_size = params.board_size
    square_size = params.square_size
    
    # Extract aruco specific
    marker_size = 0.025
    aruco_dict = cv2.aruco.DICT_6X6_250
    
    if board_type == "aruco" and hasattr(params, "marker_size"):
        marker_size = params.marker_size
        try:
            aruco_dict = getattr(cv2.aruco, params.aruco_dict)
        except:
            raise HTTPException(422, f"wrong aruco_dict: {params.aruco_dict}")
            
    cal.sync_params(board_type, board_size, square_size, marker_size, aruco_dict)
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
    if not cal._workers:
        raise HTTPException(
            400, 
            "Calibration session not started. Please click 'Sync Parameters' first."
        )
    cal.capture_all()


@router.post("/start/{device_id}")
async def start_worker(
    device_id: str,
    mode: str = "auto",
    params: BoardParametersData | None = None,
    cal: CalibrationManager = Depends(get_calibrator),
):
    if not params:
        board_size = (8, 6)
        square_size = 0.025
    else:
        board_size = params.board_size
        square_size = params.square_size
    cal.start(device_id, mode, "chess", board_size, square_size)


@router.post("/start/aruco/{device_id}")
async def start_aruco_worker(
    device_id: str,
    mode: str = "auto",
    params: ArucoBoardParametersData | None = None,
    cal: CalibrationManager = Depends(get_calibrator),
):
    if not params:
        board_size = (10, 8)
        square_size = 0.05
        marker_size = 0.035
        aruco_dict = cv2.aruco.DICT_5X5_100
    else:
        board_size = params.board_size
        square_size = params.square_size
        marker_size = params.marker_size
        try:
            aruco_dict = getattr(cv2.aruco, params.aruco_dict)
        except:
            raise HTTPException(422, f"wrong aruco_dict: {params.aruco_dict}")
    cal.start(
        device_id, mode, "aruco", board_size, square_size, marker_size, aruco_dict
    )


@router.get("/status/{device_id}")
async def status(device_id: str, cal: CalibrationManager = Depends(get_calibrator)):
    # logger.debug(f"calibration status for {device_id}: {cal.status(device_id)}")
    return cal.status(device_id)


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


@router.post("/extrinsics", response_model=list[ExtrinsicsResponse])
async def extrinsics_post_all(
    cal: CalibrationManager = Depends(get_calibrator),
    mgr: CameraManager = Depends(get_manager),
):
    results = []
    active_devices = mgr.active_device_ids()
    if not active_devices:
        raise HTTPException(400, "No active cameras to calibrate")

    for device_id in active_devices:
        try:
            extr = cal.extrinsics_calibration(device_id, EXTRINSICS_FILENAME)
            results.append(ExtrinsicsResponse(
                rvec=extr.rvec.flatten().tolist(),
                tvec=extr.tvec.flatten().tolist(),
            ))
        except Exception as e:
            logger.error(f"Extrinsics calibration failed for {device_id}: {e}")
            continue
    
    if not results:
        # If we got here, it means all active devices failed to calibrate
        # (e.g. due to lack of samples)
        raise HTTPException(
            422, 
            "Extrinsics calibration failed for all devices. Make sure you have captured enough samples."
        )
        
    return results



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
