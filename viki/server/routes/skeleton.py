"""
viki.server.routes.skeleton
--------------------------
Endpoints for controlling skeleton estimation and recording,
and a WebSocket for streaming the latest skeleton frame.
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
from viki.skeleton.models import LM

from argparse import Namespace
from viki.optimization.optimization.retarget_rgb_only import run_single

from viki.server.deps import get_worker, get_processor
from viki.server.skeleton_worker import SkeletonWorker
from viki.skeleton.processor import SkeletonProcessor


def sanitize_nan(val):
    """Recursively replace NaN with None for JSON serialization."""
    if isinstance(val, dict):
        return {k: sanitize_nan(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_nan(x) for x in val]
    if isinstance(val, np.ndarray):
        return sanitize_nan(val.tolist())
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


router = APIRouter(prefix="/api/skeleton", tags=["skeleton"])
logger = logging.getLogger(__name__)


class ToggleRequest(BaseModel):
    enabled: bool


class SmoothRequest(BaseModel):
    window_length: int = 7
    polyorder: int = 2


@router.post("/toggle")
async def toggle_estimation(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    worker.set_enabled(req.enabled)
    return {"status": "updated", "enabled": worker.is_enabled}


@router.post("/record")
async def toggle_recording(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    worker.set_recording(req.enabled)
    return {"status": "updated", "recording": worker.is_recording}


@router.get("/status")
async def get_status(worker: SkeletonWorker = Depends(get_worker)):
    return {
        "enabled": worker.is_enabled,
        "recording": worker.is_recording,
    }


@router.get("/dataset_recordings")
async def list_dataset_recordings(
    page: int = 0,
    limit: int = 10,
):
    # This whole function is just a copy of the SkeletonProcessor's __init__ and list_recordings methods
    # TODO: add config.SKELETON_OPTIMIZED_RECS_DIR
    optzd_recs_dir = Path("./data/robot_out")
    page_size = limit
    # TODO:  The line below should be moved somewhere else outside of this endpoint
    optzd_recs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in optzd_recs_dir.glob("*.h5")], reverse=True)
    start = page * page_size
    end = start + page_size
    return files[start:end]


@router.post("/optimize/{filename}")
async def optimize_recording(
    filename: str
):
    
    summary = run_single(Namespace(
            sample="./data/skeleton_smoothed/" + filename,
            robot="ur10",
            working_hand="right",
            out="./data/robot_out/" + filename + ".h5",
            target_mode="hand_se3",
            ik_position_cost=5.0,
            ik_orientation_cost=0.3,
            ik_posture_cost=1e-3,
            ik_substeps=20,
            is_solver="quadprog",
            approach_sec=5.0,
            joint_sg_window=0,
            joint_sg_polyorder=3,
            sg_window=0,
            sg_polyorder=3,
            limit_frames=None,
            recenter_to_neutral=True,
            trajectory_scale=0.25,
            align_initial_orientation=True,
            evaluate=True,
            eval_align="rigid",
    ))


@router.get("/recordings")
async def list_recordings(
    page: int = 0, 
    limit: int = 10, 
    processor: SkeletonProcessor = Depends(get_processor)
):
    return {"recordings": processor.list_recordings(page=page, page_size=limit)}


@router.post("/smooth/{filename}")
async def smooth_recording(
    filename: str,
    req: SmoothRequest,
    processor: SkeletonProcessor = Depends(get_processor)
):
    try:
        path, _ = processor.smooth_recording(
            filename, 
            window_length=req.window_length, 
            polyorder=req.polyorder
        )
        return {"status": "success", "path": path}
    except FileNotFoundError:
        raise HTTPException(404, f"Recording {filename} not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Smoothing failed")
        raise HTTPException(500, f"Smoothing failed: {str(e)}")


@router.websocket("/stream")

async def skeleton_stream(websocket: WebSocket):
    await websocket.accept()
    # logger.debug("ROUTES/SKELETON: stream endpoint engaged")
    worker: SkeletonWorker = websocket.app.state.skeleton_worker
    try:
        while True:
            frame = worker.get_latest_frame()
            detections = worker.get_latest_detections()

            if frame or detections:
                # Serialize result to dict
                data = {
                    "ts": frame.timestamp_us if frame else time.time_ns() // 1000,
                    "landmarks": (
                        sanitize_nan(frame.points) if frame else {}
                    ),
                    "end_effector": (
                        sanitize_nan(frame.end_effector.as_dict()) if frame and frame.end_effector else None
                    ),
                    "detections": {
                        dev_id: (sanitize_nan(det.points) if det else {})
                        for dev_id, det in detections.items()
                    },
                }
                await websocket.send_json(data)

            # Stream at ~20 fps (comment out for unbound stream)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
