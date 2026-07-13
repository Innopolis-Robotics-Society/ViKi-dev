from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal

from viki.optimization.optimization.convert_viki23_json import estimate_fps
from viki.optimization.optimization.retarget_rgb_only import (
    normalize_robot,
    retarget_from_poses,
    RunConfig,
    transform_points,
    transform_rotations_to_robot,
)
from viki.config import (
    RETARGET_DEFAULT_ROBOT,
    RETARGET_TARGET_MODE,
    RETARGET_IK_POSITION_COST,
    RETARGET_IK_ORIENTATION_COST,
    RETARGET_IK_POSTURE_COST,
    RETARGET_IK_SUBSTEPS,
    RETARGET_IK_SOLVER,
    RETARGET_APPROACH_SEC,
    RETARGET_JOINT_SG_WINDOW,
    RETARGET_JOINT_SG_POLYORDER,
    RETARGET_RECENTER_TO_NEUTRAL,
    RETARGET_WRIST_SCALE,
    SKELETON_SMOOTHED_DIR,
)
from viki.server.robot_viz import robot_trajectory_stream

router = APIRouter(prefix="/api/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
ROBOT_OUT_DIR = Path("data/robot_out")

_dataset_jobs: dict[str, dict] = {}
_dataset_jobs_lock = threading.Lock()


@router.get("/recordings")
async def list_smoothed_recordings(page: int = 0, limit: int = 10):
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    smoothed_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in smoothed_dir.glob("cln-*.npz")], reverse=True)
    start = page * limit
    end = start + limit
    return {"recordings": files[start:end]}


class OptimizeRequest(BaseModel):
    filename: str
    robot: str = RETARGET_DEFAULT_ROBOT
    target_mode: Literal["wrist_position", "hand_se3"] = "wrist_position"
    ik_position_cost: float | None = None
    ik_orientation_cost: float | None = None
    ik_posture_cost: float | None = None
    ik_substeps: int | None = None
    ik_solver: str | None = None
    approach_sec: float | None = None
    joint_sg_window: int | None = None
    joint_sg_polyorder: int | None = None
    recenter_to_neutral: bool | None = None
    wrist_scale: float | None = None
    align_initial_orientation: bool | None = None


@router.post("/optimize")
async def optimize_recording(
    req: OptimizeRequest,
):
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    cln_path = smoothed_dir / req.filename
    if not cln_path.exists():
        raise HTTPException(status_code=404, detail=f"Recording not found: {req.filename}")

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "filename": req.filename,
        "robot": req.robot,
        "status": "queued",
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    with _dataset_jobs_lock:
        _dataset_jobs[job_id] = job

    thread = threading.Thread(
        target=_run_optimize, args=(job_id, cln_path, req), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@router.get("/optimize/status/{job_id}")
async def optimize_status(job_id: str):
    with _dataset_jobs_lock:
        job = _dataset_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job


@router.get("/outputs")
async def list_outputs():
    ROBOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [f.name for f in ROBOT_OUT_DIR.glob("*.h5") if f.is_file()],
        reverse=True,
    )
    return {"outputs": files}


@router.get("/viz-stream")
async def robot_viz_stream(filename: str, loop: bool = True):
    h5_path = ROBOT_OUT_DIR / filename
    if not h5_path.exists():
        raise HTTPException(status_code=404, detail=f"Output not found: {filename}")
    return StreamingResponse(
        robot_trajectory_stream(h5_path, loop=loop),
        media_type=_MJPEG_MEDIA,
        headers=_STREAM_HEADERS,
    )


@router.get("/optimize/jobs")
async def list_optimize_jobs():
    with _dataset_jobs_lock:
        jobs = sorted(
            _dataset_jobs.values(), key=lambda j: j["created_at"], reverse=True
        )
    return {"jobs": jobs}


def _run_optimize(job_id: str, cln_path: Path, req: OptimizeRequest):
    with _dataset_jobs_lock:
        _dataset_jobs[job_id]["status"] = "running"
        _dataset_jobs[job_id]["started_at"] = time.time()
    try:
        with np.load(cln_path) as data:
            positions = data["positions"]
            rotations = data["rotations"]
            validity = data["valid"]
            timestamps = data["timestamps"]

        # Apply coordinate transform to robot frame before retargeting
        positions = transform_points(positions)
        rotations = transform_rotations_to_robot(rotations)

        fps = estimate_fps(timestamps)
        robot = normalize_robot(req.robot)
        cfg = RunConfig(
            robot=robot,
            working_hand="right",
            landmark_sg_window=0,
            landmark_sg_polyorder=0,
            ik_position_cost=float(req.ik_position_cost) if req.ik_position_cost is not None else float(RETARGET_IK_POSITION_COST),
            ik_orientation_cost=float(req.ik_orientation_cost) if req.ik_orientation_cost is not None else float(RETARGET_IK_ORIENTATION_COST),
            ik_posture_cost=float(req.ik_posture_cost) if req.ik_posture_cost is not None else float(RETARGET_IK_POSTURE_COST),
            target_mode=req.target_mode,
            ik_substeps=req.ik_substeps if req.ik_substeps is not None else RETARGET_IK_SUBSTEPS,
            ik_solver=req.ik_solver if req.ik_solver is not None else RETARGET_IK_SOLVER,
            approach_sec=float(req.approach_sec) if req.approach_sec is not None else float(RETARGET_APPROACH_SEC),
            joint_sg_window=req.joint_sg_window if req.joint_sg_window is not None else RETARGET_JOINT_SG_WINDOW,
            joint_sg_polyorder=req.joint_sg_polyorder if req.joint_sg_polyorder is not None else RETARGET_JOINT_SG_POLYORDER,
            limit_frames=None,
            recenter_to_neutral=req.recenter_to_neutral if req.recenter_to_neutral is not None else RETARGET_RECENTER_TO_NEUTRAL,
            wrist_scale=float(req.wrist_scale) if req.wrist_scale is not None else float(RETARGET_WRIST_SCALE),
            align_initial_orientation=req.align_initial_orientation if req.align_initial_orientation is not None else True,
        )

        ROBOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ROBOT_OUT_DIR / req.filename.replace(".npz", ".h5")

        summary = retarget_from_poses(
            positions, rotations, validity, fps, out_path, cfg
        )
        logger.info("Retarget summary: %s", summary)

        with _dataset_jobs_lock:
            _dataset_jobs[job_id]["status"] = "completed"
            _dataset_jobs[job_id]["finished_at"] = time.time()
            _dataset_jobs[job_id]["result"] = summary
    except Exception as exc:
        logger.error("Optimization failed: %s\n%s", exc, traceback.format_exc())
        with _dataset_jobs_lock:
            _dataset_jobs[job_id]["status"] = "failed"
            _dataset_jobs[job_id]["finished_at"] = time.time()
            _dataset_jobs[job_id]["error"] = str(exc)
