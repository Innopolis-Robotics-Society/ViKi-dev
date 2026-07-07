from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from viki.optimization.optimization.convert_viki23_json import estimate_fps
from viki.optimization.optimization.retarget_rgb_only import (
    normalize_robot,
    retarget_from_poses,
    RunConfig,
)
from viki.config import RETARGET_DEFAULT_ROBOT, SKELETON_SMOOTHED_DIR

router = APIRouter(prefix="/api/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)

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


@router.post("/optimize/{filename}")
async def optimize_recording(
    filename: str,
    robot: str = Query(default=RETARGET_DEFAULT_ROBOT),
):
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    cln_path = smoothed_dir / filename
    if not cln_path.exists():
        raise HTTPException(status_code=404, detail=f"Recording not found: {filename}")

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "filename": filename,
        "robot": robot,
        "status": "queued",
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    with _dataset_jobs_lock:
        _dataset_jobs[job_id] = job

    thread = threading.Thread(
        target=_run_optimize, args=(job_id, cln_path, robot, filename), daemon=True
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


@router.get("/optimize/jobs")
async def list_optimize_jobs():
    with _dataset_jobs_lock:
        jobs = sorted(
            _dataset_jobs.values(), key=lambda j: j["created_at"], reverse=True
        )
    return {"jobs": jobs}


def _run_optimize(job_id: str, cln_path: Path, robot_name: str, filename: str):
    with _dataset_jobs_lock:
        _dataset_jobs[job_id]["status"] = "running"
        _dataset_jobs[job_id]["started_at"] = time.time()
    try:
        with np.load(cln_path) as data:
            positions = data["positions"]
            rotations = data["rotations"]
            validity = data["valid"]
            timestamps = data["timestamps"]

        fps = estimate_fps(timestamps)
        robot = normalize_robot(robot_name)
        cfg = RunConfig(
            robot=robot,
            working_hand="right",
            landmark_sg_window=0,
            landmark_sg_polyorder=0,
            ik_position_cost=5.0,
            ik_orientation_cost=0.3,
            ik_posture_cost=1e-3,
            target_mode="hand_se3",
            ik_substeps=20,
            ik_solver="quadprog",
            approach_sec=5.0,
            joint_sg_window=0,
            joint_sg_polyorder=3,
            limit_frames=None,
            recenter_to_neutral=True,
            trajectory_scale=0.25,
        )

        out_dir = Path("data/robot_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename.replace(".npz", ".h5")

        summary = retarget_from_poses(
            positions, rotations, validity, fps, out_path, cfg
        )
        logger.info("Retarget summary: %s", summary)

        with _dataset_jobs_lock:
            _dataset_jobs[job_id]["status"] = "completed"
            _dataset_jobs[job_id]["finished_at"] = time.time()
            _dataset_jobs[job_id]["result"] = summary
    except Exception as exc:
        logger.error("Optimization failed: %s", exc)
        with _dataset_jobs_lock:
            _dataset_jobs[job_id]["status"] = "failed"
            _dataset_jobs[job_id]["finished_at"] = time.time()
            _dataset_jobs[job_id]["error"] = str(exc)
