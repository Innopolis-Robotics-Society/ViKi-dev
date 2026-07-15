"""
viki.server.routes.dataset
--------------------------
Dataset handling and optimisation endpoints: listing recorded skeleton data,
triggering retargeting optimisation, and streaming robot trajectories.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from viki.optimization.preparation.processor import estimate_fps
from viki.optimization.retarget.retarget_rgb_only import (
    normalize_robot,
    retarget_from_poses,
    RunConfig,
)
from viki.config import (
    HAND_TO_DETECT,
    RETARGET_APPROACH_SEC,
    RETARGET_DEFAULT_ROBOT,
    RETARGET_IK_ORIENTATION_COST,
    RETARGET_IK_POSITION_COST,
    RETARGET_IK_POSTURE_COST,
    RETARGET_IK_SOLVER,
    RETARGET_IK_SUBSTEPS,
    RETARGET_JOINT_SG_POLYORDER,
    RETARGET_JOINT_SG_WINDOW,
    RETARGET_LANDMARK_SG_POLYORDER,
    RETARGET_LANDMARK_SG_WINDOW,
    RETARGET_RECENTER_TO_NEUTRAL,
    RETARGET_TARGET_MODE,
    RETARGET_TRAJECTORY_SCALE,
    SKELETON_SMOOTHED_DIR,
)
from viki.server.robot_viz import robot_trajectory_stream

router = APIRouter(prefix="/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)

_MJPEG_MEDIA = "multipart/x-mixed-replace; boundary=frame"
_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
ROBOT_OUT_DIR = Path("data/robot_out")

_dataset_jobs: dict[str, dict] = {}
_dataset_jobs_lock = threading.Lock()


@router.get("/recordings")
async def list_smoothed_recordings(page: int = 0, limit: int = 10):
    """
    List smoothed skeleton recordings (cln-*.npz files) with pagination.

    Parameters
    ----------
    page : int, default=0
        Page number (zero-based).
    limit : int, default=10
        Number of recordings per page.

    Returns
    -------
    dict
        {"recordings": list[str]} – list of filenames.
    """
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    smoothed_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in smoothed_dir.glob("cln-*.npz")], reverse=True)
    start = page * limit
    end = start + limit
    return {"recordings": files[start:end]}


class OptimizeRequest(BaseModel):
    filename: str
    robot: str = RETARGET_DEFAULT_ROBOT


@router.post("/optimize")
async def optimize_recording(
    req: OptimizeRequest,
):
    """
    Start a background optimisation job for a smoothed recording.

    Parameters
    ----------
    req : OptimizeRequest
        Filename (cln-*.npz) and robot name.

    Returns
    -------
    dict
        {"job_id": str, "status": "queued"}

    Raises
    ------
    HTTPException 404
        If the recording file does not exist.
    """
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    cln_path = smoothed_dir / req.filename
    if not cln_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Recording not found: {req.filename}"
        )

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
        target=_run_optimize,
        args=(job_id, cln_path, req.robot, req.filename),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@router.get("/optimize/status/{job_id}")
async def optimize_status(job_id: str):
    """
    Get the status of an optimisation job.

    Parameters
    ----------
    job_id : str
        Job identifier returned by `/optimize`.

    Returns
    -------
    dict
        Job details.

    Raises
    ------
    HTTPException 404
        If job not found.
    """
    with _dataset_jobs_lock:
        job = _dataset_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job


@router.get("/outputs")
async def list_outputs():
    """
    List all generated robot trajectory output files (.h5).

    Returns
    -------
    dict
        {"outputs": list[str]} – filenames.
    """
    ROBOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [f.name for f in ROBOT_OUT_DIR.glob("*.h5") if f.is_file()],
        reverse=True,
    )
    return {"outputs": files}


@router.get("/debug-viz")
async def retarget_debug_viz():
    """Return the latest retargeting debug overlay as a PNG."""
    from viki.optimization.retarget.debug import render_debug_viz_png

    png = render_debug_viz_png()
    if png is None:
        raise HTTPException(
            status_code=404,
            detail="No retargeting debug data available. Run a retargeting job first.",
        )
    return Response(content=png, media_type="image/png", headers=_STREAM_HEADERS)


@router.get("/viz-stream")
async def robot_viz_stream(filename: str, loop: bool = True):
    """
    MJPEG stream visualising a robot trajectory from an HDF5 file.

    Parameters
    ----------
    filename : str
        Output filename (.h5).
    loop : bool, default=True
        Repeat the trajectory indefinitely.

    Returns
    -------
    StreamingResponse
        MJPEG stream of the 3D robot visualisation.

    Raises
    ------
    HTTPException 404
        If the file does not exist.
    """
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
    """
    List all optimisation jobs (history).

    Returns
    -------
    dict
        {"jobs": list[dict]} – each job details, sorted by creation time descending.
    """
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
            working_hand=HAND_TO_DETECT,
            landmark_sg_window=RETARGET_LANDMARK_SG_WINDOW,
            landmark_sg_polyorder=RETARGET_LANDMARK_SG_POLYORDER,
            ik_position_cost=float(RETARGET_IK_POSITION_COST),
            ik_orientation_cost=float(RETARGET_IK_ORIENTATION_COST),
            ik_posture_cost=float(RETARGET_IK_POSTURE_COST),
            target_mode=RETARGET_TARGET_MODE,
            ik_substeps=RETARGET_IK_SUBSTEPS,
            ik_solver=RETARGET_IK_SOLVER,
            approach_sec=RETARGET_APPROACH_SEC,
            joint_sg_window=RETARGET_JOINT_SG_WINDOW,
            joint_sg_polyorder=RETARGET_JOINT_SG_POLYORDER,
            limit_frames=None,
            recenter_to_neutral=RETARGET_RECENTER_TO_NEUTRAL,
            trajectory_scale=RETARGET_TRAJECTORY_SCALE,
            trajectory_scale_origin="initial_wrist",
            align_initial_orientation=False,
        )

        ROBOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ROBOT_OUT_DIR / filename.replace(".npz", ".h5")

        summary = retarget_from_poses(
            positions, rotations, validity, fps, out_path, cfg
        )
        logger.info("Retarget summary: %s", summary)

        with _dataset_jobs_lock:
            _dataset_jobs[job_id]["status"] = "completed"
            _dataset_jobs[job_id]["finished_at"] = time.time()
            _dataset_jobs[job_id]["result"] = summary
    except Exception as exc:
        logger.exception("Optimization failed")
        with _dataset_jobs_lock:
            _dataset_jobs[job_id]["status"] = "failed"
            _dataset_jobs[job_id]["finished_at"] = time.time()
            _dataset_jobs[job_id]["error"] = str(exc)
