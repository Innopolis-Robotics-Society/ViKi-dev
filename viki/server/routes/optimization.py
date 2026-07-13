"""
viki.server.routes.optimization
-------------------------------
Backend endpoints for experimental skeleton retargeting/optimisation.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from viki.optimization.optimization.convert_viki23_json import convert
from viki.optimization.optimization.retarget_rgb_only import (
    RunConfig,
    evaluate_saved_traj,
    normalize_robot,
    output_traj_path,
    retarget,
)

router = APIRouter(prefix="/api/optimization", tags=["optimization"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZATION_DIR = PROJECT_ROOT / "viki" / "optimization" / "optimization"
SMOOTHED_INPUT_DIR = PROJECT_ROOT / "data" / "skeleton_smoothed"
LEGACY_SAMPLES_DIR = PROJECT_ROOT / "data" / "optimization_samples"
OUTPUT_DIR = PROJECT_ROOT / "data" / "robot_out"
RECORDING_DIRS = (PROJECT_ROOT, PROJECT_ROOT / "data" / "skeleton_recs")
OUTPUT_SUFFIXES = {".h5", ".hdf5", ".json", ".png"}
TAIL_CHARS = 12000


class ConvertRequest(BaseModel):
    recording: str
    output_name: str
    hand: Literal["right", "left"] = "right"
    include_arm: bool | None = None


class RetargetRequest(BaseModel):
    sample: str
    robot: str = "ur10"
    output_name: str
    target_mode: Literal["wrist_position", "hand_se3"] = "wrist_position"
    ik_position_cost: float | None = None
    ik_orientation_cost: float | None = None
    ik_solver: str | None = None
    joint_sg_window: int = 0
    sg_window: int = 0
    recenter_to_neutral: bool = False
    trajectory_scale: float | None = Field(default=None, gt=0.0)
    trajectory_scale_origin: Literal["auto", "initial_wrist", "robot_base"] = "auto"
    align_initial_orientation: bool | None = None
    evaluate: bool = True


@dataclass(frozen=True)
class RobotRetargetDefaults:
    ik_position_cost: float
    ik_orientation_cost: float
    ik_solver: str
    trajectory_scale: float
    align_initial_orientation: bool


@dataclass
class OptimizationJob:
    job_id: str
    status: str
    description: str
    output_stem: str
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None


_jobs: dict[str, OptimizationJob] = {}
_jobs_lock = threading.Lock()
_job_queue: queue.Queue[tuple[str, Path, Path, RunConfig, bool]] = queue.Queue()
_worker_started = False


@router.get("/recordings")
async def list_recordings() -> dict[str, Any]:
    return {"recordings": [_file_info(path) for path in _recording_paths()]}


@router.post("/convert")
async def convert_recording(req: ConvertRequest) -> dict[str, Any]:
    _ensure_dirs()
    recording = _resolve_recording(req.recording)
    output_name = _safe_filename(req.output_name, expected_suffix=".npz")
    output_path = LEGACY_SAMPLES_DIR / output_name

    try:
        summary = convert(recording, output_path, req.hand)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "output_path": _relative_path(output_path),
        "frames": summary["frames"],
        "fps": summary["fps"],
        "working_hand": summary["working_hand"],
        "orientation_valid_frames": summary["orientation_valid_frames"],
        "orientation_total_frames": summary["orientation_total_frames"],
    }


@router.get("/samples")
async def list_samples() -> dict[str, Any]:
    _ensure_dirs()
    return {
        "samples": [
            _file_info(path) for path in sorted(SMOOTHED_INPUT_DIR.glob("*.npz"))
        ]
    }


@router.post("/retarget")
async def retarget_endpoint(req: RetargetRequest) -> dict[str, Any]:
    _ensure_dirs()
    sample_name = _safe_filename(req.sample, expected_suffix=".npz")
    sample_path = SMOOTHED_INPUT_DIR / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_name}")

    output_name = _safe_filename(req.output_name)
    output_path = OUTPUT_DIR / output_name
    output_path = output_traj_path(output_path, sample_path, normalize_robot(req.robot))
    output_stem = output_path.stem.replace("_traj", "")
    defaults = _retarget_defaults(req.robot, req.target_mode)

    robot_cfg = normalize_robot(req.robot)
    cfg = RunConfig(
        robot=robot_cfg,
        working_hand="right",
        landmark_sg_window=req.sg_window,
        landmark_sg_polyorder=2,
        ik_position_cost=(
            req.ik_position_cost
            if req.ik_position_cost is not None
            else defaults.ik_position_cost
        ),
        ik_orientation_cost=(
            req.ik_orientation_cost
            if req.ik_orientation_cost is not None
            else defaults.ik_orientation_cost
        ),
        ik_posture_cost=1e-3,
        target_mode=req.target_mode,
        ik_substeps=20,
        ik_solver="quadprog",
        approach_sec=5.0,
        joint_sg_window=req.joint_sg_window,
        joint_sg_polyorder=3,
        limit_frames=None,
        recenter_to_neutral=req.recenter_to_neutral,
        trajectory_scale=(
            req.trajectory_scale
            if req.trajectory_scale is not None
            else defaults.trajectory_scale
        ),
        trajectory_scale_origin=req.trajectory_scale_origin,
        align_initial_orientation=(
            req.align_initial_orientation
            if req.align_initial_orientation is not None
            else defaults.align_initial_orientation
        ),
        trajectory_scale_origin=req.trajectory_scale_origin,
    )

    do_evaluate = req.evaluate
    description = (
        f"retarget robot={req.robot} sample={sample_name} mode={req.target_mode}"
    )
    job = _enqueue_job(
        description, output_stem, sample_path, output_path, cfg, do_evaluate
    )
    return _job_response(job)


@router.get("/jobs")
async def list_jobs() -> dict[str, Any]:
    with _jobs_lock:
        jobs = [_job_response(job) for job in _jobs.values()]
    jobs.sort(key=lambda item: item["created_at"])
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return _job_response(job)


@router.get("/outputs")
async def list_outputs() -> dict[str, Any]:
    _ensure_dirs()
    paths = [
        path
        for path in sorted(OUTPUT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in OUTPUT_SUFFIXES
    ]
    return {"outputs": [_file_info(path) for path in paths]}


@router.get("/outputs/download")
async def download_output(filename: str) -> FileResponse:
    return _output_response(filename)


@router.get("/outputs/{filename}")
async def download_output_by_filename(filename: str) -> FileResponse:
    return _output_response(filename)


def _output_response(filename: str) -> FileResponse:
    name = _safe_filename(filename)
    path = OUTPUT_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Output not found: {name}")
    if path.suffix.lower() not in OUTPUT_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported output file type")
    return FileResponse(path, filename=name)


def _ensure_dirs() -> None:
    SMOOTHED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _recording_paths() -> list[Path]:
    paths: dict[str, Path] = {}
    for directory in RECORDING_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("rec_*.json")):
            paths.setdefault(path.name, path)
    return sorted(paths.values(), key=lambda item: item.name)


def _resolve_recording(recording: str) -> Path:
    name = _safe_filename(recording, expected_suffix=".json")
    for path in _recording_paths():
        if path.name == name:
            return path
    raise HTTPException(status_code=404, detail=f"Recording not found: {name}")


def _safe_filename(filename: str, expected_suffix: str | None = None) -> str:
    path = Path(filename)
    name = path.name
    if not name or name in {".", ".."} or name != filename:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename}")
    if expected_suffix and Path(name).suffix.lower() != expected_suffix:
        name = name + expected_suffix
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename}")
    return name


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "path": _relative_path(path),
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "looks_smoothed": path.parent == SMOOTHED_INPUT_DIR
        or "smoothed" in path.stem.lower()
        or path.stem.lower().startswith("cln-"),
    }


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _retarget_defaults(robot: str, target_mode: str) -> RobotRetargetDefaults:
    robot_key = robot.strip().lower()
    if target_mode == "wrist_position":
        return RobotRetargetDefaults(
            ik_position_cost=(
                2.0 if robot_key in {"iiwa14", "iiwa14_description"} else 5.0
            ),
            ik_orientation_cost=0.0,
            ik_solver="quadprog",
            trajectory_scale=(
                0.55 if robot_key in {"iiwa14", "iiwa14_description"} else 0.8
            ),
            align_initial_orientation=False,
        )
    if robot_key in {"iiwa14", "iiwa14_description"}:
        return RobotRetargetDefaults(
            ik_position_cost=2.0,
            ik_orientation_cost=0.3,
            ik_solver="quadprog",
            trajectory_scale=0.55,
            align_initial_orientation=False,
        )
    return RobotRetargetDefaults(
        ik_position_cost=5.0,
        ik_orientation_cost=0.6,
        ik_solver="quadprog",
        trajectory_scale=0.75,
        align_initial_orientation=True,
    )


def _enqueue_job(
    description: str,
    output_stem: str,
    sample_path: Path,
    output_path: Path,
    cfg: RunConfig,
    do_evaluate: bool,
) -> OptimizationJob:
    global _worker_started
    job = OptimizationJob(
        job_id=uuid.uuid4().hex,
        status="queued",
        description=description,
        output_stem=output_stem,
    )
    with _jobs_lock:
        _jobs[job.job_id] = job
        if not _worker_started:
            thread = threading.Thread(target=_job_worker, daemon=True)
            thread.start()
            _worker_started = True
    _job_queue.put((job.job_id, sample_path, output_path, cfg, do_evaluate))
    return job


def _job_worker() -> None:
    while True:
        job_id, sample_path, output_path, cfg, do_evaluate = _job_queue.get()
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
        try:
            summary = retarget(sample_path, output_path, cfg)
            if do_evaluate:
                traj_path = output_path
                eval_prefix = traj_path.with_name(
                    traj_path.stem.replace("_traj", "") + "_eval"
                )
                robot = cfg.robot
                summary.update(
                    evaluate_saved_traj(
                        sample_path, traj_path, robot, "rigid", eval_prefix
                    )
                )

            with _jobs_lock:
                job.exit_code = 0
                job.stdout_tail = _tail(
                    f"Saved trajectory: {summary.get('traj_path', str(output_path))}"
                )
                job.status = "succeeded"
                job.finished_at = time.time()
        except Exception as exc:
            with _jobs_lock:
                job.status = "failed"
                job.error = str(exc)
                job.exit_code = 1
                job.finished_at = time.time()
        finally:
            _job_queue.task_done()


def _tail(text: str) -> str:
    return text[-TAIL_CHARS:] if text and len(text) > TAIL_CHARS else (text or "")


def _job_response(job: OptimizationJob) -> dict[str, Any]:
    data = asdict(job)
    data["outputs"] = [
        _file_info(path)
        for path in sorted(OUTPUT_DIR.glob(f"{job.output_stem}*"))
        if path.is_file() and path.suffix.lower() in OUTPUT_SUFFIXES
    ]
    return data
