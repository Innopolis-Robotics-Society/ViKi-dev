"""
viki.server.routes.optimization
-------------------------------
Backend endpoints for experimental skeleton retargeting/optimisation.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
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

from skeleton_tests.optimization.convert_viki23_json import convert


router = APIRouter(prefix="/api/optimization", tags=["optimization"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKELETON_TESTS_DIR = PROJECT_ROOT / "skeleton_tests"
OPTIMIZATION_DIR = SKELETON_TESTS_DIR / "optimization"
SAMPLES_DIR = SKELETON_TESTS_DIR / "samples"
OUTPUT_DIR = SKELETON_TESTS_DIR / "output"
RECORDING_DIRS = (PROJECT_ROOT, PROJECT_ROOT / "data" / "skeleton_recs")
OUTPUT_SUFFIXES = {".h5", ".hdf5", ".json", ".png"}
COND_ENV_VAR = "VIKI_OPT_CONDA_EXE"
COND_ENV_NAME_VAR = "VIKI_OPT_CONDA_ENV"
DEFAULT_CONDA_EXE = r"C:\Users\minim\miniforge3\Scripts\conda.exe"
DEFAULT_CONDA_ENV = "viki-fk"
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
    ik_position_cost: float = 1.0
    ik_orientation_cost: float = 0.0
    joint_sg_window: int = 0
    sg_window: int = 7
    recenter_to_neutral: bool = True
    trajectory_scale: float = Field(default=0.25, gt=0.0)
    evaluate: bool = True


@dataclass
class OptimizationJob:
    job_id: str
    status: str
    command: list[str]
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
_job_queue: queue.Queue[str] = queue.Queue()
_worker_started = False


@router.get("/recordings")
async def list_recordings() -> dict[str, Any]:
    return {"recordings": [_file_info(path) for path in _recording_paths()]}


@router.post("/convert")
async def convert_recording(req: ConvertRequest) -> dict[str, Any]:
    _ensure_dirs()
    recording = _resolve_recording(req.recording)
    output_name = _safe_filename(req.output_name, expected_suffix=".npz")
    output_path = SAMPLES_DIR / output_name

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
    return {"samples": [_file_info(path) for path in sorted(SAMPLES_DIR.glob("*.npz"))]}


@router.post("/retarget")
async def retarget(req: RetargetRequest) -> dict[str, Any]:
    _ensure_dirs()
    sample_name = _safe_filename(req.sample, expected_suffix=".npz")
    sample_path = SAMPLES_DIR / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_name}")

    output_name = _safe_filename(req.output_name)
    output_path = OUTPUT_DIR / output_name
    output_stem = output_path.stem if output_path.suffix else output_path.name
    output_stem = output_stem.replace("_traj", "")
    conda_exe = _resolve_conda_exe()
    conda_env = os.getenv(COND_ENV_NAME_VAR, DEFAULT_CONDA_ENV)

    command = [
        conda_exe,
        "run",
        "-n",
        conda_env,
        "python",
        str(OPTIMIZATION_DIR / "retarget_rgb_only.py"),
        "--sample",
        str(sample_path),
        "--robot",
        req.robot,
        "--out",
        str(output_path),
        "--target-mode",
        req.target_mode,
        "--ik-position-cost",
        str(req.ik_position_cost),
        "--ik-orientation-cost",
        str(req.ik_orientation_cost),
        "--joint-sg-window",
        str(req.joint_sg_window),
        "--sg-window",
        str(req.sg_window),
        "--trajectory-scale",
        str(req.trajectory_scale),
    ]
    if req.recenter_to_neutral:
        command.append("--recenter-to-neutral")
    if req.evaluate:
        command.append("--evaluate")

    job = _enqueue_job(command, output_stem)
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


@router.get("/outputs/{filename:path}")
async def download_output(filename: str) -> FileResponse:
    name = _safe_filename(filename)
    path = OUTPUT_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Output not found: {name}")
    if path.suffix.lower() not in OUTPUT_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported output file type")
    return FileResponse(path, filename=name)


def _ensure_dirs() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
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
        "looks_smoothed": "smoothed" in path.stem.lower(),
    }


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _resolve_conda_exe() -> str:
    configured = os.getenv(COND_ENV_VAR, DEFAULT_CONDA_EXE)
    if Path(configured).exists():
        return configured
    resolved = shutil.which(configured)
    if resolved:
        return resolved
    raise HTTPException(
        status_code=503,
        detail=(
            f"Retargeting conda executable is unavailable: {configured}. "
            f"Set {COND_ENV_VAR} to a valid conda executable."
        ),
    )


def _enqueue_job(command: list[str], output_stem: str) -> OptimizationJob:
    global _worker_started
    job = OptimizationJob(
        job_id=uuid.uuid4().hex,
        status="queued",
        command=command,
        output_stem=output_stem,
    )
    with _jobs_lock:
        _jobs[job.job_id] = job
        if not _worker_started:
            thread = threading.Thread(target=_job_worker, daemon=True)
            thread.start()
            _worker_started = True
    _job_queue.put(job.job_id)
    return job


def _job_worker() -> None:
    while True:
        job_id = _job_queue.get()
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
        try:
            completed = subprocess.run(
                job.command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            with _jobs_lock:
                job.exit_code = completed.returncode
                job.stdout_tail = _tail(completed.stdout)
                job.stderr_tail = _tail(completed.stderr)
                job.status = "succeeded" if completed.returncode == 0 else "failed"
                job.finished_at = time.time()
                if completed.returncode != 0:
                    job.error = f"Retarget command exited with {completed.returncode}"
        except Exception as exc:  # pragma: no cover - defensive background path
            with _jobs_lock:
                job.status = "failed"
                job.error = str(exc)
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
