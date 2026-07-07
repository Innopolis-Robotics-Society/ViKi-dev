from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from viki.optimization.optimization.convert_viki23_json import estimate_fps
from viki.optimization.optimization.retarget_rgb_only import (
    normalize_robot,
    retarget_from_poses,
    RunConfig,
)
from viki.config import SKELETON_SMOOTHED_DIR

router = APIRouter(prefix="/api/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)


@router.get("/recordings")
async def list_smoothed_recordings(page: int = 0, limit: int = 10):
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    smoothed_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in smoothed_dir.glob("cln-*.npz")], reverse=True)
    start = page * limit
    end = start + limit
    return {"recordings": files[start:end]}


@router.post("/optimize/{filename}")
async def optimize_recording(filename: str):
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    cln_path = smoothed_dir / filename
    if not cln_path.exists():
        raise HTTPException(status_code=404, detail=f"Recording not found: {filename}")

    with np.load(cln_path) as data:
        positions = data["positions"]
        rotations = data["rotations"]
        validity = data["valid"]
        timestamps = data["timestamps"]

    fps = estimate_fps(timestamps)
    robot = normalize_robot("ur10")
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

    summary = retarget_from_poses(positions, rotations, validity, fps, out_path, cfg)
    logger.info("Retarget summary: %s", summary)
    return summary
