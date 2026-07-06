from __future__ import annotations

from argparse import Namespace
import logging
from pathlib import Path
from fastapi import APIRouter

from viki.optimization.optimization.retarget_rgb_only import run_single
from viki.config import SKELETON_SMOOTHED_DIR

router = APIRouter(prefix="/api/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)


# @router.get("/recordings")
# async def list_dataset_recordings(
#     page: int = 0,
#     limit: int = 10,
# ):
#     # This whole function is just a copy of the SkeletonProcessor's __init__ and list_recordings methods
#     # TODO: add config.SKELETON_OPTIMIZED_RECS_DIR
#     robot_out = Path("./data/robot_out")
#     # TODO:  The line below should be moved somewhere else outside of this endpoint
#     robot_out.mkdir(parents=True, exist_ok=True)
#     files = sorted([f.name for f in robot_out.glob("*.h5")], reverse=True)
#     start = page * limit
#     end = start + limit
#     return {"recordings": files[start:end]}


@router.get("/recordings")
async def list_smoothed_recordings(
    page: int = 0,
    limit: int = 10,
):
    # This whole function is just a copy of the SkeletonProcessor's __init__ and list_recordings methods
    # TODO: add config.SKELETON_OPTIMIZED_RECS_DIR
    smoothed_dir = Path(SKELETON_SMOOTHED_DIR)
    # TODO:  The line below should be moved somewhere else outside of this endpoint
    smoothed_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in smoothed_dir.glob("cln-*.npz")], reverse=True)
    start = page * limit
    end = start + limit
    return {"recordings": files[start:end]}


@router.post("/optimize/{filename}")
async def optimize_recording(filename: str):

    logging.info(
        run_single(
            Namespace(
                sample="./data/skeleton_smoothed/" + filename,
                robot="ur10",
                working_hand="right",
                out="./data/robot_out/" + filename.split(".")[0] + ".h5",
                target_mode="hand_se3",
                ik_position_cost=5.0,
                ik_orientation_cost=0.3,
                ik_posture_cost=1e-3,
                ik_substeps=20,
                ik_solver="quadprog",
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
            )
        )
    )
