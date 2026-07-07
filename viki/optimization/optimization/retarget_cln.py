"""Retarget cln-*.npz end-effector data (positions + rotations) to robot IK.

This is a lighter counterpart to retarget_rgb_only.py for when the
landmark-to-EE step has already been done by SkeletonProcessor (which
produces cln-*.npz files).  It loads positions/rotations/valid/timestamps
directly, builds SE3 targets, and runs the same PINK IK pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

try:
    from archive_io import write_hdf5_archive
except ImportError:
    try:
        from .archive_io import write_hdf5_archive
    except ImportError:
        from viki.optimization.optimization.archive_io import write_hdf5_archive

try:
    from retarget_rgb_only import (
        ROBOT_CONFIGS,
        RobotConfig,
        RunConfig,
        effective_orientation_cost,
        evaluate_saved_traj,
        fill_invalid_rotations,
        neutral_ee_position,
        normalize_robot,
        output_traj_path,
        parse_float_list,
        parse_int_list,
        require_ik_dependencies,
        run_approach,
        run_scene_ik,
        safe_float_label,
        select_recommended_candidate,
        smooth_joint_trajectory,
        sweep_candidate_path,
        write_recommendation,
        write_sweep_outputs,
    )
except ImportError:
    try:
        from .retarget_rgb_only import (
            ROBOT_CONFIGS,
            RobotConfig,
            RunConfig,
            effective_orientation_cost,
            evaluate_saved_traj,
            fill_invalid_rotations,
            neutral_ee_position,
            normalize_robot,
            output_traj_path,
            parse_float_list,
            parse_int_list,
            require_ik_dependencies,
            run_approach,
            run_scene_ik,
            safe_float_label,
            select_recommended_candidate,
            smooth_joint_trajectory,
            sweep_candidate_path,
            write_recommendation,
            write_sweep_outputs,
        )
    except ImportError:
        from viki.optimization.optimization.retarget_rgb_only import (
            ROBOT_CONFIGS,
            RobotConfig,
            RunConfig,
            effective_orientation_cost,
            evaluate_saved_traj,
            fill_invalid_rotations,
            neutral_ee_position,
            normalize_robot,
            output_traj_path,
            parse_float_list,
            parse_int_list,
            require_ik_dependencies,
            run_approach,
            run_scene_ik,
            safe_float_label,
            select_recommended_candidate,
            smooth_joint_trajectory,
            sweep_candidate_path,
            write_recommendation,
            write_sweep_outputs,
        )


def load_cln_ee(
    sample_path: Path,
    limit_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load end-effector pose data from a cln-*.npz smoothed skeleton file."""
    with np.load(sample_path) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        rotations = np.asarray(data["rotations"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        timestamps = np.asarray(data["timestamps"], dtype=np.int64)

    T = len(positions)
    if limit_frames is not None:
        if limit_frames <= 0:
            raise ValueError("--limit-frames must be positive when set.")
        positions = positions[:limit_frames]
        rotations = rotations[:limit_frames]
        valid = valid[:limit_frames]
        timestamps = timestamps[:limit_frames]

    dt_us = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 33_333.0
    fps = float(1_000_000.0 / max(dt_us, 1.0))
    return positions, rotations, valid, fps


def retarget(sample_path: Path, out_path: Path, cfg: RunConfig) -> dict[str, Any]:
    """Run one retargeting job from a cln-*.npz file."""
    (
        pin,
        _pink,
        Configuration,
        solve_ik,
        FrameTask,
        PostureTask,
        load_robot_description,
    ) = require_ik_dependencies()

    robot = load_robot_description(cfg.robot.description)
    if robot.model.getFrameId(cfg.robot.ee_frame) >= len(robot.model.frames):
        raise ValueError(
            f"End-effector frame '{cfg.robot.ee_frame}' not found in {cfg.robot.description}."
        )

    positions, rotations, valid, fps = load_cln_ee(sample_path, cfg.limit_frames)

    recenter_offset = np.zeros(3, dtype=np.float64)
    if cfg.recenter_to_neutral:
        offset = neutral_ee_position(pin, robot, cfg.robot.ee_frame) - positions[0]
        positions = positions + offset
        recenter_offset = offset

    orientation_valid = None
    if cfg.target_mode == "hand_se3":
        rot_list = [rotations[t] if valid[t] else None for t in range(len(rotations))]
        filled_rots, orientation_valid = fill_invalid_rotations(rot_list)
        targets = [pin.SE3(filled_rots[t], positions[t]) for t in range(len(positions))]
    elif cfg.target_mode == "wrist_position":
        identity = np.eye(3, dtype=np.float64)
        targets = [pin.SE3(identity, positions[t]) for t in range(len(positions))]
    else:
        raise ValueError(f"Unknown target_mode: {cfg.target_mode}")
    target_pos = np.vstack([target.translation for target in targets])
    target_rot = (
        np.stack([target.rotation for target in targets], axis=0)
        if cfg.target_mode == "hand_se3"
        else None
    )

    q_approach = run_approach(
        pin,
        Configuration,
        solve_ik,
        FrameTask,
        PostureTask,
        robot,
        cfg.robot.ee_frame,
        targets[0],
        cfg,
        fps,
    )
    q_scene_raw = run_scene_ik(
        Configuration,
        solve_ik,
        FrameTask,
        PostureTask,
        robot,
        cfg.robot.ee_frame,
        targets,
        q_approach[-1],
        cfg,
        fps,
    )
    q_scene_smooth = smooth_joint_trajectory(
        q_scene_raw, cfg.joint_sg_window, cfg.joint_sg_polyorder
    )
    pos_err_smooth, ori_err_smooth = compute_tracking_error(
        pin, robot, cfg.robot.ee_frame, q_scene_smooth, targets
    )
    ori_err_smooth_deg = np.degrees(ori_err_smooth)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    archive = {
        "q_approach": q_approach,
        "q_scene_raw": q_scene_raw,
        "q_scene_smooth": q_scene_smooth,
        "ee_target_pos": target_pos,
        "pos_err_smooth": pos_err_smooth,
        "ori_err_smooth": ori_err_smooth,
        "ori_err_smooth_deg": ori_err_smooth_deg,
        "fps": float(fps),
        "dt": float(1.0 / max(fps, 1e-9)),
        "robot": cfg.robot.description,
        "ee_frame": cfg.robot.ee_frame,
        "working_hand": cfg.working_hand,
        "target_mode": cfg.target_mode,
        "input_mode": "cln",
        "sg_window": 0,
        "sg_polyorder": 0,
        "joint_sg_window": int(cfg.joint_sg_window),
        "joint_sg_polyorder": int(cfg.joint_sg_polyorder),
        "ik_position_cost": float(cfg.ik_position_cost),
        "ik_orientation_cost": float(cfg.ik_orientation_cost),
        "effective_orientation_cost": float(effective_orientation_cost(cfg)),
        "ik_posture_cost": float(cfg.ik_posture_cost),
        "ik_substeps": int(cfg.ik_substeps),
        "ik_solver": cfg.ik_solver,
        "recenter_to_neutral": bool(cfg.recenter_to_neutral),
        "recenter_offset": recenter_offset,
        "trajectory_scale": 1.0,
        "source_npz": str(sample_path),
    }
    if target_rot is not None:
        archive["ee_target_rot"] = target_rot
    if orientation_valid is not None:
        archive["orientation_valid"] = orientation_valid
    write_hdf5_archive(out_path, archive)

    summary = {
        "traj_path": str(out_path),
        "robot": cfg.robot.description,
        "ee_frame": cfg.robot.ee_frame,
        "frames": int(len(q_scene_smooth)),
        "fps": float(fps),
        "working_hand": cfg.working_hand,
        "ik_position_cost": float(cfg.ik_position_cost),
        "ik_orientation_cost": float(cfg.ik_orientation_cost),
        "effective_orientation_cost": float(effective_orientation_cost(cfg)),
        "ik_posture_cost": float(cfg.ik_posture_cost),
        "ik_substeps": int(cfg.ik_substeps),
        "ik_solver": cfg.ik_solver,
        "target_mode": cfg.target_mode,
        "joint_sg_window": int(cfg.joint_sg_window),
        "recenter_to_neutral": bool(cfg.recenter_to_neutral),
        "recenter_offset": recenter_offset.tolist(),
        "mean_pos_error_mm": float(1000.0 * np.mean(pos_err_smooth)),
        "median_pos_error_mm": float(1000.0 * np.median(pos_err_smooth)),
        "mean_orientation_error_deg": float(np.mean(ori_err_smooth_deg)),
        "median_orientation_error_deg": float(np.median(ori_err_smooth_deg)),
        "p95_orientation_error_deg": float(np.percentile(ori_err_smooth_deg, 95)),
        "max_orientation_error_deg": float(np.max(ori_err_smooth_deg)),
    }
    if orientation_valid is not None:
        summary["orientation_valid_frames"] = int(orientation_valid.sum())
        summary["orientation_total_frames"] = int(len(orientation_valid))
    print(
        f"Saved trajectory: {out_path} "
        f"(mean pos error={summary['mean_pos_error_mm']:.1f} mm, "
        f"orientation={summary['mean_orientation_error_deg']:.1f} deg)"
    )
    return summary


def compute_tracking_error(pin, robot, ee_frame, q_traj, targets):
    frame_id = robot.model.getFrameId(ee_frame)
    pos_err = np.zeros(len(q_traj), dtype=np.float64)
    ori_err = np.zeros(len(q_traj), dtype=np.float64)
    for i, q in enumerate(q_traj):
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        ee_pose = robot.data.oMf[frame_id]
        delta = targets[i].actInv(ee_pose)
        pos_err[i] = float(np.linalg.norm(ee_pose.translation - targets[i].translation))
        ori_err[i] = float(np.linalg.norm(pin.log3(delta.rotation)))
    return pos_err, ori_err


def build_run_config(
    args: argparse.Namespace,
    robot: RobotConfig,
    pos_cost: float,
    ori_cost: float,
    substeps: int,
    joint_window: int,
) -> RunConfig:
    return RunConfig(
        robot=robot,
        working_hand=args.working_hand,
        landmark_sg_window=0,
        landmark_sg_polyorder=0,
        ik_position_cost=pos_cost,
        ik_orientation_cost=ori_cost,
        ik_posture_cost=args.ik_posture_cost,
        target_mode=args.target_mode,
        ik_substeps=substeps,
        ik_solver=args.ik_solver,
        approach_sec=args.approach_sec,
        joint_sg_window=joint_window,
        joint_sg_polyorder=args.joint_sg_polyorder,
        limit_frames=args.limit_frames,
        recenter_to_neutral=args.recenter_to_neutral,
        trajectory_scale=1.0,
    )


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    sample_path = Path(args.sample)
    robot = normalize_robot(args.robot)
    cfg = build_run_config(
        args,
        robot,
        args.ik_position_cost,
        args.ik_orientation_cost,
        args.ik_substeps,
        args.joint_sg_window,
    )
    traj_path = output_traj_path(Path(args.out), sample_path, robot)
    summary = retarget(sample_path, traj_path, cfg)
    if args.evaluate:
        eval_prefix = traj_path.with_name(traj_path.stem.replace("_traj", "") + "_eval")
        summary.update(
            evaluate_saved_traj(
                sample_path, traj_path, robot, args.eval_align, eval_prefix
            )
        )
    return summary


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    sample_path = Path(args.sample)
    robot = normalize_robot(args.robot)
    out_dir = Path(args.out)
    if out_dir.suffix:
        out_dir = out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    position_costs = parse_float_list(args.sweep_position_costs)
    ori_costs = parse_float_list(args.sweep_ori_costs)
    substeps_values = parse_int_list(args.sweep_substeps)
    joint_windows = parse_int_list(args.sweep_joint_windows)
    for pos_cost, ori_cost, substeps, joint_window in product(
        position_costs, ori_costs, substeps_values, joint_windows
    ):
        cfg = build_run_config(args, robot, pos_cost, ori_cost, substeps, joint_window)
        traj_path = sweep_candidate_path(out_dir, sample_path, robot, cfg)
        row = retarget(sample_path, traj_path, cfg)
        if args.evaluate:
            eval_prefix = out_dir / f"eval_{traj_path.stem.replace('_traj', '')}"
            row.update(
                evaluate_saved_traj(
                    sample_path, traj_path, robot, args.eval_align, eval_prefix
                )
            )
        rows.append(row)
        write_sweep_outputs(out_dir, rows)
        write_recommendation(out_dir, select_recommended_candidate(rows))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget cln-*.npz end-effector data to robot trajectories."
    )
    parser.add_argument(
        "--sample", required=True, help="Path to cln-*.npz smoothed skeleton file."
    )
    parser.add_argument(
        "--robot", default="ur10", help="Robot alias/name: ur10 or iiwa14."
    )
    parser.add_argument(
        "--working-hand",
        default="right",
        choices=["right", "left"],
        help="Hand used (metadata only).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .h5 path for a single run, or output directory for --sweep.",
    )
    parser.add_argument(
        "--ik-position-cost", type=float, default=1.0, help="PINK frame position cost."
    )
    parser.add_argument(
        "--ik-orientation-cost",
        type=float,
        default=0.3,
        help="PINK frame orientation cost.",
    )
    parser.add_argument(
        "--ik-posture-cost", type=float, default=1e-3, help="PINK posture task cost."
    )
    parser.add_argument(
        "--target-mode",
        default="hand_se3",
        choices=["hand_se3", "wrist_position"],
        help="Use full SE3 or wrist position only.",
    )
    parser.add_argument(
        "--ik-substeps",
        type=int,
        default=20,
        help="IK integration substeps per video frame.",
    )
    parser.add_argument(
        "--ik-solver", default="quadprog", help="qpsolvers backend name."
    )
    parser.add_argument(
        "--approach-sec",
        type=float,
        default=5.0,
        help="Approach duration before scene tracking.",
    )
    parser.add_argument(
        "--joint-sg-window",
        type=int,
        default=15,
        help="Joint smoothing window; <=0 disables.",
    )
    parser.add_argument(
        "--joint-sg-polyorder",
        type=int,
        default=3,
        help="Joint smoothing polynomial order.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Only retarget the first N frames.",
    )
    parser.add_argument(
        "--recenter-to-neutral",
        action="store_true",
        help="Shift so frame-0 EE starts at robot neutral position.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run FK evaluator after writing each trajectory.",
    )
    parser.add_argument(
        "--eval-align",
        default="rigid",
        choices=["rigid", "none"],
        help="Evaluator alignment mode.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a parameter sweep instead of one configuration.",
    )
    parser.add_argument(
        "--sweep-position-costs",
        default="1",
        help="Comma-separated PINK position costs.",
    )
    parser.add_argument(
        "--sweep-ori-costs",
        default="0,0.05,0.1,0.3",
        help="Comma-separated orientation costs.",
    )
    parser.add_argument(
        "--sweep-substeps", default="20,40", help="Comma-separated IK substep counts."
    )
    parser.add_argument(
        "--sweep-joint-windows",
        default="0,15,25",
        help="Comma-separated joint smoothing windows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.sweep:
            rows = run_sweep(args)
            recommendation = select_recommended_candidate(rows)
            print("Recommended candidate:")
            print(json.dumps(recommendation, indent=2, sort_keys=True))
        else:
            summary = run_single(args)
            print("Run summary:")
            print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
