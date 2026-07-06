"""RGB-only landmark retargeting script for ViKi experiments.

This is a script version of the current exploration notebook. It consumes the
existing RGB-derived human landmark archives, runs PINK IK against Pinocchio
robot descriptions, writes a trajectory archive compatible with
eval_tracking_error.py, and can optionally sweep a small set of IK settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from viki.skeleton.hand_angles import compute_palm_rotation

try:
    from archive_io import load_archive, write_hdf5_archive
except ImportError:  # pragma: no cover - allows package-style imports later.
    try:
        from .archive_io import load_archive, write_hdf5_archive
    except ImportError:
        from experiments.archive_io import load_archive, write_hdf5_archive

try:
    from smoothing import adjusted_savgol_window, smooth_savgol
except ImportError:  # pragma: no cover - allows package-style imports later.
    try:
        from .smoothing import adjusted_savgol_window, smooth_savgol
    except ImportError:
        from experiments.smoothing import adjusted_savgol_window, smooth_savgol


RIGHT_BODY_WRIST = 16
LEFT_BODY_WRIST = 15
HAND_IDXS = {"wrist": 0, "thumb_cmc": 1, "middle_mcp": 9}

# Same transform used in the exploration notebook: MediaPipe RGB coordinates
# into the robot-facing convention used by the saved trajectory archives.
R_DEFAULT = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
T_DEFAULT = np.zeros(3, dtype=np.float64)


@dataclass(frozen=True)
class RobotConfig:
    description: str
    ee_frame: str
    joint_names: tuple[str, ...]


ROBOT_CONFIGS = {
    "ur10": RobotConfig(
        "ur10_description",
        "tool0",
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
    ),
    "ur10_description": RobotConfig(
        "ur10_description",
        "tool0",
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
    ),
    "iiwa14": RobotConfig(
        "iiwa14_description",
        "iiwa_link_ee",
        (
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ),
    ),
    "iiwa14_description": RobotConfig(
        "iiwa14_description",
        "iiwa_link_ee",
        (
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ),
    ),
}


@dataclass(frozen=True)
class RunConfig:
    robot: RobotConfig
    working_hand: str
    landmark_sg_window: int
    landmark_sg_polyorder: int
    ik_position_cost: float
    ik_orientation_cost: float
    ik_posture_cost: float
    target_mode: str
    ik_substeps: int
    ik_solver: str
    approach_sec: float
    joint_sg_window: int
    joint_sg_polyorder: int
    limit_frames: int | None
    recenter_to_neutral: bool
    trajectory_scale: float


def require_ik_dependencies():
    """Import PINK/Pinocchio dependencies with a direct runtime message."""
    try:
        import pinocchio as pin
        import pink
        from pink import Configuration, solve_ik
        from pink.tasks import FrameTask, PostureTask
        from robot_descriptions.loaders.pinocchio import load_robot_description
    except ImportError as exc:
        raise RuntimeError(
            "RGB-only retargeting requires robotics Pinocchio, PINK "
            "(pin-pink), robot_descriptions, and a QP solver such as quadprog. "
            "In the conda env used here: `python -m pip install pin-pink "
            "robot_descriptions qpsolvers quadprog typing_extensions`."
        ) from exc

    missing = [name for name in ("SE3", "neutral") if not hasattr(pin, name)]
    if missing:
        raise RuntimeError(
            "The imported 'pinocchio' module is not the robotics Pinocchio "
            f"runtime. Missing attributes: {', '.join(missing)}."
        )
    return pin, pink, Configuration, solve_ik, FrameTask, PostureTask, load_robot_description


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def npz_scalar(value: Any, default: Any = None) -> Any:
    """Return a Python scalar from a 0-D npz value."""
    if value is None:
        return default
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def should_apply_legacy_transform(coordinate_frame: Any) -> bool:
    """Legacy samples need the MediaPipe-to-robot convention transform."""
    frame = str(npz_scalar(coordinate_frame, "") or "").strip().lower()
    return frame not in {"robot_base", "robot-base", "robot base"}


def normalize_robot(robot: str) -> RobotConfig:
    key = robot.strip()
    if key not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot '{robot}'. Expected one of: {', '.join(sorted(ROBOT_CONFIGS))}.")
    return ROBOT_CONFIGS[key]


def output_traj_path(out: Path, sample_path: Path, robot: RobotConfig) -> Path:
    """Resolve --out into a trajectory archive path."""
    if out.suffix.lower() in {".h5", ".hdf5"}:
        return out
    if out.suffix.lower() == ".npz":
        return out.with_suffix(".h5")
    robot_alias = robot.description.replace("_description", "")
    name = out.name
    if not name.endswith("_traj"):
        name = f"{name}_traj"
    if out.name in {"", "."}:
        name = f"{sample_path.stem}_{robot_alias}_traj"
    return out.with_name(name + ".h5")


def safe_float_label(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text or "0"


def sweep_candidate_path(out_dir: Path, sample_path: Path, robot: RobotConfig, cfg: RunConfig) -> Path:
    robot_alias = robot.description.replace("_description", "")
    name = (
        f"{sample_path.stem}_{robot_alias}_ori{safe_float_label(cfg.ik_orientation_cost)}"
        f"_pos{safe_float_label(cfg.ik_position_cost)}"
        f"_sub{cfg.ik_substeps}_jtw{cfg.joint_sg_window}_traj.h5"
    )
    return out_dir / name


def transform_points(points: np.ndarray, rotation: np.ndarray = R_DEFAULT, translation: np.ndarray = T_DEFAULT) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return arr @ rotation.T + translation


def load_landmarks(
    sample_path: Path,
    working_hand: str,
    landmark_sg_window: int,
    landmark_sg_polyorder: int,
    limit_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load, transform, and smooth MediaPipe body/hand landmarks."""
    with np.load(sample_path, allow_pickle=True) as data:
        if "body" not in data.files:
            raise KeyError(f"{sample_path} does not contain 'body'.")
        hand_key = "right_hand" if working_hand == "right" else "left_hand"
        if hand_key not in data.files:
            raise KeyError(f"{sample_path} does not contain '{hand_key}'.")

        body = np.asarray(data["body"], dtype=np.float64)
        hand = np.asarray(data[hand_key], dtype=np.float64)
        fps = float(data["fps"].item()) if "fps" in data.files and np.asarray(data["fps"]).shape == () else 30.0
        coordinate_frame = npz_scalar(data["coordinate_frame"], "") if "coordinate_frame" in data.files else ""

    if limit_frames is not None:
        if limit_frames <= 0:
            raise ValueError("--limit-frames must be positive when set.")
        body = body[:limit_frames]
        hand = hand[:limit_frames]

    if body.ndim != 3 or body.shape[2] != 3:
        raise ValueError(f"Expected body shape (T, N, 3), got {body.shape}.")
    if hand.ndim != 3 or hand.shape[2] != 3:
        raise ValueError(f"Expected hand shape (T, N, 3), got {hand.shape}.")
    if len(body) != len(hand):
        raise ValueError(f"Body/hand frame counts differ: body={len(body)}, hand={len(hand)}.")

    if should_apply_legacy_transform(coordinate_frame):
        body_robot = transform_points(body)
        hand_robot = transform_points(hand)
    else:
        body_robot = body.copy()
        hand_robot = hand.copy()
    body_smooth = smooth_savgol(body_robot, window=landmark_sg_window, polyorder=landmark_sg_polyorder, axis=0)
    hand_smooth = smooth_savgol(hand_robot, window=landmark_sg_window, polyorder=landmark_sg_polyorder, axis=0)
    return body_smooth, hand_smooth, fps


def load_orientation_valid(sample_path: Path, limit_frames: int | None) -> np.ndarray | None:
    """Load the optional raw hand-orientation validity mask from a sample."""
    with load_archive(sample_path) as data:
        if "orientation_valid" not in data.files:
            return None
        mask = np.asarray(data["orientation_valid"], dtype=bool)
    if limit_frames is not None:
        mask = mask[:limit_frames]
    return mask


def body_wrist_index(working_hand: str) -> int:
    return RIGHT_BODY_WRIST if working_hand == "right" else LEFT_BODY_WRIST


def hand_palm_rotation(hand_frame: np.ndarray) -> np.ndarray | None:
    return compute_palm_rotation(
        hand_frame[HAND_IDXS["wrist"]],
        hand_frame[HAND_IDXS["thumb_cmc"]],
        hand_frame[HAND_IDXS["middle_mcp"]],
    )


def fill_invalid_rotations(rotations: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.array([rotation is not None for rotation in rotations], dtype=bool)
    if not valid.any():
        raise ValueError(
            "target_mode=hand_se3 requires at least one valid hand orientation "
            "from landmarks 0 (wrist), 1 (thumb CMC), and 9 (middle MCP)."
        )

    valid_indices = np.flatnonzero(valid)
    filled = np.zeros((len(rotations), 3, 3), dtype=np.float64)
    for frame_idx, rotation in enumerate(rotations):
        if rotation is not None:
            filled[frame_idx] = np.asarray(rotation, dtype=np.float64)
            continue
        nearest = valid_indices[np.argmin(np.abs(valid_indices - frame_idx))]
        nearest_rotation = rotations[int(nearest)]
        assert nearest_rotation is not None
        filled[frame_idx] = np.asarray(nearest_rotation, dtype=np.float64)
    return filled, valid


def extract_se3(pin: Any, body_frame: np.ndarray, rotation: np.ndarray, wrist_body_idx: int) -> Any:
    """Build a hand target SE3 from body wrist translation and hand orientation."""
    p = np.asarray(body_frame[wrist_body_idx], dtype=np.float64)
    return pin.SE3(np.asarray(rotation, dtype=np.float64), p)


def build_targets(
    pin: Any,
    body: np.ndarray,
    hand: np.ndarray,
    working_hand: str,
    orientation_valid_hint: np.ndarray | None = None,
) -> tuple[list[Any], np.ndarray]:
    wrist_idx = body_wrist_index(working_hand)
    rotations = [hand_palm_rotation(hand[t]) for t in range(len(hand))]
    if orientation_valid_hint is not None:
        hint = np.asarray(orientation_valid_hint, dtype=bool)
        if len(hint) != len(rotations):
            raise ValueError(
                "orientation_valid length does not match hand landmarks: "
                f"{len(hint)} != {len(rotations)}."
            )
        rotations = [rotation if hint[t] else None for t, rotation in enumerate(rotations)]
    rotations, valid = fill_invalid_rotations(rotations)
    targets = [extract_se3(pin, body[t], rotations[t], wrist_idx) for t in range(len(body))]
    return targets, valid


def build_wrist_position_targets(pin: Any, body: np.ndarray, working_hand: str) -> list[Any]:
    wrist_idx = body_wrist_index(working_hand)
    identity = np.eye(3, dtype=np.float64)
    return [pin.SE3(identity, np.asarray(body[t, wrist_idx], dtype=np.float64)) for t in range(len(body))]


def effective_orientation_cost(cfg: RunConfig) -> float:
    """Orientation is intentionally disabled for wrist-position-only targets."""
    return 0.0 if cfg.target_mode == "wrist_position" else cfg.ik_orientation_cost


def neutral_ee_position(pin: Any, robot: Any, ee_frame: str) -> np.ndarray:
    """Return the end-effector position at the robot neutral configuration."""
    frame_id = robot.model.getFrameId(ee_frame)
    q0 = pin.neutral(robot.model)
    pin.forwardKinematics(robot.model, robot.data, q0)
    pin.updateFramePlacements(robot.model, robot.data)
    return np.asarray(robot.data.oMf[frame_id].translation, dtype=np.float64)


def recenter_landmarks_to_neutral(
    pin: Any,
    robot: Any,
    ee_frame: str,
    body: np.ndarray,
    hand: np.ndarray,
    working_hand: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate all landmarks so the first wrist starts at neutral EE position."""
    wrist_idx = body_wrist_index(working_hand)
    offset = neutral_ee_position(pin, robot, ee_frame) - body[0, wrist_idx]
    return body + offset, hand + offset, offset


def scale_landmarks_about_initial_wrist(
    body: np.ndarray,
    hand: np.ndarray,
    working_hand: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly scale the motion around the first wrist position."""
    if scale <= 0.0:
        raise ValueError("trajectory_scale must be positive.")
    wrist_idx = body_wrist_index(working_hand)
    anchor = body[0, wrist_idx].copy()
    return anchor + (body - anchor) * scale, anchor + (hand - anchor) * scale


def run_approach(
    pin: Any,
    Configuration: Any,
    solve_ik: Any,
    FrameTask: Any,
    PostureTask: Any,
    robot: Any,
    ee_frame: str,
    first_target: Any,
    cfg: RunConfig,
    fps: float,
) -> np.ndarray:
    """Move from neutral to the first target before scene tracking."""
    q0 = pin.neutral(robot.model)
    configuration = Configuration(robot.model, robot.data, q0)
    frame_task = FrameTask(ee_frame, position_cost=cfg.ik_position_cost, orientation_cost=effective_orientation_cost(cfg))
    posture_task = PostureTask(cost=cfg.ik_posture_cost)
    posture_task.set_target_from_configuration(configuration)

    frames = max(1, int(round(cfg.approach_sec * fps)))
    dt = 1.0 / max(fps, 1e-9) / max(cfg.ik_substeps, 1)
    q_traj = np.zeros((frames, robot.model.nq), dtype=np.float64)
    for i in range(frames):
        alpha = (i + 1) / frames
        target = pin.SE3(first_target.rotation, alpha * first_target.translation)
        frame_task.set_target(target)
        for _ in range(cfg.ik_substeps):
            velocity = solve_ik(configuration, [frame_task, posture_task], dt, solver=cfg.ik_solver)
            configuration.integrate_inplace(velocity, dt)
        q_traj[i] = configuration.q
    return q_traj


def run_scene_ik(
    Configuration: Any,
    solve_ik: Any,
    FrameTask: Any,
    PostureTask: Any,
    robot: Any,
    ee_frame: str,
    targets: list[Any],
    q_start: np.ndarray,
    cfg: RunConfig,
    fps: float,
) -> np.ndarray:
    """Track all scene targets with differential IK."""
    configuration = Configuration(robot.model, robot.data, np.asarray(q_start, dtype=np.float64).copy())
    frame_task = FrameTask(ee_frame, position_cost=cfg.ik_position_cost, orientation_cost=effective_orientation_cost(cfg))
    posture_task = PostureTask(cost=cfg.ik_posture_cost)
    posture_task.set_target_from_configuration(configuration)

    dt = 1.0 / max(fps, 1e-9) / max(cfg.ik_substeps, 1)
    q_traj = np.zeros((len(targets), robot.model.nq), dtype=np.float64)
    for i, target in enumerate(targets):
        frame_task.set_target(target)
        for _ in range(cfg.ik_substeps):
            velocity = solve_ik(configuration, [frame_task, posture_task], dt, solver=cfg.ik_solver)
            configuration.integrate_inplace(velocity, dt)
        q_traj[i] = configuration.q
    return q_traj


def smooth_joint_trajectory(q_scene: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if window <= 0:
        return np.asarray(q_scene, dtype=np.float64).copy()
    adjusted = adjusted_savgol_window(len(q_scene), window, polyorder)
    return smooth_savgol(q_scene, window=adjusted, polyorder=polyorder, axis=0)


def compute_tracking_error(pin: Any, robot: Any, ee_frame: str, q_traj: np.ndarray, targets: list[Any]) -> tuple[np.ndarray, np.ndarray]:
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


def retarget(sample_path: Path, out_path: Path, cfg: RunConfig) -> dict[str, Any]:
    """Run one retargeting job and save a compatible trajectory archive."""
    pin, _pink, Configuration, solve_ik, FrameTask, PostureTask, load_robot_description = require_ik_dependencies()
    body, hand, fps = load_landmarks(
        sample_path,
        cfg.working_hand,
        cfg.landmark_sg_window,
        cfg.landmark_sg_polyorder,
        cfg.limit_frames,
    )
    sample_orientation_valid = load_orientation_valid(sample_path, cfg.limit_frames)

    robot = load_robot_description(cfg.robot.description)
    if robot.model.getFrameId(cfg.robot.ee_frame) >= len(robot.model.frames):
        raise ValueError(f"End-effector frame '{cfg.robot.ee_frame}' not found in {cfg.robot.description}.")

    if abs(cfg.trajectory_scale - 1.0) > 1e-12:
        body, hand = scale_landmarks_about_initial_wrist(
            body,
            hand,
            cfg.working_hand,
            cfg.trajectory_scale,
        )

    recenter_offset = np.zeros(3, dtype=np.float64)
    if cfg.recenter_to_neutral:
        body, hand, recenter_offset = recenter_landmarks_to_neutral(
            pin,
            robot,
            cfg.robot.ee_frame,
            body,
            hand,
            cfg.working_hand,
        )

    orientation_valid = None
    if cfg.target_mode == "hand_se3":
        targets, orientation_valid = build_targets(
            pin,
            body,
            hand,
            cfg.working_hand,
            sample_orientation_valid,
        )
    elif cfg.target_mode == "wrist_position":
        targets = build_wrist_position_targets(pin, body, cfg.working_hand)
    else:
        raise ValueError(f"Unknown target_mode: {cfg.target_mode}")
    target_pos = np.vstack([target.translation for target in targets])
    target_rot = np.stack([target.rotation for target in targets], axis=0) if cfg.target_mode == "hand_se3" else None

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
    q_scene_smooth = smooth_joint_trajectory(q_scene_raw, cfg.joint_sg_window, cfg.joint_sg_polyorder)
    pos_err_smooth, ori_err_smooth = compute_tracking_error(pin, robot, cfg.robot.ee_frame, q_scene_smooth, targets)
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
        "sg_window": int(cfg.landmark_sg_window),
        "sg_polyorder": int(cfg.landmark_sg_polyorder),
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
        "trajectory_scale": float(cfg.trajectory_scale),
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
        "trajectory_scale": float(cfg.trajectory_scale),
        "mean_not_aligned_pos_error_mm": float(1000.0 * np.mean(pos_err_smooth)),
        "median_not_aligned_pos_error_mm": float(1000.0 * np.median(pos_err_smooth)),
        "mean_not_aligned_orientation_error_deg": float(np.mean(ori_err_smooth_deg)),
        "median_not_aligned_orientation_error_deg": float(np.median(ori_err_smooth_deg)),
        "p95_not_aligned_orientation_error_deg": float(np.percentile(ori_err_smooth_deg, 95)),
        "max_not_aligned_orientation_error_deg": float(np.max(ori_err_smooth_deg)),
    }
    if orientation_valid is not None:
        summary["orientation_valid_frames"] = int(orientation_valid.sum())
        summary["orientation_total_frames"] = int(len(orientation_valid))
    print(
        f"Saved trajectory: {out_path} "
        f"(mean unaligned notebook-style error={summary['mean_not_aligned_pos_error_mm']:.1f} mm, "
        f"orientation={summary['mean_not_aligned_orientation_error_deg']:.1f} deg)"
    )
    return summary


def evaluate_saved_traj(sample_path: Path, traj_path: Path, robot: RobotConfig, align: str, out_prefix: Path) -> dict[str, Any]:
    """Run eval_tracking_error.py's evaluator for a saved trajectory."""
    try:
        from eval_tracking_error import evaluate
    except ImportError:  # pragma: no cover - allows package-style imports later.
        try:
            from .eval_tracking_error import evaluate
        except ImportError:
            from experiments.eval_tracking_error import evaluate

    args = argparse.Namespace(
        human=str(sample_path),
        robot_traj=str(traj_path),
        robot=robot.description,
        ee_frame=robot.ee_frame,
        q_key="auto",
        target_source="auto",
        hand=None,
        smoothing="none",
        smooth_window=15,
        smooth_polyorder=3,
        align=align,
        threshold_mm=50.0,
        out=str(out_prefix),
    )
    return evaluate(args)


def write_sweep_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sweep_results.json"
    csv_path = out_dir / "sweep_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved sweep JSON: {json_path}")
    print(f"Saved sweep CSV:  {csv_path}")


def select_recommended_candidate(
    rows: list[dict[str, Any]],
    p95_position_limit_mm: float = 50.0,
    min_orientation_reduction_pct: float = 10.0,
) -> dict[str, Any] | None:
    """Select the lowest useful nonzero orientation-cost candidate.

    Orientation-cost 0 is treated as a position-feasibility baseline. A
    recommended manipulation candidate must keep p95 position error below the
    requested limit and reduce mean orientation error by at least
    min_orientation_reduction_pct relative to the zero-cost baseline when that
    baseline is present.
    """
    if not rows:
        return None

    baselines = [
        row for row in rows
        if abs(float(row.get("ik_orientation_cost", 0.0))) < 1e-12
        and "mean_orientation_error_deg" in row
    ]
    baseline_mean_ori = float(baselines[0]["mean_orientation_error_deg"]) if baselines else None

    candidates = sorted(
        rows,
        key=lambda row: (
            float(row.get("ik_orientation_cost", 0.0)),
            float(row.get("mean_orientation_error_deg", np.inf)),
            float(row.get("p95_error_mm", np.inf)),
        ),
    )
    for row in candidates:
        orientation_cost = float(row.get("ik_orientation_cost", 0.0))
        if orientation_cost <= 0.0:
            continue
        if float(row.get("p95_error_mm", np.inf)) >= p95_position_limit_mm:
            continue
        if baseline_mean_ori is None:
            return row
        mean_ori = float(row.get("mean_orientation_error_deg", np.inf))
        reduction_pct = 100.0 * (baseline_mean_ori - mean_ori) / max(baseline_mean_ori, 1e-9)
        if reduction_pct >= min_orientation_reduction_pct:
            enriched = dict(row)
            enriched["orientation_reduction_vs_cost0_pct"] = float(reduction_pct)
            return enriched
    return None


def write_recommendation(out_dir: Path, recommendation: dict[str, Any] | None) -> None:
    path = out_dir / "sweep_recommendation.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved sweep recommendation: {path}")


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
        landmark_sg_window=args.sg_window,
        landmark_sg_polyorder=args.sg_polyorder,
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
        trajectory_scale=args.trajectory_scale,
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
        summary.update(evaluate_saved_traj(sample_path, traj_path, robot, args.eval_align, eval_prefix))
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
    for pos_cost, ori_cost, substeps, joint_window in product(position_costs, ori_costs, substeps_values, joint_windows):
        cfg = build_run_config(args, robot, pos_cost, ori_cost, substeps, joint_window)
        traj_path = sweep_candidate_path(out_dir, sample_path, robot, cfg)
        row = retarget(sample_path, traj_path, cfg)
        if args.evaluate:
            eval_prefix = out_dir / f"eval_{traj_path.stem.replace('_traj', '')}"
            row.update(evaluate_saved_traj(sample_path, traj_path, robot, args.eval_align, eval_prefix))
        rows.append(row)
        write_sweep_outputs(out_dir, rows)
        write_recommendation(out_dir, select_recommended_candidate(rows))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retarget RGB-only MediaPipe landmarks to saved robot trajectories.")
    parser.add_argument("--sample", required=True, help="Path to experiments/samples/*.npz.")
    parser.add_argument("--robot", default="ur10", help="Robot alias/name: ur10 or iiwa14.")
    parser.add_argument("--working-hand", default="right", choices=["right", "left"], help="Hand to retarget.")
    parser.add_argument("--out", required=True, help="Output .h5 path for a single run, or output directory for --sweep.")
    parser.add_argument("--sg-window", type=int, default=35, help="Landmark Savitzky-Golay window.")
    parser.add_argument("--sg-polyorder", type=int, default=3, help="Landmark Savitzky-Golay polynomial order.")
    parser.add_argument("--ik-position-cost", type=float, default=1.0, help="PINK frame position cost.")
    parser.add_argument("--ik-orientation-cost", type=float, default=0.3, help="PINK frame orientation cost.")
    parser.add_argument("--ik-posture-cost", type=float, default=1e-3, help="PINK posture task cost.")
    parser.add_argument(
        "--target-mode",
        default="wrist_position",
        choices=["hand_se3", "wrist_position"],
        help="Use full hand-derived SE3 target or wrist position only.",
    )
    parser.add_argument("--ik-substeps", type=int, default=20, help="IK integration substeps per video frame.")
    parser.add_argument("--ik-solver", default="quadprog", help="qpsolvers backend name.")
    parser.add_argument("--approach-sec", type=float, default=5.0, help="Approach duration before scene tracking.")
    parser.add_argument("--joint-sg-window", type=int, default=15, help="Joint smoothing window; <=0 disables smoothing.")
    parser.add_argument("--joint-sg-polyorder", type=int, default=3, help="Joint smoothing polynomial order.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Only retarget the first N frames for smoke tests.")
    parser.add_argument(
        "--recenter-to-neutral",
        action="store_true",
        help="Translate the input skeleton so frame-0 wrist starts at the robot neutral EE position.",
    )
    parser.add_argument(
        "--trajectory-scale",
        type=float,
        default=1.0,
        help="Uniformly scale all landmark motion about the first wrist before retargeting.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run FK evaluator after writing each trajectory.")
    parser.add_argument("--eval-align", default="rigid", choices=["rigid", "none"], help="Evaluator alignment mode.")
    parser.add_argument("--sweep", action="store_true", help="Run a parameter sweep instead of one configuration.")
    parser.add_argument("--sweep-position-costs", default="1", help="Comma-separated PINK position costs.")
    parser.add_argument("--sweep-ori-costs", default="0,0.05,0.1,0.3", help="Comma-separated orientation costs.")
    parser.add_argument("--sweep-substeps", default="20,40", help="Comma-separated IK substep counts.")
    parser.add_argument("--sweep-joint-windows", default="0,15,25", help="Comma-separated joint smoothing windows.")
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
