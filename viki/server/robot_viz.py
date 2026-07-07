from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from viki.viz.mjpeg import mjpeg_chunk, placeholder


def _load_robot(description: str):
    os.environ.setdefault(
        "ROBOT_DESCRIPTIONS_CACHE",
        "/app/models/robot_descriptions",
    )
    from robot_descriptions.loaders.pinocchio import load_robot_description
    return load_robot_description(description)


def _fk_positions(model, data, q_all: np.ndarray, ee_frame: str) -> tuple[np.ndarray, np.ndarray]:
    import pinocchio as pin
    n_frames = q_all.shape[0]
    all_positions = []
    ee_positions = np.zeros((n_frames, 3), dtype=np.float64)
    for i in range(n_frames):
        pin.forwardKinematics(model, data, q_all[i])
        pin.updateFramePlacements(model, data)
        frame_pos = []
        for name in model.names:
            if name == "universe":
                continue
            fid = model.getFrameId(name)
            frame_pos.append(data.oMf[fid].translation.copy())
        ee_fid = model.getFrameId(ee_frame)
        ee_positions[i] = data.oMf[ee_fid].translation.copy()
        frame_pos.append(ee_positions[i])
        all_positions.append(np.array(frame_pos))
    return np.array(all_positions), ee_positions


def robot_trajectory_stream(
    h5_path: Path,
    q_key: str = "q_scene_raw",
    loop: bool = False,
) -> Iterator[bytes]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not h5_path.exists():
        while True:
            yield mjpeg_chunk(placeholder(640, 480, f"File not found: {h5_path.name}"))
            time.sleep(1)

    with h5py.File(h5_path, "r") as f:
        if q_key not in f:
            q_key = "q_scene_smooth" if "q_scene_smooth" in f else "q_scene_raw"
        if q_key not in f:
            while True:
                yield mjpeg_chunk(placeholder(640, 480, f"No trajectory key in {h5_path.name}"))
                time.sleep(1)
        q_all = f[q_key][:]
        robot_name = f["robot"][()] if "robot" in f else "iiwa14_description"
        ee_frame = f["ee_frame"][()] if "ee_frame" in f else "iiwa_link_ee"
        fps = float(f["fps"][()]) if "fps" in f else 15.0
        if isinstance(robot_name, bytes):
            robot_name = robot_name.decode()
        if isinstance(ee_frame, bytes):
            ee_frame = ee_frame.decode()
    n_frames, n_joints = q_all.shape

    robot = _load_robot(robot_name)
    model = robot.model
    data = robot.data

    joint_positions_all, ee_positions = _fk_positions(model, data, q_all, ee_frame)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    n_pts = joint_positions_all.shape[1]
    lines = []
    for _ in range(n_pts - 2):
        (line,) = ax.plot([], [], [], "o-", color="tab:blue", lw=2, ms=3)
        lines.append(line)
    (ee_line,) = ax.plot([], [], [], "o", color="red", ms=5)
    lines.append(ee_line)
    (traj_line,) = ax.plot([], [], [], "--", color="tab:orange", lw=1, alpha=0.6)
    f_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{h5_path.stem} — {robot_name}")

    all_pts_flat = joint_positions_all.reshape(-1, 3)
    margin = 0.1
    half_range = max(
        all_pts_flat[:, 0].ptp(),
        all_pts_flat[:, 1].ptp(),
        all_pts_flat[:, 2].ptp(),
        0.3,
    ) / 2
    mid = all_pts_flat.mean(axis=0)
    ax.set_xlim(mid[0] - half_range - margin, mid[0] + half_range + margin)
    ax.set_ylim(mid[1] - half_range - margin, mid[1] + half_range + margin)
    ax.set_zlim(mid[2] - half_range - margin, mid[2] + half_range + margin)

    interval_s = 1.0 / max(fps, 1)
    frame_idx = 0
    while True:
        if frame_idx >= n_frames:
            if not loop:
                yield mjpeg_chunk(placeholder(640, 480, "Done — trajectory complete"))
                time.sleep(2)
                continue
            frame_idx = 0

        jp = joint_positions_all[frame_idx]
        for i in range(len(lines) - 1):
            if i < len(jp) - 1:
                lines[i].set_data([jp[i][0], jp[i + 1][0]], [jp[i][1], jp[i + 1][1]])
                lines[i].set_3d_properties([jp[i][2], jp[i + 1][2]])
        ee_line.set_data([ee_positions[frame_idx][0]], [ee_positions[frame_idx][1]])
        ee_line.set_3d_properties([ee_positions[frame_idx][2]])
        traj_line.set_data(ee_positions[:frame_idx + 1, 0], ee_positions[:frame_idx + 1, 1])
        traj_line.set_3d_properties(ee_positions[:frame_idx + 1, 2])
        f_text.set_text(f"Frame {frame_idx + 1} / {n_frames}")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = img[:, :, ::-1]
        yield mjpeg_chunk(img, 85)
        frame_idx += 1
        time.sleep(interval_s)
