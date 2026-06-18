import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from viki.capture.base import Frame
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.hand_detector import HandDetector
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.models import LM
from viki.skeleton.stats import SkeletonStats

MISSED_DIR = Path("missed_frames")
MISSED_DIR.mkdir(exist_ok=True)
MAX_SAVED = 50
missed_frames: list[np.ndarray] = []

cap = cv2.VideoCapture("/home/tomatocoder/Desktop/ViKi-dev/skeleton_tests/move_open.mp4")
detector = HandDetector(hand="left", mirrored=True, mode="video")
stats = SkeletonStats(window=150)
cache = UndistortCache()

ret, bgr = cap.read()
if not ret:
    print("Cannot open video")
    cap.release()
    exit()

h, w = bgr.shape[:2]
K = np.array([[w * 0.8, 0, w / 2],
              [0, w * 0.8, h / 2],
              [0, 0, 1]], dtype=np.float32)
dist = np.zeros(5, dtype=np.float32)
depth_fake = np.full((h, w), 700, dtype=np.uint16)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ── Speed knobs ───────────────────────────────────────────────────────────────
FRAME_SKIP = 1    # process every Nth frame (1 = all, 2 = half, 3 = third …)
MAX_WIDTH  = 640  # downscale to this width before detection (None = no resize)
# ─────────────────────────────────────────────────────────────────────────────

# Precompute scaled intrinsics and depth if resizing
if MAX_WIDTH and w > MAX_WIDTH:
    scale = MAX_WIDTH / w
    proc_w = MAX_WIDTH
    proc_h = int(h * scale)
    K_proc = K.copy()
    K_proc[0, 0] *= scale   # fx
    K_proc[1, 1] *= scale   # fy
    K_proc[0, 2] *= scale   # cx
    K_proc[1, 2] *= scale   # cy
else:
    scale = 1.0
    proc_w, proc_h = w, h
    K_proc = K

depth_proc = np.full((proc_h, proc_w), 700, dtype=np.uint16)

frame_idx = 0
detected = 0
missed_count = 0

# (N, 23, 3) — all landmarks per detected frame
all_points: list[np.ndarray] = []


def _post_analysis(
    stats: SkeletonStats,
    landmarks: list[int],
    save_anim: str | None,
    plots_dir: str | None,
) -> None:
    """Display position/speed/acceleration plots and 3-D skeleton viz after recording."""
    import os
    import matplotlib.pyplot as plt

    pos, t, _ = stats.position_over_time(landmarks)
    if pos.shape[0] < 3:
        print("Not enough detected frames for post-analysis (need ≥ 3).")
        return

    print(f"\nPost-analysis: {pos.shape[0]} detected frames over {t[-1]:.1f}s "
          f"— {len(landmarks)} landmarks selected.")

    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Saving plots to {plots_dir}/\n")
    else:
        print("Close each plot window to advance to the next one.\n")

    for fig, title, filename in [
        (stats.plot_position(landmarks, axes="xyz"), "Position over time",    "position.png"),
        (stats.plot_speed(landmarks),                "Speed over time",       "speed.png"),
        (stats.plot_acceleration(landmarks),         "Acceleration over time","acceleration.png"),
        (stats.plot_3d_trace(landmarks),             "3-D landmark traces",  "trace_3d.png"),
    ]:
        if plots_dir is not None:
            path = os.path.join(plots_dir, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  saved {filename}")
            plt.close(fig)
        else:
            plt.show()

    print("3-D animation — close the window to exit.")
    anim = stats.animate_3d(landmarks, fps=30.0, save_path=save_anim)
    if anim is not None:
        plt.show()
    elif save_anim:
        print(f"Animation saved → {save_anim}")

while True:
    ret, bgr = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    if scale != 1.0:
        bgr = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

    frame = Frame(
        color=bgr,
        depth=depth_proc,
        timestamp_us=frame_idx * 33333,
        device_id="phone",
    )
    
    prepared = prepare_frame(frame, K_proc, dist, cache)
    detection = detector.detect(prepared)

    if detection is None:
        print(f"frame {frame_idx:4d}: not detected")
        missed_count += 1
        # Reservoir sampling (Algorithm R): uniform random sample of MAX_SAVED
        if missed_count <= MAX_SAVED:
            missed_frames.append(bgr.copy())
        else:
            j = random.randint(0, missed_count - 1)
            if j < MAX_SAVED:
                missed_frames[j] = bgr.copy()
        continue

    detected += 1
    lm3d = lift_to_3d(detection, prepared)
    all_points.append(lm3d.points.copy())  # (23, 3)

    stats.update(lm3d)

    wrist = lm3d.points[LM.WRIST]
    n_depth   = sum(s == "depth"   for s in lm3d.source)
    n_mp_z    = sum(s == "mp_z"    for s in lm3d.source)
    n_missing = sum(s == "missing" for s in lm3d.source)
    print(f"frame {frame_idx:4d}: wrist={wrist.round(3)}  D={n_depth} MP={n_mp_z} M={n_missing}")

cap.release()
detector.close()
print(f"\nDone: {detected}/{frame_idx} frames detected")


# ── 3D visualisation ──────────────────────────────────────────────────────────

_post_analysis(stats, landmarks=[0, 16, 20], save_anim="skeleton_animation", plots_dir="post_analysis_plots")


# if not all_points:
#     print("No detections to visualise.")
#     exit()

# pts = np.array(all_points)  # (N, 23, 3)

# # Finger chains: thumb, index, middle, ring, pinky + arm chain
# CHAINS = [
#     [LM.WRIST, LM.THUMB_CMC,  LM.THUMB_MCP,  LM.THUMB_IP,   LM.THUMB_TIP],
#     [LM.WRIST, LM.INDEX_MCP,  LM.INDEX_PIP,  LM.INDEX_DIP,  LM.INDEX_TIP],
#     [LM.WRIST, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP],
#     [LM.WRIST, LM.RING_MCP,   LM.RING_PIP,   LM.RING_DIP,   LM.RING_TIP],
#     [LM.WRIST, LM.PINKY_MCP,  LM.PINKY_PIP,  LM.PINKY_DIP,  LM.PINKY_TIP],
#     [LM.SHOULDER, LM.ELBOW, LM.WRIST],
# ]

# fig = plt.figure(figsize=(12, 6))

# # Left: last detected frame skeleton
# ax1 = fig.add_subplot(121, projection="3d")
# ax1.set_title("Last frame skeleton")
# last = pts[-1]  # (23, 3)
# for chain in CHAINS:
#     chain_pts = last[chain]
#     # skip chains with nan
#     if np.isnan(chain_pts).any():
#         continue
#     ax1.plot(chain_pts[:, 0], chain_pts[:, 2], -chain_pts[:, 1],
#              "o-", linewidth=2, markersize=4)
# ax1.set_xlabel("X (m)")
# ax1.set_ylabel("Z (m)")
# ax1.set_zlabel("-Y (m)")

# # Right: wrist trace with time-based heatmap
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.set_title("Wrist trace — colour = time (cool→warm)")

# cmap = plt.get_cmap("plasma")
# N = len(pts)

# # Draw all landmark traces faintly
# for lm_idx in range(LM.N):
#     if lm_idx == LM.WRIST:
#         continue
#     xs = pts[:, lm_idx, 0]
#     ys = pts[:, lm_idx, 2]
#     zs = -pts[:, lm_idx, 1]
#     valid = ~np.isnan(xs)
#     if not valid.any():
#         continue
#     ax2.plot(xs[valid], ys[valid], zs[valid],
#              color="silver", linewidth=0.4, alpha=0.2)

# # Wrist trace: colour segments by time index
# wx = pts[:, LM.WRIST, 0]
# wy = pts[:, LM.WRIST, 2]
# wz = -pts[:, LM.WRIST, 1]
# valid = ~np.isnan(wx)
# idx = np.where(valid)[0]

# for k in range(len(idx) - 1):
#     i, j = idx[k], idx[k + 1]
#     t = k / max(len(idx) - 2, 1)   # 0 → 1
#     ax2.plot([wx[i], wx[j]], [wy[i], wy[j]], [wz[i], wz[j]],
#              color=cmap(t), linewidth=2)

# # Start (blue) and end (red) markers
# if len(idx) >= 1:
#     ax2.scatter([wx[idx[0]]],  [wy[idx[0]]],  [wz[idx[0]]],
#                 color=cmap(0.0), s=60, zorder=5, label="start")
#     ax2.scatter([wx[idx[-1]]], [wy[idx[-1]]], [wz[idx[-1]]],
#                 color=cmap(1.0), s=60, marker="*", zorder=5, label="end")

# # Colourbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N))
# sm.set_array([])
# plt.colorbar(sm, ax=ax2, label="frame index", shrink=0.6, pad=0.1)

# ax2.set_xlabel("X (m)")
# ax2.set_ylabel("Z (m)")
# ax2.set_zlabel("-Y (m)")
# ax2.legend()

# plt.tight_layout()
# plt.savefig("skeleton_trace.png", dpi=150)
# print("Saved skeleton_trace.png")
# plt.show()

print(stats.summary())