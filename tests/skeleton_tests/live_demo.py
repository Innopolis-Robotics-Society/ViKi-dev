"""
tests/skeleton_tests/live_demo.py
---------------------------------
Live skeleton overlay from a 2D webcam using the modular detector stack in
LIVE_STREAM mode.

This demo is intentionally standalone:

  * `viki.skeleton.geometry.lift_to_3d` on this branch requires a live
    KinectBackend for color→depth reprojection and cannot be used with a
    webcam. We synthesise Z from MediaPipe's per-landmark relative-z. The
    resulting metres are demo-only, not physically calibrated.

  * `viki.skeleton.stats.SkeletonStats` still expects the old array-shaped
    `Landmarks3D` with a `.source` field. On this branch `Landmarks3D` is a
    `dict[LM, np.ndarray]` with no source, so we keep a tiny local history
    instead.

Usage:
    python tests/skeleton_tests/live_demo.py [--camera 0] [--hand right] [--width 640]

Controls:
    q / Esc — quit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from viki.capture.base import Frame
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.detectors import (
    CompositeLandmarkDetector,
    FusionMode,
    MediaPipeArm,
    MediaPipeHand,
)
from viki.skeleton.hand_angles import HandAngles, compute_hand_angles
from viki.skeleton.models import LM, HandDetection, Landmarks3D, PreparedFrame


# ── constants ────────────────────────────────────────────────────────────────

# Webcam Z synthesis: base depth + lm_z_rel * span.
_WEBCAM_Z_BASE_M = 1.0
_WEBCAM_Z_SPAN_M = 0.3

_DEFAULT_ANALYSIS_LM: tuple[LM, ...] = (
    LM.WRIST,
    LM.THUMB_TIP,
    LM.INDEX_TIP,
    LM.MIDDLE_TIP,
    LM.RING_TIP,
    LM.PINKY_TIP,
)

CHAINS: list[list[LM]] = [
    [LM.WRIST, LM.THUMB_CMC, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP],
    [LM.WRIST, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP],
    [LM.WRIST, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP],
    [LM.WRIST, LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP],
    [LM.WRIST, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP],
    [LM.SHOULDER, LM.ELBOW, LM.WRIST],
]

CHAIN_COLORS = [
    (0, 165, 255),    # thumb  — orange
    (0, 255, 0),      # index  — green
    (255, 255, 0),    # middle — yellow
    (255, 0, 255),    # ring   — magenta
    (0, 255, 255),    # pinky  — cyan
    (255, 100, 100),  # arm    — light blue
]


# ── 3-D lift ────────────────────────────────────────────────────────────────

def _lift_webcam(detection: HandDetection, prepared: PreparedFrame) -> Landmarks3D:
    """Pinhole back-projection with Z synthesised from MediaPipe lm_z_rel."""
    K = prepared.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    points: dict[LM, np.ndarray] = {}
    for i in range(LM.N):
        lm = LM(i)
        uv = detection.points[lm]
        u, v = float(uv[0]), float(uv[1])
        if np.isnan(u) or np.isnan(v):
            points[lm] = np.full(3, np.nan, dtype=np.float32)
            continue
        z_rel = float(detection.lm_z_rel[i])
        Z = _WEBCAM_Z_BASE_M + (z_rel if not np.isnan(z_rel) else 0.0) * _WEBCAM_Z_SPAN_M
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points[lm] = np.array([X, Y, Z], dtype=np.float32)
    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )


# ── drawing ─────────────────────────────────────────────────────────────────

def draw_skeleton(frame_bgr: np.ndarray, detection: HandDetection) -> np.ndarray:
    """Draw 2D skeleton overlay from HandDetection's dict-of-points."""
    img = frame_bgr.copy()

    for chain, color in zip(CHAINS, CHAIN_COLORS):
        pts = [detection.points[lm] for lm in chain]
        if any(np.isnan(p).any() for p in pts):
            continue
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     color, 2, cv2.LINE_AA)

    for uv in detection.points.values():
        u, v = uv[0], uv[1]
        if np.isnan(u) or np.isnan(v):
            continue
        cv2.circle(img, (int(u), int(v)), 4, (255, 255, 255), -1)
        cv2.circle(img, (int(u), int(v)), 4, (0, 0, 0), 1)

    return img


def _draw_angles(img: np.ndarray, angles: HandAngles | None) -> None:
    """Overlay flexion/deviation/roll (deg) on the top-right corner."""
    h, w = img.shape[:2]
    if angles is None or not angles.valid:
        cv2.putText(
            img, "angles: n/a", (w - 170, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
        )
        return
    lines = [
        f"flex   {angles.flexion_deg:+6.1f}",
        f"deviat {angles.deviation_deg:+6.1f}",
        f"roll   {angles.roll_deg:+6.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (w - 170, 20 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
        )


# ── history + post-analysis ─────────────────────────────────────────────────

@dataclass
class _History:
    """Minimal replacement for SkeletonStats — records 3-D points over time."""

    times: list[float] = field(default_factory=list)
    frames: list[dict[LM, np.ndarray]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    angles: list[HandAngles] = field(default_factory=list)
    total_frames: int = 0
    detected_frames: int = 0
    valid_angle_frames: int = 0

    def update(
        self,
        lm3d: Landmarks3D | None,
        confidence: float,
        t_rel: float,
        angles: HandAngles | None = None,
    ) -> None:
        self.total_frames += 1
        if lm3d is None:
            return
        self.detected_frames += 1
        self.times.append(t_rel)
        self.frames.append({k: v.copy() for k, v in lm3d.points.items()})
        self.confidences.append(confidence)
        if angles is None:
            angles = HandAngles(
                flexion_deg=float("nan"),
                deviation_deg=float("nan"),
                roll_deg=float("nan"),
                valid=False,
            )
        self.angles.append(angles)
        if angles.valid:
            self.valid_angle_frames += 1

    def summary(self) -> str:
        det_rate = (
            (self.detected_frames / self.total_frames) * 100
            if self.total_frames
            else 0.0
        )
        avg_conf = float(np.mean(self.confidences)) if self.confidences else 0.0
        ang_rate = (
            (self.valid_angle_frames / self.detected_frames) * 100
            if self.detected_frames
            else 0.0
        )
        return (
            f"Session: {self.total_frames} frames, "
            f"{self.detected_frames} detected ({det_rate:.1f}%), "
            f"avg confidence {avg_conf:.2f}, "
            f"hand-angles valid {self.valid_angle_frames} ({ang_rate:.1f}%)"
        )

    def angles_series(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, angles_deg) with shape (T, 3) — flexion, deviation, roll."""
        T = len(self.angles)
        arr = np.full((T, 3), np.nan, dtype=np.float32)
        for i, a in enumerate(self.angles):
            arr[i] = (a.flexion_deg, a.deviation_deg, a.roll_deg)
        return np.asarray(self.times, dtype=np.float32), arr

    def series(self, lms: list[LM]) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, pos) where pos has shape (T, L, 3); NaN where missing."""
        T = len(self.times)
        pos = np.full((T, len(lms), 3), np.nan, dtype=np.float32)
        nan3 = np.full(3, np.nan, dtype=np.float32)
        for ti, snap in enumerate(self.frames):
            for li, lm in enumerate(lms):
                pos[ti, li] = snap.get(lm, nan3)
        return np.asarray(self.times, dtype=np.float32), pos


def _post_analysis(
    history: _History,
    landmarks: list[LM],
    save_plots: str | None,
) -> None:
    """Position/speed/acceleration + 3-D trace plots after recording."""
    import os

    import matplotlib.pyplot as plt

    if history.detected_frames < 3:
        print("Not enough detected frames for post-analysis (need ≥ 3).")
        return

    t, pos = history.series(landmarks)                 # (T,), (T, L, 3)
    dt = np.diff(t)
    dt = np.where(dt > 0, dt, np.nan)
    vel = np.diff(pos, axis=0) / dt[:, None, None]     # (T-1, L, 3)
    acc = np.diff(vel, axis=0) / dt[1:, None, None]    # (T-2, L, 3)

    axis_names = ("x", "y", "z")
    lm_names = [lm.name for lm in landmarks]
    print(
        f"\nPost-analysis: {history.detected_frames} detected frames over "
        f"{t[-1] - t[0]:.1f}s — {len(landmarks)} landmarks."
    )
    if save_plots is not None:
        os.makedirs(save_plots, exist_ok=True)
        print(f"Saving plots to {save_plots}/\n")

    def _make(data: np.ndarray, t_axis: np.ndarray, unit: str, title: str):
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for ax_i, ax in enumerate(axes):
            for li, name in enumerate(lm_names):
                ax.plot(t_axis, data[:, li, ax_i], label=name)
            ax.set_ylabel(f"{axis_names[ax_i]} ({unit})")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time (s)")
        axes[0].set_title(title)
        axes[0].legend(loc="upper right", fontsize=8, ncol=len(lm_names))
        fig.tight_layout()
        return fig

    figs = [
        (_make(pos, t, "m", "Position over time"), "position.png"),
        (_make(vel, t[1:], "m/s", "Speed over time"), "speed.png"),
        (_make(acc, t[2:], "m/s²", "Acceleration over time"), "acceleration.png"),
    ]

    # Hand angles over time — separate figure because it's a single series set.
    ta, ang = history.angles_series()  # ang: (T, 3) — flexion, deviation, roll
    if history.valid_angle_frames >= 3:
        fig_ang, ax_ang = plt.subplots(figsize=(10, 4))
        for j, name in enumerate(("flexion", "deviation", "roll")):
            ax_ang.plot(ta, ang[:, j], label=name)
        ax_ang.set_xlabel("time (s)")
        ax_ang.set_ylabel("angle (deg)")
        ax_ang.set_title("Hand angles over time (forearm-local)")
        ax_ang.grid(True, alpha=0.3)
        ax_ang.legend()
        fig_ang.tight_layout()
        figs.append((fig_ang, "hand_angles.png"))
    else:
        print(
            f"  hand-angles panel skipped: only {history.valid_angle_frames} "
            "valid frames (need ≥ 3)"
        )

    fig3d = plt.figure(figsize=(6, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    for li, name in enumerate(lm_names):
        ax3d.plot(pos[:, li, 0], pos[:, li, 1], pos[:, li, 2], label=name, alpha=0.8)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3-D landmark traces")
    ax3d.legend(fontsize=8)
    figs.append((fig3d, "trace_3d.png"))

    if save_plots is not None:
        for fig, filename in figs:
            path = os.path.join(save_plots, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  saved {filename}")
            plt.close(fig)
    else:
        plt.show()


# ── main ────────────────────────────────────────────────────────────────────

def _parse_landmarks(s: str) -> list[LM]:
    out: list[LM] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if not (0 <= i < LM.N):
            raise ValueError(f"landmark index {i} out of range 0..{LM.N - 1}")
        out.append(LM(i))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--landmarks",
        default=",".join(str(int(lm)) for lm in _DEFAULT_ANALYSIS_LM),
        help=(
            "Comma-separated landmark indices for post-analysis plots "
            f"(default: wrist + fingertips = {[lm.name for lm in _DEFAULT_ANALYSIS_LM]})"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip post-analysis plots after recording ends.",
    )
    parser.add_argument(
        "--save-plots",
        metavar="DIR",
        default=None,
        help=(
            "Save all post-analysis plots as PNGs to this folder "
            "(created if it does not exist). Skips interactive display."
        ),
    )
    args = parser.parse_args()

    try:
        analysis_landmarks = _parse_landmarks(args.landmarks)
    except ValueError as exc:
        parser.error(f"--landmarks: {exc}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    detector = CompositeLandmarkDetector(
        detectors=[
            MediaPipeArm(hand=args.hand, mode="live"),
            MediaPipeHand(hand=args.hand, mode="live"),
        ],
        mode=FusionMode.ANY,
    )

    history = _History()
    cache = UndistortCache()

    ret, bgr = cap.read()
    if not ret:
        print("Cannot read from camera")
        cap.release()
        return

    h, w = bgr.shape[:2]
    if w > args.width:
        scale = args.width / w
        proc_w = args.width
        proc_h = int(h * scale)
    else:
        scale = 1.0
        proc_w, proc_h = w, h

    K = np.array(
        [[proc_w * 0.8, 0, proc_w / 2], [0, proc_w * 0.8, proc_h / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    dist = np.zeros(5, dtype=np.float32)
    # Non-zero depth: prepare_frame turns 0 into NaN. 700 mm → 0.7 m after scaling.
    depth_fake = np.full((proc_h, proc_w), 700, dtype=np.uint16)

    frame_idx = 0
    t0 = time.perf_counter()
    fps_display = 0.0

    print("Running — press q or Esc to quit")

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            frame_idx += 1

            proc = (
                cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                if scale != 1.0
                else bgr
            )

            frame = Frame(
                color=proc,
                depth=depth_fake,
                timestamp_us=frame_idx * 33333,
                device_id="webcam",
            )
            prepared = prepare_frame(frame, K, dist, cache)
            detection = detector.detect(prepared)

            t_rel = time.perf_counter() - t0

            angles: HandAngles | None = None
            if detection is not None:
                lm3d = _lift_webcam(detection, prepared)
                angles = compute_hand_angles(lm3d.points)
                print("ANG:",angles)
                history.update(
                    lm3d,
                    confidence=detection.confidence,
                    t_rel=t_rel,
                    angles=angles,
                )
            else:
                history.update(None, confidence=0.0, t_rel=t_rel)

            if t_rel >= 1.0:
                fps_display = frame_idx / t_rel

            display = proc.copy()
            if detection is not None:
                display = draw_skeleton(display, detection)
                cv2.putText(
                    display,
                    f"conf={detection.confidence:.2f}",
                    (10, proc_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                _draw_angles(display, angles)
            else:
                cv2.putText(
                    display,
                    "no hand",
                    (10, proc_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                f"FPS {fps_display:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ViKi live demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

    print(history.summary())

    if not args.no_plots:
        _post_analysis(history, analysis_landmarks, args.save_plots)


if __name__ == "__main__":
    main()
