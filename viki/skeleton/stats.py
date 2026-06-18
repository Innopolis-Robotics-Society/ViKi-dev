"""
viki.skeleton.stats
-------------------
Accumulates per-session and rolling-window statistics about the skeleton pipeline.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional, Sequence, Union

import numpy as np

from viki.skeleton.models import LM, LandmarkSource, Landmarks3D, SkeletonFrame

# Skeleton bone connections for 3-D plots.
_BONES: list[tuple[int, int]] = [
    (LM.WRIST, LM.THUMB_CMC),  (LM.THUMB_CMC, LM.THUMB_MCP),
    (LM.THUMB_MCP, LM.THUMB_IP),(LM.THUMB_IP,  LM.THUMB_TIP),
    (LM.WRIST, LM.INDEX_MCP),  (LM.INDEX_MCP,  LM.INDEX_PIP),
    (LM.INDEX_PIP, LM.INDEX_DIP),(LM.INDEX_DIP, LM.INDEX_TIP),
    (LM.WRIST, LM.MIDDLE_MCP), (LM.MIDDLE_MCP, LM.MIDDLE_PIP),
    (LM.MIDDLE_PIP,LM.MIDDLE_DIP),(LM.MIDDLE_DIP,LM.MIDDLE_TIP),
    (LM.WRIST, LM.RING_MCP),   (LM.RING_MCP,   LM.RING_PIP),
    (LM.RING_PIP,  LM.RING_DIP),(LM.RING_DIP,   LM.RING_TIP),
    (LM.WRIST, LM.PINKY_MCP),  (LM.PINKY_MCP,  LM.PINKY_PIP),
    (LM.PINKY_PIP, LM.PINKY_DIP),(LM.PINKY_DIP, LM.PINKY_TIP),
    (LM.INDEX_MCP, LM.MIDDLE_MCP),(LM.MIDDLE_MCP, LM.RING_MCP),
    (LM.RING_MCP,  LM.PINKY_MCP),
    (LM.SHOULDER,  LM.ELBOW),  (LM.ELBOW, LM.WRIST),
]

LandmarkSpec = Optional[Union[Sequence[int], int]]

# Human-readable names indexed by LM constants
_LM_NAMES: list[str] = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP",  "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP",  "RING_PIP",  "RING_DIP",  "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    "ELBOW",
    "SHOULDER",
]


def _resolve_landmarks(landmarks: LandmarkSpec) -> list[int]:
    """Normalise a landmark selector to a sorted list of valid int indices."""
    if landmarks is None:
        return list(range(LM.N))
    if isinstance(landmarks, int):
        landmarks = [landmarks]
    result = [int(lm) for lm in landmarks]
    if any(i < 0 or i >= LM.N for i in result):
        raise ValueError(f"Landmark index out of range [0, {LM.N-1}]")
    return result


class SkeletonStats:
    """
    Thread-safe statistics accumulator for the skeleton pipeline.

    Parameters
    ----------
    window : int
        Rolling window size (number of detected frames) used for jitter,
        variance, and confidence statistics. Default 150 (~5 s at 30 fps).
    """

    def __init__(self, window: int = 150) -> None:
        self._window  = window
        self._lock    = threading.Lock()
        self._start_t = time.monotonic()
        self._reset_state()

    def update(
        self,
        frame: Optional[SkeletonFrame | Landmarks3D],
        confidence: Optional[float] = None,
    ) -> None:
        """
        Push one pipeline output into the accumulator.

        Accepts either:
          - SkeletonFrame  — full two-camera fusion output
          - Landmarks3D    — single-camera output; pass scalar confidence separately
          - None           — missed frame (no detection)

        Must be called once per capture cycle to keep frame counts accurate.
        """
        with self._lock:
            self._total_frames += 1

            if frame is None:
                return

            self._detected_frames += 1

            # Normalise to (points, source, confidence_value)
            if isinstance(frame, SkeletonFrame):
                pts        = frame.landmarks          # (23, 3)
                source     = frame.source             # (23,) LandmarkSource
                conf_value = float(np.nanmean(frame.confidence))
            else:  # Landmarks3D
                pts        = frame.points             # (23, 3)
                source     = frame.source             # (23,) LandmarkSource
                conf_value = float(confidence) if confidence is not None else 0.0

            self._conf_window.append(conf_value)
            self._pos_window.append(pts.copy())
            self._pos_history.append(pts.copy())
            self._ts_history.append(time.monotonic() - self._start_t)

            for i in range(LM.N):
                src = source[i]
                if src == LandmarkSource.DEPTH:
                    self._src_depth[i] += 1
                elif src == LandmarkSource.MP_Z:
                    self._src_mpz[i] += 1
                else:
                    self._src_missing[i] += 1

    def summary(self) -> dict:
        """
        Compute and return a JSON-serialisable statistics snapshot.

        All rolling statistics are computed over the current window.
        Global counters cover the full session since creation or last reset().
        """
        with self._lock:
            return self._compute()

    def reset(self) -> None:
        """Reset all counters and windows. Keeps window size."""
        with self._lock:
            self._start_t = time.monotonic()
            self._reset_state()
            

    def position_over_time(
        self, landmarks: LandmarkSpec = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Return per-landmark 3-D positions for every detected frame.

        Returns
        -------
        pos : (T, L, 3) float32  — XYZ in metres; NaN where missing
        t   : (T,)      float32  — seconds since session start
        ids : list[int]          — landmark indices matching axis-1 of pos
        """
        with self._lock:
            ids = _resolve_landmarks(landmarks)
            if not self._pos_history:
                empty = np.empty((0, len(ids), 3), dtype=np.float32)
                return empty, np.empty((0,), dtype=np.float32), ids
            pos = np.stack(self._pos_history, axis=0)[:, ids, :].astype(np.float32)
            t   = np.array(self._ts_history, dtype=np.float32)
        return pos, t, ids

    def speed_over_time(
        self, landmarks: LandmarkSpec = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Return scalar speed (m/s) for each landmark between consecutive frames.

        Returns
        -------
        speed : (T-1, L) float32
        t     : (T-1,)   float32  — midpoint timestamps
        ids   : list[int]
        """
        pos, t, ids = self.position_over_time(landmarks)
        if pos.shape[0] < 2:
            return np.empty((0, len(ids)), dtype=np.float32), np.empty((0,), dtype=np.float32), ids
        dt    = np.diff(t)[:, None]                        # (T-1, 1)
        dp    = np.diff(pos, axis=0)                       # (T-1, L, 3)
        speed = np.linalg.norm(dp, axis=-1) / np.where(dt > 0, dt, np.nan)  # (T-1, L)
        t_mid = (t[:-1] + t[1:]) / 2
        return speed.astype(np.float32), t_mid, ids

    def acceleration_over_time(
        self, landmarks: LandmarkSpec = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Return scalar acceleration (m/s²) for each landmark.

        Returns
        -------
        accel : (T-2, L) float32
        t     : (T-2,)   float32  — midpoint timestamps
        ids   : list[int]
        """
        speed, t_mid, ids = self.speed_over_time(landmarks)
        if speed.shape[0] < 2:
            return np.empty((0, len(ids)), dtype=np.float32), np.empty((0,), dtype=np.float32), ids
        dt    = np.diff(t_mid)[:, None]
        accel = np.diff(speed, axis=0) / np.where(dt > 0, dt, np.nan)
        t_acc = (t_mid[:-1] + t_mid[1:]) / 2
        return accel.astype(np.float32), t_acc, ids

    # ── 2-D plots ──────────────────────────────────────────────────────────────

    def plot_position(
        self,
        landmarks: LandmarkSpec = None,
        axes: str = "xyz",
    ):
        """
        Plot XYZ position vs time for each selected landmark.

        Parameters
        ----------
        landmarks : None | int | list[int]
            Which landmarks to draw. None = all 23.
        axes : str
            Any combination of 'x', 'y', 'z' — which coordinate axes to show.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        pos, t, ids = self.position_over_time(landmarks)
        axis_map  = {"x": 0, "y": 1, "z": 2}
        ax_indices = [axis_map[a] for a in axes.lower() if a in axis_map]
        colors     = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
        axis_names = {0: "X", 1: "Y", 2: "Z"}

        n = len(ids)
        fig, axs = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True, squeeze=False)
        fig.suptitle("Position over time (m)")

        for row, (lm_idx, ax) in enumerate(zip(ids, axs[:, 0])):
            for ai in ax_indices:
                series = pos[:, row, ai]
                label  = axis_names[ai]
                ax.plot(t, series, label=label, color=colors[list(axis_map)[ai]], linewidth=0.9)
            ax.set_ylabel(_LM_NAMES[lm_idx], fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, linewidth=0.4)

        axs[-1, 0].set_xlabel("Time (s)")
        fig.tight_layout()
        return fig

    def plot_speed(self, landmarks: LandmarkSpec = None):
        """
        Plot speed (m/s) vs time for each selected landmark.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        speed, t, ids = self.speed_over_time(landmarks)
        n = len(ids)
        fig, axs = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True, squeeze=False)
        fig.suptitle("Speed over time (m/s)")

        for row, (lm_idx, ax) in enumerate(zip(ids, axs[:, 0])):
            ax.plot(t, speed[:, row], linewidth=0.9, color="tab:orange")
            ax.set_ylabel(_LM_NAMES[lm_idx], fontsize=8)
            ax.grid(True, linewidth=0.4)

        axs[-1, 0].set_xlabel("Time (s)")
        fig.tight_layout()
        return fig

    def plot_acceleration(self, landmarks: LandmarkSpec = None):
        """
        Plot acceleration (m/s²) vs time for each selected landmark.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        accel, t, ids = self.acceleration_over_time(landmarks)
        n = len(ids)
        fig, axs = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True, squeeze=False)
        fig.suptitle("Acceleration over time (m/s²)")

        for row, (lm_idx, ax) in enumerate(zip(ids, axs[:, 0])):
            ax.plot(t, accel[:, row], linewidth=0.9, color="tab:purple")
            ax.set_ylabel(_LM_NAMES[lm_idx], fontsize=8)
            ax.grid(True, linewidth=0.4)

        axs[-1, 0].set_xlabel("Time (s)")
        fig.tight_layout()
        return fig

    def plot_3d_trace(self):
        """
        Static 3-D trace plot for all 23 landmarks.

        Each landmark's full trajectory is drawn as a coloured line.
        Skeleton bones are drawn at every recorded frame (low alpha) and
        at the last frame (more prominent) so the hand shape is visible.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        pos, _, ids = self.position_over_time(None)
        id_to_row = {lm: r for r, lm in enumerate(ids)}

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")

        cmap   = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(ids))]

        for row, lm_idx in enumerate(ids):
            xyz   = pos[:, row, :]
            valid = ~np.isnan(xyz[:, 0])
            if valid.sum() < 2:
                continue
            x, y, z = xyz[valid, 0], xyz[valid, 1], xyz[valid, 2]
            ax.plot(x, y, z, linewidth=0.8, color=colors[row], alpha=0.7,
                    label=_LM_NAMES[lm_idx])
            ax.scatter(x[-1], y[-1], z[-1], s=20, color=colors[row], zorder=5)

        T      = pos.shape[0]
        stride = max(1, T // 60)
        for t_idx in range(0, T, stride):
            frame_pts = pos[t_idx]
            for a, b in _BONES:
                pa, pb = frame_pts[id_to_row[a]], frame_pts[id_to_row[b]]
                if np.isnan(pa).any() or np.isnan(pb).any():
                    continue
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                        color="gray", linewidth=0.6, alpha=0.15)

        # Last frame bones drawn prominently
        last_valid_idx = np.where(~np.isnan(pos[:, 0, 0]))[0]
        if len(last_valid_idx):
            last = pos[last_valid_idx[-1]]
            for a, b in _BONES:
                pa, pb = last[id_to_row[a]], last[id_to_row[b]]
                if not (np.isnan(pa).any() or np.isnan(pb).any()):
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                            "k-", linewidth=1.4, alpha=0.7)

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title("3-D landmark traces")
        ax.view_init(elev=90, azim=-90)
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        fig.tight_layout()
        return fig

    def animate_3d(
        self,
        fps: float = 30.0,
        save_path: Optional[str] = None,
    ):
        """
        Animated 3-D skeleton over recorded frames (all 23 landmarks).

        Parameters
        ----------
        fps : float
            Playback frame rate (used for interval and saved video fps).
        save_path : str | None
            If given, save as MP4 (requires ffmpeg) or GIF. Otherwise return
            the FuncAnimation so you can call plt.show() yourself.

        Returns
        -------
        matplotlib.animation.FuncAnimation  (if save_path is None)
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        pos, t, ids = self.position_over_time(None)
        T = pos.shape[0]
        if T == 0:
            raise ValueError("No recorded frames to animate.")

        id_to_row = {lm: r for r, lm in enumerate(ids)}
        cmap      = plt.get_cmap("tab20")
        colors    = [cmap(i % 20) for i in range(len(ids))]

        # Pre-compute axis limits from valid data
        valid_pos = pos[~np.isnan(pos[:, :, 0].any(axis=1))]
        all_xyz   = pos.reshape(-1, 3)
        all_xyz   = all_xyz[~np.isnan(all_xyz[:, 0])]
        pad       = 0.05
        xlim = (all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad)
        ylim = (all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad)
        zlim = (all_xyz[:, 2].min() - pad, all_xyz[:, 2].max() + pad)

        fig = plt.figure(figsize=(9, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.view_init(elev=90, azim=90)

        # Scatter artists for landmark dots
        scatters = [
            ax.plot([], [], [], "o", markersize=4, color=colors[r],
                    label=_LM_NAMES[lm])[0]
            for r, lm in enumerate(ids)
        ]
        # Line artists for bones
        bone_lines = []
        for a, b in _BONES:
            line, = ax.plot([], [], [], "k-", linewidth=1.0, alpha=0.6)
            bone_lines.append((line, id_to_row[a], id_to_row[b]))

        title = ax.set_title("")

        def _init():
            for sc in scatters:
                sc.set_data([], [])
                sc.set_3d_properties([])
            for line, _, _ in bone_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [sc for sc in scatters] + [l for l, _, _ in bone_lines]

        def _update(frame_idx: int):
            pts = pos[frame_idx]  # (L, 3)
            for r, sc in enumerate(scatters):
                p = pts[r]
                if np.isnan(p[0]):
                    sc.set_data([], [])
                    sc.set_3d_properties([])
                else:
                    sc.set_data([p[0]], [p[1]])
                    sc.set_3d_properties([p[2]])
            for line, ra, rb in bone_lines:
                pa, pb = pts[ra], pts[rb]
                if np.isnan(pa[0]) or np.isnan(pb[0]):
                    line.set_data([], [])
                    line.set_3d_properties([])
                else:
                    line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                    line.set_3d_properties([pa[2], pb[2]])
            title.set_text(f"t = {t[frame_idx]:.2f} s  (frame {frame_idx+1}/{T})")
            return [sc for sc in scatters] + [l for l, _, _ in bone_lines] + [title]

        interval_ms = 1000.0 / fps
        anim = FuncAnimation(fig, _update, frames=T, init_func=_init,
                             interval=interval_ms, blit=False)

        if save_path is not None:
            import os
            from matplotlib.animation import FFMpegWriter
            ext = os.path.splitext(save_path)[1].lower()
            if not ext:
                save_path += ".gif"
                ext = ".gif"
            video_exts = {".mp4", ".avi", ".mov"}
            if ext in video_exts and not FFMpegWriter.isAvailable():
                new_path = save_path[: -len(ext)] + ".gif"
                print(f"ffmpeg not found — saving as GIF instead: {new_path}")
                save_path = new_path
                ext = ".gif"
            if ext in video_exts:
                anim.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])
            else:
                anim.save(save_path, fps=fps)
            plt.close(fig)
            return None
        
        if len(ids) <= 10:
            ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        return anim

    def _reset_state(self) -> None:
        self._total_frames    = 0
        self._detected_frames = 0

        self._conf_window: deque[float]       = deque(maxlen=self._window)
        self._pos_window:  deque[np.ndarray]  = deque(maxlen=self._window)  # each (23,3)

        # Full session history (unbounded) — used by analytics / viz methods
        self._pos_history: list[np.ndarray] = []   # each (23, 3)
        self._ts_history:  list[float]       = []   # monotonic seconds

        # Global source counts per landmark (not windowed — full session)
        self._src_depth   = np.zeros(LM.N, dtype=np.int64)
        self._src_mpz     = np.zeros(LM.N, dtype=np.int64)
        self._src_missing = np.zeros(LM.N, dtype=np.int64)

    def _compute(self) -> dict:
        total    = self._total_frames
        detected = self._detected_frames
        elapsed  = time.monotonic() - self._start_t

        out: dict = {
            "frame_count":      total,
            "detected_count":   detected,
            "detection_rate":   round(detected / total, 4) if total else 0.0,
            "session_duration_s": round(elapsed, 2),
            "confidence":       self._conf_stats(),
            "landmarks":        self._landmark_stats(),
        }
        return out

    def _conf_stats(self) -> dict:
        if not self._conf_window:
            return {}
        arr = np.array(self._conf_window, dtype=np.float32)
        return {
            "mean":   _f(np.mean(arr)),
            "median": _f(np.median(arr)),
            "std":    _f(np.std(arr)),
            "min":    _f(np.min(arr)),
            "max":    _f(np.max(arr)),
        }

    def _landmark_stats(self) -> list[dict]:
        n_det = self._detected_frames
        landmarks = []

        # Stack rolling positions: (W, 23, 3)
        if len(self._pos_window) >= 2:
            pos_arr = np.stack(self._pos_window, axis=0)  # (W, 23, 3)
        else:
            pos_arr = None

        for i in range(LM.N):
            total_src = self._src_depth[i] + self._src_mpz[i] + self._src_missing[i]

            entry: dict = {
                "index": i,
                "name":  _LM_NAMES[i],
                "source": {
                    "depth":   _f(self._src_depth[i]   / total_src) if total_src else 0.0,
                    "mp_z":    _f(self._src_mpz[i]     / total_src) if total_src else 0.0,
                    "missing": _f(self._src_missing[i] / total_src) if total_src else 0.0,
                },
            }

            if pos_arr is not None:
                lm_pts = pos_arr[:, i, :]  # (W, 3)
                valid  = ~np.isnan(lm_pts[:, 0])

                if valid.sum() >= 2:
                    lm_valid = lm_pts[valid]  # (V, 3)

                    # Position stats (metres)
                    entry["position"] = {
                        "mean": _xyz(np.mean(lm_valid, axis=0)),
                        "std":  _xyz(np.std(lm_valid,  axis=0)),
                        "min":  _xyz(np.min(lm_valid,  axis=0)),
                        "max":  _xyz(np.max(lm_valid,  axis=0)),
                    }

                    # Jitter: frame-to-frame displacement in mm.
                    # Zero-displacement diffs are stale results (LIVE mode returns
                    # cached result while callback hasn't fired yet) — exclude them
                    # so median/mean reflect actual motion, not polling artifacts.
                    diffs = np.linalg.norm(np.diff(lm_pts[valid], axis=0), axis=1) * 1000
                    real_diffs = diffs[diffs > 1e-6]  # exclude exact duplicates
                    stale_rate = _f(1.0 - len(real_diffs) / len(diffs)) if len(diffs) else 0.0
                    if len(real_diffs) >= 1:
                        entry["jitter_mm"] = {
                            "mean":       _f(np.mean(real_diffs)),
                            "median":     _f(np.median(real_diffs)),
                            "std":        _f(np.std(real_diffs)),
                            "p95":        _f(np.percentile(real_diffs, 95)),
                            "max":        _f(np.max(real_diffs)),
                            "stale_rate": stale_rate,  # fraction of duplicate frames
                        }

            landmarks.append(entry)

        return landmarks


# ── Helpers ───────────────────────────────────────────────────────────────────

def pretty_print(summary: dict) -> None:
    """Print a SkeletonStats.summary() dict in a readable format."""
    s = summary
    print("─" * 52)
    print(f"  Frames   {s['detected_count']:>5} / {s['frame_count']:<5}  "
          f"({s['detection_rate']*100:.1f}% detected)  "
          f"{s['session_duration_s']:.1f}s")

    c = s.get("confidence", {})
    if c:
        print(f"  Conf     mean={c['mean']:.3f}  "
              f"median={c['median']:.3f}  "
              f"std={c['std']:.3f}  "
              f"[{c['min']:.2f}–{c['max']:.2f}]")

    print("─" * 52)
    print(f"  {'Landmark':<14} {'miss%':>5}  {'jit med':>7}  {'jit p95':>7}  {'jit max':>8}")
    print("─" * 52)
    for lm in s.get("landmarks", []):
        miss  = lm["source"]["missing"] * 100
        jit   = lm.get("jitter_mm", {})
        med   = f"{jit['median']:>7.1f}" if jit else "      -"
        p95   = f"{jit['p95']:>7.1f}"   if jit else "      -"
        jmax  = f"{jit['max']:>8.1f}"   if jit else "       -"
        flag  = " !" if miss > 5 else ""
        print(f"  {lm['name']:<14} {miss:>4.1f}%  {med}  {p95}  {jmax}{flag}")
    print("─" * 52)


def _f(x) -> float: 
    """Round float to 4 decimal places for clean JSON output."""
    return round(float(x), 4)


def _xyz(arr: np.ndarray) -> list[float]:
    """(3,) → [x, y, z] rounded."""
    return [_f(v) for v in arr]
