"""
viki.skeleton.stats
-------------------
Accumulates per-session and rolling-window statistics about the skeleton pipeline.

Usage
-----
    stats = SkeletonStats(window=150)      # ~5 s at 30 fps

    # in capture loop:
    frame = pipeline.process(group)
    stats.update(frame)                    # accepts None (missed frame)

    # anywhere:
    data = stats.summary()                 # JSON-ready dict
    stats.reset()                          # new recording session
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from viki.skeleton.models import LM, LandmarkSource, Landmarks3D, SkeletonFrame

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

    # ── Public API ────────────────────────────────────────────────────────────

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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._total_frames    = 0
        self._detected_frames = 0

        self._conf_window: deque[float]       = deque(maxlen=self._window)
        self._pos_window:  deque[np.ndarray]  = deque(maxlen=self._window)  # each (23,3)

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

        # Stack rolling positions: (W, 23, 3) — may be shorter than window
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
