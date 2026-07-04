"""Convert ViKi-dev 23-landmark skeleton JSON to experiment sample npz.

The ViKi-dev skeleton layout is:
  0..20: MediaPipe hand landmarks
  21: elbow
  22: shoulder

The retargeting experiments expect MediaPipe-style arrays:
  body: (T, 33, 3)
  right_hand: (T, 21, 3)
  left_hand: (T, 21, 3)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15


def interpolate_nans(points: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs over time for each landmark coordinate."""
    out = np.asarray(points, dtype=np.float64).copy()
    frames = np.arange(out.shape[0], dtype=np.float64)
    for landmark in range(out.shape[1]):
        for dim in range(out.shape[2]):
            series = out[:, landmark, dim]
            valid = np.isfinite(series)
            if valid.all():
                continue
            if not valid.any():
                continue
            if valid.sum() == 1:
                series[~valid] = series[valid][0]
            else:
                series[~valid] = np.interp(frames[~valid], frames[valid], series[valid])
            out[:, landmark, dim] = series
    return out


def estimate_fps(timestamps_us: np.ndarray) -> float:
    if len(timestamps_us) < 2:
        return 30.0
    dt = np.diff(timestamps_us.astype(np.float64)) / 1_000_000.0
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def load_viki23_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        frames: list[dict[str, Any]] = json.load(f)
    if not frames:
        raise ValueError(f"{path} contains no frames.")

    landmarks = np.full((len(frames), 23, 3), np.nan, dtype=np.float64)
    timestamps = np.zeros(len(frames), dtype=np.int64)
    for frame_idx, frame in enumerate(frames):
        timestamps[frame_idx] = int(frame.get("ts", frame_idx))
        raw = frame.get("landmarks")
        if raw is None:
            continue
        if isinstance(raw, dict):
            items = ((int(key), value) for key, value in raw.items())
        else:
            items = enumerate(raw)
        for idx, value in items:
            if 0 <= idx < 23:
                vec = np.asarray(value, dtype=np.float64)
                if vec.shape == (3,):
                    landmarks[frame_idx, idx] = vec
    return interpolate_nans(landmarks), timestamps


def convert(input_path: Path, output_path: Path, hand: str, include_arm: bool) -> dict[str, Any]:
    landmarks, timestamps = load_viki23_json(input_path)
    body = np.full((len(landmarks), 33, 3), np.nan, dtype=np.float64)
    right_hand = np.full((len(landmarks), 21, 3), np.nan, dtype=np.float64)
    left_hand = np.full((len(landmarks), 21, 3), np.nan, dtype=np.float64)

    if hand == "right":
        right_hand[:] = landmarks[:, :21, :]
        body[:, RIGHT_WRIST, :] = landmarks[:, 0, :]
        if include_arm:
            body[:, RIGHT_ELBOW, :] = landmarks[:, 21, :]
            body[:, RIGHT_SHOULDER, :] = landmarks[:, 22, :]
    elif hand == "left":
        left_hand[:] = landmarks[:, :21, :]
        body[:, LEFT_WRIST, :] = landmarks[:, 0, :]
        if include_arm:
            body[:, LEFT_ELBOW, :] = landmarks[:, 21, :]
            body[:, LEFT_SHOULDER, :] = landmarks[:, 22, :]
    else:
        raise ValueError("hand must be 'right' or 'left'.")

    # Fill unused body slots from the active arm landmarks so whole-array
    # smoothing does not propagate NaNs.
    active_wrist = RIGHT_WRIST if hand == "right" else LEFT_WRIST
    for idx in range(body.shape[1]):
        missing = ~np.isfinite(body[:, idx, :]).all(axis=1)
        body[missing, idx, :] = body[missing, active_wrist, :]
    if hand == "right":
        left_hand[:] = right_hand[:, :1, :]
    else:
        right_hand[:] = left_hand[:, :1, :]

    fps = estimate_fps(timestamps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        body=body,
        right_hand=right_hand,
        left_hand=left_hand,
        body_conf=np.isfinite(body).all(axis=2).astype(np.float32),
        right_conf=np.isfinite(right_hand).all(axis=2).astype(np.float32),
        left_conf=np.isfinite(left_hand).all(axis=2).astype(np.float32),
        fps=fps,
        frame_count=len(landmarks),
        timestamps_us=timestamps,
        source_json=str(input_path),
        source="viki23_depth_skeleton",
        coordinate_frame="viki_world_or_camera",
        working_hand=hand,
        include_arm=bool(include_arm),
    )
    summary = {
        "output_path": str(output_path),
        "frames": int(len(landmarks)),
        "fps": float(fps),
        "working_hand": hand,
        "include_arm": bool(include_arm),
    }
    print(f"Saved {output_path}")
    print(f"frames={len(landmarks)}, fps={fps:.3f}, hand={hand}, include_arm={include_arm}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ViKi-dev 23-landmark skeleton JSON to experiment sample npz.")
    parser.add_argument("--input", required=True, help="Input rec_*.json file.")
    parser.add_argument("--out", required=True, help="Output sample .npz path.")
    parser.add_argument("--hand", default="right", choices=["right", "left"], help="Which hand the 23 landmarks represent.")
    parser.add_argument("--include-arm", action="store_true", help="Also copy elbow/shoulder into body slots.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert(Path(args.input), Path(args.out), args.hand, args.include_arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
