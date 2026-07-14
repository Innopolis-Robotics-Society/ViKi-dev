"""
tests/skeleton_tests/benchmark_core.py
--------------------------------------
Reproducible metrics harness for the RTMPose skeleton detector.

Runs the SkeletonPipeline detector stack against a fixed input (a video
file, or a live webcam window) and dumps a JSON of quality / latency
stats. Use ``--label`` to tag runs; JSON dumps are named
``benchmark_<label>_<ts>.json``.

The lift step (``_lift_pinhole``) uses a constant fake depth for pure
2-D inputs — this is a synthetic benchmark of a 2-D detector, not a
measurement of metric accuracy. Detection quality (2-D pixel jitter,
detection rate, per-LM coverage) is what these numbers reflect.

Usage
-----
    # Against a recorded RGB video:
    python tests/skeleton_tests/benchmark_core.py \\
        --input /path/to/hand.mp4 --label rtmpose --output data/bench

    # Against a live webcam for N seconds:
    python tests/skeleton_tests/benchmark_core.py \\
        --input webcam:0 --duration 30 --label rtmpose --output data/bench
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from viki.capture.base import Frame
from viki.skeleton.camera_prep import prepare_frame
from viki.skeleton.detectors import (
    CompositeLandmarkDetector,
    FusionMode,
    RTMPoseWholeBody,
)
from viki.skeleton.hand_angles import compute_end_effector_pose, compute_hand_angles
from viki.skeleton.models import LM, HandDetection, Landmarks3D

_WEBCAM_Z_BASE_M = 1.0
_WEBCAM_Z_SPAN_M = 0.3


def _lift_pinhole(detection: HandDetection, K: np.ndarray) -> Landmarks3D:
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
        z_rel = float(detection.lm_z_rel[i]) if i < len(detection.lm_z_rel) else 0.0
        if np.isnan(z_rel):
            z_rel = 0.0
        Z = _WEBCAM_Z_BASE_M + z_rel * _WEBCAM_Z_SPAN_M
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points[lm] = np.array([X, Y, Z], dtype=np.float32)
    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )


@dataclass
class _Accumulator:
    """Streaming metrics buffer — one instance per benchmark run."""

    total_frames: int = 0
    detected_frames: int = 0
    confidences: list[float] = field(default_factory=list)

    # Latency samples (ms) per stage.
    lat_detect_ms: list[float] = field(default_factory=list)
    lat_lift_ms: list[float] = field(default_factory=list)
    lat_pose_ms: list[float] = field(default_factory=list)
    lat_total_ms: list[float] = field(default_factory=list)

    # Per-LM finite counts (u,v both finite in the 2-D detection).
    per_lm_finite: np.ndarray = field(
        default_factory=lambda: np.zeros(LM.N, dtype=np.int64)
    )

    # Validity of downstream computations.
    ee_full_count: int = 0        # true palm frame from WRIST + THUMB_CMC + MIDDLE_MCP
    ee_fallback_count: int = 0    # centroid + identity rotation (compute_end_effector_pose's fallback)
    angles_valid_count: int = 0

    # Rolling buffers for jitter (last seen finite value per LM).
    _last_px: dict[LM, np.ndarray] = field(default_factory=dict)
    _last_xyz: dict[LM, np.ndarray] = field(default_factory=dict)
    jitter_2d_sq: np.ndarray = field(
        default_factory=lambda: np.zeros(LM.N, dtype=np.float64)
    )
    jitter_2d_n: np.ndarray = field(
        default_factory=lambda: np.zeros(LM.N, dtype=np.int64)
    )
    jitter_3d_sq: np.ndarray = field(
        default_factory=lambda: np.zeros(LM.N, dtype=np.float64)
    )
    jitter_3d_n: np.ndarray = field(
        default_factory=lambda: np.zeros(LM.N, dtype=np.int64)
    )

    def add(
        self,
        detection: HandDetection | None,
        lm3d: Landmarks3D | None,
        angles_valid: bool,
        ee_full: bool,
        ee_fallback: bool,
        lat_detect_ms: float,
        lat_lift_ms: float,
        lat_pose_ms: float,
    ) -> None:
        self.total_frames += 1
        total_ms = lat_detect_ms + lat_lift_ms + lat_pose_ms
        self.lat_detect_ms.append(lat_detect_ms)
        self.lat_lift_ms.append(lat_lift_ms)
        self.lat_pose_ms.append(lat_pose_ms)
        self.lat_total_ms.append(total_ms)

        if detection is None:
            return

        self.detected_frames += 1
        self.confidences.append(float(detection.confidence))
        if ee_full:
            self.ee_full_count += 1
        if ee_fallback:
            self.ee_fallback_count += 1
        if angles_valid:
            self.angles_valid_count += 1

        for i in range(LM.N):
            lm = LM(i)
            uv = detection.points[lm]
            u, v = float(uv[0]), float(uv[1])
            if np.isnan(u) or np.isnan(v):
                continue
            self.per_lm_finite[i] += 1
            cur = np.array([u, v], dtype=np.float64)
            prev = self._last_px.get(lm)
            if prev is not None:
                d = float(np.linalg.norm(cur - prev))
                self.jitter_2d_sq[i] += d * d
                self.jitter_2d_n[i] += 1
            self._last_px[lm] = cur

        if lm3d is not None:
            for i in range(LM.N):
                lm = LM(i)
                p = lm3d.points[lm]
                if not np.all(np.isfinite(p)):
                    continue
                cur3 = np.asarray(p, dtype=np.float64)
                prev3 = self._last_xyz.get(lm)
                if prev3 is not None:
                    d3 = float(np.linalg.norm(cur3 - prev3))
                    self.jitter_3d_sq[i] += d3 * d3
                    self.jitter_3d_n[i] += 1
                self._last_xyz[lm] = cur3


def _stats(samples: list[float] | np.ndarray) -> dict:
    a = np.asarray(samples, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def _summary(acc: _Accumulator, wall_seconds: float, label: str, source: str) -> dict:
    det_rate = acc.detected_frames / acc.total_frames if acc.total_frames else 0.0
    per_lm_rate = (
        acc.per_lm_finite / acc.detected_frames
        if acc.detected_frames
        else np.zeros(LM.N)
    )
    rms_2d = np.where(
        acc.jitter_2d_n > 0, np.sqrt(acc.jitter_2d_sq / np.maximum(acc.jitter_2d_n, 1)), np.nan
    )
    rms_3d = np.where(
        acc.jitter_3d_n > 0, np.sqrt(acc.jitter_3d_sq / np.maximum(acc.jitter_3d_n, 1)), np.nan
    )
    return {
        "label": label,
        "source": source,
        "wall_seconds": wall_seconds,
        "session": {
            "total_frames": acc.total_frames,
            "detected_frames": acc.detected_frames,
            "detection_rate": det_rate,
            "throughput_fps": acc.total_frames / wall_seconds if wall_seconds > 0 else 0.0,
        },
        "confidence": _stats(acc.confidences),
        "latency_ms": {
            "detect": _stats(acc.lat_detect_ms),
            "lift": _stats(acc.lat_lift_ms),
            "pose_compute": _stats(acc.lat_pose_ms),
            "total": _stats(acc.lat_total_ms),
        },
        "validity_over_detected": {
            # True palm frame — requires WRIST + THUMB_CMC + MIDDLE_MCP all finite.
            "end_effector_full": (
                acc.ee_full_count / acc.detected_frames if acc.detected_frames else 0.0
            ),
            # Centroid-only fallback (identity rotation, no real orientation).
            "end_effector_fallback": (
                acc.ee_fallback_count / acc.detected_frames if acc.detected_frames else 0.0
            ),
            "hand_angles": (
                acc.angles_valid_count / acc.detected_frames if acc.detected_frames else 0.0
            ),
        },
        "per_lm_detection_rate": {LM(i).name: float(per_lm_rate[i]) for i in range(LM.N)},
        "per_lm_jitter_2d_px_rms": {
            LM(i).name: (float(rms_2d[i]) if np.isfinite(rms_2d[i]) else None)
            for i in range(LM.N)
        },
        "per_lm_jitter_3d_m_rms": {
            LM(i).name: (float(rms_3d[i]) if np.isfinite(rms_3d[i]) else None)
            for i in range(LM.N)
        },
    }


def _print_console_summary(s: dict) -> None:
    print(f"\n=== Benchmark: {s['label']} ===")
    print(f"source           : {s['source']}")
    print(f"wall seconds     : {s['wall_seconds']:.2f}")
    sess = s["session"]
    print(
        f"frames           : {sess['total_frames']} total, "
        f"{sess['detected_frames']} detected "
        f"({sess['detection_rate'] * 100:.1f}%)"
    )
    print(f"throughput       : {sess['throughput_fps']:.1f} fps (wall)")
    lat = s["latency_ms"]["total"]
    print(
        f"latency total ms : mean={lat['mean']:.1f}  "
        f"p50={lat['p50']:.1f}  p95={lat['p95']:.1f}"
    )
    for stage in ("detect", "lift", "pose_compute"):
        st = s["latency_ms"][stage]
        print(f"  {stage:<12s}   : mean={st['mean']:.2f}  p95={st['p95']:.2f}")
    print(f"confidence mean  : {s['confidence']['mean']:.3f}")
    v = s["validity_over_detected"]
    print(
        f"validity         : ee_full={v['end_effector_full'] * 100:.1f}%  "
        f"ee_fallback={v['end_effector_fallback'] * 100:.1f}%  "
        f"hand_angles={v['hand_angles'] * 100:.1f}%"
    )
    print("per-LM detection (worst 5):")
    per = s["per_lm_detection_rate"]
    worst = sorted(per.items(), key=lambda kv: kv[1])[:5]
    for name, rate in worst:
        print(f"  {name:<14s} {rate * 100:.1f}%")
    print("per-LM 2D jitter (px RMS, worst 5):")
    jit = {k: v for k, v in s["per_lm_jitter_2d_px_rms"].items() if v is not None}
    worst = sorted(jit.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for name, rms in worst:
        print(f"  {name:<14s} {rms:.2f} px")


def _open_source(input_spec: str) -> tuple[cv2.VideoCapture, bool]:
    """Return (capture, is_live). Accepts a file path or 'webcam:N'."""
    if input_spec.startswith("webcam:"):
        idx = int(input_spec.split(":", 1)[1])
        cap = cv2.VideoCapture(idx)
        return cap, True
    cap = cv2.VideoCapture(input_spec)
    return cap, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a video file, or 'webcam:N' for live capture.",
    )
    parser.add_argument("--label", required=True, help="Tag for the output JSON.")
    parser.add_argument(
        "--output",
        default="data/bench",
        help="Directory to write the metrics JSON into.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Webcam capture length in seconds. Ignored for file inputs.",
    )
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the frames while benchmarking (slower, but useful live).",
    )
    args = parser.parse_args()

    cap, is_live = _open_source(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open input {args.input!r}")

    detector = CompositeLandmarkDetector(
        detectors=[
            RTMPoseWholeBody(hand=args.hand, model_mode="balanced", device="cpu"),
        ],
        mode=FusionMode.ANY,
    )
    acc = _Accumulator()

    ret, bgr = cap.read()
    if not ret:
        raise SystemExit("Empty input.")
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
    depth_fake = np.full((proc_h, proc_w), 700, dtype=np.uint16)

    wall_t0 = time.perf_counter()
    frame_idx = 0
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_us = int(1_000_000 / fps_in) if fps_in > 0 else 33_333

    print(f"[bench] running label={args.label!r}, source={args.input!r}")
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        if scale != 1.0:
            proc = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc = bgr

        frame = Frame(
            color=proc,
            depth=depth_fake,
            timestamp_us=frame_idx * dt_us,
            device_id="benchmark",
        )
        prepared = prepare_frame(frame)

        t0 = time.perf_counter()
        detection = detector.detect(prepared)
        t_detect = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        lm3d = _lift_pinhole(detection, K) if detection is not None else None
        t_lift = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if lm3d is not None:
            ee = compute_end_effector_pose(lm3d.points, prepared.timestamp_us)
            ang = compute_hand_angles(lm3d.points)
        else:
            ee = None
            ang = None
        t_pose = (time.perf_counter() - t0) * 1000.0

        # Distinguish the true palm-frame pose from compute_end_effector_pose's
        # centroid+identity fallback: the fallback is signalled by R == I and
        # rpy == (0, 0, 0). This lets the benchmark expose fallback-inflated
        # valid rates when finger landmarks are missing.
        ee_full = False
        ee_fallback = False
        if ee is not None and ee.valid:
            is_identity = (
                bool(np.allclose(ee.R_world_palm, np.eye(3), atol=1e-6))
                and bool(np.allclose(ee.rpy_deg, 0.0, atol=1e-6))
            )
            ee_fallback = is_identity
            ee_full = not is_identity
        ang_valid = bool(ang.valid) if ang is not None else False

        acc.add(
            detection=detection,
            lm3d=lm3d,
            angles_valid=ang_valid,
            ee_full=ee_full,
            ee_fallback=ee_fallback,
            lat_detect_ms=t_detect,
            lat_lift_ms=t_lift,
            lat_pose_ms=t_pose,
        )

        if args.preview:
            cv2.imshow("bench", proc)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

        if is_live and (time.perf_counter() - wall_t0) >= args.duration:
            break

    wall_seconds = time.perf_counter() - wall_t0
    cap.release()
    detector.close()
    if args.preview:
        cv2.destroyAllWindows()

    s = _summary(acc, wall_seconds, args.label, args.input)
    _print_console_summary(s)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_{args.label}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)
    print(f"\n[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
