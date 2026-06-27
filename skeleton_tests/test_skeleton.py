"""
Smoke-test for the skeleton pipeline (two Kinect cameras).

Run inside Docker:
    docker compose run --rm terminal python3 skeleton_tests/test_skeleton.py

What it checks
--------------
1. Cameras start and produce frames.
2. Calibration loads via CalibrationManager (per-device intrinsics + extrinsics).
3. PreparedFrame is correct shape and dtype.
4. CompositeLandmarkDetector returns a 23-slot HandDetection when an arm
   is in frame.
5. lift_to_3d produces at least one 3-D-backed landmark
   (DEPTH or MediaPipe-z fallback) for the master camera.
6. SkeletonPipeline produces a SkeletonFrame whose wrist is finite.
7. fusion origin counts are not all-missing.
8. SkeletonStats accumulates and reports correctly.

Default detector configuration is arm-only (MediaPipeArm). Hand slots
1..20 are expected to be MISSING.
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.capture.sync import MultiCameraSync
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.detectors import (
    CompositeLandmarkDetector,
    FusionMode,
    MediaPipeArm,
)
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.models import LM, LandmarkSource
from viki.skeleton.pipeline import SkeletonPipeline
from viki.skeleton.stats import SkeletonStats, pretty_print

MASTER_ID = "kinect_0"
SUB_ID = "kinect_1"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


# ── 1. Start cameras ──────────────────────────────────────────────────────────
print("Starting cameras...")
manager = CameraManager()

try:
    manager.start(SUB_ID, fps=30, color_width=1280, color_height=720)
    time.sleep(0.5)
    manager.start(MASTER_ID, fps=30, color_width=1280, color_height=720)
    time.sleep(1.0)
except Exception as e:
    fail(f"Could not start cameras: {e}")

# Wait for first frames
for _ in range(30):
    f0 = manager.latest_frame(MASTER_ID)
    f1 = manager.latest_frame(SUB_ID)
    if f0 and f1:
        break
    time.sleep(0.1)
else:
    fail("Timed out waiting for frames")

print(
    f"[1/8] cameras OK  "
    f"kinect_0={f0.color.shape}  kinect_1={f1.color.shape}"
)

# ── 2. Calibration ────────────────────────────────────────────────────────────
calibrator = CalibrationManager(manager)
calibrator.load_intrinsics(MASTER_ID)
calibrator.load_intrinsics(SUB_ID)
calibrator.load_extrinsics(MASTER_ID)
calibrator.load_extrinsics(SUB_ID)

intr0 = calibrator.get_intrinsics(MASTER_ID)
intr1 = calibrator.get_intrinsics(SUB_ID)
if intr0 is None or intr1 is None:
    fail("Intrinsics not loaded — run intrinsics calibration first")

ext0 = calibrator.get_extrinsics(MASTER_ID)
ext1 = calibrator.get_extrinsics(SUB_ID)
if ext0 is None or ext1 is None:
    fail("Extrinsics not loaded — run extrinsics calibration first")

# Build the same relative transform that SkeletonPipeline uses internally,
# just to sanity-check it on its own (det(R) ≈ 1, |T| reported in cm).
R_master = np.asarray(ext0.rotation_matrix, dtype=np.float64)
R_sub = np.asarray(ext1.rotation_matrix, dtype=np.float64)
t_master = np.asarray(ext0.tvec, dtype=np.float64).reshape(3, 1)
t_sub = np.asarray(ext1.tvec, dtype=np.float64).reshape(3, 1)
R = R_master @ R_sub.T
T = t_master - R @ t_sub

if abs(np.linalg.det(R) - 1.0) > 1e-4:
    fail(f"det(R_relative) = {np.linalg.det(R):.6f}, expected ≈ 1.0")

print(
    f"[2/8] calibration OK  det(R)={np.linalg.det(R):.6f}  "
    f"|T|={np.linalg.norm(T) * 100:.1f}cm"
)

# ── 3. prepare_frame ──────────────────────────────────────────────────────────
K0 = intr0.camera_matrix
dist0 = intr0.dist_coeffs
cache = UndistortCache()
prep = prepare_frame(f0, K0, dist0, cache)

if prep.rgb.dtype != np.uint8:
    fail(f"rgb dtype={prep.rgb.dtype}, expected uint8")
if prep.rgb.shape[2] != 3:
    fail(f"rgb channels={prep.rgb.shape[2]}, expected 3")
if prep.depth_m.dtype != np.float32:
    fail(f"depth_m dtype={prep.depth_m.dtype}, expected float32")

valid_depth = int(np.sum(~np.isnan(prep.depth_m)))
total = prep.depth_m.size
print(
    f"[3/8] prepare_frame OK  "
    f"rgb={prep.rgb.shape}  depth_m={prep.depth_m.shape}  "
    f"valid_depth={valid_depth / total * 100:.1f}%"
)

# ── 4. HandDetection via composite (arm-only) ────────────────────────────────
print("     >>> Put your RIGHT arm in front of the cameras <<<")
time.sleep(3.0)

detector = CompositeLandmarkDetector(
    detectors=[MediaPipeArm(hand="right", mode="image")],
    mode=FusionMode.ANY,
)
detection = None
for _ in range(60):
    f0 = manager.latest_frame(MASTER_ID)
    prep = prepare_frame(f0, K0, dist0, cache)
    detection = detector.detect(prep)
    if detection is not None:
        break
    time.sleep(0.05)

if detection is None:
    fail("No arm detected on kinect_0 after 3s — is arm in frame?")
if detection.px.shape != (LM.N, 2):
    fail(f"px.shape={detection.px.shape}, expected ({LM.N}, 2)")
if detection.lm_z_rel.shape != (LM.N,):
    fail(f"lm_z_rel.shape={detection.lm_z_rel.shape}, expected ({LM.N},)")

print(
    f"[4/8] arm detected on kinect_0  confidence={detection.confidence:.2f}  "
    f"wrist_px=({detection.px[LM.WRIST, 0]:.0f}, "
    f"{detection.px[LM.WRIST, 1]:.0f})"
)

# ── 5. lift_to_3d ─────────────────────────────────────────────────────────────
lm3d = lift_to_3d(detection, prep)

n_depth = int(np.sum(lm3d.source == LandmarkSource.DEPTH))
n_mpz = int(np.sum(lm3d.source == LandmarkSource.MP_Z))
n_missing = int(np.sum(lm3d.source == LandmarkSource.MISSING))

if lm3d.points.shape != (LM.N, 3):
    fail(f"points.shape={lm3d.points.shape}, expected ({LM.N}, 3)")

# Arm-only: expect 3 filled slots (WRIST, ELBOW, SHOULDER). At least one
# must be backed by a real 3-D origin (depth or MediaPipe-z fallback).
if (n_depth + n_mpz) == 0:
    fail("No DEPTH or MP_Z landmarks — depth and z-fallback both failed")

wrist_z = lm3d.points[LM.WRIST, 2]
if np.isnan(wrist_z) or not (0.1 < wrist_z < 2.0):
    fail(f"Wrist Z={wrist_z:.3f}m out of plausible range [0.1, 2.0]")

print(
    f"[5/8] DEPTH={n_depth}  MP_Z={n_mpz}  MISSING={n_missing}  (kinect_0)  "
    f"wrist_Z={wrist_z:.3f}m"
)

# ── 6. SkeletonFrame via full pipeline ────────────────────────────────────────
detector.close()
sync = MultiCameraSync(manager, sync_fps=15)
pipeline = SkeletonPipeline(calibrator)

skeleton = None
for _ in range(60):
    group = sync.get_synced_frame()
    if group is None:
        time.sleep(0.05)
        continue
    result = pipeline.process(group)
    if result.fused_frame is not None:
        skeleton = result.fused_frame
        break
    time.sleep(0.05)

if skeleton is None:
    fail("SkeletonPipeline returned no fused_frame for 3s — check both cameras see the arm")
if skeleton.landmarks.shape != (LM.N, 3):
    fail(f"landmarks.shape={skeleton.landmarks.shape}")

wrist = skeleton.landmarks[LM.WRIST]
if np.isnan(wrist).any():
    fail("Wrist is nan in fused SkeletonFrame")

print(
    f"[6/8] SkeletonFrame OK  "
    f"wrist={wrist.tolist()}  "
    f"ts={skeleton.timestamp_us}"
)

# ── 7. Fusion origin ──────────────────────────────────────────────────────────
from collections import Counter

origin_counts = Counter(skeleton.origin.tolist())
print(
    "[7/8] origin counts: "
    + "  ".join(f"{k}={v}" for k, v in sorted(origin_counts.items()))
)

if origin_counts.get("missing", 0) == LM.N:
    fail("All 23 points are missing — fusion produced empty frame")

# ── 8. SkeletonStats ─────────────────────────────────────────────────────────
stats = SkeletonStats(window=150)

# One real frame + one missed frame.
stats.update(skeleton)
stats.update(None)

summary = stats.summary()

# These checks may be too strict — comment out if necessary.
if summary["frame_count"] != 2:
    fail(f"frame_count={summary['frame_count']}, expected 2")
if summary["detected_count"] != 1:
    fail(f"detected_count={summary['detected_count']}, expected 1")
if abs(summary["detection_rate"] - 0.5) > 1e-4:
    fail(f"detection_rate={summary['detection_rate']}, expected 0.5")
if len(summary["landmarks"]) != LM.N:
    fail(f"landmarks list len={len(summary['landmarks'])}, expected {LM.N}")
if "confidence" not in summary or not summary["confidence"]:
    fail("summary missing confidence stats")

# reset() clears all counters.
stats.reset()
summary2 = stats.summary()
if summary2["frame_count"] != 0 or summary2["detected_count"] != 0:
    fail(
        f"after reset: frame_count={summary2['frame_count']} "
        f"detected={summary2['detected_count']}"
    )

print(
    f"[8/8] SkeletonStats OK  "
    f"detection_rate={summary['detection_rate']:.2f}  "
    f"landmarks={len(summary['landmarks'])}"
)
pretty_print(summary)

# ── Cleanup ───────────────────────────────────────────────────────────────────
pipeline.close()
manager.stop_all()

print()
print("PASS")
