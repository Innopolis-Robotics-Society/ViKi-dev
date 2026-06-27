"""
Smoke-test for the skeleton pipeline.

Run inside Docker:
    docker compose run --rm terminal python3 skeleton_tests/test_skeleton.py

What it checks
--------------
1. Cameras start and produce frames
2. Calibration loads (intrinsics + extrinsics)
3. PreparedFrame is correct shape and dtype
4. HandDetection returns 23 landmarks when hand is in frame
5. Landmarks3D has at least some DEPTH points
6. SkeletonFrame is produced and landmarks are in plausible metric range
7. fusion origin fields are set correctly
8. SkeletonStats accumulates and reports correctly

Expected output with hand in front of cameras:
    [1/8] cameras OK
    [2/8] calibration OK
    [3/8] prepare_frame OK  rgb=(720,1280,3) depth_m=(720,1280)
    [4/8] hand detected on kinect_0  confidence=0.97
    [5/8] DEPTH=18  MP_Z=2  MISSING=3  (kinect_0)  wrist_Z=0.521m
    [6/8] SkeletonFrame OK  wrist=[0.012, -0.043, 0.521]
    [7/8] origin counts: kinect_0=19  kinect_1=3  missing=1
    [8/8] SkeletonStats OK  detection_rate=0.50  landmarks=23
    PASS
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from viki.capture.manager import CameraManager
from viki.capture.sync import MultiCameraSync
from viki.skeleton.camera_prep import UndistortCache, prepare_frame
from viki.skeleton.hand_detector import HandDetector
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.fusion import fuse, load_extrinsics
from viki.skeleton.models import LM, LandmarkSource
from viki.skeleton.pipeline import SkeletonPipeline
from viki.skeleton.stats import SkeletonStats, pretty_print

CALIB_PATH = "viki/capture/calibration_results.npz"
MASTER_ID = "kinect_0"
SUB_ID = "kinect_1"


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


# ── 1. Start cameras ──────────────────────────────────────────────────────────
print("Starting cameras...")
manager = CameraManager()
manager.load_calibration(CALIB_PATH)

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

print(f"[1/8] cameras OK  " f"kinect_0={f0.color.shape}  kinect_1={f1.color.shape}")

# ── 2. Calibration ────────────────────────────────────────────────────────────
cal0 = manager.get_calibration(MASTER_ID)
cal1 = manager.get_calibration(SUB_ID)
if cal0 is None or cal1 is None:
    fail("Calibration not loaded — run calibration first")

R, T = load_extrinsics(CALIB_PATH)
if abs(np.linalg.det(R) - 1.0) > 1e-4:
    fail(f"det(R) = {np.linalg.det(R):.6f}, expected 1.0")

print(
    f"[2/8] calibration OK  det(R)={np.linalg.det(R):.6f}  "
    f"|T|={np.linalg.norm(T)*100:.1f}cm"
)

# ── 3. prepare_frame ──────────────────────────────────────────────────────────
import cv2

K0 = np.asarray(cal0["mtx"], dtype=np.float32)
dist0 = np.asarray(cal0["dist"], dtype=np.float32)
cache = UndistortCache()
prep = prepare_frame(f0, K0, dist0, cache)

if prep.rgb.dtype != np.uint8:
    fail(f"rgb dtype={prep.rgb.dtype}, expected uint8")
if prep.rgb.shape[2] != 3:
    fail(f"rgb channels={prep.rgb.shape[2]}, expected 3")
if prep.depth_m.dtype != np.float32:
    fail(f"depth_m dtype={prep.depth_m.dtype}, expected float32")

valid_depth = np.sum(~np.isnan(prep.depth_m))
total = prep.depth_m.size
print(
    f"[3/8] prepare_frame OK  "
    f"rgb={prep.rgb.shape}  depth_m={prep.depth_m.shape}  "
    f"valid_depth={valid_depth/total*100:.1f}%"
)

# ── 4. HandDetection ─────────────────────────────────────────────────────────
print("     >>> Put your RIGHT hand in front of the cameras <<<")
time.sleep(3.0)

detector = HandDetector(hand="right")
detection = None
for _ in range(60):
    f0 = manager.latest_frame(MASTER_ID)
    prep = prepare_frame(f0, K0, dist0, cache)
    detection = detector.detect(prep)
    if detection is not None:
        break
    time.sleep(0.05)

if detection is None:
    fail("No hand detected on kinect_0 after 3s — is hand in frame?")
if detection.points.shape != (23, 2):
    fail(f"px.shape={detection.points.shape}, expected (23, 2)")
if detection.lm_z_rel.shape != (23,):
    fail(f"lm_z_rel.shape={detection.lm_z_rel.shape}, expected (23,)")

print(
    f"[4/8] hand detected on kinect_0  confidence={detection.confidence:.2f}  "
    f"wrist_px=({detection.points[LM.WRIST,0]:.0f}, {detection.points[LM.WRIST,1]:.0f})"
)

# ── 5. lift_to_3d ─────────────────────────────────────────────────────────────
lm3d = lift_to_3d(detection, prep)

n_depth = np.sum(lm3d.source == LandmarkSource.DEPTH)
n_mpz = np.sum(lm3d.source == LandmarkSource.MP_Z)
n_missing = np.sum(lm3d.source == LandmarkSource.MISSING)

if lm3d.points.shape != (23, 3):
    fail(f"points.shape={lm3d.points.shape}, expected (23, 3)")
if n_depth == 0:
    fail("No DEPTH landmarks — depth sensor may not be working")

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
pipeline = SkeletonPipeline(manager, calib_path=CALIB_PATH)

skeleton = None
for _ in range(60):
    group = sync.get_synced_frame()
    if group is None:
        time.sleep(0.05)
        continue
    skeleton = pipeline.process(group)
    if skeleton is not None:
        break
    time.sleep(0.05)

if skeleton is None:
    fail("SkeletonPipeline returned None for 3s — check both cameras see the hand")
if skeleton.landmarks.shape != (23, 3):
    fail(f"landmarks.shape={skeleton.landmarks.shape}")

wrist = skeleton.landmarks[LM.WRIST]
if np.isnan(wrist).any():
    fail(f"Wrist is nan in fused SkeletonFrame")

print(
    f"[6/8] SkeletonFrame OK  "
    f"wrist={wrist.tolist()}  "
    f"ts={skeleton.timestamp_us}"
)

# ── 7. Fusion origin ──────────────────────────────────────────────────────────
from collections import Counter

origin_counts = Counter(skeleton.origin.tolist())
print(
    f"[7/8] origin counts: "
    + "  ".join(f"{k}={v}" for k, v in sorted(origin_counts.items()))
)

if origin_counts.get("missing", 0) == 23:
    fail("All 23 points are missing — fusion produced empty frame")

# ── 8. SkeletonStats ─────────────────────────────────────────────────────────
stats = SkeletonStats(window=150)

# One real frame + one missed frame
stats.update(skeleton)
stats.update(None)

summary = stats.summary()


# Эти ограничения могут быть too strict, так что можно закомменить есч
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

# reset() clears all counters
stats.reset()
summary2 = stats.summary()
if summary2["frame_count"] != 0 or summary2["detected_count"] != 0:
    fail(
        f"after reset: frame_count={summary2['frame_count']} detected={summary2['detected_count']}"
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
