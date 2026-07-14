# ViKi — Vision-based Kinematic Imitation

> Video-to-Kinematics for robotics: capture human demonstrations with RGB-D cameras, retarget motions to robots, and generate LeRobot datasets.

ViKi is an open-source pipeline that turns RGB-D video of a human doing a manipulation task into a robot-ready demonstration dataset — no teleoperation rig required.

---

## How it works

```
Human demo (RGB-D video)
        │
        ▼
  Multi-view capture        ← RealSense D435i + Azure Kinect DK
        │
        ▼
  3D skeleton extraction    ← MediaPipe + depth fusion
        │
        ▼
        
  Trajectory optimisation   ← Object-relative IK via PINK / Pinocchio
        │
        ▼
  LeRobot dataset           ← Ready for ACT or Diffusion Policy training
```

---

## Why ViKi?

Teleoperation is expensive, slow, and tied to one robot. Human video is cheap and abundant — but naive retargeting from human to robot kinematics produces noisy, jerky trajectories that hurt policy quality. ViKi closes that gap with a full pipeline that:

- Captures **synchronised multi‑view RGB‑D** streams (RealSense + Azure Kinect).
- Extracts **3D skeletons** via MediaPipe with depth fusion for robust hand tracking.
- **Smooths and interpolates** trajectories to reduce jitter.
- **Retargets** motions to arbitrary robot kinematics (PINK/Pinocchio IK) with object‑relative costs.
- Exports **LeRobot‑compatible HDF5 datasets** ready for ACT or Diffusion Policy training.
---

## Setup

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for full installation instructions including USB configuration, Docker setup, and multi-Kinect sync wiring.

Quick start:

```bash
sudo ./scripts/host_setup.sh   # run once
docker compose up --build
# open http://localhost:8501
```

The web UI is a **Streamlit** app in `viki/streamlit_app/`, started as its own service by
`docker compose` on port `8501`. It talks to the FastAPI capture server (port `8000`) over
HTTP; `http://localhost:8000/` redirects to the Streamlit UI.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Capture | ✅ Done | Multi-view RGB-D capture server, per-camera controls, depth streaming |
| 2 — Skeleton | ✅ Done | MediaPipe pose estimation, depth-fused 3D keypoints, multi-view fusion |
| 3 — Smoothing | 🔧 in progress | One Euro Filter, outlier rejection, smoothness metrics |
| 4 — Retargeting | ⬜ planned | URDF IK via PINK/Pinocchio, object-relative cost, gripper inference |
| 5 — Dataset | ⬜ planned | LeRobot HDF5 writer, RGB + depth + joints + actions packaging |
| 6 — Evaluation | ⬜ planned | ACT and Diffusion Policy on UR3, naive vs ViKi success rate comparison |

---

## Development

### Running Tests
Unit tests are executed in a dedicated test container to ensure all system dependencies (RealSense/Kinect SDKs) are present:
```bash
docker compose -f docker-compose.test.yml run --rm tests
```

### Project Architecture
- `viki/capture`: Abstracts camera backends (RealSense, Kinect), manages multi‑camera synchronisation, and serves MJPEG streams.
- `viki/calibration`: Handles intrinsic/extrinsic calibration of all cameras using chessboard or ChArUco boards.
- `viki/viz`: Pure pixel processing (depth colorization, MJPEG encoding).
- `viki/server`: FastAPI backend that exposes frontend files and REST endpoints for controlling cameras, calibration, skeleton processing, and recording.
- `viki/skeleton`: Runs MediaPipe Hand/Pose detectors, lifts 2D detections to 3D using depth, and fuses observations from multiple cameras.
- `viki/optimization`: Smooths trajectories, runs IK retargeting to robot URDFs, and exports evaluation metrics.
