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

Teleoperation is expensive, slow, and tied to one robot. Human video is cheap and abundant — but naive retargeting from human to robot kinematics produces noisy, jerky trajectories that hurt policy quality. ViKi closes that gap with trajectory optimisation that respects joint limits, smoothness, and object-relative task structure.

---

## Setup

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for full installation instructions including USB configuration, Docker setup, and multi-Kinect sync wiring.

Quick start:

```bash
sudo ./scripts/host_setup.sh   # run once
docker compose up --build
# open http://localhost:8000
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Capture | 🔧 in progress | Multi-view RGB-D capture server, per-camera controls, depth streaming |
| 2 — Skeleton | ⬜ planned | MediaPipe pose estimation, depth-fused 3D keypoints, multi-view fusion |
| 3 — Smoothing | ⬜ planned | One Euro Filter, outlier rejection, smoothness metrics |
| 4 — Retargeting | ⬜ planned | URDF IK via PINK/Pinocchio, object-relative cost, gripper inference |
| 5 — Dataset | ⬜ planned | LeRobot HDF5 writer, RGB + depth + joints + actions packaging |
| 6 — Evaluation | ⬜ planned | ACT and Diffusion Policy on UR3, naive vs ViKi success rate comparison |
