# ViKi — Vision-based Kinematic Imitation

> Collect robot manipulation datasets from human video. No robot required during data collection.

ViKi is an open-source pipeline that turns consumer-grade RGB-D video of a human doing a task into a clean, robot-ready demonstration dataset. You record yourself, ViKi figures out what your arm did, optimises the motion for your specific robot, and spits out a LeRobot-compatible dataset ready for ACT or Diffusion Policy training.

---

## Why ViKi?

Teleoperation rigs are expensive, slow, and tied to one robot. Human video is cheap and abundant — but raw retargeting from human to robot kinematics produces noisy, jerky trajectories that hurt policy quality. ViKi closes that gap with:

- **Multi-view RGB-D capture** — synchronized Intel RealSense D435i + Azure Kinect, minimising occlusion
- **Skeleton smoothing** — temporal filtering and outlier rejection on raw pose estimates before any retargeting
- **Trajectory optimisation** — object-relative IK via PINK/Pinocchio, respecting joint limits and smoothness constraints
- **URDF-agnostic** — swap in any robot model; the pipeline adapts automatically
- **LeRobot output** — drop-in compatible with ACT and Diffusion Policy training scripts

---

## Pipeline at a Glance

```
Human Demo (RGB-D)
        │
        ▼
┌─────────────────────┐
│  Multi-view Capture │  <- RealSense D435i + Azure Kinect, hardware sync
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 3D Skeleton Extract │  <- MediaPipe / depth-fused body + hand pose
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Skeleton Smoothing  │  <- Temporal filtering, outlier rejection
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Trajectory Optimise │  <- Object-relative IK, PINK + Pinocchio, URDF target
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Dataset Generation │  <- LeRobot format, train/eval split
└─────────────────────┘
```

---

## Roadmap / Plan of Work

### Phase 1 — Multi-view Capture System
- [ ] Hardware setup: RealSense D435i × 2 + Azure Kinect
- [ ] Hardware sync via GPIO trigger (RealSense primary → secondary + Azure)
- [ ] Extrinsic calibration with ArUco marker board
- [ ] Capture node: synchronized recording, timestamped bags
- [ ] Depth quality validation tooling

### Phase 2 — 3D Skeleton Extraction
- [ ] RGB pose estimation (MediaPipe Holistic or similar) per camera view
- [ ] Depth-fusion: lift 2D keypoints to 3D using registered depth frames
- [ ] Multi-view triangulation / fusion for improved accuracy
- [ ] Object pose estimation (ArUco / FoundationPose) for object-relative frame

### Phase 3 — Skeleton Smoothing
- [ ] One Euro Filter / Savitzky-Golay on raw joint trajectories
- [ ] Outlier detection and keypoint interpolation
- [ ] Smoothness metrics and visualisation tools
- [ ] Benchmark: smoothed vs raw trajectory quality

### Phase 4 — Trajectory Optimisation
- [ ] URDF loader and robot model interface (Pinocchio)
- [ ] Object-relative task formulation for end-effector target
- [ ] Cost function: task error + joint velocity + posture regulariser + joint limit penalty
- [ ] PINK solver integration, per-frame IK loop
- [ ] Weight ablation tooling (systematic w1–w4 sweep)
- [ ] Gripper state inference from hand aperture

### Phase 5 — Dataset Generation
- [ ] LeRobot HDF5 format writer
- [ ] RGB + depth + joint state + action packaging
- [ ] Train/eval split utility
- [ ] Dataset statistics and visualisation dashboard

### Phase 6 — Evaluation
- [ ] Tasks: pick-and-place, object handover, pouring
- [ ] Baselines: naive retargeting vs ViKi optimised
- [ ] Policies: ACT and Diffusion Policy, 200 epochs each
- [ ] Evaluation on physical UR3: 30 rollouts per task per condition
- [ ] Success rate table + failure mode analysis