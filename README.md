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

## System Setup

This section covers everything you need to run ViKi on a fresh Ubuntu machine. Follow these steps in order — they only need to be done once per machine.

### Prerequisites

- Ubuntu 22.04 (x86_64)
- Intel RealSense D435i
- Azure Kinect DK (one or two)
- 3.5mm mono audio cable for multi-Kinect sync
- Docker and Docker Compose

---

### 1. Host setup script

Run once as root. Installs Docker, udev rules for all cameras, and adds your user to the required groups:

```bash
chmod +x scripts/host_setup.sh
sudo ./scripts/host_setup.sh
```

Then **log out and back in** for group membership to take effect.

---

### 2. USB memory limit (required for two Azure Kinects)

Azure Kinect depth streams require significantly more USB DMA memory than the Linux kernel default (16 MB). Without this fix, the second device fails with `LIBUSB_ERROR_IO` / `errno=12` (ENOMEM).

Edit GRUB:

```bash
sudo nano /etc/default/grub
```

Find this line:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```

Replace with:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=1000"
```

Apply and reboot:

```bash
sudo update-grub
sudo reboot
```

Verify after reboot:

```bash
cat /sys/module/usbcore/parameters/usbfs_memory_mb
# should print: 1000
```

**Reference:** [Azure Kinect SDK issue #485](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/485)

---

### 3. X11 access for Kinect depth engine

The Azure Kinect depth engine uses OpenGL (via llvmpipe software rasterizer) to process raw depth frames. It needs access to the X display. Add this to `~/.bashrc` so it runs automatically at login:

```bash
echo 'xhost +local: > /dev/null 2>&1' >> ~/.bashrc
source ~/.bashrc
```

---

### 4. Multi-Kinect USB wiring

For two Azure Kinects on a single PC:

- Each Kinect must be on a **separate USB hub** with dedicated bandwidth. Check with `lsusb -t` — look for two distinct hubs, each running at 10000M.
- Connect the **3.5mm mono audio cable**: `SYNC OUT` of the master → `SYNC IN` of the subordinate. This enables hardware-synchronized depth captures.

> Note: The plastic cover on the Kinect must be removed to access the SYNC ports.

**Startup order matters.** Always start the subordinate camera first, then the master. In the ViKi UI: click Start on kinect_1 (subordinate), wait for LIVE indicator, then Start on kinect_0 (master).

---

### 5. Running ViKi

```bash
docker compose up --build
```

Open `http://localhost:8000` in your browser.

On subsequent runs (no code changes):

```bash
docker compose up
```

To open a debug terminal inside the container:

```bash
docker compose run --rm terminal
```

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| Kinect fails with `depth engine error 204` | `xhost +local:` not run before `docker compose up` |
| Second Kinect fails with `LIBUSB_ERROR_IO` | GRUB `usbfs_memory_mb` fix not applied, or both Kinects on same USB hub |
| RealSense `Couldn't resolve requests` | Wrong resolution selected; RealSense supports 640×480 / 1280×720 / 1920×1080 |
| Depth stream black in UI | Start camera, wait 2 seconds, then refresh the page |
| `Authorization required` in logs | Run `xhost +local:` on the host before starting the container |

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
- [x] Hardware setup: RealSense D435i × 2 + Azure Kinect
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
