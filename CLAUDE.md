# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

ViKi (Vision-based Kinematic Imitation) turns RGB-D video of a human manipulation task into a robot-ready LeRobot dataset. The pipeline is: multi-view RGB-D capture → 3D skeleton extraction → trajectory optimisation → LeRobot HDF5 dataset. Only Phase 1 (capture) is currently implemented.

## Running the server

Everything runs inside Docker. The `viki` directory is bind-mounted at `/app/viki`, so Python changes take effect on restart without a rebuild.

```bash
# First run (builds image, ~5 min)
docker compose up --build

# Subsequent runs
docker compose up

# Debug terminal inside the container
docker compose run --rm terminal
```

The web UI is at `http://localhost:8000`.

## Frontend

The web UI is a React + Vite + TypeScript SPA in `viki/frontend/`, built into
`viki/server/static/` (a gitignored build artifact) which FastAPI serves unchanged
(`GET /` → `static/index.html`, `/static` mount for assets; Vite `base` is `/static/`).

```bash
./scripts/build_frontend.sh          # build into viki/server/static/ (run before docker compose up)
cd viki/frontend && npm run dev      # dev server :5173, proxies /api + skeleton WS to :8000
cd viki/frontend && npm run build    # production build
cd viki/frontend && npm test         # vitest
```

Structure is **feature-based**: `src/features/{cameras,config,calibration,skeleton,topbar}/`
each own their components, Zustand slice (`*.slice.ts`), API calls (`*.api.ts`), types and
CSS Modules; cross-cutting code lives in `src/shared/` (`api/client.ts`, `store/store.ts`,
`ui/`, `hooks/`). State is a single Zustand store (UI = f(state)); components never call
`fetch` directly — only through a feature `*.api.ts`. MJPEG streams stay as `<img src>`;
the skeleton uses a WebSocket + canvas. This is deliberately NOT the backend's horizontal
layering — feature-slicing is the idiomatic React structure.

## One-time host setup

```bash
sudo ./scripts/host_setup.sh   # installs Docker, udev rules, groups
# then log out and back in
```

For two Azure Kinects: also add `usbcore.usbfs_memory_mb=1000` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub && sudo reboot`.

For Kinect depth engine (OpenGL via llvmpipe): add `xhost +local: > /dev/null 2>&1` to `~/.bashrc`.

## Architecture

```
viki/
  config.py          # Centralised tunables (stream/depth/encoding constants, start defaults)
  capture/
    base.py          # CameraBackend ABC + Frame/CameraIntrinsics/SyncedFrameGroup dataclasses
    realsense.py     # RealSenseBackend (pyrealsense2)
    kinect.py        # KinectBackend (ctypes over libk4a.so — no pyk4a)
    manager.py       # CameraManager: device discovery, start/stop, per-camera worker threads
    calibration.py   # CalibrationManager: chessboard detection, intrinsics + stereo solve
    sync.py          # MultiCameraSync: cross-camera frame alignment by host timestamp
  viz/               # Pure pixel helpers (numpy/cv2) — no FastAPI, no camera, no I/O
    mjpeg.py         # encode_jpeg / mjpeg_chunk / placeholder
    depth.py         # DepthColorizer (EMA-normalised depth → BGR), Undistorter (cached remap)
  server/
    app.py           # App assembly only: lifespan, static mount, include_router
    deps.py          # FastAPI DI: get_manager / get_calibrator (resolve from app.state)
    streams.py       # MJPEG stream generators: poll manager + timing, delegate pixels to viz
    routes/
      cameras.py     # APIRouter: /api/devices, /api/cameras/{id}/start|stop|info|stream|depth
      calibration.py # APIRouter: /api/calibrate/stream|capture|run|status|clear
    static/          # index.html UI
```

**Layering (server):** `routes/` (thin handlers) → `deps.py` (DI) → `streams.py` (poll + timing) → `viz/` (pure pixel work) → `config.py` (constants). `viz/` depends on neither FastAPI nor the camera layer, so it is reusable (e.g. Phase 2 overlays) and testable without hardware. `app.py` only wires these together.

**Data flow:** `CameraManager` owns one `_CameraWorker` (daemon thread) per active camera. Each worker calls `backend.get_frame()` in a tight loop and stores the result in a per-camera ring buffer (`deque`, last-value cache) under a lock. The MJPEG generators in `streams.py` poll `manager.latest_frame()` at ~30 fps — never blocking the FastAPI event loop. All consumers (colour/depth streams, calibration preview, calibration capture) read this one shared source; there is no push pub/sub, and frames are pulled independently (a captured frame may be 1–2 frames newer than the displayed one — fine for the static-board calibration workflow).

**Adding a new camera backend:** subclass `CameraBackend` (implement `start`, `stop`, `get_frame`, `device_id`, `is_running`), then add detection in `CameraManager.list_devices()` and routing in `CameraManager._make_backend()`.

**Adding a new endpoint:** put the handler in the relevant `server/routes/*.py` router (resolve `manager`/`calibrator` via `Depends` from `deps.py`); keep streaming/timing logic in `streams.py` and any pixel processing in `viz/`. Handlers should stay thin — delegate, don't compute.

**Frame format:** `Frame.color` is HxWx3 uint8 BGR (OpenCV convention). `Frame.depth` is HxW uint16 in millimetres.

## Key implementation details

- **Kinect backend uses ctypes directly over `libk4a.so`** — no `pyk4a`. All function signatures are declared in `kinect.py`. This was chosen to avoid compilation inside Docker.
- **`KinectBackend.align_depth_to_color`** is present but marked as bugged — do not enable it.
- **`WFOV_UNBINNED` depth mode** is capped at 15 fps by hardware; the backend raises `ValueError` at 30 fps.
- **USB release delay:** `KinectBackend.stop()` sleeps 2 seconds after closing to let USB fully release before the next open.
- **Kinect sync startup order:** always start subordinate (`kinect_1`) before master (`kinect_0`). Currently both run in standalone `wired_sync_mode=0`; hardware sync mode is planned.

## Planned phases (not yet implemented)

| Phase | Description |
|---|---|
| 2 — Skeleton | MediaPipe pose estimation, depth-fused 3D keypoints, multi-view fusion |
| 3 — Smoothing | One Euro Filter, outlier rejection |
| 4 — Retargeting | URDF IK via PINK/Pinocchio, object-relative cost, gripper inference |
| 5 — Dataset | LeRobot HDF5 writer (RGB + depth + joints + actions) |
| 6 — Evaluation | ACT and Diffusion Policy on UR3 |
