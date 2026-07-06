# Skeleton Tests Optimisation

This directory is the temporary home for the skeleton optimisation and retargeting workflow while the APIs and data contracts are still being developed. It is intentionally separate from the main `viki/` package so the experimental code can move quickly without changing the capture/skeleton runtime path.

No frontend code or Docker mounts are changed for this workflow.

## Directory Layout

```text
skeleton_tests/
  optimization/
    convert_viki23_json.py     # ViKi skeleton JSON -> optimiser .npz sample
    retarget_rgb_only.py       # IK retargeting entry point
    eval_tracking_error.py     # FK/evaluation helpers
    smoothing.py               # Savitzky-Golay helpers used by retarget/eval
    test_*.py                  # Lightweight tests
  samples/                     # Generated optimiser input .npz files
  output/                      # Retargeted HDF5 trajectories, metrics, and plots
```

`skeleton_tests/` is ignored by the repo, so files in this directory must be added with `git add -f` when they are ready to commit.

## Input Recording Format

The converter reads ViKi-dev skeleton recordings named `rec_*.json` from:

```text
<repo root>/
<repo root>/data/skeleton_recs/
```

Each frame is expected to contain `landmarks` in either list or dict form. The current ViKi-dev skeleton layout is:

```text
0      wrist / hand root target
0..20  MediaPipe hand landmarks
21     elbow
22     shoulder
```

Recordings with only landmarks `0..20` are also accepted by the converter. For optimisation, elbow `21` and shoulder `22` are ignored. Landmark `0` supplies the wrist position target. Landmarks `0`, `1`, and `9` supply the palm orientation for `hand_se3`.

## Conversion Contract

`optimization/convert_viki23_json.py` converts a ViKi skeleton JSON recording into the optimiser sample format:

```text
body:          (T, 33, 3)
right_hand:    (T, 21, 3)
left_hand:     (T, 21, 3)
body_conf:     (T, 33)
right_conf:    (T, 21)
left_conf:     (T, 21)
fps
frame_count
timestamps_us
source_json
source
coordinate_frame
working_hand
orientation_valid
orientation_valid_frames
```

For right-hand wrist-only export:

```text
ViKi landmark 0 -> MediaPipe body index 16
```

For left-hand wrist-only export:

```text
ViKi landmark 0 -> MediaPipe body index 15
```

The converter stores `coordinate_frame="viki_world_or_camera"` by default. Retargeting keeps the legacy coordinate transform for this frame. Samples marked `coordinate_frame="robot_base"` skip that legacy transform.

`orientation_valid` is computed from the raw recording before the converter fills missing landmark coordinates for smoothing. That keeps missing or collinear palm-bone frames visible to `hand_se3`; retargeting fills those rotations from the nearest valid frame and records the validity mask in the HDF5 output.

## CLI Usage

Convert a smoothed recording into an optimiser sample:

```powershell
python skeleton_tests\optimization\convert_viki23_json.py `
  --input rec_1782584807_smoothed.json `
  --out skeleton_tests\samples\rec_1782584807_smoothed_wrist_only.npz `
  --hand right
```

Run retargeting directly through the expected FK conda environment:

```powershell
& 'C:\Users\minim\miniforge3\Scripts\conda.exe' run -n viki-fk python skeleton_tests\optimization\retarget_rgb_only.py `
  --sample skeleton_tests\samples\rec_1782584807_smoothed_wrist_only.npz `
  --robot ur10 `
  --out skeleton_tests\output\real_wrist_ur10.h5 `
  --target-mode wrist_position `
  --ik-position-cost 5 `
  --ik-orientation-cost 0 `
  --joint-sg-window 0 `
  --sg-window 0 `
  --recenter-to-neutral `
  --trajectory-scale 0.25 `
  --evaluate
```

For calibrated robot-base samples, use `--trajectory-scale 1.0` and do not use `--recenter-to-neutral`.

## FastAPI Endpoints

The backend router is mounted at:

```text
/api/optimization
```

Implemented endpoints:

```text
GET  /api/optimization/recordings
POST /api/optimization/convert
GET  /api/optimization/samples
POST /api/optimization/retarget
GET  /api/optimization/jobs
GET  /api/optimization/jobs/{job_id}
GET  /api/optimization/outputs
GET  /api/optimization/outputs/{filename}
```

### List Recordings

```text
GET /api/optimization/recordings
```

Returns usable `rec_*.json` files with filename, relative path, size, modified time, and `looks_smoothed`.

### Convert Recording

```text
POST /api/optimization/convert
```

Request:

```json
{
  "recording": "rec_1782584807_smoothed.json",
  "output_name": "rec_1782584807_smoothed_wrist_only.npz",
  "hand": "right"
}
```

The output is written under:

```text
skeleton_tests/samples/
```

The response includes output path, frame count, fps, working hand, and orientation-valid frame counts.

### List Samples

```text
GET /api/optimization/samples
```

Returns `.npz` files under `skeleton_tests/samples/`.

### Start Retargeting

```text
POST /api/optimization/retarget
```

Request:

```json
{
  "sample": "rec_1782584807_smoothed_wrist_only.npz",
  "robot": "ur10",
  "output_name": "real_wrist_ur10",
  "target_mode": "wrist_position",
  "ik_position_cost": 5,
  "ik_orientation_cost": 0,
  "joint_sg_window": 0,
  "sg_window": 7,
  "recenter_to_neutral": true,
  "trajectory_scale": 0.25,
  "evaluate": true
}
```

The endpoint returns a `job_id` immediately. Retargeting runs in a background worker as a subprocess. Only one worker thread is started, and jobs are processed through an in-memory queue.

If `output_name` has no suffix, the retargeting script writes `<output_name>_traj.h5`. If it ends in `.h5` or `.hdf5`, that exact HDF5 path is used. A legacy `.npz` suffix is converted to `.h5`.

The default subprocess command uses:

```text
C:\Users\minim\miniforge3\Scripts\conda.exe run -n viki-fk python skeleton_tests\optimization\retarget_rgb_only.py ...
```

These can be overridden with environment variables:

```text
VIKI_OPT_CONDA_EXE
VIKI_OPT_CONDA_ENV
```

If conda is unavailable, the retarget endpoint returns `503`. Conversion and file listing endpoints do not require conda.

### Job Status

```text
GET /api/optimization/jobs
GET /api/optimization/jobs/{job_id}
```

Job states are:

```text
queued
running
succeeded
failed
```

The job detail response includes the command, status, stdout/stderr tails, exit code, matching output files, and any error message.

Job state is in memory only. Restarting the server clears the job list but does not delete generated output files.

### Outputs

```text
GET /api/optimization/outputs
GET /api/optimization/outputs/{filename}
```

Outputs are read from:

```text
skeleton_tests/output/
```

Download is restricted to simple filenames and supported output extensions:

```text
.h5
.hdf5
.json
.png
```

Path traversal is rejected.

## Retargeting Behavior

Use `target_mode="wrist_position"` for position-only retargeting. In this mode, `effective_orientation_cost()` is forced to `0`, so orientation does not drive the robot trajectory.

Use `target_mode="hand_se3"` to use wrist position plus palm orientation. The palm frame is derived from hand bones:

```text
x_palm = normalize(MIDDLE_MCP - WRIST)
z_palm = normalize((MIDDLE_MCP - WRIST) x (THUMB_CMC - WRIST))
y_palm = z_palm x x_palm
```

`hand_se3` writes `ee_target_rot` and `orientation_valid` into the HDF5 trajectory output.

For uncalibrated debug runs, `--align-initial-orientation` maps the first valid palm orientation onto the robot's neutral end-effector rotation, then tracks relative hand rotation after that. Leave it off when the sample orientation is already calibrated to the robot tool frame.

The intended current flow is:

```text
ViKi skeleton recording -> hand-only optimiser sample -> wrist_position or hand_se3 retargeting
```

Debug mode:

```text
--recenter-to-neutral --trajectory-scale 0.25
```

Real calibrated mode:

```text
--trajectory-scale 1.0
```

Do not use `--recenter-to-neutral` for calibrated robot-base trajectories because it hides the real spatial relationship between the skeleton and robot.

## Tests

Run the focused tests from the repo root:

```powershell
python -m unittest discover viki\skeleton
python -m unittest viki.server.routes.test_optimization
python -m unittest discover skeleton_tests\optimization
```

The retarget logic tests do not require PINK/Pinocchio. Full IK execution still requires the `viki-fk` conda environment.

## Git Notes

Because this directory is ignored, stage it explicitly:

```powershell
git add -f skeleton_tests\README.md skeleton_tests\__init__.py skeleton_tests\optimization skeleton_tests\samples\.gitkeep skeleton_tests\output\.gitkeep
```

Then stage the API files normally:

```powershell
git add viki\server\app.py viki\server\routes\optimization.py viki\server\routes\test_optimization.py
```
