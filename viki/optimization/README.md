# Skeleton Tests Optimisation

This directory holds the skeleton optimisation and retargeting workflow, organised into `preparation/` (recorded landmarks -> end-effector pose) and `retarget/` (end-effector pose -> IK solution).

No frontend code or Docker mounts are changed for this workflow.

## Directory Layout

```text
viki/optimization/
  preparation/                # landmarks -> end-effector rotation + position
    processor.py              # PreparationPipeline: interpolate, fuse, smooth, EE pose -> cln-*.npz
    smoothing.py              # Savitzky-Golay helpers for landmark sequences
    fusion.py                 # cross-camera trajectory fusion
    test_processor_orientation.py
  retarget/                   # end-effector pose -> IK solution (.h5)
    retarget_rgb_only.py      # IK retargeting entry point
    archive_io.py             # HDF5 trajectory writer/reader
    smoothing.py              # Savitzky-Golay helpers for joint trajectories
    eval_tracking_error.py    # FK/evaluation helpers
    debug.py                  # retarget debug visualisation
    test_*.py                 # Lightweight tests
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

## Pipeline

The legacy ViKi2.3 JSON converter (`convert_viki23_json.py`) and its
`/api/optimization/convert` + `/api/optimization/recordings` endpoints have been
removed — there is no legacy format to import.

The current flow is:

```text
skeleton recording (rec-*.npz) -> PreparationPipeline
  (per-camera interpolate, cross-camera fuse, smooth, compute EE pose)
  -> cln-*.npz (positions, rotations, valid, timestamps)
  -> retarget (PINK IK) -> robot_out/*.h5
```

`PreparationPipeline.smooth_recording` reads a raw `rec-*.npz`, interpolates and
smooths per-camera landmark trajectories, fuses the cameras onto a common time
grid, and computes end-effector poses. The resulting `cln-*.npz` is the input to
the retarget endpoints.

## CLI Usage

Run retargeting directly through the expected FK conda environment:

```powershell
& 'C:\Users\minim\miniforge3\Scripts\conda.exe' run -n viki-fk   python viki\optimization\retarget\retarget_rgb_only.py `
  --sample data\skeleton_smoothed\cln-17.20-12.07.2026.npz `
  --robot ur10 `
  --out data\robot_out\boardbase_ur10 `
  --target-mode hand_se3 `
  --ik-position-cost 5 `
  --ik-orientation-cost 0.6 `
  --joint-sg-window 0 `
  --sg-window 0 `
  --trajectory-scale-origin auto `
  --trajectory-scale 0.75 `
  --align-initial-orientation `
  --evaluate
```

The tested iiwa14 settings are scale `0.55`, orientation cost `0.3`, and no
initial-orientation alignment. Do not use `--recenter-to-neutral` for the
ChArUco robot-base flow.

## FastAPI Endpoints

The backend router is mounted at:

```text
/api/optimization
```

Implemented endpoints:

```text
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
  "sample": "cln-17.20-12.07.2026.npz",
  "robot": "ur10",
  "output_name": "real_wrist_ur10",
  "target_mode": "hand_se3",
  "ik_position_cost": 5,
  "ik_orientation_cost": 0.6,
  "joint_sg_window": 0,
  "sg_window": 0,
  "recenter_to_neutral": false,
  "trajectory_scale": 0.75,
  "trajectory_scale_origin": "auto",
  "align_initial_orientation": true,
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

Current board-base defaults:

```text
UR10 hand_se3: --trajectory-scale-origin auto --trajectory-scale 0.75 --ik-orientation-cost 0.6 --align-initial-orientation
iiwa14 hand_se3: --trajectory-scale-origin auto --trajectory-scale 0.55 --ik-orientation-cost 0.3
```

Literal calibrated scale, when the target is reachable:

```text
--trajectory-scale-origin robot_base --trajectory-scale 1.0
```

Do not use `--recenter-to-neutral` for calibrated robot-base trajectories because it hides the real spatial relationship between the skeleton and robot.

## Tests

Run the focused tests from the repo root:

```powershell
python -m unittest discover viki\optimization\preparation
python -m unittest discover viki\optimization\retarget
python -m unittest viki.server.routes.test_optimization
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
