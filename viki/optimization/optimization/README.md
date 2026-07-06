# ViKi Optimisation Package

This package contains the active backend optimisation workflow. The FastAPI
route in `viki/server/routes/optimization.py` lists already-smoothed target
archives from `data/skeleton_smoothed/`, launches this retargeter in the
external `viki-fk` conda environment, and writes robot HDF5 outputs to
`data/robot_out/`.

## Active Smoothed Input Format

The current frontend/backend flow expects `.npz` files in
`data/skeleton_smoothed/` with these arrays:

- `positions`: `(T, 3)` wrist target positions.
- `rotations`: `(T, 3, 3)` palm/tool target rotations.
- `valid`: `(T,)` orientation validity mask.
- `timestamps`: `(T,)` frame timestamps, normally in microseconds.

These files are already smoothed. The retargeter detects this format directly
and skips landmark Savitzky-Golay smoothing even if `--sg-window` is provided.
Missing wrist positions are linearly filled over time so short gaps do not
crash the IK run; orientation validity is still preserved in the HDF5 output.

## Retargeting Modes

`wrist_position` uses only the wrist position and forces the effective
orientation cost to `0`.

`hand_se3` uses wrist position plus the provided palm/tool rotation. With
`--align-initial-orientation`, frame 0 of the input rotation sequence is mapped
onto the robot neutral end-effector rotation before IK. This is intended for
the current debug flow where the smoothed data is not yet fully calibrated into
robot base coordinates.

Recommended debug defaults for smoothed inputs are:

```powershell
--sg-window 0 `
--joint-sg-window 0 `
--recenter-to-neutral `
--trajectory-scale 0.25 `
--align-initial-orientation
```

Use `--trajectory-scale 1.0` and omit `--recenter-to-neutral` only after the
input positions are calibrated into the robot base frame.

## Outputs

Single-run and evaluation outputs are HDF5-first. Robot trajectory files use
`.h5`/`.hdf5` and include:

- `q_approach`, `q_scene_raw`, `q_scene_smooth`
- `ee_target_pos`
- `ee_target_rot` and `orientation_valid` for `hand_se3`
- solver/configuration metadata
- `timestamps_us` for smoothed input archives

The API only exposes generated artifacts from `data/robot_out/`.

## Legacy Conversion

`convert_viki23_json.py` remains available for old frame-based skeleton JSON
recordings. It creates legacy optimiser `.npz` samples and ignores arm
landmarks `21` and `22`; the active API retarget path does not use these
converted samples.
