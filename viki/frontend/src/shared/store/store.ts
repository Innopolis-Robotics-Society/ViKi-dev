// The single source of truth. One Zustand store composed of feature slices,
// so any panel reads/writes shared state (devices, statuses, config, ...) and
// re-renders automatically when it changes. UI = f(state).
//
// Each feature exports a slice creator `create<Feature>Slice` with its own state
// + actions; they are combined here. Slices read each other through `get()`,
// which is typed against the full RootState.

import { create } from "zustand";
import {
  createCamerasSlice,
  type CamerasSlice,
} from "../../features/cameras/cameras.slice";
import {
  createConfigSlice,
  type ConfigSlice,
} from "../../features/config/config.slice";
import {
  createCalibrationSlice,
  type CalibrationSlice,
} from "../../features/calibration/calibration.slice";
import {
  createSkeletonSlice,
  type SkeletonSlice,
} from "../../features/skeleton/skeleton.slice";
import { createLogSlice, type LogSlice } from "./log.slice";

export type RootState = LogSlice &
  ConfigSlice &
  CamerasSlice &
  CalibrationSlice &
  SkeletonSlice;

export const useStore = create<RootState>()((...a) => ({
  ...createLogSlice(...a),
  ...createConfigSlice(...a),
  ...createCamerasSlice(...a),
  ...createCalibrationSlice(...a),
  ...createSkeletonSlice(...a),
}));
