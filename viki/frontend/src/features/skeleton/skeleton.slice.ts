// Skeleton feature slice: panel + estimation/recording state, the selected viz
// camera and 3D view mode, and the latest stream data (per-device 2D detections
// + fused 3D landmarks). Ports toggleSkeleton/toggleEstimation/toggleRecording/
// updateSkelStatus/toggleSkelView/updateSkelVizCam and the stream data handling
// from startSkelStream.

import type { StateCreator } from "zustand";
import type { RootState } from "../../shared/store/store";
import { skeletonApi } from "./skeleton.api";
import type {
  DetectionPoints,
  Landmarks,
  SkelViewMode,
  SkeletonFrame,
} from "./skeleton.types";

const VIEW_MODES: SkelViewMode[] = ["projections", "isometric", "camera"];

export interface SkeletonSlice {
  skelOpen: boolean;
  estimating: boolean;
  recording: boolean;
  vizCam: string | null;
  viewMode: SkelViewMode;
  detections: Record<string, DetectionPoints | null>;
  detectionStatus: Record<string, boolean>;
  landmarks: Landmarks;

  toggleSkeleton: () => void;
  fetchSkelStatus: () => Promise<void>;
  toggleEstimation: (enable: boolean) => Promise<void>;
  toggleRecording: (enable: boolean) => Promise<void>;
  setVizCam: (id: string) => void;
  cycleViewMode: () => void;
  onStreamFrame: (frame: SkeletonFrame) => void;
}

export const createSkeletonSlice: StateCreator<RootState, [], [], SkeletonSlice> = (
  set,
  get,
) => ({
  skelOpen: false,
  estimating: false,
  recording: false,
  vizCam: null,
  viewMode: "projections",
  detections: {},
  detectionStatus: {},
  landmarks: {},

  toggleSkeleton: () => {
    const open = !get().skelOpen;
    set({ skelOpen: open });
    if (open) {
      // Default the viz camera to the first device (ports populateSkelVizCams).
      const first = get().devices[0]?.id ?? null;
      if (!get().vizCam && first) set({ vizCam: first });
      void get().fetchSkelStatus();
    }
  },

  fetchSkelStatus: async () => {
    try {
      const data = await skeletonApi.status();
      set({ estimating: data.enabled, recording: data.recording });
    } catch {
      get().log("Skel status check failed", "error");
    }
  },

  toggleEstimation: async (enable) => {
    get().log(`Turning skeleton estimation ${enable ? "ON" : "OFF"}...`);
    try {
      await skeletonApi.toggle(enable);
      await get().fetchSkelStatus();
      if (!enable) {
        // Clear overlays + 3D view when estimation stops.
        set({ detections: {}, detectionStatus: {}, landmarks: {} });
      }
    } catch (e) {
      get().log("Estimation toggle failed: " + e);
    }
  },

  toggleRecording: async (enable) => {
    get().log(`Turning skeleton recording ${enable ? "ON" : "OFF"}...`);
    try {
      await skeletonApi.record(enable);
      await get().fetchSkelStatus();
    } catch (e) {
      get().log("Recording toggle failed: " + e, "error");
    }
  },

  setVizCam: (id) => {
    // Clear the current viz when switching camera (ports updateSkelVizCam).
    set({ vizCam: id, landmarks: {} });
  },

  cycleViewMode: () =>
    set((s) => ({
      viewMode: VIEW_MODES[(VIEW_MODES.indexOf(s.viewMode) + 1) % VIEW_MODES.length],
    })),

  onStreamFrame: (frame) => {
    const detections = frame.detections ?? {};
    const detectionStatus: Record<string, boolean> = {};
    for (const d of get().devices) detectionStatus[d.id] = !!detections[d.id];

    const patch: Partial<SkeletonSlice> = { detections, detectionStatus };
    // Keep the last 3D pose if this frame had no fused landmarks.
    if (frame.landmarks && Object.keys(frame.landmarks).length > 0) {
      patch.landmarks = frame.landmarks;
    }
    set(patch);
  },
});
