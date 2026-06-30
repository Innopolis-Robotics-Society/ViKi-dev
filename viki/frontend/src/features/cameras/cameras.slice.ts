// Cameras feature slice: device list, per-camera running state, live info, and
// the per-card start config (single source of truth for the control selects).
// Ports scanDevices/startCamera/stopCamera/startAll/stopAll/updateInfo/checkServer
// and the RGB-D record action.

import type { StateCreator } from "zustand";
import type { RootState } from "../../shared/store/store";
import { camerasApi } from "./cameras.api";
import { recordingApi } from "./recording.api";
import type { CameraInfo, Device, StartConfig } from "./cameras.types";
import type { CameraTypeConfig } from "../config/config.types";

function defaultCardConfig(cfg: CameraTypeConfig): StartConfig {
  const [w, h] = cfg.defaultRes.split("x").map(Number);
  return {
    color_width: w,
    color_height: h,
    fps: cfg.defaultFps,
    depth_mode: cfg.defaultDepth ?? "NFOV_UNBINNED",
  };
}

export interface CamerasSlice {
  devices: Device[];
  running: Record<string, boolean>;
  info: Record<string, CameraInfo | undefined>;
  cardConfig: Record<string, StartConfig>;
  serverAlive: boolean;
  isRecordingRgbd: boolean;

  scanDevices: () => Promise<void>;
  checkServer: () => Promise<void>;
  setCardConfig: (id: string, patch: Partial<StartConfig>) => void;
  startCamera: (id: string) => Promise<void>;
  stopCamera: (id: string) => Promise<void>;
  startAll: () => Promise<void>;
  stopAll: () => Promise<void>;
  fetchInfo: (id: string) => Promise<void>;
  recordRgbd: () => Promise<number | null>;
}

export const createCamerasSlice: StateCreator<RootState, [], [], CamerasSlice> = (
  set,
  get,
) => ({
  devices: [],
  running: {},
  info: {},
  cardConfig: {},
  serverAlive: false,
  isRecordingRgbd: false,

  scanDevices: async () => {
    get().log("Scanning for devices...");
    try {
      const data = await camerasApi.listDevices();
      get().log(
        `Found: ${data.realsense.length} RealSense, ${data.kinect.length} Kinect`,
        "ok",
      );
      const devices: Device[] = [
        ...data.realsense.map((id): Device => ({ id, type: "realsense" })),
        ...data.kinect.map((id): Device => ({ id, type: "kinect" })),
      ];
      const active = data.active ?? [];
      const cameraConfig = get().cameraConfig;
      const prevConfig = get().cardConfig;
      const running: Record<string, boolean> = {};
      const cardConfig: Record<string, StartConfig> = {};
      for (const d of devices) {
        running[d.id] = active.includes(d.id);
        const typeCfg =
          d.type === "kinect" ? cameraConfig.kinect : cameraConfig.realsense;
        cardConfig[d.id] = prevConfig[d.id] ?? defaultCardConfig(typeCfg);
      }
      set({ devices, running, cardConfig });
    } catch (e) {
      get().log("Scan failed: " + e, "error");
    }
  },

  checkServer: async () => {
    try {
      await camerasApi.listDevices();
      set({ serverAlive: true });
    } catch {
      set({ serverAlive: false });
    }
  },

  setCardConfig: (id, patch) =>
    set((s) => ({
      cardConfig: { ...s.cardConfig, [id]: { ...s.cardConfig[id], ...patch } },
    })),

  startCamera: async (id) => {
    const cfg = get().cardConfig[id];
    if (!cfg) return;
    get().log(`Starting ${id} @ ${cfg.color_width}×${cfg.color_height} ${cfg.fps}fps...`);
    try {
      await camerasApi.start(id, cfg);
      get().log(`${id} started`, "ok");
      set((s) => ({ running: { ...s.running, [id]: true } }));
    } catch (e) {
      get().log(`Failed to start ${id}: ${e}`, "error");
    }
  },

  stopCamera: async (id) => {
    get().log(`Stopping ${id}...`);
    try {
      await camerasApi.stop(id);
      get().log(`${id} stopped`);
    } catch (e) {
      get().log(`Failed to stop ${id}: ${e}`, "error");
    }
    set((s) => ({
      running: { ...s.running, [id]: false },
      info: { ...s.info, [id]: undefined },
    }));
  },

  startAll: async () => {
    for (const d of get().devices) {
      if (!get().running[d.id]) await get().startCamera(d.id);
    }
  },

  stopAll: async () => {
    for (const d of get().devices) {
      if (get().running[d.id]) await get().stopCamera(d.id);
    }
  },

  fetchInfo: async (id) => {
    try {
      const info = await camerasApi.info(id);
      set((s) => ({ info: { ...s.info, [id]: info } }));
    } catch {
      /* camera stopped */
    }
  },

  recordRgbd: async () => {
    const { duration } = get().frontendConfig.recording;
    get().log("Starting RGB-D recording...");
    try {
      const res = await recordingApi.start({
        duration,
        fps: get().frontendConfig.recording.fps,
      });
      get().log(`Recording started: ${res.status}`, "ok");
      set({ isRecordingRgbd: true });
      return duration;
    } catch (e) {
      get().log("Recording failed: " + e, "error");
      return null;
    }
  },
});
