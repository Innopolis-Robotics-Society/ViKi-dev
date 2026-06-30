// Config feature slice: the single-source server config plus the camera /
// calibration / recording defaults derived from it (ports initializeFrontendConfig,
// loadConfig, saveConfig, resetConfig, restartServer, toggleConfig, toggleConfigHelp).

import type { StateCreator } from "zustand";
import type { RootState } from "../../shared/store/store";
import { configApi } from "./config.api";
import type {
  CameraConfig,
  FrontendConfig,
  ServerConfig,
} from "./config.types";

// Hardcoded for the UI, mapping to aruco dict ID 4 (see old initializeFrontendConfig).
const DEFAULT_ARUCO_DICT = "DICT_5X5_50";

// Minimal defaults used if the initial /api/config fetch fails (ported from init()).
const FALLBACK_CONFIG: ServerConfig = {
  DEFAULT_FPS: 15,
  DEFAULT_COLOR_WIDTH: 1280,
  DEFAULT_COLOR_HEIGHT: 720,
  DEFAULT_DEPTH_MODE: "NFOV_UNBINNED",
  CALIB_CHESS_BOARD_SIZE: [8, 6],
  CALIB_CHESS_SQUARE_SIZE: 0.025,
  CALIB_ARUCO_BOARD_SIZE: [10, 8],
  CALIB_ARUCO_SQUARE_SIZE: 0.05,
  CALIB_ARUCO_MARKER_SIZE: 0.035,
  RECORDING_DURATION: 10.0,
  RECORDING_FPS: 15,
  REALSENSE_RESOLUTIONS: ["640x480", "1280x720", "1920x1080"],
  REALSENSE_FPS: [15, 30],
  KINECT_RESOLUTIONS: ["1280x720", "1920x1080", "2048x1536"],
  KINECT_FPS: [5, 15, 30],
  KINECT_DEPTH_MODES: ["NFOV_UNBINNED", "NFOV_2X2BINNED", "WFOV_UNBINNED", "WFOV_2X2BINNED"],
  KINECT_DEPTH_MODE_MAX_FPS: { WFOV_UNBINNED: 15 },
};

function deriveCameraConfig(c: ServerConfig): CameraConfig {
  const defaultRes = `${c.DEFAULT_COLOR_WIDTH}x${c.DEFAULT_COLOR_HEIGHT}`;
  return {
    realsense: {
      resolutions: c.REALSENSE_RESOLUTIONS,
      defaultRes,
      fps: c.REALSENSE_FPS,
      defaultFps: c.DEFAULT_FPS,
      depthModes: null,
    },
    kinect: {
      resolutions: c.KINECT_RESOLUTIONS,
      defaultRes,
      fps: c.KINECT_FPS,
      defaultFps: c.DEFAULT_FPS,
      depthModes: c.KINECT_DEPTH_MODES,
      defaultDepth: c.DEFAULT_DEPTH_MODE,
      depthModeMaxFps: c.KINECT_DEPTH_MODE_MAX_FPS,
    },
  };
}

function deriveFrontendConfig(c: ServerConfig): FrontendConfig {
  return {
    calibration: {
      chess: {
        boardSize: c.CALIB_CHESS_BOARD_SIZE,
        squareSize: c.CALIB_CHESS_SQUARE_SIZE,
      },
      aruco: {
        boardSize: c.CALIB_ARUCO_BOARD_SIZE,
        squareSize: c.CALIB_ARUCO_SQUARE_SIZE,
        markerSize: c.CALIB_ARUCO_MARKER_SIZE,
        defaultDict: DEFAULT_ARUCO_DICT,
      },
    },
    recording: {
      duration: c.RECORDING_DURATION,
      fps: c.RECORDING_FPS,
    },
  };
}

export interface ConfigSlice {
  configOpen: boolean;
  helpOpen: boolean;
  configText: string;
  cameraConfig: CameraConfig;
  frontendConfig: FrontendConfig;
  /** Load config on startup and derive camera/calibration/recording defaults. */
  initConfig: () => Promise<void>;
  loadConfig: () => Promise<void>;
  saveConfig: () => Promise<void>;
  resetConfig: () => Promise<void>;
  restartServer: () => Promise<void>;
  toggleConfig: () => void;
  toggleConfigHelp: () => void;
  setConfigText: (text: string) => void;
}

export const createConfigSlice: StateCreator<RootState, [], [], ConfigSlice> = (
  set,
  get,
) => ({
  configOpen: false,
  helpOpen: false,
  configText: "",
  cameraConfig: deriveCameraConfig(FALLBACK_CONFIG),
  frontendConfig: deriveFrontendConfig(FALLBACK_CONFIG),

  initConfig: async () => {
    try {
      const config = await configApi.get();
      set({
        cameraConfig: deriveCameraConfig(config),
        frontendConfig: deriveFrontendConfig(config),
      });
      get().log("Configuration loaded from server", "ok");
    } catch (e) {
      get().log("Failed to load initial config: " + e, "error");
      set({
        cameraConfig: deriveCameraConfig(FALLBACK_CONFIG),
        frontendConfig: deriveFrontendConfig(FALLBACK_CONFIG),
      });
    }
  },

  loadConfig: async () => {
    try {
      const config = await configApi.get();
      set({ configText: JSON.stringify(config, null, 2) });
    } catch (e) {
      get().log("Failed to load config: " + e, "error");
    }
  },

  saveConfig: async () => {
    try {
      const config = JSON.parse(get().configText) as ServerConfig;
      await configApi.save(config);
      get().log("Configuration saved successfully", "ok");
    } catch (e) {
      get().log("Failed to save config: " + e, "error");
    }
  },

  resetConfig: async () => {
    if (
      !window.confirm(
        "Reset all settings to defaults? This will overwrite your current configuration.",
      )
    )
      return;
    try {
      await configApi.reset();
      await get().loadConfig();
      get().log("Configuration reset to defaults", "ok");
    } catch (e) {
      get().log("Failed to reset config: " + e, "error");
    }
  },

  restartServer: async () => {
    if (
      !window.confirm("Restart the server? You will be disconnected for a few seconds.")
    )
      return;
    try {
      await configApi.restart();
      get().log("Restarting server... please wait.", "ok");
    } catch (e) {
      get().log("Restart request failed: " + e, "error");
    }
  },

  toggleConfig: () => {
    const open = !get().configOpen;
    set({ configOpen: open });
    if (open) void get().loadConfig();
  },

  toggleConfigHelp: () => set((s) => ({ helpOpen: !s.helpOpen })),

  setConfigText: (text) => set({ configText: text }),
});
