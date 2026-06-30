// Calibration feature slice: board parameters, per-device sample counts, the
// computed extrinsics (also consumed by the skeleton 3D view), and the session
// lifecycle. Ports startCalibrationSession/syncBoardParameters/captureSample/
// clearCalibration/intrinsicsCalibration/extrinsicsCalibration/updateCalibStatus/
// resetCalibration/toggleBoardFields/toggleCalibration.

import type { StateCreator } from "zustand";
import type { RootState } from "../../shared/store/store";
import { calibrationApi } from "./calibration.api";
import type {
  ArucoParams,
  BoardType,
  ChessParams,
  Extrinsics,
} from "./calibration.types";

export type { Extrinsics };

export interface CalibrationSlice {
  calibOpen: boolean;
  boardType: BoardType;
  boardWidth: number;
  boardHeight: number;
  squareSize: number;
  markerSize: number;
  arucoDict: string;
  sampleCounts: Record<string, string>;
  extrinsics: Record<string, Extrinsics>;

  toggleCalibration: () => void;
  initBoardFields: () => void;
  setBoardType: (type: BoardType) => Promise<void>;
  setBoardField: (
    field: "boardWidth" | "boardHeight" | "squareSize" | "markerSize" | "arucoDict",
    value: number | string,
  ) => Promise<void>;
  syncBoardParams: () => Promise<void>;
  startSession: (reset: boolean) => Promise<void>;
  captureSample: () => Promise<void>;
  clearSamples: () => Promise<void>;
  intrinsicsCalibration: () => Promise<void>;
  extrinsicsCalibration: () => Promise<void>;
  updateStatus: () => Promise<void>;
  resetCalibration: () => Promise<void>;
}

// Build the request body for the current board parameters.
function buildParams(s: CalibrationSlice): ChessParams | ArucoParams {
  const base: ChessParams = {
    board_size: [s.boardWidth, s.boardHeight],
    square_size: s.squareSize,
  };
  if (s.boardType === "aruco") {
    return { ...base, marker_size: s.markerSize, aruco_dict: s.arucoDict };
  }
  return base;
}

export const createCalibrationSlice: StateCreator<
  RootState,
  [],
  [],
  CalibrationSlice
> = (set, get) => ({
  calibOpen: false,
  boardType: "chess",
  boardWidth: 8,
  boardHeight: 6,
  squareSize: 0.05,
  markerSize: 0.035,
  arucoDict: "DICT_5X5_50",
  sampleCounts: {},
  extrinsics: {},

  toggleCalibration: () => set((s) => ({ calibOpen: !s.calibOpen })),

  // Populate board fields from the loaded config (ports toggleBoardFields' field set).
  initBoardFields: () => {
    const calib = get().frontendConfig.calibration;
    const t = get().boardType;
    if (t === "chess") {
      set({
        boardWidth: calib.chess.boardSize[0],
        boardHeight: calib.chess.boardSize[1],
        squareSize: calib.chess.squareSize,
        arucoDict: calib.aruco.defaultDict,
      });
    } else {
      set({
        boardWidth: calib.aruco.boardSize[0],
        boardHeight: calib.aruco.boardSize[1],
        squareSize: calib.aruco.squareSize,
        markerSize: calib.aruco.markerSize,
        arucoDict: calib.aruco.defaultDict,
      });
    }
  },

  setBoardType: async (type) => {
    set({ boardType: type });
    get().initBoardFields();
    await get().syncBoardParams();
    await get().startSession(true);
  },

  setBoardField: async (field, value) => {
    set({ [field]: value } as Partial<CalibrationSlice>);
    await get().syncBoardParams();
  },

  syncBoardParams: async () => {
    try {
      await calibrationApi.sync(get().boardType, buildParams(get()));
    } catch (e) {
      get().log("Parameter sync failed: " + e, "error");
    }
  },

  startSession: async (reset) => {
    if (reset) await get().resetCalibration();
    const boardType = get().boardType;
    get().log(`Starting calibration session (${boardType} board)...`);
    const params = buildParams(get());
    for (const d of get().devices) {
      try {
        if (boardType === "chess") {
          await calibrationApi.startChess(d.id, params as ChessParams);
        } else {
          await calibrationApi.startAruco(d.id, params as ArucoParams);
        }
      } catch (e) {
        get().log(`Calibration session start failed: ${e}`, "error");
      }
    }
    await get().updateStatus();
  },

  captureSample: async () => {
    get().log("Capturing calibration sample...");
    try {
      await get().updateStatus();
      await calibrationApi.capture();
      get().log("Sample captured successfully", "ok");
      await get().updateStatus();
    } catch (e) {
      get().log(`Capture failed: ${e}`, "error");
    }
  },

  clearSamples: async () => {
    for (const d of get().devices) {
      await calibrationApi.clear(d.id);
    }
    get().log("Calibration samples cleared");
    await get().updateStatus();
  },

  intrinsicsCalibration: async () => {
    get().log("Intrinsics calibration started...");
    for (const d of get().devices) {
      try {
        await calibrationApi.intrinsics(d.id);
        get().log(`Intrinsics calibration for ${d.id} successful`, "ok");
      } catch (e) {
        get().log(`Intrinsics calibration for ${d.id} failed: ${e}`, "error");
      }
    }
  },

  extrinsicsCalibration: async () => {
    get().log("Extrinsics calibration started...");
    try {
      const results = await calibrationApi.extrinsics();
      const extrinsics: Record<string, Extrinsics> = {};
      results.forEach((extr) => {
        extrinsics[extr.device_id] = { rvec: extr.rvec, tvec: extr.tvec };
      });
      set({ extrinsics });
      get().log("Extrinsics calibration successful", "ok");
    } catch (e) {
      get().log(`Extrinsics calibration failed: ${e}`, "error");
    }
  },

  updateStatus: async () => {
    await Promise.all(
      get().devices.map(async (d) => {
        try {
          const data = await calibrationApi.status(d.id);
          const text = data.started ? `(${data.samples_count} samples)` : "(0 samples)";
          set((s) => ({ sampleCounts: { ...s.sampleCounts, [d.id]: text } }));
        } catch {
          set((s) => ({ sampleCounts: { ...s.sampleCounts, [d.id]: "(error)" } }));
          get().log(`Status check for ${d.id} failed`, "error");
        }
      }),
    );
  },

  resetCalibration: async () => {
    await calibrationApi.reset();
  },
});
