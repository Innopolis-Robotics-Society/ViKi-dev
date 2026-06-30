// Shapes for the calibration endpoints, verified against
// viki/server/routes/calibration.py and routes/models.py.

export type BoardType = "chess" | "aruco";

export interface ChessParams {
  board_size: [number, number];
  square_size: number;
}

export interface ArucoParams extends ChessParams {
  marker_size: number;
  aruco_dict: string;
}

/** GET /api/calibration/status/{id} */
export interface CalibStatus {
  samples_count: number;
  started: boolean;
}

/** POST /api/calibration/extrinsics -> ExtrinsicsResponse[] */
export interface ExtrinsicsResult {
  device_id: string;
  rvec: number[];
  tvec: number[];
}

/** Solved extrinsics for one device (rotation/translation vectors). */
export interface Extrinsics {
  rvec: number[];
  tvec: number[];
}

/** POST/GET /api/calibration/intrinsics/{id} -> IntrinsicsResponse */
export interface IntrinsicsResult {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  dist_coeffs: number[];
}

// OpenCV ArUco dictionary names offered in the board params (ported verbatim).
export const ARUCO_DICTS = [
  "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
  "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
  "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
  "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
  "DICT_ARUCO_ORIGINAL",
] as const;
