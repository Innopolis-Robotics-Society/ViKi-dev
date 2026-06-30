// Shapes for the single-source server config (GET/POST /api/config). The raw
// config is a free-form JSON object edited as text; these interfaces describe
// only the keys the frontend derives camera/calibration/recording defaults from
// (see initializeFrontendConfig in the old index.html).

/** Raw config JSON as stored on the server. Loosely typed: the editor round-trips
 *  the whole object, and only the listed keys are consumed by the UI. */
export interface ServerConfig {
  DEFAULT_FPS: number;
  DEFAULT_COLOR_WIDTH: number;
  DEFAULT_COLOR_HEIGHT: number;
  DEFAULT_DEPTH_MODE: string;
  REALSENSE_RESOLUTIONS: string[];
  REALSENSE_FPS: number[];
  KINECT_RESOLUTIONS: string[];
  KINECT_FPS: number[];
  KINECT_DEPTH_MODES: string[];
  KINECT_DEPTH_MODE_MAX_FPS: Record<string, number>;
  CALIB_CHESS_BOARD_SIZE: [number, number];
  CALIB_CHESS_SQUARE_SIZE: number;
  CALIB_ARUCO_BOARD_SIZE: [number, number];
  CALIB_ARUCO_SQUARE_SIZE: number;
  CALIB_ARUCO_MARKER_SIZE: number;
  RECORDING_DURATION: number;
  RECORDING_FPS: number;
  [key: string]: unknown;
}

/** Per-camera-type options used to build the start controls (CAMERA_CONFIG). */
export interface CameraTypeConfig {
  resolutions: string[];
  defaultRes: string;
  fps: number[];
  defaultFps: number;
  depthModes: string[] | null;
  defaultDepth?: string;
  depthModeMaxFps?: Record<string, number>;
}

export interface CameraConfig {
  realsense: CameraTypeConfig;
  kinect: CameraTypeConfig;
}

/** Derived defaults consumed by the calibration and recording features. */
export interface FrontendConfig {
  calibration: {
    chess: { boardSize: [number, number]; squareSize: number };
    aruco: {
      boardSize: [number, number];
      squareSize: number;
      markerSize: number;
      defaultDict: string;
    };
  };
  recording: { duration: number; fps: number };
}
