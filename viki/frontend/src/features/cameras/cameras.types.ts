// Shapes for the cameras API, verified against viki/server/routes/cameras.py
// and viki/capture/manager.py.

export type CameraType = "realsense" | "kinect" | "web_camera";

export interface Device {
  id: string;
  type: CameraType;
}

/** GET /api/devices -> CameraManager.list_devices() */
export interface DevicesResponse {
  realsense: string[];
  kinect: string[];
  web_camera: string[];
  active: string[];
  realsense_error?: string;
  kinect_error?: string;
  web_camera_error?: string;
}

export interface ColorIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

/** GET /api/cameras/{id}/info -> CameraManager.get_info() */
export interface CameraInfo {
  device_id: string;
  running: boolean;
  has_frame: boolean;
  /** [height, width, channels] */
  color_shape?: [number, number, number];
  /** [height, width] */
  depth_shape?: [number, number];
  timestamp_us?: number;
  color_intrinsics?: ColorIntrinsics;
}

/** Body for POST /api/cameras/{id}/start (subset of StartRequest). */
export interface StartConfig {
  color_width: number;
  color_height: number;
  fps: number;
  depth_mode: string;
}
