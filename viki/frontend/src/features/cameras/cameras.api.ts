// Cameras feature API calls. Built on shared/api/client; the only place that
// knows camera endpoint paths. MJPEG streams are returned as URLs for direct
// <img src> loading, never piped through fetch.

import { api, streamUrl } from "../../shared/api/client";
import type { CameraInfo, DevicesResponse, StartConfig } from "./cameras.types";

export const camerasApi = {
  listDevices: () => api.get<DevicesResponse>("/devices"),
  start: (id: string, cfg: StartConfig) =>
    api.post<{ status: string; device_id: string }>(`/cameras/${id}/start`, cfg),
  stop: (id: string) =>
    api.post<{ status: string; device_id: string }>(`/cameras/${id}/stop`),
  info: (id: string) => api.get<CameraInfo>(`/cameras/${id}/info`),
  colorStreamUrl: (id: string, undistort = true) =>
    streamUrl(`/cameras/${id}/stream?undistort=${undistort}&t=${Date.now()}`),
  depthStreamUrl: (id: string) => streamUrl(`/cameras/${id}/depth?t=${Date.now()}`),
};
