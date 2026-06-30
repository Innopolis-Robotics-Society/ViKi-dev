// Calibration endpoints. Built on shared/api/client.

import { api } from "../../shared/api/client";
import type {
  ArucoParams,
  BoardType,
  CalibStatus,
  ChessParams,
  ExtrinsicsResult,
  IntrinsicsResult,
} from "./calibration.types";

export const calibrationApi = {
  reset: () => api.post<{ status: string }>("/calibration/reset"),
  sync: (boardType: BoardType, params: ChessParams | ArucoParams) =>
    api.post<{ status: string }>(`/calibration/sync?board_type=${boardType}`, params),
  startChess: (id: string, params: ChessParams) =>
    api.post<unknown>(`/calibration/start/${id}?mode=manual`, params),
  startAruco: (id: string, params: ArucoParams) =>
    api.post<unknown>(`/calibration/start/aruco/${id}?mode=manual`, params),
  capture: () => api.post<unknown>("/calibration/capture"),
  status: (id: string) =>
    api.get<CalibStatus>(`/calibration/status/${id}?t=${Date.now()}`),
  clear: (id: string) => api.post<{ status: string }>(`/calibration/clear/${id}`),
  intrinsics: (id: string) =>
    api.post<IntrinsicsResult>(`/calibration/intrinsics/${id}`),
  extrinsics: () => api.post<ExtrinsicsResult[]>("/calibration/extrinsics"),
};
