// RGB-D recording endpoint (POST /api/record/start). Lives with cameras since
// recording is driven from the camera toolbar.

import { api } from "../../shared/api/client";

export interface RecordRequest {
  duration: number;
  fps: number;
}

export const recordingApi = {
  start: (req: RecordRequest) =>
    api.post<{ status: string; duration: number; fps: number }>("/record/start", req),
};
