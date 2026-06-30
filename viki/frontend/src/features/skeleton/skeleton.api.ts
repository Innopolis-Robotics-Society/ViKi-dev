// Skeleton estimation control endpoints. Built on shared/api/client.
// The live stream is a WebSocket handled separately in useSkeletonSocket.

import { api } from "../../shared/api/client";
import type { SkeletonStatus } from "./skeleton.types";

export const skeletonApi = {
  toggle: (enabled: boolean) =>
    api.post<{ status: string; enabled: boolean }>("/skeleton/toggle", { enabled }),
  record: (enabled: boolean) =>
    api.post<{ status: string; recording: boolean }>("/skeleton/record", { enabled }),
  status: () => api.get<SkeletonStatus>("/skeleton/status"),
};
