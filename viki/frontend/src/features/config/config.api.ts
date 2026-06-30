// Config + server lifecycle endpoints. Built on shared/api/client.

import { api } from "../../shared/api/client";
import type { ServerConfig } from "./config.types";

export const configApi = {
  get: () => api.get<ServerConfig>("/config"),
  save: (config: ServerConfig) => api.post<{ status: string }>("/config", config),
  reset: () => api.post<{ status: string }>("/config/reset"),
  restart: () => api.post<unknown>("/restart"),
};
