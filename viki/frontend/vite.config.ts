/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// FastAPI serves `GET /` -> server/static/index.html and mounts /static (StaticFiles).
// So built assets must be referenced at /static/assets/... -> base '/static/',
// and the build output goes straight into server/static so the server serves it as-is.
export default defineConfig({
  base: "/static/",
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL("../server/static", import.meta.url)),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
