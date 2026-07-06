# Frontend migration: React SPA → Streamlit

**Date:** 2026-07-03
**Status:** Approved, in implementation
**Supersedes the frontend of:** `2026-06-30-frontend-react-refactor-design.md` (React app is removed)

## Requirement

Migrate the web frontend from the React + Vite + TS SPA to **Streamlit** (Python).
Streamlit becomes THE frontend; the React app (`viki/frontend/`) and the static build
(`viki/server/static/`) are removed.

## Architecture

Streamlit cannot replace the camera/calibration/skeleton logic — that lives in FastAPI
(`viki/server/`, `viki/capture/`). So:

- **Streamlit is a new frontend process** that talks to the existing FastAPI backend over
  HTTP (`requests`) for all control/config/calibration/skeleton actions.
- FastAPI keeps all `/api/*` routes, MJPEG stream endpoints, and the skeleton WebSocket
  **unchanged**.
- FastAPI stops serving the SPA: the `/static` mount and `GET /` index are removed; `/`
  becomes a redirect to the Streamlit UI.
- Two processes: FastAPI on `:8000`, Streamlit on `:8501`. Compose runs both;
  `network_mode: host` means both bind directly on the host, and `localhost:8000` is
  reachable from the Streamlit container AND the user's browser.

### Decisions (locked)
- **3D skeleton:** embedded HTML/JS component (`st.components.v1.html`) running the same
  WebSocket + `<canvas>` render loop as the React app — live ~30fps. Ported from the
  React `skeleton3d.ts` / `overlay.ts` / `useSkeletonSocket.ts` into self-contained
  vanilla JS in an HTML string.
- **MJPEG camera streams:** embedded `<img src=".../api/cameras/{id}/stream">` via
  `st.components.v1.html`. The browser pulls MJPEG directly from FastAPI (same as React) —
  full live-video parity, no piping through Streamlit reruns.
- **React app is removed** entirely after the port.
- **Polling** (status, sample counts, skeleton detected-state) uses Streamlit's built-in
  `@st.fragment(run_every=...)` — no extra autorefresh dependency.

## Backend base URLs

- `VIKI_BACKEND_URL` (server-side `requests`, default `http://localhost:8000`).
- `VIKI_BROWSER_BACKEND_URL` (for the embedded `<img>`/WebSocket, i.e. what the browser
  hits, default `http://localhost:8000`; the WS URL is derived by swapping the scheme).
  Both default the same because compose uses host networking.

## Target structure

```
viki/streamlit_app/
  app.py               # entry: st.set_page_config, topbar (server dot, scan,
                       #   start/stop all, record), st.tabs(Cameras/Calibration/
                       #   Skeleton/Config)
  api.py               # ViKiApi: thin requests wrapper over /api/* (typed helpers,
                       #   error handling); the only module that knows endpoint paths
  settings.py          # BACKEND_URL / BROWSER_BACKEND_URL / WS url helpers from env
  embeds.py            # HTML builders: mjpeg_img(url), skeleton_canvas(ws_url, ...)
  sections/
    cameras.py         # device cards: embedded color+depth MJPEG, res/fps/depth
                       #   selects, start/stop, info
    calibration.py     # board params (chess/aruco), per-device preview + sample
                       #   count (fragment poll), capture, intrinsics, extrinsics, clear
    skeleton.py        # start/stop estimation + record, embedded canvas component,
                       #   joint table + detected status (fragment poll), view toggle
    config_panel.py    # JSON editor + load/save/reset/restart + help expander
```

Session state (`st.session_state`) holds the discovered device list, per-card start
config, and toggles — Streamlit's equivalent of the Zustand store.

## API surface (unchanged backend — verify shapes against React `*.api.ts` / `*.types.ts`
and `viki/server/routes/*.py`)

Cameras: `GET /api/devices`, `POST /api/cameras/{id}/start` (body: color_width,
color_height, fps, depth_mode), `POST /api/cameras/{id}/stop`, `GET /api/cameras/{id}/info`,
MJPEG `GET /api/cameras/{id}/stream?undistort=&t=`, depth `GET /api/cameras/{id}/depth`.
Config: `GET/POST /api/config`, `POST /api/config/reset`, `POST /api/restart`.
Calibration: `POST /api/calibration/start/{id}`, `.../start/aruco/{id}`, `.../capture`,
`.../clear`, `.../clear/{id}`, `.../reset`, `.../sync`, `GET .../status/{id}`,
`GET .../samples_count/{id}`, `GET/POST .../intrinsics/{id}`, `POST .../extrinsics`.
Skeleton: `POST /api/skeleton/toggle`, `POST /api/skeleton/record`,
`GET /api/skeleton/status`, WebSocket `/api/skeleton/stream`.
Recording: `POST /api/record/start`.

## Docker

Add a `streamlit` service to `docker-compose.yml` reusing the same image/mounts/host
network, command
`streamlit run viki/streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0
 --server.headless true`. Add `streamlit` and `requests` to `pyproject.toml` deps so the
image ships them. The `viki` (FastAPI) service is otherwise unchanged.

## Testing (no cameras connected)

`docker compose up --build`, then verify:
- FastAPI up on :8000 (`GET /api/devices` returns JSON with empty realsense/kinect lists).
- Streamlit up on :8501, page renders, tabs present, "No cameras detected" empty state,
  no exception in logs on load or when clicking Scan/tabs.
- `GET /` on :8000 redirects to Streamlit.
Full camera/stream/skeleton behavior needs hardware and is out of scope for this test.

## Removal

Delete `viki/frontend/` and `viki/server/static/`; drop the frontend-build script
(`scripts/build_frontend.sh`) and revert the React-specific notes in README/CLAUDE.md to
describe the Streamlit UI instead.
