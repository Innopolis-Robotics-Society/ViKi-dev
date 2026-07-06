# Recordings API contract

Status: **frontend implemented, backend NOT implemented.**

The Streamlit "Recordings" tab (`viki/streamlit_app/sections/recordings.py`) is
already wired to the three endpoints below via typed helpers in
`viki/streamlit_app/api.py` (`ViKiApi.list_recordings`, `ViKiApi.get_recording`,
and the browser download URL built in the section). Until the backend serves
these, `list_recordings()` returns HTTP 404 and the tab degrades gracefully with
an `st.info` pointing at this document.

The backend team must implement these three read-only `GET` endpoints in a new
router (suggested `viki/server/routes/recordings.py`, following the existing
`APIRouter` + `Depends(get_manager)` pattern) and include it in
`viki/server/app.py`.

---

## On-disk layout (source of truth)

Recordings are written by `viki/capture/recorder.py` (`RGBDRecorder`) under a
base directory `data/videos/` (constructor arg `output_base_dir`, default
`"data/videos"`). One directory per session:

```
data/videos/
  rec_<YYYYmmdd_HHMMSS>/           # session, e.g. rec_20260705_162225
    <device_id>/                   # e.g. kinect_0, kinect_1, realsense_...
      color.mp4                    # RGB video               -> kind "color"
      depth_viz.mp4                # colorised depth video   -> kind "depth_viz"
      depth/                       # only if viki.config.RECORD_DEPTH is True
        000000.npy                 # raw uint16 depth (mm)   -> kind "depth_raw"
        000001.npy
        ...
      intrinsics.json              # per-camera intrinsics   -> kind "intrinsics"
    timestamps.json                # top-level, device_id null -> kind "meta"
    extrinsics.json                # top-level, device_id null -> kind "meta"
```

- The session **name** is exactly the directory name (`rec_<YYYYmmdd_HHMMSS>`).
- `created_at` should be derived from the timestamp embedded in the name
  (`datetime.strptime(name[4:], "%Y%m%d_%H%M%S")`) or from the directory mtime.
- `cameras` is the list of `<device_id>` subdirectories present.

### `kind` classification (backend must assign)

| file / pattern                | `kind`        | `device_id`     |
| ----------------------------- | ------------- | --------------- |
| `<device_id>/color.mp4`       | `color`       | `<device_id>`   |
| `<device_id>/depth_viz.mp4`   | `depth_viz`   | `<device_id>`   |
| `<device_id>/depth/*.npy`     | `depth_raw`   | `<device_id>`   |
| `<device_id>/intrinsics.json` | `intrinsics`  | `<device_id>`   |
| `timestamps.json`             | `meta`        | `null`          |
| `extrinsics.json`             | `meta`        | `null`          |

The frontend maps unknown `kind` values through verbatim, so adding new kinds
later is non-breaking.

> Note on raw depth: a session can contain thousands of `depth/*.npy` files.
> The backend MAY collapse them into a single synthetic tree entry (e.g.
> `path: "<device_id>/depth/", kind: "depth_raw"`) if per-file listing is too
> heavy — but the frontend currently renders one download row per returned
> entry, so returning individual `.npy` entries is fine and the default
> expectation. Whatever is returned in the tree must be downloadable via
> endpoint 3.

---

## 1. List sessions

```
GET /api/recordings
```

No query parameters.

### Response `200 application/json`

An array of session summaries, ordering not required (the frontend sorts newest
first by `created_at`, falling back to `name`).

```json
[
  {
    "name": "rec_20260705_162225",
    "created_at": "2026-07-05T16:22:25",
    "size_bytes": 12345678,
    "cameras": ["kinect_0", "kinect_1"]
  },
  {
    "name": "rec_20260704_090000",
    "created_at": "2026-07-04T09:00:00",
    "size_bytes": 987654,
    "cameras": ["realsense_0"]
  }
]
```

| field        | type        | notes                                             |
| ------------ | ----------- | ------------------------------------------------- |
| `name`       | string      | session directory name; used as `{session}` below |
| `created_at` | string      | ISO 8601; from name timestamp or dir mtime        |
| `size_bytes` | integer     | total recursive size of the session directory     |
| `cameras`    | string[]    | `<device_id>` subdirectories present              |

If `data/videos/` does not exist or is empty, return `[]` (HTTP 200), **not** a
404 — the frontend shows a friendly "No recordings yet" message for an empty
list.

---

## 2. Session file tree

```
GET /api/recordings/{session}
```

| path param | type   | notes                              |
| ---------- | ------ | ---------------------------------- |
| `session`  | string | session name from endpoint 1       |

### Response `200 application/json`

```json
{
  "name": "rec_20260705_162225",
  "files": [
    {
      "path": "kinect_0/color.mp4",
      "device_id": "kinect_0",
      "kind": "color",
      "size_bytes": 8123456
    },
    {
      "path": "kinect_0/depth_viz.mp4",
      "device_id": "kinect_0",
      "kind": "depth_viz",
      "size_bytes": 4123456
    },
    {
      "path": "kinect_0/intrinsics.json",
      "device_id": "kinect_0",
      "kind": "intrinsics",
      "size_bytes": 512
    },
    {
      "path": "timestamps.json",
      "device_id": null,
      "kind": "meta",
      "size_bytes": 45678
    },
    {
      "path": "extrinsics.json",
      "device_id": null,
      "kind": "meta",
      "size_bytes": 320
    }
  ]
}
```

| field                | type            | notes                                                    |
| -------------------- | --------------- | -------------------------------------------------------- |
| `name`               | string          | echoes the session name                                  |
| `files`              | object[]        | flat list, one entry per downloadable file               |
| `files[].path`       | string          | **relative to the session dir**, forward slashes; this exact string is passed back to endpoint 3 as `?path=` |
| `files[].device_id`  | string \| null  | owning camera, or `null` for session-level metadata      |
| `files[].kind`       | string          | one of `color`, `depth_viz`, `depth_raw`, `intrinsics`, `meta` (see table above) |
| `files[].size_bytes` | integer         | file size in bytes                                       |

### Error cases

- **404 Not Found** — `{session}` does not exist (or is not a directory) under
  `data/videos/`. The frontend shows "session no longer available".
- **400 Bad Request** — `{session}` contains path-traversal characters
  (`..`, `/`, absolute path, etc.). Reject before touching the filesystem.

---

## 3. Download one file

```
GET /api/recordings/{session}/download?path=<relative_path>
```

| param     | in    | type   | notes                                                        |
| --------- | ----- | ------ | ------------------------------------------------------------ |
| `session` | path  | string | session name from endpoint 1                                 |
| `path`    | query | string | URL-encoded, **exactly** a `files[].path` from endpoint 2, relative to the session dir (e.g. `kinect_0%2Fcolor.mp4`) |

### Response `200`

A streamed `FileResponse` for the single file, with:

```
Content-Disposition: attachment; filename="<basename>"
Content-Type: <appropriate type, e.g. video/mp4, application/json, application/octet-stream>
Content-Length: <size>
```

The frontend renders this as a direct `st.link_button` to
`browser_url("/api/recordings/{session}/download?path=<encoded>")`, so the
user's browser downloads straight from FastAPI. **Bytes must NOT be proxied
through Streamlit/Python** — videos can be hundreds of MB.

### Security — path traversal (MUST enforce)

Both `session` and `path` are attacker-controllable. The handler MUST:

1. Reject `session` or `path` containing `..` segments, `~`, or a leading `/`
   (absolute path) → **400 Bad Request**.
2. Resolve the final target and confirm it stays inside the intended session
   directory, e.g.:

   ```python
   base = (RECORDINGS_ROOT / session).resolve()
   target = (base / path).resolve()
   if not str(target).startswith(str(base) + os.sep) and target != base:
       raise HTTPException(status_code=400, detail="Invalid path")
   ```

3. Confirm the resolved target exists and is a **file** → else **404 Not
   Found**.

Do not follow symlinks out of the recordings root. When in doubt, 400.

### Error cases

- **400 Bad Request** — missing `path`, path traversal attempt, or resolved
  target escapes the session directory.
- **404 Not Found** — unknown `{session}`, or `path` points at a non-existent
  file / a directory.

---

## Frontend wiring reference (already in place)

| Contract endpoint                              | Frontend caller                                                        |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| `GET /api/recordings`                          | `ViKiApi.list_recordings()` in `viki/streamlit_app/api.py`             |
| `GET /api/recordings/{session}`                | `ViKiApi.get_recording(session)` in `viki/streamlit_app/api.py`       |
| `GET /api/recordings/{session}/download?path=` | `st.link_button` → `browser_url(...)` in `sections/recordings.py`      |

The tab is registered in `viki/streamlit_app/app.py` as the "Recordings" tab.
When the backend returns 404 for endpoint 1, the tab shows an `st.info` message
referencing this document instead of erroring.
