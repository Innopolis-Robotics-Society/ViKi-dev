"""viki.streamlit_app.embeds
-----------------------------
HTML builders returned as strings for ``st.components.v1.html``.

- :func:`mjpeg_img` -- an ``<img>`` whose ``src`` points at a FastAPI MJPEG
  endpoint. The browser pulls the stream itself; bytes never flow through
  Python/requests.
- :func:`skeleton_canvas` -- a fully self-contained ``<canvas>`` + ``<script>``
  that opens the ``/api/skeleton/stream`` WebSocket and runs the same render
  loop as the React app (``skeleton3d.ts`` / ``overlay.ts`` /
  ``useSkeletonSocket.ts``), ported to vanilla JS. No external JS/CDN/imports.
"""

from __future__ import annotations

import html
import json
from typing import Any


def mjpeg_img(url: str, height: int = 320, label: str | None = None) -> str:
    """An ``<img>`` element that loads an MJPEG stream directly in the browser."""
    safe_url = html.escape(url, quote=True)
    label_html = ""
    if label:
        label_html = (
            f'<span style="position:absolute;top:6px;left:8px;font:600 10px monospace;'
            f'letter-spacing:.08em;color:#cfd3ff;background:rgba(10,12,24,.6);'
            f'padding:2px 6px;border-radius:4px;">{html.escape(label)}</span>'
        )
    return f"""
<div style="position:relative;width:100%;background:#0b0d18;border-radius:8px;
            overflow:hidden;display:flex;align-items:center;justify-content:center;
            min-height:{height}px;">
  <img src="{safe_url}" alt="stream"
       style="width:100%;height:{height}px;object-fit:contain;display:block;"
       onerror="this.style.opacity=0.15;" />
  {label_html}
</div>
""".strip()


# --- skeleton constants, ported verbatim from skeleton.constants.ts ---------
SKEL_NAMES = [
    "Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip",
    "Elbow", "Shoulder",
]
# Drawn bones: Shoulder(22)-Elbow(21)-Wrist(0).
SKEL_CONNS = [[22, 21], [21, 0]]
# Landmarks rendered as joints.
SKEL_DRAW_INDICES = [0, 21, 22]


def skeleton_canvas(
    ws_url: str,
    devices: list[dict],
    extrinsics: dict[str, Any] | None = None,
    viz_cam: str | None = None,
    view_mode: str = "projections",
    width: int | None = None,
    height: int = 360,
) -> str:
    """Self-contained skeleton visualiser.

    Opens ``ws_url`` (the ``/api/skeleton/stream`` WebSocket), decodes each
    ``{ts, landmarks, detections}`` frame, and draws the fused 3D pose exactly as
    ``drawSkeleton3D`` does (projections / isometric / camera views, EMA-smoothed
    center, 1 m spatial filter). Also renders the per-device detected status and
    the live joint table in the browser -- that data only exists on the WebSocket,
    so it cannot be fetched server-side without changing the backend.

    Everything is inline vanilla JS; there are no imports/CDN.
    """
    cfg = {
        "wsUrl": ws_url,
        "deviceIds": [d["id"] for d in devices],
        "extrinsics": extrinsics or {},
        "vizCam": viz_cam if viz_cam is not None else (devices[0]["id"] if devices else None),
        "viewMode": view_mode,
        "skelNames": SKEL_NAMES,
        "skelConns": SKEL_CONNS,
        "skelDrawIndices": SKEL_DRAW_INDICES,
    }
    cfg_json = json.dumps(cfg)
    width_css = f"{width}px" if width else "100%"

    # Template uses __TOKENS__ so JS braces don't collide with str.format/f-strings.
    template = r"""
<div id="viki-skel-root" style="width:__WIDTH__;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#e6e8f2;">
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;">
    <button id="viki-view-btn"
      style="background:#20233a;color:#e6e8f2;border:1px solid #33375a;border-radius:6px;
             padding:6px 10px;cursor:pointer;font-size:12px;">View: Projections</button>
    <label style="font-size:12px;color:#9aa0c0;">Visualize camera</label>
    <select id="viki-cam-select"
      style="background:#20233a;color:#e6e8f2;border:1px solid #33375a;border-radius:6px;
             padding:5px 8px;font-size:12px;"></select>
    <span id="viki-ws-state" style="font-size:11px;color:#7f86ad;margin-left:auto;">connecting...</span>
  </div>
  <canvas id="viki-skel-canvas"
    style="width:100%;height:__HEIGHT__px;background:#0b0d18;border-radius:8px;display:block;"></canvas>
  <div id="viki-detections" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;font-size:12px;"></div>
  <div style="max-height:220px;overflow:auto;margin-top:8px;border:1px solid #23263f;border-radius:8px;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="position:sticky;top:0;background:#161a2e;color:#9aa0c0;text-align:left;">
          <th style="padding:6px 10px;">Joint</th><th style="padding:6px 10px;">X</th>
          <th style="padding:6px 10px;">Y</th><th style="padding:6px 10px;">Z</th>
        </tr>
      </thead>
      <tbody id="viki-joint-body"></tbody>
    </table>
  </div>
</div>
<script>
(function() {
  const CFG = __CFG_JSON__;
  const VIEW_MODES = ["projections", "isometric", "camera"];
  const VIEW_LABELS = {projections: "Projections", isometric: "Isometric", camera: "Camera"};
  const SKEL_CONNS = CFG.skelConns;
  const DRAW_IDX = CFG.skelDrawIndices;
  const NAMES = CFG.skelNames;

  let viewMode = CFG.viewMode || "projections";
  let vizCam = CFG.vizCam;
  let smoothedCenter = null;        // EMA-smoothed center, carried across frames
  let latest = { landmarks: {}, detections: {} };

  const canvas = document.getElementById("viki-skel-canvas");
  const ctx = canvas.getContext("2d");
  const viewBtn = document.getElementById("viki-view-btn");
  const camSelect = document.getElementById("viki-cam-select");
  const detDiv = document.getElementById("viki-detections");
  const jointBody = document.getElementById("viki-joint-body");
  const wsState = document.getElementById("viki-ws-state");

  // Populate camera selector.
  (CFG.deviceIds || []).forEach(function(id) {
    const o = document.createElement("option");
    o.value = id; o.textContent = id;
    if (id === vizCam) o.selected = true;
    camSelect.appendChild(o);
  });
  camSelect.addEventListener("change", function() {
    vizCam = camSelect.value;
    smoothedCenter = null;          // reset smoothing on camera switch
  });
  viewBtn.textContent = "View: " + (VIEW_LABELS[viewMode] || viewMode);
  viewBtn.addEventListener("click", function() {
    const i = VIEW_MODES.indexOf(viewMode);
    viewMode = VIEW_MODES[(i + 1) % VIEW_MODES.length];
    viewBtn.textContent = "View: " + (VIEW_LABELS[viewMode] || viewMode);
  });

  // Rodrigues rotation vector -> 3x3 matrix (ported verbatim).
  function rodrigues(rvec) {
    const theta = Math.sqrt(rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2]);
    if (theta < 1e-6) return [[1,0,0],[0,1,0],[0,0,1]];
    const ux = rvec[0]/theta, uy = rvec[1]/theta, uz = rvec[2]/theta;
    const c = Math.cos(theta), s = Math.sin(theta), k = 1 - c;
    return [
      [c + ux*ux*k, ux*uy*k - uz*s, ux*uz*k + uy*s],
      [uy*ux*k + uz*s, c + uy*uy*k, uy*uz*k - ux*s],
      [uz*ux*k - uy*s, uz*uy*k + ux*s, c + uz*uz*k],
    ];
  }

  // landmarks dict {index: [x,y,z]|null} -> dense array 0..NAMES.length-1
  function toArray(landmarks) {
    const arr = [];
    for (let i = 0; i < NAMES.length; i++) {
      const p = landmarks ? landmarks[i] : null;
      arr[i] = (p && p.length === 3) ? p : null;
    }
    return arr;
  }

  function drawSkeleton3D(landmarks) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(cssW * dpr));
    canvas.height = Math.max(1, Math.floor(cssH * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = cssW, H = cssH;

    ctx.clearRect(0, 0, W, H);
    if (!landmarks || landmarks.length === 0) return;

    const plausible = landmarks.filter(function(p){ return p && !isNaN(p[0]) && p[2] > 0.1; });
    if (plausible.length === 0) return;

    const rawCenter = {
      x: plausible.reduce(function(s,p){return s+p[0];},0)/plausible.length,
      y: plausible.reduce(function(s,p){return s+p[1];},0)/plausible.length,
      z: plausible.reduce(function(s,p){return s+p[2];},0)/plausible.length,
    };
    if (!smoothedCenter) {
      smoothedCenter = {x: rawCenter.x, y: rawCenter.y, z: rawCenter.z};
    } else {
      const a = 0.1;
      smoothedCenter.x += a*(rawCenter.x - smoothedCenter.x);
      smoothedCenter.y += a*(rawCenter.y - smoothedCenter.y);
      smoothedCenter.z += a*(rawCenter.z - smoothedCenter.z);
    }

    // Shoulder (index 22) is the moving origin for the view.
    let projCenter = {x:0, y:0, z:0};
    const shoulder = landmarks[22];
    if (shoulder && !isNaN(shoulder[0])) {
      projCenter = {x: shoulder[0], y: shoulder[1], z: shoulder[2]};
    } else if (smoothedCenter) {
      projCenter = {x: smoothedCenter.x, y: smoothedCenter.y, z: smoothedCenter.z};
    }
    const fc = smoothedCenter;

    function isValid(p) {
      if (!p || isNaN(p[0]) || p[2] <= 0.1) return false;
      const d = Math.sqrt(Math.pow(p[0]-fc.x,2) + Math.pow(p[1]-fc.y,2) + Math.pow(p[2]-fc.z,2));
      return d < 1.0;
    }

    const validPoints = landmarks.filter(isValid);
    if (validPoints.length === 0) return;

    const scale = W / 5;

    if (viewMode === "projections") {
      const viewW = W / 3, viewH = H;
      const views = [
        {label:"TOP (X-Z)", proj:function(p){return {x:-(p[0]-projCenter.x), y:p[2]-projCenter.z};}, offset:0},
        {label:"FRONT (X-Y)", proj:function(p){return {x:-(p[0]-projCenter.x), y:p[1]-projCenter.y};}, offset:viewW},
        {label:"SIDE (Z-Y)", proj:function(p){return {x:p[2]-projCenter.z, y:p[1]-projCenter.y};}, offset:viewW*2},
      ];
      views.forEach(function(view){
        const cX = view.offset + viewW/2, cY = viewH/2;
        ctx.fillStyle = "#6b6b80"; ctx.font = "10px monospace";
        ctx.fillText(view.label, view.offset + 10, 20);
        ctx.strokeStyle = "#5b7fff"; ctx.lineWidth = 2; ctx.fillStyle = "#fff";
        SKEL_CONNS.forEach(function(pair){
          const a = pair[0], b = pair[1];
          if (isValid(landmarks[a]) && isValid(landmarks[b])) {
            const p1 = view.proj(landmarks[a]), p2 = view.proj(landmarks[b]);
            ctx.beginPath();
            ctx.moveTo(cX + p1.x*scale, cY - p1.y*scale);
            ctx.lineTo(cX + p2.x*scale, cY - p2.y*scale);
            ctx.stroke();
          }
        });
        landmarks.forEach(function(p,i){
          if (DRAW_IDX.indexOf(i) < 0 || !isValid(p)) return;
          const pr = view.proj(p);
          ctx.beginPath();
          ctx.arc(cX + pr.x*scale, cY - pr.y*scale, 3, 0, Math.PI*2);
          ctx.fill();
        });
      });
    } else if (viewMode === "isometric") {
      const cX = W/2, cY = H/2;
      function projectIso(p) {
        const dx = -(p[0]-projCenter.x), dy = p[1]-projCenter.y, dz = p[2]-projCenter.z;
        const x1 = dx*Math.cos(Math.PI/4) + dz*Math.sin(Math.PI/4);
        const z1 = -dx*Math.sin(Math.PI/4) + dz*Math.cos(Math.PI/4);
        const y2 = dy*Math.cos(Math.PI/6) - z1*Math.sin(Math.PI/6);
        return {x:x1, y:y2};
      }
      const axisLen = 0.2;
      const projOrigin = projectIso([0,0,0]);
      const axes = [
        {p:[axisLen,0,0], color:"#ff5b5b", label:"X"},
        {p:[0,axisLen,0], color:"#5bff5b", label:"Y"},
        {p:[0,0,axisLen], color:"#5b7fff", label:"Z"},
      ];
      axes.forEach(function(ax){
        const pEnd = projectIso(ax.p);
        ctx.strokeStyle = ax.color; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cX + projOrigin.x*scale, cY - projOrigin.y*scale);
        ctx.lineTo(cX + pEnd.x*scale, cY - pEnd.y*scale);
        ctx.stroke();
        ctx.fillStyle = ax.color; ctx.font = "10px monospace";
        ctx.fillText(ax.label, cX + pEnd.x*scale + 2, cY - pEnd.y*scale);
      });
      ctx.strokeStyle = "#5b7fff"; ctx.lineWidth = 2; ctx.fillStyle = "#fff";
      SKEL_CONNS.forEach(function(pair){
        const a = pair[0], b = pair[1];
        if (isValid(landmarks[a]) && isValid(landmarks[b])) {
          const p1 = projectIso(landmarks[a]), p2 = projectIso(landmarks[b]);
          ctx.beginPath();
          ctx.moveTo(cX + p1.x*scale, cY - p1.y*scale);
          ctx.lineTo(cX + p2.x*scale, cY - p2.y*scale);
          ctx.stroke();
        }
      });
      landmarks.forEach(function(p,i){
        if (DRAW_IDX.indexOf(i) < 0 || !isValid(p)) return;
        const pr = projectIso(p);
        ctx.beginPath();
        ctx.arc(cX + pr.x*scale, cY - pr.y*scale, 3, 0, Math.PI*2);
        ctx.fill();
      });
    } else if (viewMode === "camera") {
      const extr = vizCam ? CFG.extrinsics[vizCam] : null;
      if (!extr) {
        ctx.fillStyle = "#fff"; ctx.textAlign = "center";
        ctx.fillText("No extrinsics calibrated for this camera", W/2, H/2);
        ctx.textAlign = "left";
        return;
      }
      const R = rodrigues(extr.rvec), t = extr.tvec;
      const cX = W/2, cY = H/2;
      function projectCam(p) {
        const xc = R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2] + t[0];
        const yc = R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2] + t[1];
        const zc = R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2] + t[2];
        if (zc <= 0.1) return null;
        return {x: xc/zc, y: yc/zc};
      }
      ctx.strokeStyle = "#5b7fff"; ctx.lineWidth = 2; ctx.fillStyle = "#fff";
      SKEL_CONNS.forEach(function(pair){
        const a = pair[0], b = pair[1];
        if (isValid(landmarks[a]) && isValid(landmarks[b])) {
          const p1 = projectCam(landmarks[a]), p2 = projectCam(landmarks[b]);
          if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(cX + p1.x*scale, cY - p1.y*scale);
            ctx.lineTo(cX + p2.x*scale, cY - p2.y*scale);
            ctx.stroke();
          }
        }
      });
      landmarks.forEach(function(p,i){
        if (DRAW_IDX.indexOf(i) < 0 || !isValid(p)) return;
        const pr = projectCam(p);
        if (pr) {
          ctx.beginPath();
          ctx.arc(cX + pr.x*scale, cY - pr.y*scale, 3, 0, Math.PI*2);
          ctx.fill();
        }
      });
    }
  }

  function updateDetections(detections) {
    detDiv.innerHTML = "";
    (CFG.deviceIds || []).forEach(function(id){
      const on = !!(detections && detections[id]);
      const span = document.createElement("span");
      span.textContent = id + ": " + (on ? "Detected" : "Not Detected");
      span.style.padding = "3px 8px";
      span.style.borderRadius = "6px";
      span.style.border = "1px solid " + (on ? "#2f6b3a" : "#4a2f2f");
      span.style.color = on ? "#7be08f" : "#e08f8f";
      span.style.background = on ? "rgba(47,107,58,.15)" : "rgba(74,47,47,.15)";
      detDiv.appendChild(span);
    });
  }

  function updateTable(landmarks) {
    let html = "";
    for (let i = 0; i < NAMES.length; i++) {
      const p = landmarks ? landmarks[i] : null;
      if (!p || p.length !== 3 || p[0] === null || isNaN(p[0])) continue;
      html += "<tr>" +
        "<td style='padding:4px 10px;'>" + (NAMES[i] || i) + "</td>" +
        "<td style='padding:4px 10px;'>" + p[0].toFixed(3) + "</td>" +
        "<td style='padding:4px 10px;'>" + p[1].toFixed(3) + "</td>" +
        "<td style='padding:4px 10px;'>" + p[2].toFixed(3) + "</td></tr>";
    }
    jointBody.innerHTML = html;
  }

  // --- animation loop (draws latest frame ~ display refresh) ---
  let lastLandmarks = {};
  function loop() {
    drawSkeleton3D(toArray(lastLandmarks));
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // --- WebSocket ---
  let ws = null;
  let reconnectTimer = null;
  function connect() {
    try {
      ws = new WebSocket(CFG.wsUrl);
    } catch (e) {
      wsState.textContent = "ws error";
      return;
    }
    ws.onopen = function(){ wsState.textContent = "live"; wsState.style.color = "#7be08f"; };
    ws.onmessage = function(ev){
      let data;
      try { data = JSON.parse(ev.data); } catch (e) { return; }
      const detections = data.detections || {};
      updateDetections(detections);
      // Keep last 3D pose if this frame had no fused landmarks.
      if (data.landmarks && Object.keys(data.landmarks).length > 0) {
        lastLandmarks = data.landmarks;
        updateTable(lastLandmarks);
      }
    };
    ws.onclose = function(){
      wsState.textContent = "disconnected"; wsState.style.color = "#e08f8f";
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 2000);
    };
    ws.onerror = function(){ try { ws.close(); } catch (e) {} };
  }
  connect();

  window.addEventListener("beforeunload", function(){ try { ws && ws.close(); } catch(e){} });
})();
</script>
"""
    return (
        template.replace("__CFG_JSON__", cfg_json)
        .replace("__WIDTH__", width_css)
        .replace("__HEIGHT__", str(height))
    )
