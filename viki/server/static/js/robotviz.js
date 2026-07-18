// Robot trajectory visualization: comprehensive extrinsics + FK animation.
// Displays an MJPEG stream with full scene: cameras, board, human path, robot path, FK arm.
import { api } from './core.js';

let robotVizStreamUrl = null;
let robotVizInterval = null;
let previousFilename = null;

export function toggleRobotViz() {
  const panel = document.getElementById('robotviz-panel');
  const visible = panel.style.display === 'block';
  panel.style.display = visible ? 'none' : 'block';
  if (!visible) {
    loadRobotVizOutputs();
  } else {
    stopStream();
  }
}

function stopStream() {
  const img = document.querySelector('#robotviz-stream img');
  if (img) img.src = '';
  if (robotVizInterval) { clearInterval(robotVizInterval); robotVizInterval = null; }
  robotVizStreamUrl = null;
  previousFilename = null;
}

export async function loadRobotVizOutputs() {
  const listEl = document.getElementById('robotviz-output-list');
  const infoEl = document.getElementById('robotviz-info');
  infoEl.textContent = 'Loading...';
  try {
    const data = await api('GET', '/api/dataset/outputs');
    const files = data.outputs || [];
    listEl.innerHTML = files.length === 0
      ? '<div style="padding:8px;color:var(--muted);">No outputs found</div>'
      : files.map(f =>
        `<div style="padding:6px 8px;cursor:pointer;border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;"
                    data-action="selectRobotVizOutput" data-filename="${f}">
                <span>${f}</span>
              </div>`
      ).join('');
    infoEl.textContent = `${files.length} output(s)`;
  } catch (e) {
    infoEl.textContent = `Failed: ${e}`;
    listEl.innerHTML = '<div style="padding:8px;color:var(--red);">Failed to load outputs</div>';
  }
}

export function selectRobotVizOutput(filename, el) {
  document.querySelectorAll('#robotviz-output-list div').forEach(d => d.style.background = '');
  if (el) el.style.background = 'var(--surface)';
  previousFilename = filename;
  startStream();
}

function buildConfig() {
  const getChecked = id => document.getElementById(id)?.checked ?? true;
  const centerOn = document.querySelector('input[name="robotviz-center"]:checked')?.value ?? 'world';
  const axesLength = parseFloat(document.getElementById('robotviz-axes-length')?.value) || 2.0;
  return {
    center_on: centerOn,
    axes_length: axesLength,
    show_cameras: getChecked('rv-toggle-cameras'),
    show_board: getChecked('rv-toggle-board'),
    show_neutral_ee: getChecked('rv-toggle-neutral'),
    show_human_trail: getChecked('rv-toggle-human-trail'),
    show_robot_trail: getChecked('rv-toggle-robot-trail'),
    show_base_to_ee: getChecked('rv-toggle-base-to-ee'),
    show_debug_overlay: getChecked('rv-toggle-debug'),
    show_reach_sphere: getChecked('rv-toggle-reach'),
    show_fk_arm: getChecked('rv-toggle-fk'),
    show_ee_target: getChecked('rv-toggle-ee-target'),
  };
}

function startStream() {
  const filename = previousFilename;
  if (!filename) return;

  const cfg = buildConfig();
  const params = new URLSearchParams({ filename, ...cfg });
  const url = `/api/dataset/viz-stream?${params}&t=${Date.now()}`;

  const streamDiv = document.getElementById('robotviz-stream');
  if (robotVizStreamUrl) {
    const oldImg = streamDiv.querySelector('img');
    if (oldImg) oldImg.src = '';
  }
  robotVizStreamUrl = url;
  streamDiv.innerHTML = `<img src="${url}" alt="robot trajectory">`;
}

export function applyRobotVizConfig() {
  startStream();
}
