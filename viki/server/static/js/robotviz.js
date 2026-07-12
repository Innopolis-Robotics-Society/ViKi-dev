// Robot trajectory visualization: list dataset outputs, stream selected one.
import { api } from './core.js';

let robotVizInterval = null;
let robotVizStreamUrl = null;

export function toggleRobotViz() {
  const panel = document.getElementById('robotviz-panel');
  const visible = panel.style.display === 'block';
  panel.style.display = visible ? 'none' : 'block';
  if (!visible) {
    loadRobotVizOutputs();
  } else {
    const img = document.querySelector('#robotviz-stream img');
    if (img) img.src = '';
    if (robotVizInterval) { clearInterval(robotVizInterval); robotVizInterval = null; }
  }
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

  const streamDiv = document.getElementById('robotviz-stream');
  const url = `/api/dataset/viz-stream?filename=${encodeURIComponent(filename)}&t=${Date.now()}`;

  if (robotVizStreamUrl) {
    // Remove old img to stop previous stream
    const oldImg = streamDiv.querySelector('img');
    if (oldImg) oldImg.src = '';
  }
  robotVizStreamUrl = url;
  streamDiv.innerHTML = `<img src="${url}" alt="robot trajectory">`;
}
