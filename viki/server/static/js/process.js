// Process Recordings panel: skeleton smoothing + robot-dataset conversion jobs.
import { api, log } from './core.js';

let processPage = 0;
let processMode = 'smooth';
let processSelectedRec = null;
let processJobs = {};
let processJobsPollInterval = null;
let processJobPolls = {};

export function toggleProcess() {
  const panel = document.getElementById('process-panel');
  const visible = panel.style.display === 'block';
  panel.style.display = visible ? 'none' : 'block';
  if (!visible) {
    loadProcessRecordings();
    pollProcessJobs();
  } else {
    clearProcessJobsPoll();
    Object.values(processJobPolls).forEach(clearInterval);
    processJobPolls = {};
  }
}

export function setProcessMode(mode) {
  processMode = mode;
  document.getElementById('btn-process-mode-smooth').className = mode === 'smooth' ? 'primary' : '';
  document.getElementById('btn-process-mode-dataset').className = mode === 'dataset' ? 'primary' : '';
  const robotRow = document.getElementById('process-robot-row');
  const winLabel = document.querySelector('label[for="process-win-len"]');
  const winInput = document.getElementById('process-win-len');
  const polyLabel = document.querySelector('label[for="process-poly"]');
  const polyInput = document.getElementById('process-poly');
  if (mode === 'dataset') {
    robotRow.style.display = 'flex';
    winLabel.style.display = 'none';
    winInput.style.display = 'none';
    polyLabel.style.display = 'none';
    polyInput.style.display = 'none';
    document.getElementById('btn-process-smooth').textContent = 'Convert to Robot Dataset';
  } else {
    robotRow.style.display = 'none';
    winLabel.style.display = '';
    winInput.style.display = '';
    polyLabel.style.display = '';
    polyInput.style.display = '';
    document.getElementById('btn-process-smooth').textContent = 'Smooth Selected';
  }
  document.getElementById('process-status').textContent = '';
  document.getElementById('process-conversion-status').style.display = 'none';
  loadProcessRecordings();
}

export async function loadProcessRecordings() {
  const statusEl = document.getElementById('process-status');
  const listEl = document.getElementById('process-rec-list');
  statusEl.textContent = 'Loading...';
  try {
    let data;
    if (processMode === 'dataset') {
      data = await api('GET', `/api/dataset/recordings?page=${processPage}&limit=10`);
    } else {
      data = await api('GET', `/api/optimization/recordings?page=${processPage}&limit=10`);
    }
    const recs = data.recordings || [];
    listEl.innerHTML = recs.length === 0
      ? '<div style="padding:8px;color:var(--muted);">No recordings found</div>'
      : recs.map(f =>
        `<div style="padding:6px 8px;cursor:pointer;border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;"
                    data-action="selectProcessRec" data-filename="${f}">
                <input type="radio" name="process-rec" value="${f}" style="accent-color:var(--accent);">
                <span>${f}</span>
              </div>`
      ).join('');
    document.getElementById('btn-process-prev').disabled = processPage === 0;
    const nextBtn = document.getElementById('btn-process-next');
    nextBtn.disabled = recs.length < 10;
    document.getElementById('process-page-info').textContent = `Page ${processPage + 1}`;
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = `Failed to load: ${e}`;
  }
}

export function selectProcessRec(filename, el) {
  processSelectedRec = filename;
  document.querySelectorAll('#process-rec-list div').forEach(d => d.style.background = '');
  if (el) el.style.background = 'var(--surface)';
  document.getElementById('btn-process-smooth').disabled = false;
}

export function processPrevPage() {
  if (processPage > 0) { processPage--; loadProcessRecordings(); }
}

export function processNextPage() {
  processPage++;
  loadProcessRecordings();
}

function pollProcessJobs() {
  if (processJobsPollInterval) clearInterval(processJobsPollInterval);
  processJobsPollInterval = setInterval(renderProcessJobs, 2000);
  renderProcessJobs();
}

function clearProcessJobsPoll() {
  if (processJobsPollInterval) {
    clearInterval(processJobsPollInterval);
    processJobsPollInterval = null;
  }
}

async function renderProcessJobs() {
  try {
    const data = await api('GET', '/api/dataset/optimize/jobs');
    const container = document.getElementById('process-conversion-status');
    data.jobs.forEach(j => { processJobs[j.job_id] = j; });
    const entries = Object.values(processJobs).slice(0, 20);
    if (entries.length === 0) {
      container.style.display = 'none';
      return;
    }
    container.style.display = 'block';
    container.innerHTML = entries.map(j => {
      const icons = { queued: '⏳', running: '⟳', completed: '✅', failed: '❌' };
      const colors = { queued: 'var(--muted)', running: 'var(--yellow)', completed: 'var(--green)', failed: 'var(--red)' };
      return `<div style="padding:2px 0;color:${colors[j.status] || 'var(--muted)'}">${icons[j.status] || '?'} ${j.filename} → ${j.robot} [${j.status}]</div>`;
    }).join('');
  } catch (e) { /* ignore poll errors */ }
}

export async function processSmoothSelected() {
  if (!processSelectedRec) return;
  const statusEl = document.getElementById('process-status');
  const btn = document.getElementById('btn-process-smooth');
  btn.disabled = true;

  if (processMode === 'dataset') {
    const robot = document.getElementById('process-robot').value;
    statusEl.textContent = `Queueing ${processSelectedRec} for robot ${robot}...`;
    try {
      const res = await api('POST', '/api/dataset/optimize', { filename: processSelectedRec, robot });
      log(`Dataset conversion queued: ${processSelectedRec} -> ${robot} (job: ${res.job_id})`, 'ok');
      statusEl.innerHTML = `⏳ Job queued — <span id="job-status-${res.job_id}" style="color:var(--yellow);">pending</span>`;
      processJobs[res.job_id] = { job_id: res.job_id, filename: processSelectedRec, robot, status: 'queued' };
      renderProcessJobs();

      const poll = setInterval(async () => {
        try {
          const j = await api('GET', `/api/dataset/optimize/status/${res.job_id}`);
          processJobs[res.job_id] = j;
          renderProcessJobs();
          const span = document.getElementById(`job-status-${res.job_id}`);
          if (span) {
            if (j.status === 'running') { span.innerHTML = '<span style="color:var(--yellow);">⟳ converting...</span>'; }
            else if (j.status === 'completed') {
              span.innerHTML = '<span style="color:var(--green);">✅ done</span>';
              clearInterval(poll);
              delete processJobPolls[res.job_id];
            } else if (j.status === 'failed') {
              span.innerHTML = `<span style="color:var(--red);">❌ ${j.error || 'failed'}</span>`;
              clearInterval(poll);
              delete processJobPolls[res.job_id];
            }
          }
          if (j.status === 'completed' || j.status === 'failed') {
            statusEl.innerHTML = `${j.status === 'completed' ? '✅' : '❌'} ${processSelectedRec} → ${robot} [${j.status}]`;
          }
        } catch (e) { clearInterval(poll); }
      }, 1000);
      processJobPolls[res.job_id] = poll;
    } catch (e) {
      statusEl.textContent = `❌ ${e}`;
      log(`Dataset conversion failed: ${e}`, 'error');
    }
    btn.disabled = false;
    return;
  }

  const winLen = parseInt(document.getElementById('process-win-len').value) || 7;
  const poly = parseInt(document.getElementById('process-poly').value) || 2;
  statusEl.textContent = `Smoothing ${processSelectedRec}...`;
  try {
    const res = await api('POST', '/api/optimization/smooth', { filename: processSelectedRec, window_length: winLen, polyorder: poly });
    statusEl.textContent = `✅ ${res.path}`;
    log(`Smoothed: ${res.path}`, 'ok');
    const plotContainer = document.getElementById('process-smooth-plot');
    const plotName = res.path.split('/').pop() || res.path.split('\\').pop();
    plotContainer.innerHTML = `<img src="/api/optimization/smooth-plot?filename=${encodeURIComponent(plotName)}" style="width:100%;height:100%;object-fit:contain;">`;
    plotContainer.style.display = 'flex';
  } catch (e) {
    statusEl.textContent = `❌ ${e}`;
    log(`Smoothing failed: ${e}`, 'error');
  }
  btn.disabled = false;
}
