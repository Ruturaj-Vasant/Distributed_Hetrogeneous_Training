const lossHistory = [];
const MAX_LOSS_POINTS = 200;
let latestState = null;

function initDashboard(view) {
  fetch('/api/history')
    .then(r => r.json())
    .then(d => {
      if (d.loss && d.loss.length) {
        lossHistory.push(...d.loss);
        trimLoss();
        drawLoss();
      }
    })
    .catch(() => {});

  connect(view);
}

function connect(view) {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => setConnected(true);
  ws.onmessage = e => {
    latestState = JSON.parse(e.data);
    renderShell(latestState);
    if (view === 'leader') renderLeader(latestState);
    if (view === 'worker') renderWorker(latestState);
    if (view === 'watch') renderWatch(latestState);
  };
  ws.onclose = () => {
    setConnected(false);
    setTimeout(() => connect(view), 2000);
  };
}

function setConnected(ok) {
  const dot = document.getElementById('conn-dot');
  const label = document.getElementById('conn-label');
  if (!dot || !label) return;
  dot.className = ok ? 'ok' : '';
  label.textContent = ok ? 'connected' : 'reconnecting...';
}

function renderShell(s) {
  setText('stat-phase', s.phase);
  const phaseEl = document.getElementById('stat-phase');
  if (phaseEl) phaseEl.className = `stat-value ${s.phase}`;
  setText('stat-epoch', s.phase === 'waiting' ? '-' : s.epoch);
  setText('stat-step', s.phase === 'waiting' ? '-' : s.step);
  setText('stat-loss', s.loss > 0 ? s.loss.toFixed(4) : '-');
  setText('stat-workers', s.workers.length);

  const c = s.cfg;
  setText('cfg-pill', `${c.model} / ${c.dataset} / ep=${c.epochs} / lr=${c.lr} / topk=${c.topk === 0 ? 'full' : c.topk}`);

  if (s.loss > 0) {
    lossHistory.push(s.loss);
    trimLoss();
    drawLoss();
  }
}

function renderLeader(s) {
  renderWorkersTable('leader-workers', s.workers, true);
  renderPending('pending-body', s.pending);
}

function renderWorker(s) {
  const selected = s.workers[0] || null;
  const body = document.getElementById('worker-detail');
  if (!body) return;
  if (!selected) {
    body.innerHTML = '<div class="empty">No worker has connected yet.</div>';
  } else {
    const hbClass = heartbeatClass(selected.heartbeat_ago);
    body.innerHTML = `<div class="kv">
      <span>Worker ID</span><span>${esc(selected.worker_id)}</span>
      <span>Hostname</span><span>${esc(selected.hostname)}</span>
      <span>OS</span><span>${esc(selected.os)}</span>
      <span>Accelerator</span><span>${esc(selected.accel)}</span>
      <span>Status</span><span>${statusBadge(selected)}</span>
      <span>Score</span><span>${selected.score}</span>
      <span>Shard</span><span>${selected.shard_size ? selected.shard_size.toLocaleString() : '-'}</span>
      <span>Loss</span><span>${selected.last_loss > 0 ? selected.last_loss.toFixed(4) : '-'}</span>
      <span>Steps</span><span>${selected.steps}</span>
      <span>Heartbeat</span><span class="${hbClass}">${selected.heartbeat_ago}s ago</span>
    </div>`;
  }
  renderWorkersTable('worker-list', s.workers, false);
}

function renderWatch(s) {
  renderWorkersTable('watch-workers', s.workers, false);
  renderPending('watch-pending', s.pending);
}

function renderWorkersTable(id, workers, includeActions) {
  const target = document.getElementById(id);
  if (!target) return;
  if (!workers.length) {
    target.innerHTML = '<div class="empty">No workers connected.</div>';
    return;
  }
  target.innerHTML = `<table>
    <thead><tr>
      <th>ID</th><th>Hostname</th><th>Device</th><th>Status</th>
      <th>Score</th><th>Shard</th><th>Loss</th><th>Steps</th><th>Heartbeat</th>
    </tr></thead>
    <tbody>${workers.map(w => {
      const hbClass = heartbeatClass(w.heartbeat_ago);
      return `<tr>
        <td>${esc(w.worker_id)}</td>
        <td>${esc(w.hostname)}</td>
        <td class="muted">${esc(w.accel)}</td>
        <td>${statusBadge(w)}</td>
        <td>${w.score}</td>
        <td>${w.shard_size ? w.shard_size.toLocaleString() : '-'}</td>
        <td>${w.last_loss > 0 ? w.last_loss.toFixed(4) : '-'}</td>
        <td>${w.steps}</td>
        <td class="${hbClass}">${w.heartbeat_ago}s</td>
      </tr>`;
    }).join('')}</tbody>
  </table>`;
}

function renderPending(id, pending) {
  const target = document.getElementById(id);
  if (!target) return;
  if (!pending.length) {
    target.innerHTML = '<div class="empty">No pending workers.</div>';
    return;
  }
  target.innerHTML = `<table>
    <thead><tr><th>ID</th><th>Hostname</th><th>Device</th><th>Score</th><th>Action</th></tr></thead>
    <tbody>${pending.map(p => `<tr>
      <td>${esc(p.worker_id)}</td>
      <td>${esc(p.hostname)}</td>
      <td class="muted">${esc(p.accel)}</td>
      <td>${p.score}</td>
      <td><button onclick="admitOne('${esc(p.worker_id)}')">Admit</button></td>
    </tr>`).join('')}</tbody>
  </table>`;
}

function drawLoss() {
  const canvas = document.getElementById('loss-canvas');
  if (!canvas || lossHistory.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.offsetWidth * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  const pad = { top: 12, right: 18, bottom: 14, left: 18 };
  const minV = Math.min(...lossHistory);
  const maxV = Math.max(...lossHistory);
  const range = maxV - minV || 1;
  const x = i => pad.left + (i / (lossHistory.length - 1)) * (W - pad.left - pad.right);
  const y = v => pad.top + (1 - (v - minV) / range) * (H - pad.top - pad.bottom);
  ctx.clearRect(0, 0, W, H);
  ctx.beginPath();
  ctx.moveTo(x(0), y(lossHistory[0]));
  for (let i = 1; i < lossHistory.length; i++) ctx.lineTo(x(i), y(lossHistory[i]));
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

function apiPost(url, body = {}) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(e => console.warn(url, e));
}

function admitAll() { apiPost('/api/admit', { worker_ids: [] }); }
function admitOne(id) { apiPost('/api/admit', { worker_ids: [id] }); }
function startTraining() { apiPost('/api/start'); }
function resetTraining() { apiPost('/api/reset'); }

function statusBadge(w) {
  const cls = !w.alive ? 'dead' : w.status;
  const label = !w.alive ? 'dead' : w.status;
  return `<span class="badge badge-${cls}">${label}</span>`;
}

function heartbeatClass(seconds) {
  return seconds < 10 ? 'hb-ok' : seconds < 30 ? 'hb-warn' : 'hb-dead';
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function trimLoss() {
  if (lossHistory.length > MAX_LOSS_POINTS) {
    lossHistory.splice(0, lossHistory.length - MAX_LOSS_POINTS);
  }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
}

window.addEventListener('resize', drawLoss);
