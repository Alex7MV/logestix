import { MosaicGrid } from '/js/modules/renderer.js';
import { StreamClient } from '/js/modules/stream.js';
import { decompose, AGENT_REGISTRY } from '/js/modules/mosaic.js';

const gridEl = document.getElementById('grid');
const emptyEl = document.getElementById('grid-empty');
const railEl = document.getElementById('agent-rail');
const form = document.getElementById('prompt-form');
const input = document.getElementById('prompt-input');
const submitBtn = document.getElementById('submit-btn');
const status = document.getElementById('stream-status');

const grid = new MosaicGrid(gridEl, emptyEl);
let client = null;

// Static agent-node legend.
for (const agent of AGENT_REGISTRY) {
  const chip = document.createElement('span');
  chip.className = 'agent-chip';
  chip.dataset.agent = agent;
  chip.innerHTML = '<span class="dot"></span>' + agent;
  railEl.append(chip);
}

function setStatus(text, cls = '') {
  status.textContent = text;
  status.className = 'stream-status ' + cls;
  status.hidden = false;
}

async function run() {
  form.reset();
  submitBtn.disabled = true;
  grid.clear();

  const tiles = decompose(input.value.trim());
  const payload = {
    model: tiles[0]?.ref ?? 'router', // dispatch target surfaced to the backend
    messages: [{ role: 'user', content: input.value.trim() }],
  };

  for (const chip of railEl.querySelectorAll('.agent-chip.live')) {
    chip.classList.remove('live');
  }

  // Tessellate: turn each decomposed tile into a live routing node.
  const records = new Map();
  let active = tiles.length;
  for (const t of tiles) {
    const rec = grid.addTile(t.agent, t.task);
    records.set(t.ref, rec);
    const chip = railEl.querySelector(`[data-agent="${t.agent}"]`);
    if (chip) chip.classList.add('live');
  }

  client = new StreamClient();
  setStatus(`routing ${tiles.length} tile${tiles.length > 1 ? 's' : ''} → ${payload.model}`);

  try {
    // Replay: route the whole tile-stream through the shared completion feed.
    await client.stream(payload, (delta) => {
      // Fan the delta out to every active tile in its own visual block.
      for (const rec of records.values()) {
        if (!rec.done) grid.append(rec, delta);
      }
    });
    for (const rec of records.values()) grid.complete(rec);
    setStatus('mosaic complete — all tiles green (GREEN)', 'status-ok');
  } catch (err) {
    for (const rec of records.values()) grid.fail(rec, 'stream error');
    setStatus(`stream aborted: ${err.message}`, 'status-err');
  } finally {
    client = null;
    submitBtn.disabled = false;
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  run();
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 220) + 'px';
});
