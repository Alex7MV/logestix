// MosaicGrid — draws the routed task tiles and appends streamed tokens in
// place. Token text is appended as discrete Text nodes onto a single persistent
// Span so the DOM is never re-serialized per token (no innerHTML churn, no
// allocation cascade while the stream is hot).

export class MosaicGrid {
  constructor(gridEl, emptyEl) {
    this.grid = gridEl;
    this.empty = emptyEl;
    this.tiles = new Map(); // sequence -> tile record
    this.seq = 0;
  }

  clear() {
    this.grid.textContent = '';
    this.tiles.clear();
    this.seq = 0;
    this.empty.hidden = true;
  }

  addTile(agent, task) {
    const seq = this.seq++;
    const art = document.createElement('article');
    art.className = 'tile active';
    art.dataset.seq = String(seq);

    const head = document.createElement('div');
    head.className = 'tile-head';

    const seqEl = document.createElement('span');
    seqEl.className = 'tile-seq';
    seqEl.textContent = `#${String(seq).padStart(2, '0')}`;

    const agentEl = document.createElement('span');
    agentEl.className = 'tile-agent';
    agentEl.textContent = agent;

    const statusEl = document.createElement('span');
    statusEl.className = 'tile-status status-warn';
    statusEl.textContent = 'routing';

    head.append(seqEl, agentEl, statusEl);
    art.append(head);

    if (task) {
      const taskEl = document.createElement('div');
      taskEl.className = 'tile-task';
      taskEl.textContent = task;
      art.append(taskEl);
    }

    const body = document.createElement('div');
    body.className = 'tile-body';
    const caret = document.createElement('span');
    caret.className = 'caret';
    const sink = document.createElement('span');
    body.append(sink, caret);
    art.append(body);

    this.grid.append(art);
    this.empty.hidden = true;

    const rec = { art, agent, task, body, sink, caret, statusEl, done: false };
    this.tiles.set(seq, rec);
    return rec;
  }

  // Append a streamed token fragment into the active tile. Mutates one Text
  // node in place — zero per-token DOM rebuilding.
  append(rec, text) {
    rec.sink.append(document.createTextNode(text));
    rec.art.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  setStatus(rec, label, cls) {
    rec.statusEl.textContent = label;
    rec.statusEl.className = `tile-status ${cls}`;
  }

  complete(rec) {
    rec.done = true;
    rec.art.classList.remove('active');
    rec.art.classList.add('done');
    rec.caret.remove();
    this.setStatus(rec, 'done', 'status-ok');
  }

  fail(rec, message) {
    rec.art.classList.remove('active');
    this.setStatus(rec, message, 'status-err');
    rec.caret.remove();
  }
}
