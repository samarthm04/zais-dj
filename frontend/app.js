const MAX_FILES = 10;

const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const processBtn = document.getElementById('process-btn');
const statusEl = document.getElementById('status');
const tracksEl = document.getElementById('tracks');

let tracks = [];
const cards = new Map();

function setStatus(msg) { statusEl.textContent = msg || ''; }

async function api(url, opts) {
  const res = await fetch(url, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { /* no json body */ }
  if (!res.ok) throw new Error((body && body.detail) || res.statusText || 'request failed');
  return body;
}

/* ---- reading the analyze_track() result ------------------------------- */
/* Field names are probed rather than assumed: the exact result shape has
   not been verified against the real engine yet. Whatever is not found
   shows as "—", and the full payload stays visible under "raw result". */

function pick(obj, keys) {
  if (!obj || typeof obj !== 'object') return undefined;
  for (const k of keys) {
    if (obj[k] !== undefined && obj[k] !== null) return obj[k];
  }
  return undefined;
}

function readTempo(r) {
  let v = pick(r, ['tempo', 'bpm', 'tempo_bpm', 'estimated_tempo']);
  if (v && typeof v === 'object') v = pick(v, ['value', 'bpm', 'tempo', 'median', 'global']);
  if (typeof v === 'number') return v.toFixed(1);
  return v != null ? String(v) : '—';
}

function readKey(r) {
  let v = pick(r, ['camelot', 'key_camelot', 'key', 'musical_key']);
  if (v && typeof v === 'object') {
    const parts = [pick(v, ['camelot', 'code']), pick(v, ['name', 'label', 'key'])]
      .filter(x => x != null && typeof x !== 'object');
    return parts.length ? [...new Set(parts.map(String))].join(' · ') : '—';
  }
  return v != null ? String(v) : '—';
}

function readCueTime(c) {
  if (typeof c === 'number') return c;
  const t = pick(c, ['time', 't', 'time_s', 'time_sec', 'seconds', 'sec',
                     'position', 'pos', 'start', 'start_time', 'timestamp']);
  return typeof t === 'number' ? t : null;
}

function normalizeCues(list, kind) {
  if (!Array.isArray(list)) return [];
  return list.map(c => {
    const time = readCueTime(c);
    if (time === null) return null;
    const rawKind = String(pick(c, ['type', 'kind', 'role', 'direction']) || kind || '').toLowerCase();
    const resolved = rawKind.includes('exit') || rawKind.includes('out') ? 'exit'
                   : rawKind.includes('entry') || rawKind.includes('in') ? 'entry'
                   : 'cue';
    const score = pick(c, ['score', 'rank', 'confidence', 'strength', 'weight', 'rating']);
    const label = pick(c, ['label', 'name', 'section', 'description']);
    return {
      kind: resolved,
      time,
      score: typeof score === 'number' ? score : null,
      label: label != null && typeof label !== 'object' ? String(label) : null,
    };
  }).filter(Boolean);
}

function readCues(r) {
  if (!r || typeof r !== 'object') return [];
  const out = [];
  const container = pick(r, ['cue_points', 'cues', 'cuepoints']);

  if (Array.isArray(container)) {
    out.push(...normalizeCues(container, null));
  } else if (container && typeof container === 'object') {
    out.push(...normalizeCues(pick(container, ['entry', 'entries', 'entry_points', 'in']), 'entry'));
    out.push(...normalizeCues(pick(container, ['exit', 'exits', 'exit_points', 'out']), 'exit'));
  }

  out.push(...normalizeCues(pick(r, ['entry_cues', 'entry_points', 'entries']), 'entry'));
  out.push(...normalizeCues(pick(r, ['exit_cues', 'exit_points', 'exits']), 'exit'));

  return out.sort((a, b) => a.time - b.time);
}

function fmtTime(sec) {
  if (!isFinite(sec)) return '—';
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s < 10 ? '0' : ''}${s.toFixed(1)}`;
}

/* ---- rendering -------------------------------------------------------- */

function buildCard(t) {
  const card = document.createElement('section');
  card.className = 'track';

  const head = document.createElement('div');
  head.className = 'track-head';
  const title = document.createElement('h2');
  title.textContent = t.filename;
  const remove = document.createElement('button');
  remove.className = 'remove';
  remove.textContent = 'remove';
  remove.title = 'Delete this track from the library';
  remove.addEventListener('click', () => doDelete(t));
  head.append(title, remove);
  card.appendChild(head);

  const meta = document.createElement('div');
  meta.className = 'meta';
  card.appendChild(meta);

  const audio = document.createElement('audio');
  audio.controls = true;
  audio.preload = 'metadata';
  audio.src = `/api/audio/${t.track_id}`;
  card.appendChild(audio);

  const timeline = document.createElement('div');
  timeline.className = 'timeline';
  card.appendChild(timeline);

  const playhead = document.createElement('div');
  playhead.className = 'playhead';
  playhead.style.left = '0%';
  timeline.appendChild(playhead);

  const cues = document.createElement('div');
  cues.className = 'cues';
  card.appendChild(cues);

  const body = document.createElement('div');
  card.appendChild(body);

  timeline.addEventListener('click', ev => {
    if (!isFinite(audio.duration)) return;
    const rect = timeline.getBoundingClientRect();
    audio.currentTime = ((ev.clientX - rect.left) / rect.width) * audio.duration;
  });
  audio.addEventListener('timeupdate', () => {
    if (!isFinite(audio.duration) || audio.duration === 0) return;
    playhead.style.left = `${(audio.currentTime / audio.duration) * 100}%`;
  });
  audio.addEventListener('loadedmetadata', () => drawMarkers(card));

  const el = { card, meta, audio, timeline, playhead, cues, body };
  cards.set(t.track_id, el);
  return el;
}

function drawMarkers(card) {
  const el = [...cards.values()].find(c => c.card === card);
  if (!el || !el.cueList) return;
  el.timeline.querySelectorAll('.marker').forEach(m => m.remove());

  const duration = isFinite(el.audio.duration) && el.audio.duration > 0 ? el.audio.duration : null;
  if (!duration) return;

  for (const c of el.cueList) {
    const m = document.createElement('div');
    m.className = `marker ${c.kind}`;
    m.style.left = `${Math.min(100, Math.max(0, (c.time / duration) * 100))}%`;
    m.dataset.label = c.kind === 'cue' ? fmtTime(c.time) : c.kind;
    m.title = `${c.kind} @ ${fmtTime(c.time)}`;
    el.timeline.appendChild(m);
  }
}

function updateCard(t) {
  const el = cards.get(t.track_id) || buildCard(t);
  const { meta, cues, body, audio } = el;
  meta.replaceChildren();
  cues.replaceChildren();
  body.replaceChildren();

  if (t.error) {
    const err = document.createElement('div');
    err.className = 'error';
    err.textContent = t.error;
    body.appendChild(err);
  }

  if (!t.result) {
    if (!t.error) {
      const p = document.createElement('div');
      p.className = 'empty';
      p.textContent = t.processing ? 'Analyzing…' : 'Not analyzed yet.';
      body.appendChild(p);
    }
    el.cueList = [];
    drawMarkers(el.card);
    return;
  }

  for (const [label, value] of [['Tempo', `${readTempo(t.result)} BPM`], ['Key', readKey(t.result)]]) {
    const d = document.createElement('div');
    const l = document.createElement('span');
    l.className = 'label';
    l.textContent = label;
    const v = document.createElement('span');
    v.className = 'value';
    v.textContent = value;
    d.append(l, v);
    meta.appendChild(d);
  }

  const cueList = readCues(t.result);
  el.cueList = cueList;

  const count = document.createElement('div');
  const cl = document.createElement('span');
  cl.className = 'label';
  cl.textContent = 'Cue points';
  const cv = document.createElement('span');
  cv.className = 'value';
  cv.textContent = String(cueList.length);
  count.append(cl, cv);
  meta.appendChild(count);

  if (cueList.length === 0) {
    const p = document.createElement('div');
    p.className = 'empty';
    p.textContent = 'No cue points found in the result — check the raw result below.';
    cues.appendChild(p);
  }

  cueList.forEach(c => {
    const b = document.createElement('button');
    b.className = `cue ${c.kind}`;
    const text = document.createElement('span');
    text.textContent = `${c.kind} ${fmtTime(c.time)}${c.label ? ` · ${c.label}` : ''}`;
    b.appendChild(text);
    if (c.score !== null) {
      const s = document.createElement('span');
      s.className = 'score';
      s.textContent = c.score.toFixed(2);
      b.appendChild(s);
    }
    b.addEventListener('click', () => {
      audio.currentTime = c.time;
      audio.play();
    });
    cues.appendChild(b);
  });

  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = 'raw result';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(t.result, null, 2);
  details.append(summary, pre);
  body.appendChild(details);

  drawMarkers(el.card);
}

function renderAll() {
  for (const t of tracks) {
    const existing = cards.get(t.track_id);
    if (!existing) tracksEl.appendChild(buildCard(t).card);
    updateCard(t);
  }
  processBtn.disabled = tracks.length === 0;
  refreshMixer();
}

/* ---- transition renderer ---------------------------------------------- */

const mixer = document.getElementById('mixer');
const mixFrom = document.getElementById('mix-from');
const mixFromCue = document.getElementById('mix-from-cue');
const mixTo = document.getElementById('mix-to');
const mixToCue = document.getElementById('mix-to-cue');
const mixBars = document.getElementById('mix-bars');
const mixMatch = document.getElementById('mix-match');
const mixBtn = document.getElementById('mix-btn');
const mixStatus = document.getElementById('mix-status');
const mixOut = document.getElementById('mix-out');

function analyzed() { return tracks.filter(t => t.result); }

function fillTracks(sel, keep) {
  const previous = keep && sel.value;
  sel.replaceChildren();
  for (const t of analyzed()) {
    const o = document.createElement('option');
    o.value = t.track_id;
    o.textContent = t.filename;
    sel.appendChild(o);
  }
  if (previous && [...sel.options].some(o => o.value === previous)) sel.value = previous;
}

function fillCues(sel, trackId, kind) {
  sel.replaceChildren();
  const t = tracks.find(x => x.track_id === trackId);
  if (!t || !t.result) return;
  const list = readCues(t.result).filter(c => c.kind === kind || c.kind === 'cue');
  for (const c of list) {
    const o = document.createElement('option');
    o.value = String(c.time);
    o.textContent = `${fmtTime(c.time)}${c.score != null ? `  ${c.score.toFixed(2)}` : ''}`
      + `${c.label ? `  ${c.label}` : ''}`;
    sel.appendChild(o);
  }
}

function refreshMixer() {
  const ready = analyzed();
  mixer.hidden = ready.length < 2;
  if (mixer.hidden) return;
  fillTracks(mixFrom, true);
  fillTracks(mixTo, true);
  if (!mixTo.value || mixTo.value === mixFrom.value) {
    const other = ready.find(t => t.track_id !== mixFrom.value);
    if (other) mixTo.value = other.track_id;
  }
  fillCues(mixFromCue, mixFrom.value, 'exit');
  fillCues(mixToCue, mixTo.value, 'entry');
  refreshSetBuilder();
}

mixFrom.addEventListener('change', () => fillCues(mixFromCue, mixFrom.value, 'exit'));
mixTo.addEventListener('change', () => fillCues(mixToCue, mixTo.value, 'entry'));

async function doMix() {
  if (!mixFromCue.value || !mixToCue.value) {
    mixStatus.textContent = 'Both tracks need cue points.';
    return;
  }
  mixBtn.disabled = true;
  mixStatus.textContent = 'Rendering…';
  try {
    const d = await api('/api/mix', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        from_track_id: mixFrom.value,
        from_time: parseFloat(mixFromCue.value),
        to_track_id: mixTo.value,
        to_time: parseFloat(mixToCue.value),
        crossfade_bars: parseInt(mixBars.value, 10),
        match_tempo: mixMatch.checked,
      }),
    });
    showMix(d);
    mixStatus.textContent = '';
  } catch (e) {
    mixStatus.textContent = `Render failed: ${e.message}`;
  } finally {
    mixBtn.disabled = false;
  }
}

function showMix(d) {
  const box = document.createElement('div');
  box.className = 'mix-result';

  const title = document.createElement('div');
  title.className = 'mix-title';
  title.textContent = `${d.from.filename} @ ${fmtTime(d.from.time)}  →  ${d.to.filename} @ ${fmtTime(d.to.time)}`;
  box.appendChild(title);

  const audio = document.createElement('audio');
  audio.controls = true;
  audio.preload = 'metadata';
  audio.src = d.url;
  box.appendChild(audio);

  const facts = document.createElement('div');
  facts.className = 'mix-facts';
  const stretch = d.stretch_percent === 0
    ? 'no stretch needed'
    : `${d.stretch_percent > 0 ? '+' : ''}${d.stretch_percent}% stretch`;
  facts.textContent = `${d.tempo_a} → ${d.tempo_b} BPM · ${stretch}`
    + ` · ${d.crossfade_bars} bars (${d.crossfade_seconds}s)`
    + ` · trim ${d.gain_db > 0 ? '+' : ''}${d.gain_db} dB`
    + ` · transition at ${fmtTime(d.transition_at)}`;
  box.appendChild(facts);

  if (!d.stretch_comfortable) {
    const warn = document.createElement('div');
    warn.className = 'mix-warn';
    warn.textContent = `${Math.abs(d.stretch_percent)}% is well past what a DJ would pull `
      + `(about 6%). Expect audible artefacts — this pairing is not tempo-compatible.`;
    box.appendChild(warn);
  }

  const jump = document.createElement('button');
  jump.className = 'cue';
  jump.textContent = 'jump to transition';
  jump.addEventListener('click', () => {
    audio.currentTime = Math.max(0, d.transition_at - 4);
    audio.play();
  });
  box.appendChild(jump);

  mixOut.prepend(box);
}

mixBtn.addEventListener('click', doMix);

/* ---- full set --------------------------------------------------------- */

const setBuilder = document.getElementById('setbuilder');
const setPreset = document.getElementById('set-preset');
const setSeconds = document.getElementById('set-seconds');
const setStart = document.getElementById('set-start');
const setFillers = document.getElementById('set-fillers');
const setBtn = document.getElementById('set-btn');
const setBuildStatus = document.getElementById('set-status');
const setOut = document.getElementById('set-out');

api('/api/presets')
  .then(d => {
    for (const p of d.presets) {
      const o = document.createElement('option');
      o.value = p.id;
      o.textContent = `${p.label} · ${p.crossfade_bars} bars`;
      setPreset.appendChild(o);
    }
  })
  .catch(() => { /* preset list is optional */ });

function refreshSetBuilder() {
  const ready = analyzed();
  setBuilder.hidden = ready.length < 2;
  if (setBuilder.hidden) return;
  const previous = setStart.value;
  setStart.replaceChildren();
  const auto = document.createElement('option');
  auto.value = '';
  auto.textContent = 'let it choose';
  setStart.appendChild(auto);
  for (const t of ready) {
    const o = document.createElement('option');
    o.value = t.track_id;
    o.textContent = t.filename;
    setStart.appendChild(o);
  }
  if (previous) setStart.value = previous;
}

async function doSet() {
  setBtn.disabled = true;
  setBuildStatus.textContent = 'Ordering and rendering…';
  try {
    const d = await api('/api/set', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        preset: setPreset.value || 'pop',
        max_play_seconds: parseFloat(setSeconds.value),
        start_track_id: setStart.value || null,
        use_fillers: setFillers.checked,
      }),
    });
    showSet(d);
    setBuildStatus.textContent = '';
  } catch (e) {
    setBuildStatus.textContent = `Build failed: ${e.message}`;
  } finally {
    setBtn.disabled = false;
  }
}

function voteButtons(setId, index) {
  const wrap = document.createElement('span');
  wrap.className = 'votes';
  const note = document.createElement('span');
  note.className = 'verdict';

  for (const [value, label] of [['like', '♥'], ['dislike', '✕']]) {
    const btn = document.createElement('button');
    btn.className = `cue vote ${value}`;
    btn.textContent = label;
    btn.title = value === 'like' ? 'this transition worked' : 'this transition did not work';
    btn.addEventListener('click', async () => {
      wrap.querySelectorAll('button').forEach(b => { b.disabled = true; });
      try {
        const r = await api('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ set_id: setId, index, verdict: value }),
        });
        btn.classList.add('chosen');
        note.textContent = `${r.style || 'transition'} @ ${r.bars} bars — ${value}d`;
      } catch (e) {
        note.textContent = e.message;
        wrap.querySelectorAll('button').forEach(b => { b.disabled = false; });
      }
    });
    wrap.appendChild(btn);
  }
  wrap.appendChild(note);
  return wrap;
}

function showSet(d) {
  const box = document.createElement('div');
  box.className = 'mix-result';

  const title = document.createElement('div');
  title.className = 'mix-title';
  const matched = d.all_beatmatched ? 'every transition beatmatched' : 'some transitions not beatmatched';
  title.textContent = `${d.tracks} tracks · ${fmtTime(d.duration)} · ${d.preset} · ${matched}`
    + (d.reprises ? ` · ${d.reprises} reprise${d.reprises > 1 ? 's' : ''}` : '');
  box.appendChild(title);

  const audio = document.createElement('audio');
  audio.controls = true;
  audio.preload = 'metadata';
  audio.src = d.url;
  box.appendChild(audio);

  for (const skip of d.skipped || []) {
    const warn = document.createElement('div');
    warn.className = 'mix-warn';
    warn.textContent = `Left out — ${skip.filename}: ${skip.reason}`;
    box.appendChild(warn);
  }

  const table = document.createElement('div');
  table.className = 'setlist';

  // Bridges are separate from the track list but play in between, so show
  // everything on one timeline ordered by when it actually starts.
  const rows = [
    ...d.timeline.map((t, i) => ({ kind: 'track', position: i + 1, ...t })),
    ...(d.fillers || []).map(f => ({ kind: 'bridge', ...f })),
  ].sort((a, b) => a.starts_at - b.starts_at);

  rows.forEach(row => {
    const line = document.createElement('div');
    line.className = row.kind === 'bridge' ? 'set-row bridge' : 'set-row';

    const jump = document.createElement('button');
    jump.className = 'cue';
    jump.textContent = fmtTime(row.starts_at);
    jump.addEventListener('click', () => {
      audio.currentTime = Math.max(0, row.starts_at - 3);
      audio.play();
    });
    line.appendChild(jump);

    if (row.kind === 'bridge') {
      const name = document.createElement('span');
      name.className = 'set-name';
      name.textContent = `↳ generated bridge · ${row.bars} bars`;
      line.appendChild(name);
      const meta = document.createElement('span');
      meta.className = 'set-meta';
      meta.textContent = `${row.tempo_from} → ${row.tempo_to} BPM over ${row.seconds}s`;
      line.appendChild(meta);
      table.appendChild(line);
      return;
    }

    const name = document.createElement('span');
    name.className = 'set-name';
    name.textContent = row.is_reprise
      ? `${row.position}. ${row.filename}  ↺ reprise from ${fmtTime(row.entry_cue)}`
      : `${row.position}. ${row.filename}`;
    if (row.is_reprise) line.classList.add('reprise');
    line.appendChild(name);

    const meta = document.createElement('span');
    meta.className = 'set-meta';
    const ramp = row.enters_at_tempo !== row.set_tempo || row.leaves_at_tempo !== row.set_tempo
      ? `${row.enters_at_tempo}→${row.set_tempo}→${row.leaves_at_tempo}`
      : `${row.set_tempo} throughout`;
    meta.textContent = `${row.camelot || '—'} · ${ramp} BPM`
      + (row.stretch_percent ? ` · ≤${row.stretch_percent}% stretch` : '')
      + (row.keylock ? ' · keylock' : '')
      + (row.cycle ? ` · ${row.cycle}` : '');
    line.appendChild(meta);

    if (row.transition_to_next) {
      const style = document.createElement('span');
      style.className = `set-style ${row.beatmatched ? '' : 'unmatched'}`;
      const p = row.phrasing;
      const phrase = p ? ` · ${p.bars} bars = ${p.cycles}×${p.cycle_bars}-bar cycle` : '';
      style.textContent = (row.beatmatched
        ? `${row.transition_to_next} · ${row.tempo_delta_percent > 0 ? '+' : ''}${row.tempo_delta_percent}%`
        : `${row.transition_to_next} · not beatmatched (${row.tempo_delta_percent > 0 ? '+' : ''}${row.tempo_delta_percent}%)`)
        + phrase;
      if (p && p.sizing) style.title = p.sizing.join('; ');
      line.appendChild(style);

      // Judged per handover, not per set. One bad transition in an otherwise
      // good set should only count against that transition.
      line.appendChild(voteButtons(d.set_id, row.index));
    }
    table.appendChild(line);
  });
  box.appendChild(table);
  setOut.prepend(box);
}

setBtn.addEventListener('click', doSet);


/* ---- actions ---------------------------------------------------------- */

async function doDelete(t) {
  if (!confirm(`Remove "${t.filename}" from the library?`)) return;
  try {
    await api(`/api/track/${t.track_id}`, { method: 'DELETE' });
    const el = cards.get(t.track_id);
    if (el) {
      el.audio.pause();          // otherwise it keeps playing after the card goes
      el.audio.removeAttribute('src');
      el.card.remove();
      cards.delete(t.track_id);
    }
    tracks = tracks.filter(x => x.track_id !== t.track_id);
    renderAll();
    setStatus(`Removed ${t.filename}. ${tracks.length} left.`);
  } catch (e) {
    setStatus(`Could not remove: ${e.message}`);
  }
}

async function doUpload() {
  const files = [...fileInput.files];
  if (files.length === 0) { setStatus('Pick some audio files first.'); return; }
  if (files.length > MAX_FILES) { setStatus(`Max ${MAX_FILES} files at once.`); return; }

  uploadBtn.disabled = processBtn.disabled = true;
  setStatus(`Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`);
  const form = new FormData();
  files.forEach(f => form.append('files', f));

  try {
    const data = await api('/api/upload', { method: 'POST', body: form });
    tracks.push(...data.tracks);
    renderAll();
    setStatus(`Uploaded ${data.tracks.length}. Hit Process to analyze.`);
    fileInput.value = '';
  } catch (e) {
    setStatus(`Upload failed: ${e.message}`);
  } finally {
    uploadBtn.disabled = false;
    processBtn.disabled = tracks.length === 0;
  }
}

async function doProcess() {
  const pending = tracks.filter(t => !t.result);
  if (pending.length === 0) { setStatus('Nothing left to analyze.'); return; }

  uploadBtn.disabled = processBtn.disabled = true;
  for (let i = 0; i < pending.length; i++) {
    const t = pending[i];
    setStatus(`Analyzing ${i + 1}/${pending.length}: ${t.filename}`);
    t.processing = true;
    t.error = null;
    updateCard(t);
    try {
      const data = await api('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track_ids: [t.track_id] }),
      });
      Object.assign(t, data.tracks[0]);
    } catch (e) {
      t.error = `Request failed: ${e.message}`;
    }
    t.processing = false;
    updateCard(t);
  }
  setStatus('Done.');
  uploadBtn.disabled = false;
  processBtn.disabled = false;
}

uploadBtn.addEventListener('click', doUpload);
processBtn.addEventListener('click', doProcess);

// Opening index.html straight off disk gives the buttons nothing to talk to,
// since every request here is relative to the backend that serves the page.
// Say so, rather than leaving the buttons looking merely broken.
if (!location.protocol.startsWith('http')) {
  uploadBtn.disabled = true;
  processBtn.disabled = true;
  setStatus('Not being served by the backend — start it and open http://127.0.0.1:8765/ instead.');
} else {
  api('/api/tracks')
    .then(data => { tracks = data.tracks; renderAll(); })
    .catch(e => setStatus(`Could not load tracks: ${e.message}`));
}
