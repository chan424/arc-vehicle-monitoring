// ARC Vehicle Monitoring - detection worker.
// Runs the object detector off the main thread: loads the runtime and model,
// letterboxes each frame, runs inference, decodes boxes and applies NMS.
import * as ort from './ort/ort.js';

const BASE = new URL('./', import.meta.url);
let wasmReady = null;
let session = null;
let backend = null;
let model = null;
let modelBytes = null;
let verified = true;
let canvas = null, ctx = null, inputData = null, grid = null;
let chain = Promise.resolve();

const post = (m, transfer) => self.postMessage(m, transfer || []);

async function fetchParts(parts, total, label) {
  const out = new Uint8Array(total);
  let loaded = 0;
  for (const p of parts) {
    const res = await fetch(new URL(p, BASE));
    if (!res.ok) throw new Error(`Could not load ${p} (HTTP ${res.status})`);
    const b64 = p.endsWith('.js');     // model weights travel as base64 text inside .js files
    if (b64) {
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = '', started = false;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        if (!started) { const q = buf.indexOf('"'); if (q < 0) { buf = ''; continue; } buf = buf.slice(q + 1); started = true; }
        const end = buf.indexOf('"');
        let usable = end >= 0 ? end : buf.length;
        usable -= usable % 4;
        if (!usable) continue;
        const bin = atob(buf.slice(0, usable));
        buf = buf.slice(usable);
        if (loaded + bin.length > total) throw new Error(`${label} is larger than expected`);
        for (let j = 0; j < bin.length; j++) out[loaded + j] = bin.charCodeAt(j);
        loaded += bin.length;
        post({ type: 'progress', label, loaded, total });
      }
    } else if (res.body && res.body.getReader) {
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        out.set(value, loaded);
        loaded += value.length;
        post({ type: 'progress', label, loaded, total });
      }
    } else {
      const buf = new Uint8Array(await res.arrayBuffer());
      out.set(buf, loaded);
      loaded += buf.length;
      post({ type: 'progress', label, loaded, total });
    }
  }
  if (loaded !== total) throw new Error(`${label} download incomplete (${loaded} of ${total} bytes)`);
  return out;
}

function buildGrid(size) {
  const gx = [], gy = [], gs = [];
  for (const s of [8, 16, 32]) {
    const n = Math.floor(size / s);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { gx.push(x); gy.push(y); gs.push(s); }
  }
  return { gx: Float32Array.from(gx), gy: Float32Array.from(gy), gs: Float32Array.from(gs) };
}

async function createSession(ep) {
  return ort.InferenceSession.create(modelBytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
}

async function runSession(s, tensor) {
  const res = await s.run({ [s.inputNames[0]]: tensor });
  return res[s.outputNames[0]];
}

async function init(m) {
  model = m.model;
  if (!wasmReady) {
    wasmReady = (async () => {
      const bin = await fetchParts(m.runtime.parts, m.runtime.bytes, 'Detection runtime');
      ort.env.wasm.wasmBinary = bin.buffer;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.logLevel = 'error';
    })();
  }
  await wasmReady;
  modelBytes = model.buffer ? new Uint8Array(model.buffer) : await fetchParts(model.parts, model.bytes, 'Detector model');
  post({ type: 'progress', label: 'Starting detector', loaded: 1, total: 1 });

  const S = model.size;
  canvas = new OffscreenCanvas(S, S);
  ctx = canvas.getContext('2d', { willReadFrequently: true });
  inputData = new Float32Array(3 * S * S);
  grid = model.kind === 'yolox' ? buildGrid(S) : null;

  if (session) { try { await session.release(); } catch (e) { /* ignore */ } session = null; }
  let note = '';
  let wantGpu = m.backend !== 'wasm' && !!self.navigator.gpu;
  if (wantGpu) {
    try { if (!(await self.navigator.gpu.requestAdapter())) wantGpu = false; } catch (e) { wantGpu = false; }
  }
  const blank = new ort.Tensor('float32', new Float32Array(3 * S * S), [1, 3, S, S]);
  if (wantGpu) {
    try {
      session = await createSession('webgpu');
      await runSession(session, blank);
      backend = 'webgpu';
    } catch (e) {
      note = 'Graphics acceleration could not start; using the processor instead.';
      try { if (session) await session.release(); } catch (e2) { /* ignore */ }
      session = null;
    }
  }
  if (!session) {
    session = await createSession('wasm');
    const out = await runSession(session, blank);
    backend = 'wasm';
    post({ type: 'shape', dims: Array.from(out.dims) });
  }
  // A GPU session is checked once against the processor on the first real frame.
  verified = backend === 'wasm';
  post({ type: 'ready', backend, note, size: S });
}

function preprocess(bitmap) {
  const S = model.size;
  const w = bitmap.width, h = bitmap.height;
  const r = Math.min(S / w, S / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const centered = model.kind !== 'yolox';
  const px = centered ? Math.floor((S - nw) / 2) : 0;
  const py = centered ? Math.floor((S - nh) / 2) : 0;
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, S, S);
  ctx.drawImage(bitmap, 0, 0, w, h, px, py, nw, nh);
  const d = ctx.getImageData(0, 0, S, S).data;
  const n = S * S;
  if (model.kind === 'yolox') {
    // YOLOX expects BGR, 0-255.
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      inputData[i] = d[j + 2];
      inputData[n + i] = d[j + 1];
      inputData[2 * n + i] = d[j];
    }
  } else {
    // Ultralytics-style exports expect RGB, 0-1.
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      inputData[i] = d[j] / 255;
      inputData[n + i] = d[j + 1] / 255;
      inputData[2 * n + i] = d[j + 2] / 255;
    }
  }
  return { tensor: new ort.Tensor('float32', inputData, [1, 3, S, S]), map: { r, px, py, w, h } };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9);
}

function nms(dets) {
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of dets) {
    let ok = true;
    for (const k of keep) {
      const v = iou(d, k);
      // Same class: ordinary NMS. Different class: only near-identical boxes are
      // the same vehicle seen twice (for example car + truck on one van).
      if (v > (k.cls === d.cls ? 0.45 : 0.6)) { ok = false; break; }
    }
    if (ok) keep.push(d);
  }
  return keep;
}

function decode(out, map) {
  const data = out.data, dims = out.dims;
  const classMap = model.classMap; // model class index -> category index, or -1
  const minScore = model.minScore;
  const dets = [];
  const push = (cx, cy, bw, bh, score, cls, raw) => {
    let x = (cx - bw / 2 - map.px) / map.r, y = (cy - bh / 2 - map.py) / map.r;
    let w = bw / map.r, h = bh / map.r;
    const x2 = Math.min(map.w, x + w), y2 = Math.min(map.h, y + h);
    x = Math.max(0, x); y = Math.max(0, y);
    w = x2 - x; h = y2 - y;
    if (w < 4 || h < 4) return;
    dets.push({ x, y, w, h, score, cls, raw });
  };

  if (model.kind === 'yolox') {
    const N = dims[1], C = dims[2], nc = C - 5;
    for (let i = 0; i < N; i++) {
      const o = i * C;
      const obj = data[o + 4];
      if (obj < minScore) continue;
      let best = 0, bc = -1;
      for (let c = 0; c < nc; c++) {
        if (classMap[c] < 0) continue;
        const v = data[o + 5 + c];
        if (v > best) { best = v; bc = c; }
      }
      const score = obj * best;
      if (bc < 0 || score < minScore) continue;
      const s = grid.gs[i];
      push((data[o] + grid.gx[i]) * s, (data[o + 1] + grid.gy[i]) * s, Math.exp(data[o + 2]) * s, Math.exp(data[o + 3]) * s, score, classMap[bc], bc);
    }
  } else if (dims.length === 3 && dims[2] === 6 && dims[1] <= 1000) {
    // NMS-free export: rows of x1, y1, x2, y2, score, class.
    for (let i = 0; i < dims[1]; i++) {
      const o = i * 6, score = data[o + 4], c = Math.round(data[o + 5]);
      if (score < minScore || c < 0 || c >= classMap.length || classMap[c] < 0) continue;
      const bw = data[o + 2] - data[o], bh = data[o + 3] - data[o + 1];
      push(data[o] + bw / 2, data[o + 1] + bh / 2, bw, bh, score, classMap[c], c);
    }
  } else if (dims.length === 3) {
    // Ultralytics export: [1, 4 + classes, N], or the transposed [1, N, 4 + classes].
    const chanFirst = dims[1] < dims[2];
    const N = chanFirst ? dims[2] : dims[1], C = chanFirst ? dims[1] : dims[2], nc = C - 4;
    const at = chanFirst ? (i, c) => data[c * N + i] : (i, c) => data[i * C + c];
    for (let i = 0; i < N; i++) {
      let best = 0, bc = -1;
      for (let c = 0; c < nc; c++) {
        if (c >= classMap.length || classMap[c] < 0) continue;
        const v = at(i, 4 + c);
        if (v > best) { best = v; bc = c; }
      }
      if (bc < 0 || best < minScore) continue;
      push(at(i, 0), at(i, 1), at(i, 2), at(i, 3), best, classMap[bc], bc);
    }
  } else {
    throw new Error(`Unsupported model output shape [${Array.from(dims).join(', ')}]`);
  }
  return nms(dets);
}

function relDiff(a, b) {
  if (a.length !== b.length) return 1;
  let d = 0, m = 0;
  for (let i = 0; i < a.length; i++) { d += Math.abs(a[i] - b[i]); m += Math.abs(b[i]); }
  return d / (m + 1e-9);
}

async function onFrame(m) {
  const t0 = performance.now();
  const { tensor, map } = preprocess(m.bitmap);
  m.bitmap.close();
  let out;
  try {
    out = await runSession(session, tensor);
  } catch (e) {
    if (backend !== 'webgpu') throw e;
    // GPU failed mid-run (driver reset, device lost): continue on the processor.
    try { await session.release(); } catch (e2) { /* ignore */ }
    session = await createSession('wasm');
    backend = 'wasm'; verified = true;
    post({ type: 'backend', backend, note: 'Graphics acceleration stopped working; continuing on the processor.' });
    out = await runSession(session, tensor);
  }
  if (!verified) {
    verified = true;
    try {
      const ref = await createSession('wasm');
      const refOut = await runSession(ref, tensor);
      const diff = relDiff(out.data, refOut.data);
      if (diff > 0.02) {
        try { await session.release(); } catch (e2) { /* ignore */ }
        session = ref; backend = 'wasm'; out = refOut;
        post({ type: 'backend', backend, note: 'Graphics results did not match the processor check, so the processor is used for reliable counts.' });
      } else {
        await ref.release();
        post({ type: 'verified', diff });
      }
    } catch (e) { /* keep the GPU session if the check itself cannot run */ }
  }
  const dets = decode(out, map);
  post({ type: 'result', id: m.id, t: m.t, dets, ms: performance.now() - t0, backend });
}

self.onmessage = (e) => {
  const m = e.data;
  chain = chain.then(async () => {
    try {
      if (m.type === 'init') await init(m);
      else if (m.type === 'frame') await onFrame(m);
    } catch (err) {
      post({ type: 'error', stage: m.type, id: m.id, message: (err && err.message) || String(err) });
    }
  });
};
