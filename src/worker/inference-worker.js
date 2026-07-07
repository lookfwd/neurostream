// Runs the ONNX denoiser off the main thread. Imports onnxruntime-web from a CDN
// (override ORT_URL below to self-host). Supports two model kinds:
//   single-channel: input [1,1,W], run once per channel (respects static batch=1)
//   multichannel  : input [1,C,W], run once per window
//
// Messages in:  { type:'init', modelUrl, kind, channels, window, ep }
//               { type:'infer', id, data(Float32 C*W), channels, window }
// Messages out: { type:'ready', ep, inputName, outputName } | { type:'result', id, data }
//               { type:'error', message }

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.mjs';

let ort = null, session = null, inName = null, outName = null;
let KIND = 'single-channel', C = 8, W = 1024, activeEP = 'wasm', NORM = 'none';

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      ort = await import(ORT_URL);
      // Let ORT fetch its wasm assets from the same CDN.
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
      KIND = msg.kind; C = msg.channels; W = msg.window; NORM = msg.normalize || 'none';
      const eps = (msg.ep || ['webgpu', 'wasm']).map((n) => (n === 'webgpu' ? 'webgpu' : 'wasm'));
      session = await ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: eps,
        graphOptimizationLevel: 'all',
      });
      inName = session.inputNames[0];
      outName = session.outputNames[0];
      activeEP = eps[0]; // best-effort; ORT falls back internally if needed
      // warm-up
      await runWindow(new Float32Array(C * W));
      self.postMessage({ type: 'ready', ep: activeEP, inputName: inName, outputName: outName });
    } else if (msg.type === 'infer') {
      const out = await runWindow(msg.data);
      self.postMessage({ type: 'result', id: msg.id, data: out }, [out.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};

async function runWindow(flat) {
  if (KIND === 'multichannel') {
    // z-score the whole [C,W] block, run, then rescale — mirrors IC-U-Net's reference
    // reconstruct(): (x-mean)/std before the net, *std (+mean) after.
    const { x, mean, std } = normStats(flat, 0, flat.length);
    const t = new ort.Tensor('float32', x, [1, C, W]);
    const res = await session.run({ [inName]: t });
    return denorm(Float32Array.from(res[outName].data), mean, std);
  }
  // single-channel: loop channels, reuse a [1,1,W] tensor; normalize per channel
  const out = new Float32Array(C * W);
  for (let c = 0; c < C; c++) {
    const { x, mean, std } = normStats(flat, c * W, W);
    const t = new ort.Tensor('float32', x, [1, 1, W]);
    const res = await session.run({ [inName]: t });
    out.set(denorm(Float32Array.from(res[outName].data), mean, std), c * W);
  }
  return out;
}

// Copy [off, off+len) out of `flat`; z-score it when NORM==='zscore', else pass through.
function normStats(flat, off, len) {
  if (NORM !== 'zscore') return { x: flat.subarray(off, off + len).slice(), mean: 0, std: 1 };
  let mean = 0;
  for (let i = 0; i < len; i++) mean += flat[off + i];
  mean /= len;
  let v = 0;
  for (let i = 0; i < len; i++) { const d = flat[off + i] - mean; v += d * d; }
  const std = Math.sqrt(v / len) || 1;
  const x = new Float32Array(len);
  for (let i = 0; i < len; i++) x[i] = (flat[off + i] - mean) / std;
  return { x, mean, std };
}
function denorm(o, mean, std) {
  if (NORM !== 'zscore') return o;
  for (let i = 0; i < o.length; i++) o[i] = o[i] * std + mean;
  return o;
}
