// Central configuration: defaults, validation, and derived quantities.
// Everything that shapes latency/throughput is computed here so the UI can show it.

export const DEFAULTS = {
  channels: 8,                 // 4 | 8 | 16 supported end to end
  sampleRate: 256,             // Hz the MODEL expects; source is resampled to this
  windowSec: 4.0,              // MUST equal the model's trained segment length
  overlap: 0.80,               // fraction; hop = window * (1 - overlap)
  taper: 'tukey',              // 'tukey' | 'hann' | 'rect' — overlap-add cross-fade
  taperAlpha: 0.5,             // tukey flat-fraction parameter

  ep: ['webgpu', 'wasm'],      // onnxruntime-web execution provider preference

  source: {                    // where samples come from
    type: 'synthetic',         // 'synthetic' | 'mqtt'
    // synthetic:
    artifacts: true,           // inject blink + EMG bursts
    spatial: true,             // low-rank spatially-correlated sources (EEG-like) vs per-channel independent
    chunk: 32,                 // samples per emitted chunk
    rate: null,                // source Hz; null => same as model sampleRate
    // mqtt:
    url: 'wss://broker.example:8084/mqtt',
    topicIn: 'eeg/raw',
    format: 'binary',          // 'binary' | 'json'
  },

  model: {                     // the denoising method (backend)
    // type: 'passthrough' | 'onnx' | 'ae' | 'dae' | 'novelcnn'   (neural → ONNX)
    //     | 'lms' | 'ssp' | 'oscar' | 'filter' | 'movingaverage' | 'cca' | 'atar'
    type: 'passthrough',
    kind: 'single-channel',    // onnx: 'single-channel' | 'multichannel'
    url: null,                 // onnx/ae/dae/novelcnn: path to model.onnx
    normalize: 'zscore',       // onnx: 'zscore' (per-window standardize→model→rescale) | 'none'
    params: {},                // method-specific params (see each backend / README)
  },

  output: {                    // optional republish of the cleaned stream
    republish: false,
    topicOut: 'eeg/clean',
  },

  channelMap: null,            // optional int[] reorder/subset: out[i] = in[channelMap[i]]
  filter: { highpass: 0, notch: 0 }, // Hz; 0 disables. e.g. { highpass: 1, notch: 50 }
};

function deepMerge(base, over) {
  if (over == null) return structuredCloneSafe(base);
  const out = structuredCloneSafe(base);
  for (const k of Object.keys(over)) {
    const v = over[k];
    out[k] = (v && typeof v === 'object' && !Array.isArray(v)) ? deepMerge(base[k] ?? {}, v) : v;
  }
  return out;
}
function structuredCloneSafe(o) {
  return (typeof structuredClone === 'function') ? structuredClone(o) : JSON.parse(JSON.stringify(o));
}

export function resolveConfig(user = {}) {
  const c = deepMerge(DEFAULTS, user);

  if (![4, 8, 16, 30].includes(c.channels)) {
    throw new Error(`channels must be 4, 8, 16, or 30 (got ${c.channels})`);
  }
  if (c.overlap < 0 || c.overlap >= 1) throw new Error('overlap must be in [0, 1)');
  if (c.windowSec <= 0 || c.sampleRate <= 0) throw new Error('windowSec and sampleRate must be > 0');

  const windowSamples = Math.round(c.windowSec * c.sampleRate);
  const hopSamples = Math.max(1, Math.round(windowSamples * (1 - c.overlap)));
  const sourceRate = c.source.rate ?? c.sampleRate;

  c._derived = {
    windowSamples,
    hopSamples,
    hopSec: hopSamples / c.sampleRate,
    sourceRate,
    inferencesPerSec: c.sampleRate / hopSamples,
    averaging: windowSamples / hopSamples,       // times each sample is cleaned & averaged
    latencyMinSec: (windowSamples / 2) / c.sampleRate, // well-supported estimate lag
    latencyMaxSec: windowSamples / c.sampleRate,       // full-window lag
  };
  return c;
}
