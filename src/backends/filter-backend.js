// Classic digital filtering as a denoising method. Two flavours via params.kind:
//   'movingaverage' : boxcar smoother of params.taps samples (removes high-freq/EMG-ish)
//   'iir'           : biquad chain, params.highpass / notch / lowpass in Hz
// Causal and per-sample: adds only the filter's group delay, no windowing latency.

import { FilterChain, MovingAverage } from '../filters.js';

export class FilterBackend {
  constructor({ channels, sampleRate, params = {} }) {
    this.mode = 'stream';
    this.channels = channels;
    const kind = params.kind || 'iir';
    this.kind = kind;
    if (kind === 'movingaverage') {
      const taps = params.taps || 8;
      this.units = Array.from({ length: channels }, () => new MovingAverage(taps));
      this.info = { type: 'filter', method: `moving average (${taps} taps)`, ep: 'cpu', latencySec: (taps / 2) / sampleRate };
    } else {
      const spec = { highpass: params.highpass || 0, notch: params.notch || 0, lowpass: params.lowpass || 40 };
      this.units = Array.from({ length: channels }, () => new FilterChain(sampleRate, spec));
      this.info = { type: 'filter', method: 'IIR digital filter', ep: 'cpu', latencySec: 0.01 };
    }
  }
  async init() { return this.info; }
  processStream(channels) {
    const out = [];
    for (let c = 0; c < this.channels; c++) {
      const src = channels[c], u = this.units[c], dst = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) dst[i] = u.process(src[i]);
      out.push(dst);
    }
    return out;
  }
  dispose() {}
}
