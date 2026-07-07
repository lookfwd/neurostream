// OSCAR — online signal conditioning + artifact removal.
// OSCAR is described as a real-time preprocessing module that conditions the signal and
// suppresses transient artifacts (blink, motion, cable) as samples arrive. There isn't a
// single canonical published equation, so this is a faithful composite of the stages such
// a module performs, implemented causally per channel:
//   1. conditioning: DC-blocking high-pass + optional line-noise notch
//   2. robust level tracking: running estimate of baseline and scale (EMA of |x|)
//   3. transient suppression: samples exceeding k·scale are soft-limited toward baseline,
//      so large excursions (artifacts) are attenuated while ordinary EEG passes through.
//
// params: { highpass:Hz=1, notch:Hz=0, k:2..8=4, adapt:0..1=0.02 }

import { FilterChain } from '../filters.js';

export class OscarBackend {
  constructor({ channels, sampleRate, params = {} }) {
    this.mode = 'stream';
    this.channels = channels;
    this.k = params.k ?? 4;
    this.adapt = params.adapt ?? 0.02;
    const spec = { highpass: params.highpass ?? 1, notch: params.notch ?? 0 };
    this.cond = Array.from({ length: channels }, () => new FilterChain(sampleRate, spec));
    this.scale = new Float32Array(channels).fill(1);   // running robust scale
    this.info = { type: 'oscar', method: `OSCAR (condition + suppress, k=${this.k})`, ep: 'cpu', latencySec: 0.01 };
  }
  async init() { return this.info; }

  processStream(channels) {
    const n = channels[0].length;
    const out = Array.from({ length: this.channels }, () => new Float32Array(n));
    for (let c = 0; c < this.channels; c++) {
      const src = channels[c], dst = out[c], cond = this.cond[c];
      for (let i = 0; i < n; i++) {
        const y = cond.process(src[i]);           // conditioned (DC-blocked, notched)
        const a = Math.abs(y);
        // update robust scale slowly toward |y|
        this.scale[c] += this.adapt * (a - this.scale[c]);
        const thr = this.k * this.scale[c];
        // soft-limit excursions beyond threshold (tanh knee), pass normal signal
        dst[i] = a <= thr ? y : Math.sign(y) * (thr + (a - thr) * (1 / (1 + (a - thr) / (thr + 1e-6))) * 0.15);
      }
    }
    return out;
  }
  dispose() {}
}
