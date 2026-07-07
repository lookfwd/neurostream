// ATAR — Automatic and Tunable Artifact Removal (wavelet-domain). Per channel per
// window: db4 DWT, apply a tunable operator to detail coefficients so that
// high-amplitude transients (blink/motion) are attenuated, then inverse DWT.
//
// params: { levels:4, k:1..10=5, mode:'linatten'|'elim'|'clamp'='linatten' }
//   k     : threshold in robust-sigma units (θ = k·σ per level; higher k = gentler)
//   mode  : how coefficients above θ are treated.

import { dwt, idwt, madSigma } from '../wavelet.js';

export class AtarBackend {
  constructor({ channels, window, params = {} }) {
    this.mode = 'window';
    this.C = channels;
    this.W = window;
    this.levels = params.levels ?? 4;
    this.k = params.k ?? 5;
    this.op = params.mode ?? 'linatten';
    this.info = { type: 'atar', method: `ATAR (db4, ${this.op}, k=${this.k})`, ep: 'cpu' };
  }
  async init() { return this.info; }

  process({ start, data }) {
    const C = this.C, W = this.W;
    const out = new Float32Array(C * W);
    const chan = new Float64Array(W);
    for (let c = 0; c < C; c++) {
      for (let n = 0; n < W; n++) chan[n] = data[c * W + n];
      const dec = dwt(chan, this.levels);
      for (const d of dec.details) {
        const sigma = madSigma(d) || 1e-9;
        const theta = this.k * sigma;
        for (let i = 0; i < d.length; i++) d[i] = this._operator(d[i], theta);
      }
      // low-frequency transients (blink/step) land in the approximation band too
      const aSig = madSigma(dec.approx) || 1e-9;
      const aTheta = this.k * aSig;
      for (let i = 0; i < dec.approx.length; i++) dec.approx[i] = this._operator(dec.approx[i], aTheta);
      const rec = idwt(dec);
      for (let n = 0; n < W; n++) out[c * W + n] = rec[n];
    }
    return { start, data: out };
  }

  _operator(w, theta) {
    const a = Math.abs(w);
    if (a <= theta) return w;
    if (this.op === 'elim') return 0;                         // drop artifact coeffs
    if (this.op === 'clamp') return Math.sign(w) * theta;     // limit magnitude to θ
    // linatten: keep θ, decay the excess smoothly
    return Math.sign(w) * (theta + (a - theta) * (theta / a) * 0.25);
  }
  dispose() {}
}
