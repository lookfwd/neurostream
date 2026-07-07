// Adaptive noise cancellation with a normalized LMS (NLMS) FIR filter.
// A reference channel (e.g. a frontal/EOG-like electrode) drives an adaptive filter
// whose output is subtracted from each target channel, cancelling activity correlated
// with the reference (classically ocular artifacts).
//
// params: { reference:int=0, order:int=8, mu:0..2=0.3, clean:'others'|'all'='others' }
// The reference channel itself is passed through unchanged when clean='others'.

export class LmsBackend {
  constructor({ channels, sampleRate, params = {} }) {
    this.mode = 'stream';
    this.channels = channels;
    this.ref = Math.min(channels - 1, params.reference ?? 0);
    this.order = params.order ?? 8;
    this.mu = params.mu ?? 0.1;
    this.cleanAll = (params.clean ?? 'others') === 'all';
    // per-target-channel weights and shared reference history
    this.w = Array.from({ length: channels }, () => new Float32Array(this.order));
    this.hist = new Float32Array(this.order); // most-recent-first ring, simple shift
    this.avgPow = 1;                           // running ||ref history||² estimate
    this.info = { type: 'lms', method: `NLMS (ref ch${this.ref}, order ${this.order})`, ep: 'cpu', latencySec: 0 };
  }
  async init() { return this.info; }

  processStream(channels) {
    const n = channels[0].length;
    const out = Array.from({ length: this.channels }, () => new Float32Array(n));
    for (let i = 0; i < n; i++) {
      // push reference sample into history (shift)
      for (let k = this.order - 1; k > 0; k--) this.hist[k] = this.hist[k - 1];
      this.hist[0] = channels[this.ref][i];
      let inst = 0; for (let k = 0; k < this.order; k++) inst += this.hist[k] * this.hist[k];
      this.avgPow += 0.01 * (inst - this.avgPow);
      // regularize the normalizer by a running power floor -> stable at low instantaneous power
      const denom = inst + 0.1 * this.avgPow + 1e-6;

      for (let c = 0; c < this.channels; c++) {
        if (c === this.ref && !this.cleanAll) { out[c][i] = channels[c][i]; continue; }
        const w = this.w[c];
        let yhat = 0; for (let k = 0; k < this.order; k++) yhat += w[k] * this.hist[k];
        const e = channels[c][i] - yhat;   // cleaned = error signal
        out[c][i] = e;
        const g = (this.mu * e) / denom;   // regularized NLMS update
        for (let k = 0; k < this.order; k++) w[k] += g * this.hist[k];
      }
    }
    return out;
  }
  dispose() {}
}
