// Signal-Space Projection. Given one or more artifact spatial patterns (topographies)
// measured during calibration — e.g. the scalp distribution of a blink — build the
// projector P = I - U Uᵀ (U orthonormal) and apply it to every sample vector, removing
// the artifact subspace. Memoryless and per-sample: no added latency.
//
// params: { projectors: number[][] }  each vector has length = channels.
// With no projectors it is identity (and says so), since SSP needs calibrated patterns.

function orthonormalize(vectors, n) {
  const basis = [];
  for (let v of vectors) {
    const u = Float64Array.from(v);
    for (const b of basis) {
      let dot = 0; for (let i = 0; i < n; i++) dot += u[i] * b[i];
      for (let i = 0; i < n; i++) u[i] -= dot * b[i];
    }
    let norm = 0; for (let i = 0; i < n; i++) norm += u[i] * u[i];
    norm = Math.sqrt(norm);
    if (norm > 1e-8) { for (let i = 0; i < n; i++) u[i] /= norm; basis.push(u); }
  }
  return basis;
}

export class SspBackend {
  constructor({ channels, params = {} }) {
    this.mode = 'stream';
    this.channels = channels;
    this.basis = orthonormalize(params.projectors || [], channels);
    this.info = {
      type: 'ssp',
      method: this.basis.length ? `SSP (${this.basis.length} component${this.basis.length > 1 ? 's' : ''} removed)` : 'SSP (identity — no projectors given)',
      ep: 'cpu', latencySec: 0,
    };
  }
  async init() { return this.info; }

  processStream(channels) {
    const n = channels[0].length, C = this.channels;
    const out = Array.from({ length: C }, () => new Float32Array(n));
    if (this.basis.length === 0) { for (let c = 0; c < C; c++) out[c].set(channels[c]); return out; }
    const x = new Float64Array(C);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < C; c++) x[c] = channels[c][i];
      // y = x - sum_b (bᵀx) b
      for (const b of this.basis) {
        let dot = 0; for (let c = 0; c < C; c++) dot += b[c] * x[c];
        for (let c = 0; c < C; c++) x[c] -= dot * b[c];
      }
      for (let c = 0; c < C; c++) out[c][i] = x[c];
    }
    return out;
  }
  dispose() {}

  // Helper: build a projector vector from an average artifact topography (e.g. mean
  // scalp pattern over detected blinks). Returns a length-`channels` array.
  static topography(vec) { return Array.from(vec); }
}
