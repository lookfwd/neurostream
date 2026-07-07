// BSS-CCA artifact removal (De Clercq-style). Per window, decompose the multichannel
// signal into maximally auto-correlated components via canonical correlation between the
// data and its 1-sample-lagged copy. Components with LOW canonical correlation are
// low-autocorrelation / broadband (muscle-like) and are removed; the rest are
// reconstructed. Needs multiple channels to separate sources.
//
// params: { threshold:0..1=0.9, remove:int=null }
//   remove N lowest-correlation components if `remove` is set, else remove those with
//   canonical correlation below `threshold`.

import { matmul, transpose, inverse, cholesky, jacobiEigen } from '../linalg.js';

export class CcaBackend {
  constructor({ channels, window, params = {} }) {
    this.mode = 'window';
    this.C = channels;
    this.W = window;
    this.threshold = params.threshold ?? 0.9;
    this.remove = params.remove ?? null;
    this.info = { type: 'cca', method: `BSS-CCA (${this.remove != null ? this.remove + ' comps' : 'ρ<' + this.threshold})`, ep: 'cpu' };
  }
  async init() { return this.info; }

  process({ start, data }) {
    const C = this.C, W = this.W;
    // per-channel mean-centered matrix X (C×W, row-major), keep means to add back
    const X = new Float64Array(C * W), mean = new Float64Array(C);
    for (let c = 0; c < C; c++) {
      let m = 0; const off = c * W;
      for (let n = 0; n < W; n++) m += data[off + n];
      m /= W; mean[c] = m;
      for (let n = 0; n < W; n++) X[c * W + n] = data[off + n] - m;
    }
    // lagged copy Y (1-sample): Y[:,n] = X[:,n-1], Y[:,0]=X[:,0]
    const Y = new Float64Array(C * W);
    for (let c = 0; c < C; c++) { Y[c * W] = X[c * W]; for (let n = 1; n < W; n++) Y[c * W + n] = X[c * W + n - 1]; }

    const Cxx = cov(X, X, C, W), Cyy = cov(Y, Y, C, W), Cxy = cov(X, Y, C, W), Cyx = transpose(Cxy, C, C);

    // symmetric CCA eigenproblem: K = Lx⁻¹ Cxy Cyy⁻¹ Cyx Lx⁻ᵀ  (Cxx = Lx Lxᵀ)
    const Lx = cholesky(Cxx, C);
    const Lxi = inverse(Lx, C);
    const Cyyi = inverse(Cyy, C);
    const M = matmul(matmul(Cxy, Cyyi, C, C, C), Cyx, C, C, C);   // Cxy Cyy⁻¹ Cyx
    const K = matmul(matmul(Lxi, M, C, C, C), transpose(Lxi, C, C), C, C, C);
    // symmetrize against round-off
    for (let i = 0; i < C; i++) for (let j = i + 1; j < C; j++) { const a = 0.5 * (K[i * C + j] + K[j * C + i]); K[i * C + j] = K[j * C + i] = a; }

    const { values, vectors } = jacobiEigen(K, C);            // values = ρ², vectors columns = u_k
    const rho = values.map((v) => Math.sqrt(Math.max(0, v)));

    // demixing W_ (rows w_k = Lx⁻ᵀ u_k)
    const LxiT = transpose(Lxi, C, C);
    const Wd = new Float64Array(C * C);
    for (let k = 0; k < C; k++) for (let i = 0; i < C; i++) { let s = 0; for (let j = 0; j < C; j++) s += LxiT[i * C + j] * vectors[j * C + k]; Wd[k * C + i] = s; }

    // sources S = Wd · X  (C×W)
    const S = matmul(Wd, X, C, C, W);

    // choose components to remove (low canonical correlation)
    const order = rho.map((r, k) => [r, k]).sort((a, b) => a[0] - b[0]);
    const kill = new Uint8Array(C);
    if (this.remove != null) { for (let n = 0; n < Math.min(this.remove, C); n++) kill[order[n][1]] = 1; }
    else { for (let k = 0; k < C; k++) if (rho[k] < this.threshold) kill[k] = 1; }
    for (let k = 0; k < C; k++) if (kill[k]) for (let n = 0; n < W; n++) S[k * W + n] = 0;

    // reconstruct X_clean = A · S,  A = Wd⁻¹
    const A = inverse(Wd, C);
    const Xc = matmul(A, S, C, C, W);

    const out = new Float32Array(C * W);
    for (let c = 0; c < C; c++) for (let n = 0; n < W; n++) out[c * W + n] = Xc[c * W + n] + mean[c];
    return { start, data: out };
  }
  dispose() {}
}

// covariance A Bᵀ / W  -> C×C (A, B are C×W row-major)
function cov(A, B, C, W) {
  const out = new Float64Array(C * C);
  for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) {
    let s = 0; const ai = i * W, bj = j * W;
    for (let n = 0; n < W; n++) s += A[ai + n] * B[bj + n];
    out[i * C + j] = s / W;
  }
  return out;
}
