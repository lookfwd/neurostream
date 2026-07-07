// ORICA — Online Recursive ICA for real-time EEG source separation + artifact removal.
//
// This follows the ORICA structure of Hsu, Mullen, Jung & Cauwenberghs (2016): an online
// whitening stage plus a natural-gradient demixing update with a kurtosis-sign tanh
// nonlinearity and a cooling forgetting factor. Rather than the literal RLS one-liner
// (which diverges easily if mis-tuned), whitening is a running-covariance EMA reduced to
// C^-1/2, and the demixing update is the orthogonal natural gradient  ΔW = μ (ygᵀ − gyᵀ)W
// (skew-symmetric → keeps W orthogonal on whitened data), with periodic re-orthonormalization
// for numerical safety. Equivariant and stable; faithful to ORICA in behaviour, not a
// bit-exact port of the reference MATLAB.
//
// It is a SEPARATOR, so to clean it also: tracks each component's running kurtosis, flags
// high-kurtosis (spiky, e.g. blink/ocular) components as artifacts, zeroes them, and remixes
// back to sensor space. Convergence takes ~k·N² samples (k≈25) — reported as `convergence`
// so the UI can show it; rejection is held off until the unmixing has settled.
//
// params: { mu:0..0.05=0.008, gamma:0.6, whiten:0..1=0.02, kurtosis:>0=6, convSamples:int }

import { matmul, inverse, jacobiEigen, eye } from '../linalg.js';

export class OricaBackend {
  constructor({ channels, sampleRate, params = {} }) {
    this.mode = 'stream';
    const N = channels;
    this.N = N;
    this.mu0 = params.mu ?? 0.008;               // ICA learning rate (decays with cooling)
    this.gamma = params.gamma ?? 0.6;            // cooling exponent
    this.aCov = params.whiten ?? 0.02;           // whitening covariance EMA rate
    this.kReject = params.kurtosis ?? 10;        // excess-kurtosis threshold => artifact
    this.kAdapt = params.kurtMemory ?? 0.001;    // kurtosis EMA rate (~4 s: spans blink gaps)
    this.convSamples = params.convSamples ?? 25 * N * N;   // k·N² rule of thumb
    this.rejectAt = 0.25;                        // fraction converged before rejecting

    this.mean = new Float64Array(N);
    this.cov = eye(N);                           // running covariance estimate
    this.P = eye(N);                             // whitening matrix C^-1/2
    this.W = eye(N);                             // demixing on whitened data (orthogonal)
    this.m2 = new Float64Array(N).fill(1);       // running variance per component
    this.m4 = new Float64Array(N).fill(3);       // running 4th moment per component
    this.kurt = new Float64Array(N);             // running excess kurtosis
    this.kurtSign = new Int8Array(N).fill(1);    // +1 super-Gaussian, -1 sub-Gaussian
    this.seen = 0;
    this.block = 0;
    this.info = {
      type: 'orica', method: `ORICA (kurtosis reject, k=${this.kReject})`,
      ep: 'cpu', latencySec: 0, convergence: 0, rejected: 0,
    };
  }
  async init() { return this.info; }

  processStream(channels) {
    const N = this.N, n = channels[0].length;

    // 1. center with a slow running mean, build Xc (N rows × n)
    const X = [];
    for (let c = 0; c < N; c++) {
      let s = 0; const src = channels[c];
      for (let i = 0; i < n; i++) s += src[i];
      this.mean[c] += 0.01 * (s / n - this.mean[c]);
      const row = new Float64Array(n), m = this.mean[c];
      for (let i = 0; i < n; i++) row[i] = src[i] - m;
      X.push(row);
    }

    // 2. running covariance EMA
    for (let a = 0; a < N; a++) {
      for (let b = a; b < N; b++) {
        let s = 0; const xa = X[a], xb = X[b];
        for (let i = 0; i < n; i++) s += xa[i] * xb[i];
        const v = (1 - this.aCov) * this.cov[a * N + b] + this.aCov * (s / n);
        this.cov[a * N + b] = v; this.cov[b * N + a] = v;
      }
    }
    // 3. refresh whitening P = cov^-1/2 periodically (eigen is the costly step)
    if (this.block % 4 === 0) this.P = symInvSqrt(this.cov, N);

    // 4. whiten, 5. estimate sources
    const V = mulCols(this.P, X, N, n);
    const Y = mulCols(this.W, V, N, n);

    // 6. nonlinearity + orthogonal natural-gradient demixing update
    const G = [];
    for (let c = 0; c < N; c++) {
      const g = new Float64Array(n), k = this.kurtSign[c], y = Y[c];
      for (let i = 0; i < n; i++) g[i] = k * Math.tanh(y[i]);
      G.push(g);
    }
    const B = new Float64Array(N * N);            // (1/n)(Y Gᵀ − G Yᵀ), skew-symmetric
    for (let a = 0; a < N; a++) {
      for (let b = 0; b < N; b++) {
        let s = 0; const ya = Y[a], ga = G[a], yb = Y[b], gb = G[b];
        for (let i = 0; i < n; i++) s += ya[i] * gb[i] - ga[i] * yb[i];
        B[a * N + b] = s / n;
      }
    }
    const BW = matmul(B, this.W, N, N, N);
    const mu = Math.max(this.mu0 * 0.05,
      this.mu0 / Math.pow(1 + this.seen / (N * N), this.gamma));
    // cap the Frobenius step so an outlier block (e.g. an electrode pop) can't blow W up
    let dn = 0; for (let idx = 0; idx < N * N; idx++) { const d = mu * BW[idx]; dn += d * d; }
    const step = dn > 0.01 ? mu * Math.sqrt(0.01 / dn) : mu;   // ||ΔW||_F ≤ 0.1
    for (let idx = 0; idx < N * N; idx++) this.W[idx] += step * BW[idx];
    // re-orthonormalize periodically: W <- (W Wᵀ)^-1/2 W  (keeps W orthogonal, bounded)
    if (this.block % 8 === 0) this.W = matmul(symInvSqrt(gram(this.W, N), N), this.W, N, N, N);

    // 7. running kurtosis per component -> sign (for the nonlinearity) and artifact flag
    for (let c = 0; c < N; c++) {
      const y = Y[c];
      for (let i = 0; i < n; i++) {
        const y2 = y[i] * y[i];
        this.m2[c] += this.kAdapt * (y2 - this.m2[c]);
        this.m4[c] += this.kAdapt * (y2 * y2 - this.m4[c]);
      }
      this.kurt[c] = this.m4[c] / (this.m2[c] * this.m2[c] + 1e-12) - 3;
      this.kurtSign[c] = this.kurt[c] >= 0 ? 1 : -1;
    }

    // 8. convergence bookkeeping
    this.seen += n; this.block++;
    const conv = Math.min(1, this.seen / this.convSamples);
    this.info.convergence = +conv.toFixed(3);

    // 9. artifact mask (once settled): reject components whose long-memory kurtosis exceeds
    //    the threshold, but only the MOST kurtotic few. The long memory keeps a genuine
    //    artifact component's kurtosis stably high (no flicker); the cap stops it from
    //    nuking half the subspace when artifact energy smears across many components at low
    //    channel counts. Cap grows with channels since an impulse spreads across more ICs.
    const keep = new Float64Array(N).fill(1);
    let rejected = 0;
    if (conv >= this.rejectAt) {
      const cand = [];
      for (let c = 0; c < N; c++) if (this.kurt[c] > this.kReject) cand.push(c);
      cand.sort((a, b) => this.kurt[b] - this.kurt[a]);
      const cap = Math.min(N - 1, Math.max(2, Math.floor(N / 8)));
      for (let j = 0; j < Math.min(cap, cand.length); j++) { keep[cand[j]] = 0; rejected++; }
    }
    this.info.rejected = rejected;

    // 10. remix: cleaned = A⁻¹ (keep ∘ Y) + mean,  A = W P  (unmixing)
    const Ainv = inverse(matmul(this.W, this.P, N, N, N), N);
    const out = Array.from({ length: N }, () => new Float32Array(n));
    for (let c = 0; c < N; c++) {
      const dst = out[c], m = this.mean[c];
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let k = 0; k < N; k++) if (keep[k]) s += Ainv[c * N + k] * Y[k][i];
        dst[i] = s + m;
      }
    }
    return out;
  }
  dispose() {}
}

// M (N×N row-major) times columns `cols` (N rows each length n) -> N rows each length n.
function mulCols(M, cols, N, n) {
  const out = [];
  for (let a = 0; a < N; a++) {
    const row = new Float64Array(n);
    for (let k = 0; k < N; k++) {
      const m = M[a * N + k]; if (m === 0) continue;
      const ck = cols[k];
      for (let i = 0; i < n; i++) row[i] += m * ck[i];
    }
    out.push(row);
  }
  return out;
}
// Symmetric M^-1/2 via eigen, flooring eigenvalues for stability.
function symInvSqrt(M, N) {
  const { values, vectors } = jacobiEigen(M, N);
  // floor eigenvalues RELATIVE to the largest — regularized whitening. Low-rank data
  // (30 channels, ~few real sources) has near-zero eigenvalues; without this the whitening
  // amplifies those empty directions without bound and the demixing update diverges.
  let maxv = 0;
  for (let p = 0; p < N; p++) if (values[p] > maxv) maxv = values[p];
  const floor = Math.max(1e-9, 1e-3 * maxv);
  const inv = new Float64Array(N);
  for (let p = 0; p < N; p++) inv[p] = 1 / Math.sqrt(Math.max(values[p], floor));
  const R = new Float64Array(N * N);
  for (let a = 0; a < N; a++) {
    for (let b = a; b < N; b++) {
      let s = 0;
      for (let p = 0; p < N; p++) s += vectors[a * N + p] * inv[p] * vectors[b * N + p];
      R[a * N + b] = s; R[b * N + a] = s;
    }
  }
  return R;
}
// Gram matrix W Wᵀ (N×N).
function gram(W, N) {
  const G = new Float64Array(N * N);
  for (let a = 0; a < N; a++) {
    for (let b = a; b < N; b++) {
      let s = 0;
      for (let k = 0; k < N; k++) s += W[a * N + k] * W[b * N + k];
      G[a * N + b] = s; G[b * N + a] = s;
    }
  }
  return G;
}
