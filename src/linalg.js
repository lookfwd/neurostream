// Compact dense linear algebra for small matrices (channel counts <= 16).
// Matrices are Float64Array in row-major order with an explicit dimension n (square)
// unless noted. Enough for covariance work and the CCA generalized eigenproblem.

export function zeros(n, m = n) { return new Float64Array(n * m); }
export function eye(n) { const A = zeros(n); for (let i = 0; i < n; i++) A[i * n + i] = 1; return A; }

export function matmul(A, B, n, k, m) {
  // A: n×k, B: k×m -> n×m
  const C = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p]; if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i * m + j] += a * B[p * m + j];
    }
  }
  return C;
}
export function transpose(A, n, m) {
  const T = new Float64Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j * n + i] = A[i * m + j];
  return T;
}

// Inverse of an n×n matrix via Gauss-Jordan with partial pivoting.
export function inverse(A, n) {
  const M = Float64Array.from(A);
  const I = eye(n);
  for (let col = 0; col < n; col++) {
    let piv = col, best = Math.abs(M[col * n + col]);
    for (let r = col + 1; r < n; r++) { const v = Math.abs(M[r * n + col]); if (v > best) { best = v; piv = r; } }
    if (best < 1e-12) { M[col * n + col] += 1e-9; } // regularize near-singular
    if (piv !== col) { swapRows(M, n, col, piv); swapRows(I, n, col, piv); }
    const d = M[col * n + col];
    for (let j = 0; j < n; j++) { M[col * n + j] /= d; I[col * n + j] /= d; }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r * n + col];
      if (f === 0) continue;
      for (let j = 0; j < n; j++) { M[r * n + j] -= f * M[col * n + j]; I[r * n + j] -= f * I[col * n + j]; }
    }
  }
  return I;
}
function swapRows(M, n, a, b) { for (let j = 0; j < n; j++) { const t = M[a * n + j]; M[a * n + j] = M[b * n + j]; M[b * n + j] = t; } }

// Lower-triangular Cholesky: A (SPD) = L Lᵀ. Adds jitter if not quite PD.
export function cholesky(A, n) {
  const L = zeros(n);
  let jitter = 0;
  for (let attempt = 0; attempt < 5; attempt++) {
    L.fill(0); let okay = true;
    for (let i = 0; i < n && okay; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j] + (i === j ? jitter : 0);
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) { if (s <= 0) { okay = false; break; } L[i * n + j] = Math.sqrt(s); }
        else L[i * n + j] = s / L[j * n + j];
      }
    }
    if (okay) return L;
    jitter = jitter === 0 ? 1e-9 : jitter * 10;
  }
  return L;
}

// Symmetric eigen-decomposition via cyclic Jacobi. Returns { values:[], vectors }
// where vectors is n×n column-major eigenvectors (column k is eigenvector k).
export function jacobiEigen(Ain, n, sweeps = 60) {
  const A = Float64Array.from(Ain);
  const V = eye(n);
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[p * n + p], aqq = A[q * n + q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi), sn = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - sn * akq;
          A[k * n + q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - sn * aqk;
          A[q * n + k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - sn * vkq;
          V[k * n + q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  return { values, vectors: V };
}
