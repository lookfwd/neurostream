// Daubechies-4 (db4) discrete wavelet transform with periodic boundary handling.
// Perfect-reconstruction: idwt(dwt(x)) == x. Used by the ATAR backend.

// db4 scaling (low-pass) coefficients
const H = [
  0.48296291314469025, 0.83651630373746899,
  0.22414386804185735, -0.12940952255092145,
];
// wavelet (high-pass) from quadrature mirror: g[k] = (-1)^k h[L-1-k]
const G = [H[3], -H[2], H[1], -H[0]];
const L = H.length;

function pmod(i, n) { return ((i % n) + n) % n; }

// one level: x(length N, N even) -> { a: approx(N/2), d: detail(N/2) }
function dwtStep(x) {
  const N = x.length, half = N >> 1;
  const a = new Float64Array(half), d = new Float64Array(half);
  for (let i = 0; i < half; i++) {
    let sa = 0, sd = 0;
    for (let k = 0; k < L; k++) {
      const idx = pmod(2 * i + k, N);
      sa += H[k] * x[idx];
      sd += G[k] * x[idx];
    }
    a[i] = sa; d[i] = sd;
  }
  return { a, d };
}

// inverse of one level: a,d (length half) -> x (length N)
function idwtStep(a, d) {
  const half = a.length, N = half << 1;
  const x = new Float64Array(N);
  for (let i = 0; i < half; i++) {
    for (let k = 0; k < L; k++) {
      const idx = pmod(2 * i + k, N);
      x[idx] += H[k] * a[i] + G[k] * d[i];
    }
  }
  return x;
}

// Multi-level forward. Returns { details:[level1..], approx, lengths } (level1 = finest)
export function dwt(signal, levels) {
  let a = Float64Array.from(signal);
  const details = [], lengths = [];
  for (let l = 0; l < levels; l++) {
    if (a.length < 2 || (a.length & 1)) break;
    const { a: na, d } = dwtStep(a);
    details.push(d); lengths.push(d.length);
    a = na;
  }
  return { details, approx: a };
}

// Multi-level inverse from the structure produced by dwt (details possibly modified).
export function idwt({ details, approx }) {
  let a = approx;
  for (let l = details.length - 1; l >= 0; l--) {
    a = idwtStep(a, details[l]);
  }
  return a;
}

// Robust per-array scale estimate (median absolute deviation / 0.6745 ~ std).
export function madSigma(arr) {
  const b = Float64Array.from(arr).map(Math.abs).sort();
  const med = b.length ? b[b.length >> 1] : 0;
  return med / 0.6745;
}
