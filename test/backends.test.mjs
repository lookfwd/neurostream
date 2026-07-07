// Backend tests: verify method invariants without a browser.
import { CcaBackend } from '../src/backends/cca-backend.js';
import { AtarBackend } from '../src/backends/atar-backend.js';
import { LmsBackend } from '../src/backends/lms-backend.js';
import { SspBackend } from '../src/backends/ssp-backend.js';
import { FilterBackend } from '../src/backends/filter-backend.js';
import { OscarBackend } from '../src/backends/oscar-backend.js';
import { OricaBackend } from '../src/backends/orica-backend.js';

const mulberry32 = (a) => () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };

let fail = 0;
const ok = (n, c, x = '') => { console.log(`${c ? 'PASS' : 'FAIL'}  ${n}${x ? '  — ' + x : ''}`); if (!c) fail++; };
const flat = (chs, W) => { const C = chs.length; const o = new Float32Array(C * W); for (let c = 0; c < C; c++) o.set(chs[c], c * W); return o; };
const rms = (a) => { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s / a.length); };

// signal builders
function multichan(C, W, muscleAmp = 0) {
  const chs = [];
  for (let c = 0; c < C; c++) {
    const s = new Float32Array(W);
    for (let n = 0; n < W; n++) {
      s[n] = Math.sin(2 * Math.PI * (8 + c) * n / 256) + 0.4 * Math.sin(2 * Math.PI * 18 * n / 256)
           + muscleAmp * (Math.random() - 0.5);   // broadband "muscle"
    }
    chs.push(s);
  }
  return chs;
}

// ---- CCA: with no components removed, must reconstruct the input exactly ----
{
  const C = 8, W = 512;
  const chs = multichan(C, W, 0.5);
  const cca = new CcaBackend({ channels: C, window: W, params: { remove: 0 } }); // remove nothing
  const out = cca.process({ start: 0, data: flat(chs, W) });
  let worst = 0;
  for (let c = 0; c < C; c++) for (let n = 0; n < W; n++) worst = Math.max(worst, Math.abs(out.data[c * W + n] - chs[c][n]));
  ok('CCA reconstructs input when nothing removed (maxErr<1e-3)', worst < 1e-3, `maxErr=${worst.toExponential(2)}`);
}
// ---- CCA: removing low-correlation comps should reduce broadband power ----
{
  const C = 8, W = 512;
  const chs = multichan(C, W, 1.2);
  const cca = new CcaBackend({ channels: C, window: W, params: { remove: 3 } });
  const out = cca.process({ start: 0, data: flat(chs, W) });
  let rin = 0, rout = 0;
  for (let c = 0; c < C; c++) { rin += rms(chs[c]); const o = out.data.subarray(c * W, (c + 1) * W); rout += rms(o); }
  ok('CCA reduces power when removing muscle-like comps', rout < rin, `in=${rin.toFixed(1)} out=${rout.toFixed(1)}`);
}

// ---- ATAR: very high k (θ huge) => nothing exceeds threshold => output ≈ input ----
{
  const C = 4, W = 1024;
  const chs = multichan(C, W, 0.3);
  const atar = new AtarBackend({ channels: C, window: W, params: { k: 1e12, mode: 'elim' } });
  const out = atar.process({ start: 0, data: flat(chs, W) });
  let worst = 0;
  for (let c = 0; c < C; c++) for (let n = 0; n < W; n++) worst = Math.max(worst, Math.abs(out.data[c * W + n] - chs[c][n]));
  ok('ATAR is near-identity at huge threshold (maxErr<1e-6)', worst < 1e-6, `maxErr=${worst.toExponential(2)}`);
}
// ---- ATAR: a big transient spike should be attenuated ----
{
  const C = 1, W = 1024;
  const s = multichan(1, W, 0.1)[0];
  for (let n = 500; n < 520; n++) s[n] += 80;   // artifact burst
  const before = Math.max(...Array.from(s.subarray(495, 525)).map(Math.abs));
  const atar = new AtarBackend({ channels: 1, window: W, params: { k: 3, mode: 'linatten' } });
  const out = atar.process({ start: 0, data: flat([s], W) });
  const after = Math.max(...Array.from(out.data.subarray(495, 525)).map(Math.abs));
  ok('ATAR attenuates a transient burst', after < before, `before=${before.toFixed(1)} after=${after.toFixed(1)}`);
}

// ---- LMS: cancels reference-correlated artifact injected into a target channel ----
{
  const C = 2, W = 2000;
  const ref = new Float32Array(W), tgt = new Float32Array(W);
  for (let n = 0; n < W; n++) {
    const eog = Math.sin(2 * Math.PI * 1.5 * n / 256);   // slow ocular-like reference
    ref[n] = eog + 0.05 * (Math.random() - 0.5);
    tgt[n] = Math.sin(2 * Math.PI * 10 * n / 256) + 0.8 * eog;  // EEG + leaked artifact
  }
  const lms = new LmsBackend({ channels: C, sampleRate: 256, params: { reference: 0, order: 8, mu: 0.1 } });
  const out = lms.processStream([ref, tgt], 0);
  // compare artifact leakage before/after on the second half (after adaptation)
  const half = W / 2;
  const cleanEEG = new Float32Array(half); for (let n = 0; n < half; n++) cleanEEG[n] = Math.sin(2 * Math.PI * 10 * (n + half) / 256);
  let errBefore = 0, errAfter = 0;
  for (let n = 0; n < half; n++) {
    errBefore += Math.abs(tgt[n + half] - cleanEEG[n]);
    errAfter += Math.abs(out[1][n + half] - cleanEEG[n]);
  }
  ok('LMS reduces reference-correlated artifact', errAfter < errBefore * 0.6, `before=${errBefore.toFixed(0)} after=${errAfter.toFixed(0)}`);
}

// ---- SSP: projecting out a topography removes that spatial pattern ----
{
  const C = 4, W = 300;
  const topo = [1, 0.8, 0.4, 0.1];
  const chs = Array.from({ length: C }, () => new Float32Array(W));
  for (let n = 0; n < W; n++) {
    const brain = Math.sin(2 * Math.PI * 10 * n / 256);
    const artifact = 5 * Math.sin(2 * Math.PI * 2 * n / 256);
    for (let c = 0; c < C; c++) chs[c][n] = brain * (c % 2 ? -1 : 1) + topo[c] * artifact;
  }
  const ssp = new SspBackend({ channels: C, params: { projectors: [topo] } });
  const out = ssp.processStream(chs, 0);
  // residual projection onto topo direction should be ~0
  let resid = 0; const norm = Math.hypot(...topo);
  for (let n = 0; n < W; n++) { let d = 0; for (let c = 0; c < C; c++) d += out[c][n] * topo[c] / norm; resid += Math.abs(d); }
  ok('SSP nulls the projected topography (residual≈0)', resid / W < 1e-4, `resid/N=${(resid / W).toExponential(2)}`);
}

// ---- Filter backends: moving average reduces high-freq noise power ----
{
  const C = 1, W = 2000;
  const s = new Float32Array(W);
  for (let n = 0; n < W; n++) s[n] = Math.sin(2 * Math.PI * 8 * n / 256) + 1.5 * (Math.random() - 0.5);
  const mov = new FilterBackend({ channels: 1, sampleRate: 256, params: { kind: 'movingaverage', taps: 8 } });
  const out = mov.processStream([s]);
  ok('moving average reduces noise power', rms(out[0]) < rms(s), `in=${rms(s).toFixed(2)} out=${rms(out[0]).toFixed(2)}`);
}

// ---- OSCAR: conditions + suppresses a large transient, keeps ongoing signal ----
{
  const C = 1, W = 2000;
  const s = new Float32Array(W);
  for (let n = 0; n < W; n++) s[n] = Math.sin(2 * Math.PI * 10 * n / 256);
  for (let n = 1000; n < 1030; n++) s[n] += 60;   // motion/blink transient
  const osc = new OscarBackend({ channels: 1, sampleRate: 256, params: { highpass: 0.5, k: 4 } });
  const out = osc.processStream([s]);
  const peakIn = Math.max(...Array.from(s.subarray(1000, 1030)).map(Math.abs));
  const peakOut = Math.max(...Array.from(out[0].subarray(1000, 1030)).map(Math.abs));
  ok('OSCAR suppresses transient', peakOut < peakIn, `in=${peakIn.toFixed(1)} out=${peakOut.toFixed(1)}`);
}

// ---- ORICA: with nothing rejected (huge kurtosis threshold), remix must reconstruct input ----
{
  const C = 4, n = 64, blocks = 20;
  const orica = new OricaBackend({ channels: C, sampleRate: 256, params: { kurtosis: 1e9 } });
  const rng = mulberry32(1);
  let lastIn, lastOut;
  for (let b = 0; b < blocks; b++) {
    const chs = [];
    for (let c = 0; c < C; c++) {
      const s = new Float32Array(n);
      for (let i = 0; i < n; i++) { const t = (b * n + i) / 256; s[i] = Math.sin(2 * Math.PI * (6 + c) * t) + 0.5 * (rng() - 0.5); }
      chs.push(s);
    }
    lastOut = orica.processStream(chs, b * n); lastIn = chs;
  }
  let worst = 0;
  for (let c = 0; c < C; c++) for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(lastOut[c][i] - lastIn[c][i]));
  ok('ORICA reconstructs input when nothing rejected (maxErr<1e-2)', worst < 1e-2, `maxErr=${worst.toExponential(2)}`);
}
// ---- ORICA: separates + rejects a spiky super-Gaussian source mixed across channels ----
{
  const C = 4, n = 64, blocks = 160;
  const rng = mulberry32(7);
  const M = Array.from({ length: C }, (_, a) => Array.from({ length: C }, (_, b) => (rng() * 2 - 1) + (a === b ? 1.5 : 0)));
  const orica = new OricaBackend({ channels: C, sampleRate: 256, params: { kurtosis: 5, convSamples: 400 } });
  let artIn = 0, artOut = 0;
  for (let b = 0; b < blocks; b++) {
    const S = Array.from({ length: C }, () => new Float32Array(n)), spike = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      spike[i] = rng() < 0.03 ? (rng() * 2 - 1) * 10 : 0;   // rare large spikes => very high kurtosis
      S[0][i] = spike[i];
      for (let k = 1; k < C; k++) S[k][i] = rng() * 2 - 1;   // sub-Gaussian background
    }
    const chs = Array.from({ length: C }, (_, c) => { const x = new Float32Array(n); for (let i = 0; i < n; i++) { let v = 0; for (let k = 0; k < C; k++) v += M[c][k] * S[k][i]; x[i] = v; } return x; });
    const out = orica.processStream(chs, b * n);
    if (b >= blocks - 30) for (let i = 0; i < n; i++) if (spike[i] !== 0) for (let c = 0; c < C; c++) { artIn += Math.abs(chs[c][i]); artOut += Math.abs(out[c][i]); }
  }
  ok('ORICA removes spiky artifact after convergence', artOut < artIn * 0.9, `in=${artIn.toFixed(0)} out=${artOut.toFixed(0)} conv=${orica.info.convergence}`);
}
// ---- ORICA: no false rejection on sub/near-Gaussian-only input ----
{
  const C = 4, n = 64, blocks = 80;
  const rng = mulberry32(3);
  const orica = new OricaBackend({ channels: C, sampleRate: 256, params: { kurtosis: 5, convSamples: 200 } });
  for (let b = 0; b < blocks; b++) {
    const chs = Array.from({ length: C }, () => { const s = new Float32Array(n); for (let i = 0; i < n; i++) s[i] = rng() * 2 - 1; return s; });
    orica.processStream(chs, b * n);
  }
  ok('ORICA rejects nothing on non-spiky input', orica.info.rejected === 0, `rejected=${orica.info.rejected} conv=${orica.info.convergence}`);
}

console.log(`\n${fail === 0 ? 'ALL BACKEND TESTS PASSED' : fail + ' FAILED'}`);
process.exit(fail === 0 ? 0 : 1);
