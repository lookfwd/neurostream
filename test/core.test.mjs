// Pure-JS unit tests for the signal-processing core (no browser globals).
import { resolveConfig } from '../src/config.js';
import { RingBuffer } from '../src/ring-buffer.js';
import { Windower } from '../src/windower.js';
import { OverlapAdd } from '../src/overlap-add.js';
import { encodeBinary, decodeBinary } from '../src/codec.js';

let failures = 0;
function ok(name, cond, extra = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '  — ' + extra : ''}`);
  if (!cond) failures++;
}
function maxAbsDiff(a, b) { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; }

// ---- 1. config derives hop/latency correctly ----
{
  const c = resolveConfig({ channels: 8, windowSec: 4, overlap: 0.8, sampleRate: 256 });
  const d = c._derived;
  ok('config windowSamples=1024', d.windowSamples === 1024);
  ok('config hopSamples=205', d.hopSamples === 205, `got ${d.hopSamples}`);
  ok('config averaging≈5x', Math.abs(d.averaging - 5) < 0.05, `got ${d.averaging.toFixed(2)}`);
}

// ---- 2. codec round-trips channel-major float data ----
{
  const channels = [Float32Array.from([1, 2, 3]), Float32Array.from([4, 5, 6])];
  const dec = decodeBinary(encodeBinary({ nChannels: 2, nSamples: 3, t: 123, channels }));
  ok('codec nChannels', dec.nChannels === 2);
  ok('codec ch1[2]=6', dec.channels[1][2] === 6);
}

// ---- 3. ring buffer preserves absolute-indexed data across wrap ----
{
  const rb = new RingBuffer(2, 100);           // capacity rounds to 128
  let idx = 0;
  for (let k = 0; k < 500; k++) {              // write well past capacity to force wraps
    const a = new Float32Array(10), b = new Float32Array(10);
    for (let i = 0; i < 10; i++) { a[i] = idx + i; b[i] = -(idx + i); }
    rb.write([a, b]); idx += 10;
  }
  const out = new Float32Array(50);
  rb.readInto(0, 4900, 50, out);               // last 100 samples are resident
  let good = true; for (let i = 0; i < 50; i++) good = good && out[i] === 4900 + i;
  ok('ring buffer reads correct absolute range after wrap', good);
}

// ---- 4. windower + overlap-add reconstruct a passthrough signal ----
// The heart of the framework: with identity "cleaning", output must equal the input
// (in the finalized region) to within floating-point error, for both 80% overlap and
// several channel counts.
function reconstructTest(channels, overlap) {
  const fs = 256, windowSec = 4;
  const c = resolveConfig({ channels, windowSec, overlap, sampleRate: fs });
  const { windowSamples: W, hopSamples: H } = c._derived;
  const rb = new RingBuffer(channels, 3 * W);
  const win = new Windower(rb, { channels, window: W, hop: H });
  const ola = new OverlapAdd({ channels, window: W, hop: H, taper: 'tukey', taperAlpha: 0.5 });

  // ground-truth signal per channel
  const totalSec = 12;
  const N = totalSec * fs;
  const truth = [];
  for (let ch = 0; ch < channels; ch++) {
    const s = new Float32Array(N);
    for (let i = 0; i < N; i++) s[i] = Math.sin(2 * Math.PI * (7 + ch) * i / fs) + 0.3 * Math.sin(2 * Math.PI * 23 * i / fs);
    truth.push(s);
  }

  // stream it in small chunks; collect finalized output
  const outParts = Array.from({ length: channels }, () => []);
  let firstFinal = null, lastFinal = 0;
  const chunk = 32;
  for (let p = 0; p < N; p += chunk) {
    const n = Math.min(chunk, N - p);
    const cc = [];
    for (let ch = 0; ch < channels; ch++) cc.push(truth[ch].subarray(p, p + n));
    rb.write(cc);
    let w;
    while ((w = win.next())) {
      // identity "cleaning": pass window.data straight through
      const fin = ola.push({ start: w.start, data: w.data });
      if (fin) {
        if (firstFinal === null) firstFinal = fin.start;
        for (let ch = 0; ch < channels; ch++) outParts[ch].push(fin.channels[ch]);
        lastFinal = fin.start + fin.nSamples;
      }
    }
  }

  // stitch finalized output and compare against truth over the finalized span
  let worst = 0;
  for (let ch = 0; ch < channels; ch++) {
    const merged = new Float32Array(lastFinal - firstFinal);
    let off = 0; for (const part of outParts[ch]) { merged.set(part, off); off += part.length; }
    const ref = truth[ch].subarray(firstFinal, lastFinal);
    worst = Math.max(worst, maxAbsDiff(merged, ref));
  }
  return { worst, span: lastFinal - firstFinal, firstFinal };
}

for (const ch of [4, 8, 16]) {
  const r = reconstructTest(ch, 0.8);
  ok(`OLA reconstruct passthrough, ${ch}ch @80% overlap (maxErr<1e-4)`, r.worst < 1e-4,
     `maxErr=${r.worst.toExponential(2)}, span=${r.span}, firstFinal=${r.firstFinal}`);
}
// also check a different overlap to be sure the grid math generalizes
{
  const r = reconstructTest(8, 0.5);
  ok('OLA reconstruct passthrough, 8ch @50% overlap (maxErr<1e-4)', r.worst < 1e-4, `maxErr=${r.worst.toExponential(2)}`);
}

console.log(`\n${failures === 0 ? 'ALL TESTS PASSED' : failures + ' TEST(S) FAILED'}`);
process.exit(failures === 0 ? 0 : 1);
