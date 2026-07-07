// Full-pipeline test using only the non-browser path (synthetic source + passthrough).
// Verifies: windows flow, overlap-add finalizes, and cleaned output equals raw input
// (passthrough) at matching absolute indices — i.e. the orchestration preserves data.
import { Pipeline } from '../src/pipeline.js';

const CH = 8, CAP = 200000;
const raw = Array.from({ length: CH }, () => new Float32Array(CAP).fill(NaN));
const clean = Array.from({ length: CH }, () => new Float32Array(CAP).fill(NaN));
let cleanLo = Infinity, cleanHi = 0, statsSeen = 0, cleanEvents = 0;

const p = new Pipeline({
  channels: CH,
  windowSec: 0.5,          // short window so finalized output appears quickly
  overlap: 0.8,
  source: { type: 'synthetic', artifacts: true, chunk: 32 },
  model: { type: 'passthrough' },
});

p.on('raw', (r) => { for (let c = 0; c < CH; c++) raw[c].set(r.channels[c], r.start); });
p.on('clean', (c) => {
  cleanEvents++;
  for (let ch = 0; ch < CH; ch++) clean[ch].set(c.channels[ch], c.start);
  cleanLo = Math.min(cleanLo, c.start); cleanHi = Math.max(cleanHi, c.start + c.nSamples);
});
p.on('stats', () => statsSeen++);
p.on('error', (e) => { console.error('pipeline error', e); process.exitCode = 1; });

await p.start();
await new Promise((r) => setTimeout(r, 2500));
p.stop();

let worst = 0, compared = 0;
for (let i = cleanLo; i < cleanHi; i++) {
  for (let ch = 0; ch < CH; ch++) {
    const a = clean[ch][i], b = raw[ch][i];
    if (Number.isNaN(a) || Number.isNaN(b)) continue;
    worst = Math.max(worst, Math.abs(a - b)); compared++;
  }
}

let fail = 0;
const ok = (n, c, x = '') => { console.log(`${c ? 'PASS' : 'FAIL'}  ${n}${x ? '  — ' + x : ''}`); if (!c) fail++; };
ok('clean events emitted', cleanEvents > 0, `${cleanEvents} events`);
ok('stats events emitted', statsSeen > 0, `${statsSeen}`);
ok('finalized span is non-trivial', (cleanHi - cleanLo) > 256, `span=${cleanHi - cleanLo}`);
ok('samples compared > 0', compared > 0, `${compared}`);
ok('cleaned == raw through passthrough (maxErr<1e-4)', worst < 1e-4, `maxErr=${worst.toExponential(2)}`);

console.log(`\n${fail === 0 ? 'INTEGRATION OK' : fail + ' FAILED'}`);
process.exit(fail === 0 ? 0 : 1);
