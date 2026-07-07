// Weighted overlap-add of cleaned windows into a continuous output stream.
// output[i] = sum_k( taper * cleaned_k )[i] / sum_k( taper )[i]
//
// A sample is FINAL once a window starting after it arrives: all windows that can
// cover sample i have start <= i, so when we receive a window at start S we can
// safely emit every sample in [lastEmit, S). This yields `hop` finalized samples
// per window, each averaged over all overlapping windows (window/hop of them).

function makeTaper(kind, N, alpha) {
  const w = new Float32Array(N);
  if (kind === 'rect') { w.fill(1); return w; }
  if (kind === 'hann') {
    for (let n = 0; n < N; n++) w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1));
    return w;
  }
  // tukey: flat centre, cosine-tapered edges over `alpha` fraction
  const a = Math.min(Math.max(alpha, 0), 1);
  const edge = (a * (N - 1)) / 2;
  for (let n = 0; n < N; n++) {
    if (n < edge) w[n] = 0.5 * (1 + Math.cos(Math.PI * (n / edge - 1)));
    else if (n > N - 1 - edge) w[n] = 0.5 * (1 + Math.cos(Math.PI * ((n - (N - 1 - edge)) / edge)));
    else w[n] = 1;
  }
  return w;
}

// Floor so no sample can end up with zero total weight (e.g. Hann/Tukey edges at the
// very start of a stream, where only one window covers a sample). With passthrough,
// overlapping windows share the same value, so any positive weighting reconstructs
// exactly; for a real model this just avoids a divide-by-zero at stream edges.
function floorTaper(w, floor = 1e-3) {
  for (let i = 0; i < w.length; i++) if (w[i] < floor) w[i] = floor;
  return w;
}

const EPS = 1e-8;

export class OverlapAdd {
  constructor({ channels, window, hop, taper = 'tukey', taperAlpha = 0.5 }) {
    this.channels = channels;
    this.window = window;
    this.hop = hop;
    this.taper = floorTaper(makeTaper(taper, window, taperAlpha));

    // Circular accumulators large enough to hold the live region (window + hop).
    let cap = 1; while (cap < window + hop) cap <<= 1;
    this.cap = cap; this.mask = cap - 1;
    this.acc = [];
    this.wsum = new Float32Array(cap); // taper weights are channel-independent
    for (let c = 0; c < channels; c++) this.acc.push(new Float32Array(cap));

    this.lastEmit = 0; // absolute index of next sample to emit
  }

  // window: { start, data: Float32Array(channels*window) } — cleaned, channel-major.
  // Returns { start, nSamples, channels: Float32Array[] } of finalized output, or null.
  push({ start, data }) {
    let out = null;
    if (start > this.lastEmit) out = this._finalize(this.lastEmit, start);

    for (let c = 0; c < this.channels; c++) {
      const acc = this.acc[c];
      const win = data.subarray(c * this.window, (c + 1) * this.window);
      for (let n = 0; n < this.window; n++) {
        const idx = (start + n) & this.mask;
        acc[idx] += win[n] * this.taper[n];
      }
    }
    for (let n = 0; n < this.window; n++) {
      this.wsum[(start + n) & this.mask] += this.taper[n];
    }
    return out;
  }

  _finalize(from, to) {
    const len = to - from;
    const channels = [];
    for (let c = 0; c < this.channels; c++) {
      const acc = this.acc[c];
      const dst = new Float32Array(len);
      for (let i = 0; i < len; i++) {
        const idx = (from + i) & this.mask;
        const w = this.wsum[idx];
        dst[i] = w > EPS ? acc[idx] / w : 0;
      }
      channels.push(dst);
    }
    // zero the emitted slots so the ring can be reused
    for (let i = 0; i < len; i++) {
      const idx = (from + i) & this.mask;
      this.wsum[idx] = 0;
      for (let c = 0; c < this.channels; c++) this.acc[c][idx] = 0;
    }
    this.lastEmit = to;
    return { start: from, nSamples: len, channels };
  }
}
