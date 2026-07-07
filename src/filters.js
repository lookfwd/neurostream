// Optional per-channel IIR pre-filtering. Direct-form-II transposed biquads.
// Coefficients from the RBJ audio EQ cookbook (standard biquad design).

function highpassCoeffs(fs, f0, Q = 0.707) {
  const w0 = 2 * Math.PI * f0 / fs;
  const c = Math.cos(w0), s = Math.sin(w0), alpha = s / (2 * Q);
  const b0 = (1 + c) / 2, b1 = -(1 + c), b2 = (1 + c) / 2;
  const a0 = 1 + alpha, a1 = -2 * c, a2 = 1 - alpha;
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}
function notchCoeffs(fs, f0, Q = 30) {
  const w0 = 2 * Math.PI * f0 / fs;
  const c = Math.cos(w0), s = Math.sin(w0), alpha = s / (2 * Q);
  const b0 = 1, b1 = -2 * c, b2 = 1;
  const a0 = 1 + alpha, a1 = -2 * c, a2 = 1 - alpha;
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}
function lowpassCoeffs(fs, f0, Q = 0.707) {
  const w0 = 2 * Math.PI * f0 / fs;
  const c = Math.cos(w0), s = Math.sin(w0), alpha = s / (2 * Q);
  const b0 = (1 - c) / 2, b1 = 1 - c, b2 = (1 - c) / 2;
  const a0 = 1 + alpha, a1 = -2 * c, a2 = 1 - alpha;
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}

class Biquad {
  constructor([b0, b1, b2, a1, a2]) {
    this.b0 = b0; this.b1 = b1; this.b2 = b2; this.a1 = a1; this.a2 = a2;
    this.z1 = 0; this.z2 = 0;
  }
  step(x) {
    const y = this.b0 * x + this.z1;
    this.z1 = this.b1 * x - this.a1 * y + this.z2;
    this.z2 = this.b2 * x - this.a2 * y;
    return y;
  }
}

// One filter chain per channel; call process(sample) per incoming sample.
export class FilterChain {
  constructor(fs, { highpass = 0, notch = 0, lowpass = 0 } = {}) {
    this.stages = [];
    if (highpass > 0) this.stages.push(new Biquad(highpassCoeffs(fs, highpass)));
    if (notch > 0) this.stages.push(new Biquad(notchCoeffs(fs, notch)));
    if (lowpass > 0) this.stages.push(new Biquad(lowpassCoeffs(fs, lowpass)));
  }
  get active() { return this.stages.length > 0; }
  process(x) {
    let y = x;
    for (const st of this.stages) y = st.step(y);
    return y;
  }
}

// Streaming boxcar moving average of length `taps` (stateful, per channel).
export class MovingAverage {
  constructor(taps) {
    this.taps = Math.max(1, taps | 0);
    this.buf = new Float32Array(this.taps);
    this.i = 0; this.sum = 0; this.filled = 0;
  }
  process(x) {
    this.sum -= this.buf[this.i];
    this.buf[this.i] = x;
    this.sum += x;
    this.i = (this.i + 1) % this.taps;
    if (this.filled < this.taps) this.filled++;
    return this.sum / this.filled;
  }
}
