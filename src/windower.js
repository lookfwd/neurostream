// Turns a growing ring buffer into fixed-grid windows.
// Window k covers absolute samples [k*hop, k*hop + window). A window is produced
// as soon as enough samples have arrived to fill it, in strictly increasing order.

export class Windower {
  constructor(ring, { channels, window, hop }) {
    this.ring = ring;
    this.channels = channels;
    this.window = window;
    this.hop = hop;
    this.nextStart = 0;
    // reusable scratch, channel-major flat: [c0 window][c1 window]...
    this.scratch = new Float32Array(channels * window);
    this._chanView = [];
    for (let c = 0; c < channels; c++) {
      this._chanView.push(this.scratch.subarray(c * window, (c + 1) * window));
    }
  }

  // Returns the next window as { start, data: Float32Array(channels*window) } or null.
  // `data` is a fresh copy so callers/transfers don't clash with the scratch.
  next() {
    if (this.ring.written < this.nextStart + this.window) return null;
    for (let c = 0; c < this.channels; c++) {
      this.ring.readInto(c, this.nextStart, this.window, this._chanView[c]);
    }
    const start = this.nextStart;
    this.nextStart += this.hop;
    return { start, data: this.scratch.slice() };
  }
}
