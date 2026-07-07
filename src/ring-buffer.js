// Per-channel float ring buffer indexed by an absolute, monotonically increasing
// sample counter. Writes are contiguous; reads request an absolute [start, start+len)
// range that must still be resident (i.e. within the last `capacity` samples).

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

export class RingBuffer {
  constructor(channels, minCapacity) {
    this.channels = channels;
    this.capacity = nextPow2(Math.max(2, minCapacity));
    this.mask = this.capacity - 1;
    this.written = 0; // absolute count of samples written (per channel, in lockstep)

    const bytes = channels * this.capacity * 4;
    let backing;
    if (typeof SharedArrayBuffer === 'function') {
      try { backing = new SharedArrayBuffer(bytes); } catch { backing = new ArrayBuffer(bytes); }
    } else {
      backing = new ArrayBuffer(bytes);
    }
    this.buffer = backing;
    this.data = [];
    for (let c = 0; c < channels; c++) {
      this.data.push(new Float32Array(backing, c * this.capacity * 4, this.capacity));
    }
  }

  // channelsChunk: Float32Array[] (one per channel), all the same length n.
  write(channelsChunk) {
    const n = channelsChunk[0].length;
    for (let c = 0; c < this.channels; c++) {
      const src = channelsChunk[c];
      const ring = this.data[c];
      let w = this.written & this.mask;
      const first = Math.min(n, this.capacity - w);
      ring.set(src.subarray(0, first), w);
      if (first < n) ring.set(src.subarray(first), 0);
    }
    this.written += n;
  }

  // Copy absolute range [start, start+len) for one channel into `out`.
  readInto(channel, start, len, out) {
    if (start < this.written - this.capacity) {
      throw new Error(`ring underflow: sample ${start} evicted (oldest=${this.written - this.capacity})`);
    }
    const ring = this.data[channel];
    let r = start & this.mask;
    const first = Math.min(len, this.capacity - r);
    out.set(ring.subarray(r, r + first), 0);
    if (first < len) out.set(ring.subarray(0, len - first), first);
    return out;
  }
}
