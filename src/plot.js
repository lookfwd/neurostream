// Dual-trace scope. Raw is drawn as a dim "ghost"; cleaned is drawn crisp and lags
// behind by the pipeline latency — the empty sliver at the right edge, where only raw
// exists yet, is that latency made visible.

export class Scope {
  constructor(canvas, { channels, sampleRate, spanSec = 6, smooth = 7, colors = {} }) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.channels = channels;
    this.fs = sampleRate;
    this.L = Math.round(spanSec * sampleRate);
    this.ringL = this.L + sampleRate;   // ring holds > display span so the lagged scroll
                                        // window never reads positions the newest samples overwrote
    this.ch = 0;
    this.smooth = smooth;         // scroll-follow rate (higher = tighter, less display lag)
    this._head = null;            // interpolated right-edge index (float), advances by wall-clock
    this._lastT = 0;
    this.stacked = false;         // true => draw every channel in its own lane
    this.colors = {
      bg: '#0b1621', grid: '#16273a', raw: '#456079', clean: '#5fe3c0',
      gap: '#20344a', text: '#7b93a9', ...colors,
    };
    this.raw = Array.from({ length: channels }, () => new Float32Array(this.ringL).fill(NaN));
    this.clean = Array.from({ length: channels }, () => new Float32Array(this.ringL).fill(NaN));
    this.latestRaw = 0; this.latestClean = 0;
    this._raf = null;
  }

  setChannel(c) {
    if (c === 'all') { this.stacked = true; return; }
    this.stacked = false;
    this.ch = Math.max(0, Math.min(this.channels - 1, c));
  }

  pushRaw(start, channelsData) {
    this._write(this.raw, start, channelsData);
    this.latestRaw = start + channelsData[0].length;
  }
  pushClean(start, channelsData) {
    this._write(this.clean, start, channelsData);
    this.latestClean = start + channelsData[0].length;
  }
  _write(target, start, channelsData) {
    const n = channelsData[0].length;
    for (let c = 0; c < this.channels; c++) {
      const dst = target[c], src = channelsData[c];
      for (let i = 0; i < n; i++) dst[(start + i) % this.ringL] = src[i];
    }
  }

  start() { const loop = () => { this.draw(); this._raf = requestAnimationFrame(loop); }; loop(); }
  stop() { if (this._raf) cancelAnimationFrame(this._raf); this._raf = null; }

  // Advance the displayed right-edge by wall-clock time (exponential follow of the newest
  // sample) so the trace scrolls continuously at frame rate instead of snapping forward once
  // per data chunk. Returns the fractional absolute index at the right edge.
  _advanceHead() {
    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    if (this._head == null) { this._head = this.latestRaw; this._lastT = now; return this._head; }
    let dt = (now - this._lastT) / 1000; this._lastT = now;
    if (dt <= 0) return this._head;
    if (dt > 0.5) { this._head = this.latestRaw; return this._head; }   // resumed from background: snap
    const a = 1 - Math.exp(-dt * this.smooth);        // frame-rate-independent smoothing
    this._head += (this.latestRaw - this._head) * a;  // ease toward newest sample (stays just behind)
    if (this._head > this.latestRaw) this._head = this.latestRaw;
    if (this._head < this.latestRaw - this.L) this._head = this.latestRaw - this.L; // catch up if far behind
    return this._head;
  }

  draw() {
    if (this.stacked) return this.drawStacked();
    const { ctx, cv, colors } = this;
    const W = cv.width, H = cv.height;
    ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, W, H);

    // grid
    ctx.strokeStyle = colors.grid; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 1; g < 8; g++) { const y = (H * g) / 8; ctx.moveTo(0, y); ctx.lineTo(W, y); }
    for (let g = 1; g < 12; g++) { const x = (W * g) / 12; ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    ctx.stroke();

    const startF = this._advanceHead() - this.L;   // fractional absolute index at left edge
    const start = Math.floor(startF), frac = startF - start;
    const mid = H / 2, scale = H / 220;            // ~±110 units full-scale

    // latency gap: region to the right of the newest cleaned sample
    if (this.latestClean < this.latestRaw) {
      const gx = ((this.latestClean - startF) / this.L) * W;
      ctx.fillStyle = colors.gap; ctx.globalAlpha = 0.5;
      ctx.fillRect(gx, 0, W - gx, H); ctx.globalAlpha = 1;
      ctx.fillStyle = colors.text; ctx.font = '11px ui-monospace, monospace';
      ctx.fillText('latency', Math.min(gx + 6, W - 54), 14);
    }

    this._trace(this.raw[this.ch], start, frac, W, H, mid, scale, colors.raw, 1);
    this._trace(this.clean[this.ch], start, frac, W, H, mid, scale, colors.clean, 1.6);
  }

  // All channels at once: each channel in its own horizontal lane (EEG montage view).
  drawStacked() {
    const { ctx, cv, colors } = this;
    const W = cv.width, H = cv.height;
    ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, W, H);

    const startF = this._advanceHead() - this.L;
    const start = Math.floor(startF), frac = startF - start;
    const laneH = H / this.channels;
    const scale = laneH / 80;            // amplitude that ~fills a lane; clipped to lane

    // latency gap (right of newest cleaned sample), spanning all lanes
    if (this.latestClean < this.latestRaw) {
      const gx = ((this.latestClean - startF) / this.L) * W;
      ctx.fillStyle = colors.gap; ctx.globalAlpha = 0.5;
      ctx.fillRect(gx, 0, W - gx, H); ctx.globalAlpha = 1;
      ctx.fillStyle = colors.text; ctx.font = '11px ui-monospace, monospace';
      ctx.fillText('latency', Math.min(gx + 6, W - 54), 12);
    }

    // lane separators
    ctx.strokeStyle = colors.grid; ctx.lineWidth = 1; ctx.beginPath();
    for (let c = 1; c < this.channels; c++) { const y = laneH * c; ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();

    for (let c = 0; c < this.channels; c++) {
      const mid = laneH * (c + 0.5);
      this._traceLane(this.raw[c], start, frac, W, mid, laneH, scale, colors.raw, 1);
      this._traceLane(this.clean[c], start, frac, W, mid, laneH, scale, colors.clean, 1.3);
    }

    // channel labels (thin out when lanes are tiny)
    ctx.fillStyle = colors.text; ctx.font = '9px ui-monospace, monospace';
    const step = laneH >= 14 ? 1 : (laneH >= 8 ? 2 : 5);
    for (let c = 0; c < this.channels; c += step) ctx.fillText('ch' + c, 2, laneH * (c + 0.5) + 3);
  }

  _traceLane(ring, start, frac, W, mid, laneH, scale, color, lw) {
    if (!ring) return;
    const ctx = this.ctx, half = laneH * 0.5;
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
    let pen = false;
    for (let i = 0; i < this.L; i++) {
      const idx = start + i;
      if (idx < 0) { pen = false; continue; }          // pre-stream: left stays empty during fill
      const v = ring[idx % this.ringL];
      if (Number.isNaN(v)) { pen = false; continue; }
      const x = ((i - frac) / this.L) * W;
      let y = mid - v * scale;
      if (y < mid - half) y = mid - half; else if (y > mid + half) y = mid + half; // clamp to lane
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  _trace(ring, start, frac, W, H, mid, scale, color, lw) {
    if (!ring) return;
    const ctx = this.ctx;
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
    let pen = false;
    for (let i = 0; i < this.L; i++) {
      const idx = start + i;
      if (idx < 0) { pen = false; continue; }          // pre-stream: left stays empty during fill
      const v = ring[idx % this.ringL];
      if (Number.isNaN(v)) { pen = false; continue; }
      const x = ((i - frac) / this.L) * W;
      const y = mid - v * scale;
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
