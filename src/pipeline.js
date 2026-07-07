import { Emitter } from './events.js';
import { resolveConfig } from './config.js';
import { RingBuffer } from './ring-buffer.js';
import { Windower } from './windower.js';
import { OverlapAdd } from './overlap-add.js';
import { FilterChain } from './filters.js';
import { SyntheticSource } from './sources/synthetic-source.js';
import { MqttSource } from './sources/mqtt-source.js';
import { PassthroughBackend } from './backends/passthrough.js';
import { OnnxBackend } from './backends/onnx-backend.js';
import { FilterBackend } from './backends/filter-backend.js';
import { LmsBackend } from './backends/lms-backend.js';
import { SspBackend } from './backends/ssp-backend.js';
import { OscarBackend } from './backends/oscar-backend.js';
import { CcaBackend } from './backends/cca-backend.js';
import { AtarBackend } from './backends/atar-backend.js';
import { OricaBackend } from './backends/orica-backend.js';
import { encodeBinary } from './codec.js';

// Linear resampler with cross-chunk continuity (sourceRate -> modelRate).
class Resampler {
  constructor(inRate, outRate, channels) {
    this.ratio = inRate / outRate;      // input samples per output sample
    this.channels = channels;
    this.pos = 0;                        // fractional read position into the stream
    this.consumed = 0;                   // absolute input index at start of `prev`
    this.prev = channels ? Array.from({ length: channels }, () => 0) : null;
    this.identity = Math.abs(inRate - outRate) < 1e-9;
  }
  process(channelsChunk) {
    if (this.identity) return channelsChunk;
    const n = channelsChunk[0].length;
    const out = Array.from({ length: this.channels }, () => []);
    // absolute input index space: [this.consumed-1 .. this.consumed+n-1]
    while (this.pos <= this.consumed + n - 1) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      for (let c = 0; c < this.channels; c++) {
        const a = this._sampleAt(channelsChunk[c], i);
        const b = this._sampleAt(channelsChunk[c], i + 1);
        out[c].push(a + (b - a) * frac);
      }
      this.pos += this.ratio;
    }
    for (let c = 0; c < this.channels; c++) this.prev[c] = channelsChunk[c][n - 1];
    this.consumed += n;
    return out.map((arr) => Float32Array.from(arr));
  }
  _sampleAt(chunk, absIdx) {
    const rel = absIdx - this.consumed;
    if (rel >= 0 && rel < chunk.length) return chunk[rel];
    if (rel === -1) return this.prev ? this.prev[0] : chunk[0]; // handled per channel below
    return chunk[Math.max(0, Math.min(chunk.length - 1, rel))];
  }
}

export class Pipeline extends Emitter {
  constructor(userConfig = {}) {
    super();
    this.config = resolveConfig(userConfig);
    const d = this.config._derived;
    const C = this.config.channels;

    this.ring = new RingBuffer(C, 3 * d.windowSamples);
    this.windower = new Windower(this.ring, { channels: C, window: d.windowSamples, hop: d.hopSamples });
    this.ola = new OverlapAdd({
      channels: C, window: d.windowSamples, hop: d.hopSamples,
      taper: this.config.taper, taperAlpha: this.config.taperAlpha,
    });

    this.resampler = new Resampler(d.sourceRate, this.config.sampleRate, C);
    this.filters = Array.from({ length: C }, () => new FilterChain(this.config.sampleRate, this.config.filter));

    this.source = this._makeSource();
    this.backend = this._makeBackend();

    this._pumping = false;
    this._rawWritten = 0;                       // absolute model-rate index for raw emit
    this.stats = { inferMsEMA: 0, count: 0, t0: 0, ep: this.backend.info?.ep ?? null };
  }

  _makeSource() {
    const s = this.config.source, d = this.config._derived;
    if (s.type === 'mqtt') return new MqttSource({ url: s.url, topicIn: s.topicIn, format: s.format, mqttLib: s.mqttLib });
    return new SyntheticSource({ channels: this.config.channels, rate: d.sourceRate, chunk: s.chunk, artifacts: s.artifacts, spatial: s.spatial });
  }
  _makeBackend() {
    const m = this.config.model, d = this.config._derived;
    const C = this.config.channels, fs = this.config.sampleRate, W = d.windowSamples, p = m.params || {};
    switch (m.type) {
      case 'onnx': case 'ae': case 'dae': case 'novelcnn':
        // AE / DAE / NovelCNN are neural nets: export to ONNX and run via the worker.
        if (!m.url) throw new Error(`model.url is required for '${m.type}' (export the net to ONNX)`);
        return new OnnxBackend({ modelUrl: m.url, kind: m.kind, channels: C, window: W, ep: this.config.ep, normalize: m.normalize });
      case 'orica': return new OricaBackend({ channels: C, sampleRate: fs, params: p });
      case 'lms':   return new LmsBackend({ channels: C, sampleRate: fs, params: p });
      case 'ssp':   return new SspBackend({ channels: C, params: p });
      case 'oscar': return new OscarBackend({ channels: C, sampleRate: fs, params: p });
      case 'filter': case 'movingaverage': case 'digitalfilter':
        return new FilterBackend({ channels: C, sampleRate: fs, params: m.type === 'movingaverage' ? { ...p, kind: 'movingaverage' } : p });
      case 'cca':   return new CcaBackend({ channels: C, window: W, params: p });
      case 'atar':  return new AtarBackend({ channels: C, window: W, params: p });
      case 'passthrough': default: return new PassthroughBackend();
    }
  }

  async start() {
    await this.backend.init();
    this._mode = this.backend.mode || 'window';
    this.stats.ep = this.backend.info?.ep ?? 'cpu';
    this.stats.t0 = performance.now();

    if (this.config.output.republish && this.config.source.type === 'mqtt') {
      this._pub = this.source; // reuse the same client for publishing
    }

    this.source.on('status', (st) => this.emit('status', st));
    this.source.on('error', (e) => this.emit('error', e));
    this.source.on('data', (frame) => this._onData(frame));
    this.source.start();
    this.emit('status', { pipeline: 'started', ...this.config._derived, ep: this.stats.ep });
  }

  stop() {
    this.source.stop();
    this.backend.dispose();
    this.emit('status', { pipeline: 'stopped' });
  }

  _onData(frame) {
    const C = this.config.channels;
    // 1. channel select / map
    const map = this.config.channelMap;
    let sel = [];
    for (let c = 0; c < C; c++) {
      const srcIdx = map ? map[c] : c;
      sel.push(frame.channels[srcIdx] ?? new Float32Array(frame.nSamples));
    }
    // 2. resample to model rate
    sel = this.resampler.process(sel);
    // 3. optional per-sample IIR filtering (in place, keeps filter state)
    if (this.filters[0].active) {
      for (let c = 0; c < C; c++) {
        const ch = sel[c], f = this.filters[c];
        for (let i = 0; i < ch.length; i++) ch[i] = f.process(ch[i]);
      }
    }
    // 4. absolute index + raw emit for display
    const absStart = this._rawWritten;
    this._lastN = sel[0].length;
    this.emit('raw', { start: absStart, nSamples: sel[0].length, channels: sel });
    this._rawWritten += sel[0].length;

    // 5a. stream backends: causal, process the chunk and emit cleaned immediately
    if (this._mode === 'stream') {
      const t = performance.now();
      const cleaned = this.backend.processStream(sel, absStart);
      this._recordInfer(performance.now() - t);
      const fin = { start: absStart, nSamples: sel[0].length, channels: cleaned };
      this.emit('clean', fin);
      if (this.config.output.republish) this._republish(fin);
      this.emit('stats', this._statsSnapshot());
      return;
    }
    // 5b. window backends: buffer, window on the fixed grid, denoise, overlap-add
    this.ring.write(sel);
    this._pump();
  }

  _recordInfer(dt) {
    this.stats.inferMsEMA = this.stats.inferMsEMA ? 0.9 * this.stats.inferMsEMA + 0.1 * dt : dt;
    this.stats.count++;
  }

  async _pump() {
    if (this._pumping) return;
    this._pumping = true;
    try {
      let w;
      while ((w = this.windower.next())) {
        const t = performance.now();
        const cleaned = await this.backend.process(w);
        this._recordInfer(performance.now() - t);

        const fin = this.ola.push({ start: cleaned.start, data: cleaned.data });
        if (fin) {
          this.emit('clean', fin);
          if (this.config.output.republish) this._republish(fin);
        }
        this.emit('stats', this._statsSnapshot());
      }
    } catch (e) {
      this.emit('error', e);
    } finally {
      this._pumping = false;
    }
  }

  _republish(fin) {
    if (this.config.source.type !== 'mqtt' || !this.source.client) return;
    const buf = encodeBinary({ nChannels: this.config.channels, nSamples: fin.nSamples, t: Date.now(), channels: fin.channels });
    this.source.client.publish(this.config.output.topicOut, new Uint8Array(buf), { qos: 0 });
  }

  _statsSnapshot() {
    const d = this.config._derived;
    const stream = this._mode === 'stream';
    const elapsed = (performance.now() - this.stats.t0) / 1000;
    return {
      ep: this.stats.ep,
      method: this.backend.info?.method || this.backend.info?.type,
      mode: this._mode,
      inferMs: +this.stats.inferMsEMA.toFixed(2),
      inferencesPerSec: elapsed > 0 ? +(this.stats.count / elapsed).toFixed(2) : 0,
      targetPerSec: stream ? null : +d.inferencesPerSec.toFixed(2),
      averaging: stream ? 1 : +d.averaging.toFixed(1),
      latencySec: stream ? +(this.backend.info?.latencySec ?? 0).toFixed(3) : +d.latencyMinSec.toFixed(2),
      hopMs: stream ? Math.round(1000 * (this._lastN || 1) / this.config.sampleRate) : Math.round(d.hopSec * 1000),
      convergence: this.backend.info?.convergence ?? null,   // ORICA: 0..1 adaptation progress
      rejected: this.backend.info?.rejected ?? null,         // ORICA: # artifact components zeroed
    };
  }
}
