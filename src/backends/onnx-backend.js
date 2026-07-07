// Bridges the pipeline to the inference Web Worker. Keeps inference off the main
// thread and matches async results to requests by id.

export class OnnxBackend {
  constructor({ modelUrl, kind = 'single-channel', channels, window, ep = ['webgpu', 'wasm'], normalize = 'none' }) {
    this.mode = 'window';
    this.cfg = { modelUrl, kind, channels, window, ep, normalize };
    this.info = { type: 'onnx', method: `onnx (${kind}${normalize === 'zscore' ? ', z-scored' : ''})`, ep: null };
    this._pending = new Map();
    this._id = 0;
    this._worker = null;
  }

  init() {
    return new Promise((resolve, reject) => {
      this._worker = new Worker(new URL('../worker/inference-worker.js', import.meta.url), { type: 'module' });
      this._worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') { this.info.ep = m.ep; resolve(this.info); }
        else if (m.type === 'result') {
          const p = this._pending.get(m.id);
          if (p) { this._pending.delete(m.id); p.resolve(m.data); }
        } else if (m.type === 'error') {
          const first = this._pending.values().next().value;
          if (first) { first.reject(new Error(m.message)); }
          else reject(new Error(m.message));
        }
      };
      this._worker.onerror = (err) => reject(err);
      this._worker.postMessage({
        type: 'init',
        modelUrl: this.cfg.modelUrl, kind: this.cfg.kind,
        channels: this.cfg.channels, window: this.cfg.window, ep: this.cfg.ep,
        normalize: this.cfg.normalize,
      });
    });
  }

  // window: { start, data: Float32Array(channels*window) }
  process(window) {
    return new Promise((resolve, reject) => {
      const id = ++this._id;
      this._pending.set(id, {
        resolve: (data) => resolve({ start: window.start, data }),
        reject,
      });
      // copy so the transfer doesn't detach the caller's scratch
      const data = window.data.slice();
      this._worker.postMessage({ type: 'infer', id, data, channels: this.cfg.channels, window: this.cfg.window }, [data.buffer]);
    });
  }

  dispose() { if (this._worker) { this._worker.terminate(); this._worker = null; } }
}
