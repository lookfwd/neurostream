// Identity backend. Proves the windowing/overlap-add path independent of any model,
// and serves as the "no model" graceful-degradation fallback. Runs inline (no worker).

export class PassthroughBackend {
  constructor() { this.mode = 'window'; this.info = { type: 'passthrough', method: 'passthrough (identity)', ep: 'cpu' }; }
  async init() { return this.info; }
  // window.data: Float32Array(channels*window), channel-major. Return same shape.
  async process(window) {
    return { start: window.start, data: window.data };
  }
  dispose() {}
}
