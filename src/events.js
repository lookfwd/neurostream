// Minimal event emitter. Browser- and node-safe (no globals).
export class Emitter {
  constructor() { this._h = new Map(); }
  on(evt, fn) {
    if (!this._h.has(evt)) this._h.set(evt, new Set());
    this._h.get(evt).add(fn);
    return () => this.off(evt, fn);
  }
  off(evt, fn) { this._h.get(evt)?.delete(fn); }
  emit(evt, payload) {
    const set = this._h.get(evt);
    if (!set) return;
    for (const fn of set) {
      try { fn(payload); } catch (e) { console.error(`[neurostream] listener for "${evt}" threw`, e); }
    }
  }
}
