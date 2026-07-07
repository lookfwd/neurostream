// EEG source that subscribes to a broker over MQTT-over-WebSockets and decodes
// frames per the codec contract. Requires mqtt.js, provided either as `opts.mqttLib`
// or as a global `mqtt` (e.g. loaded via <script> from a CDN).

import { Emitter } from '../events.js';
import { decodeBinary, decodeJSON } from '../codec.js';

export class MqttSource extends Emitter {
  constructor({ url, topicIn, format = 'binary', mqttLib = null }) {
    super();
    this.url = url;
    this.topicIn = topicIn;
    this.format = format;
    this.lib = mqttLib || (typeof globalThis !== 'undefined' ? globalThis.mqtt : null);
    this.client = null;
    if (!this.lib) throw new Error('mqtt.js not found: pass { mqttLib } or load it globally as `mqtt`');
  }

  start() {
    this.client = this.lib.connect(this.url, { reconnectPeriod: 1500, protocolVersion: 4 });
    this.client.on('connect', () => {
      this.client.subscribe(this.topicIn, { qos: 0 });
      this.emit('status', { source: 'mqtt', state: 'streaming', url: this.url, topic: this.topicIn });
    });
    this.client.on('reconnect', () => this.emit('status', { source: 'mqtt', state: 'reconnecting' }));
    this.client.on('close', () => this.emit('status', { source: 'mqtt', state: 'closed' }));
    this.client.on('error', (err) => this.emit('error', err));
    this.client.on('message', (_topic, payload) => {
      try {
        const frame = this.format === 'json' ? decodeJSON(payload) : decodeBinary(payload);
        this.emit('data', frame);
      } catch (e) { this.emit('error', e); }
    });
  }

  stop() {
    if (this.client) { this.client.end(true); this.client = null; }
    this.emit('status', { source: 'mqtt', state: 'stopped' });
  }
}
