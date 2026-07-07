// Wire format for EEG chunks over MQTT.
// Binary layout (little-endian): channel-major so per-channel views are contiguous.
//   uint16 nChannels
//   uint16 nSamples
//   float64 timestampMs
//   float32[nChannels * nSamples]   (ch0 all samples, ch1 all samples, ...)
//
// A `frame` in this framework is: { nChannels, nSamples, t, channels: Float32Array[] }

const HEADER_BYTES = 2 + 2 + 8;

export function encodeBinary(frame) {
  const { nChannels, nSamples, t = Date.now(), channels } = frame;
  const buf = new ArrayBuffer(HEADER_BYTES + nChannels * nSamples * 4);
  const dv = new DataView(buf);
  dv.setUint16(0, nChannels, true);
  dv.setUint16(2, nSamples, true);
  dv.setFloat64(4, t, true);
  const f32 = new Float32Array(buf, HEADER_BYTES);
  for (let c = 0; c < nChannels; c++) f32.set(channels[c], c * nSamples);
  return buf;
}

export function decodeBinary(arrayBufferOrView) {
  const buf = arrayBufferOrView instanceof ArrayBuffer
    ? arrayBufferOrView
    : arrayBufferOrView.buffer.slice(
        arrayBufferOrView.byteOffset,
        arrayBufferOrView.byteOffset + arrayBufferOrView.byteLength);
  const dv = new DataView(buf);
  const nChannels = dv.getUint16(0, true);
  const nSamples = dv.getUint16(2, true);
  const t = dv.getFloat64(4, true);
  const f32 = new Float32Array(buf, HEADER_BYTES, nChannels * nSamples);
  const channels = [];
  for (let c = 0; c < nChannels; c++) {
    channels.push(f32.subarray(c * nSamples, (c + 1) * nSamples));
  }
  return { nChannels, nSamples, t, channels };
}

// JSON fallback for debugging only. Shape mirrors the binary frame.
export function encodeJSON(frame) {
  return JSON.stringify({
    nChannels: frame.nChannels,
    nSamples: frame.nSamples,
    t: frame.t ?? Date.now(),
    channels: frame.channels.map((c) => Array.from(c)),
  });
}
export function decodeJSON(text) {
  const o = JSON.parse(typeof text === 'string' ? text : new TextDecoder().decode(text));
  return {
    nChannels: o.nChannels,
    nSamples: o.nSamples,
    t: o.t,
    channels: o.channels.map((c) => Float32Array.from(c)),
  };
}
