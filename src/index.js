// neurostream — real-time EEG denoising in the browser.
export { Pipeline } from './pipeline.js';
export { Scope } from './plot.js';
export { resolveConfig, DEFAULTS } from './config.js';
export { encodeBinary, decodeBinary, encodeJSON, decodeJSON } from './codec.js';

// Building blocks (for custom pipelines / testing)
export { RingBuffer } from './ring-buffer.js';
export { Windower } from './windower.js';
export { OverlapAdd } from './overlap-add.js';
export { FilterChain, MovingAverage } from './filters.js';
export { SyntheticSource } from './sources/synthetic-source.js';
export { MqttSource } from './sources/mqtt-source.js';

// Denoising method backends
export { PassthroughBackend } from './backends/passthrough.js';
export { OnnxBackend } from './backends/onnx-backend.js';       // AE / DAE / NovelCNN / any ONNX
export { FilterBackend } from './backends/filter-backend.js';  // moving average / IIR
export { LmsBackend } from './backends/lms-backend.js';
export { SspBackend } from './backends/ssp-backend.js';
export { OscarBackend } from './backends/oscar-backend.js';
export { CcaBackend } from './backends/cca-backend.js';
export { AtarBackend } from './backends/atar-backend.js';
