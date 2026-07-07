# neurostream

Real-time EEG denoising in the browser. Subscribe to a live EEG stream over
**MQTT-over-WebSockets**, clean it with a selectable **method** — classical adaptive
filters, blind source separation, wavelet, or a neural **ONNX** denoiser — and get a
cleaned stream back, plotted live and optionally republished. Runs at **4, 8, or 16
channels**.

[![Cleaning Streaming EEG data in the Browser (WebGPU) with ONNX denoisers e.g. IC-U-Net](https://img.youtube.com/vi/DKR9yC2xpv8/0.jpg)](https://www.youtube.com/watch?v=DKR9yC2xpv8)

Methods split two ways: **causal** ones (LMS, SSP, OSCAR, moving average, IIR) run
per-sample with no added latency; **windowed** ones (CCA, ATAR, ONNX autoencoders) run on
overlapped windows and reconstruct by overlap-add.

It ships runnable today: the default config uses a synthetic source and a passthrough
"model", so the whole pipeline works before you have a broker or an exported model.

```
MQTT (wss) ──▶ decode ──▶ resample→filter→map ──▶ ring buffer
   ──▶ windower (fixed grid) ──▶ ONNX worker (WebGPU→WASM) ──▶ overlap-add
   ──▶ cleaned stream  ──▶ scope  and/or  republish (wss)
```

## Quick start

Serve the folder over HTTP (module scripts need it) and open the demo:

```bash
npx serve .        # or: python3 -m http.server
# open http://localhost:3000/demo/
```

Press **Start**. You get a live scope immediately (synthetic EEG with injected blink +
EMG, passthrough model). Switch **Model → ONNX** and give it a `.onnx` URL to denoise
for real; switch **Source → MQTT** and point it at your broker.

## Use as a library

```js
import { Pipeline, Scope } from './src/index.js';

const pipe = new Pipeline({
  channels: 8,                    // 4 | 8 | 16
  windowSec: 4.0,                 // MUST equal the model's trained segment
  overlap: 0.80,                  // hop = window * (1 - overlap)
  source: { type: 'mqtt', url: 'wss://broker:8084/mqtt', topicIn: 'eeg/raw', format: 'binary' },
  model:  { type: 'onnx', url: '/models/denoiser.onnx', kind: 'single-channel' },
  output: { republish: true, topicOut: 'eeg/clean' },
});

pipe.on('clean', ({ start, channels }) => { /* Float32Array[] cleaned EEG */ });
pipe.on('stats', (s) => console.log(s.ep, s.inferMs, s.latencySec));
await pipe.start();
```

## Config reference

| Key | Default | Meaning |
|---|---|---|
| `channels` | `8` | 4, 8, or 16, end to end. |
| `sampleRate` | `256` | Rate the model expects; the source is resampled to this. |
| `windowSec` | `4.0` | Model's trained segment. Change it → re-export the model. |
| `overlap` | `0.80` | Fraction. Higher = smoother + more compute, same latency. |
| `taper` / `taperAlpha` | `tukey` / `0.5` | Overlap-add cross-fade window. |
| `ep` | `['webgpu','wasm']` | ONNX Runtime Web execution providers, in order. |
| `source.type` | `synthetic` | `synthetic` or `mqtt`. |
| `model.type` / `model.kind` | `passthrough` / `single-channel` | `onnx`+`single-channel`\|`multichannel`, or `passthrough`. |
| `channelMap` | `null` | `int[]`: `out[i] = in[channelMap[i]]`. |
| `filter` | `{highpass:0,notch:0}` | Optional per-channel IIR, Hz (0 disables). |
| `output.republish` | `false` | Publish cleaned frames to `topicOut` (MQTT source only). |

Derived values (`pipe.config._derived`, also surfaced via `stats`): `hopSamples`,
`inferencesPerSec`, `averaging`, `latencyMinSec`/`latencyMaxSec`.

## Denoising methods

Set `model.type` (and optional `model.params`). Causal methods add no windowing latency;
windowed methods go through the overlap-add path.

| `type` | Mode | Removes / does | Key params (defaults) |
|---|---|---|---|
| `passthrough` | windowed | identity (baseline / no model) | — |
| `filter` | causal | band-limit, line noise | `highpass:0, notch:0, lowpass:40` (Hz) |
| `movingaverage` | causal | high-freq / EMG smoothing | `taps:8` |
| `lms` | causal | reference-correlated (ocular) via NLMS | `reference:0, order:8, mu:0.1` |
| `orica` | causal | online ICA unmixing + kurtosis-based artifact-component rejection | `mu:0.008, kurtosis:10, whiten:0.02` |
| `ssp` | causal | a calibrated artifact topography | `projectors:[]` (length-`channels` vectors) |
| `oscar` | causal | conditioning + transient suppression | `highpass:1, notch:0, k:4` |
| `cca` | windowed | muscle (low-autocorrelation BSS comps) | `threshold:0.9` or `remove:N` |
| `atar` | windowed | wavelet transients, tunable | `k:5, mode:'linatten'\|'elim'\|'clamp'` |
| `onnx` / `ae` / `dae` / `novelcnn` | windowed | learned denoising | `url`, `kind` |

Notes:
- **AE / DAE / NovelCNN** are neural nets — train and export to ONNX (`scripts/export_model.py`),
  then run them through the same worker as any `onnx` model; set `type` to `ae`/`dae`/`novelcnn`
  (aliases of `onnx`) plus `url`.
- **LMS** subtracts activity correlated with a reference channel; point `reference` at a
  frontal/EOG-like electrode. It passes the reference channel through unchanged.
- **SSP** needs calibrated artifact patterns — pass the scalp topographies (e.g. average
  blink pattern) as `projectors`; with none it is identity.
- **CCA** and **ATAR** operate per window; CCA needs several channels to separate sources.
- **OSCAR** is implemented as a transparent composite of the online conditioning +
  transient-suppression stages such a module performs, not a proprietary reproduction.
- **ORICA** (online recursive ICA) separates sources sample-by-sample (online whitening +
  orthogonal natural-gradient demixing) and auto-rejects high-kurtosis (spiky, e.g. ocular)
  components, remixing the rest back to sensors. It needs ~`25·N²` samples to converge —
  progress is reported as `convergence` (0→1) and rejection is held off until it settles, so
  the header shows `converging NN%` then `converged · K rejected`. Separation quality rises
  with channel count (16/30 strong, 8 workable, 4 thin), the same way ICA always does.

```js
new Pipeline({ channels: 8, model: { type: 'atar', params: { k: 4, mode: 'elim' } } });
new Pipeline({ channels: 8, model: { type: 'lms', params: { reference: 0, mu: 0.1 } } });
new Pipeline({ channels: 8, model: { type: 'cca', params: { remove: 2 } } });
```

## MQTT payload contract

Binary, little-endian, **channel-major**:

```
uint16  nChannels
uint16  nSamples
float64 timestampMs
float32 [nChannels * nSamples]   // ch0 all samples, ch1 all samples, ...
```

`encodeBinary` / `decodeBinary` implement it; publish frames in this layout. A JSON
fallback (`format: 'json'`) exists for debugging.

## Channel counts and model choice (4 / 8 / 16)

- **`single-channel`** models (EEGdenoiseNet-family) run the same weights independently
  on each electrode — works at any count with one model. No spatial info, weaker on
  muscle. This is the safe default that runs at 4, 8, and 16 today.
- **`multichannel`** models (ART / IC-U-Net) exploit cross-channel structure and denoise
  better, but each checkpoint is tied to its trained channel count — export one per count.
- Physics: spatial denoising improves with electrodes. 16 is strong, 8 solid, 4 thin —
  at 4 channels a single-channel denoiser (or classical regression) often matches a
  starved multichannel model.

## Exporting a model

```bash
python scripts/export_model.py --checkpoint model.pt --kind single-channel --window 1024
```

Wire `build_model()` to your architecture. The script exports static-shape ONNX, runs a
PyTorch-vs-ONNX parity check, and writes a `.meta.json` sidecar. Put `denoiser.onnx`
under `models/` at the repo root (the demo's default `model.url` is `/models/denoiser.onnx`,
served from the site root), or wherever your `model.url` points.

## Example: Step by step instructions for IC-U-Net

A full worked example: take CNElab's pretrained **IC-U-Net** (a 30-channel denoiser,
trained on 1024-sample / 4 s segments at 256 Hz), convert it to ONNX, and run it here on
synthetic EEG. IC-U-Net's weights are not bundled with any package — they live on the
authors' Google Drive — so this is a one-time download-and-convert, then drop-in.

This repo already ships the pieces you need: the export helper
[`scripts/export_icunet.py`](scripts/export_icunet.py), and `channels: 30` is accepted end
to end (config + demo), so the model runs at its **native** channel count.

### 1. Download the IC-U-Net code and weights

In a scratch folder (not inside this repo):

```bash
git clone https://github.com/CNElab-Plus/ArtifactRemovalTransformer.git
cd ArtifactRemovalTransformer
```

Download only the **`ICUNet`** folder from the repo's Google Drive
(<https://drive.google.com/drive/folders/1ahbqcyBs6pwfWHaIf_N978DZD-JmGQJg>) — via the
browser (right-click → Download) or `pip install gdown && gdown --folder <url>` — and place
it so this exact path exists (the repo's loader hard-codes it):

```
ArtifactRemovalTransformer/model/ICUNet/modelsave/checkpoint.pth.tar
```

### 2. Convert IC-U-Net to ONNX (one time)

CPU PyTorch is fine — this is a one-off conversion, no GPU needed:

```bash
pip install torch onnx onnxruntime scipy numpy
cp /path/to/neurostream/scripts/export_icunet.py .   # into the ART repo root
python export_icunet.py
```

Expected output:

```
exported -> denoiser.onnx  input [1,30,1024]
parity max|torch-onnx| = 1.xxxe-06 -> OK
metadata -> denoiser.meta.json
```

`OK` means the ONNX graph matches the PyTorch model numerically. The script imports the
repo's own `model.cumbersome_model2.UNet1(n_channels=30, n_classes=30)` and loads
`checkpoint['state_dict']` non-strictly — the same construction/loading `utils.py` uses —
so it must run **from the ART repo root**. It pins `dynamo=False` because PyTorch 2.9+
defaults `torch.onnx.export` to the dynamo exporter (which needs `onnxscript`); the legacy
TorchScript path exports this plain conv U-Net without that dependency.

### 3. Drop the model into neurostream and run

```bash
mkdir -p models
cp denoiser.onnx models/denoiser.onnx    # repo root: /models is served at the site root

npx serve .        # or: python3 -m http.server 3000
# open http://localhost:3000/demo/
```

The demo's default Model URL is `/models/denoiser.onnx` — an absolute path served from the
repo root, so the file goes in `<repo>/models/`, not `demo/models/`.

In the demo, set:

| Control | Value |
|---|---|
| **Channels** | `30 (IC-U-Net)` |
| **Source** | Synthetic (blink + EMG) |
| **Window** | `4` |
| **Overlap** | `80%` |
| **Model** | Autoencoder / DAE / NovelCNN (ONNX) |
| **Model URL** | `/models/denoiser.onnx` |
| **kind** | `multichannel` |

Press **Start**. You get 30 synthetic channels with the cleaned trace lagging the raw
ghost by ~2–4 s (window latency), blinks and EMG suppressed by the real IC-U-Net weights.

### Notes

- **No tensor-name matching needed.** The worker reads the model's own input/output names
  (`session.inputNames[0]` / `outputNames[0]`), so you don't edit the worker for this.
- **This runs IC-U-Net at its native 30 channels** — the faithful path. Running it at
  4/8/16 (your hardware montage) is a *different* problem: the 30-channel model needs a
  channel-map/zero-fill layer to accept fewer inputs, which is not built in yet.

## Latency and jitter

Latency is dominated by the **window**, not compute. At `windowSec: 4`, a well-cleaned
sample lags live by ~2–4 s; overlap sets refresh cadence, not latency (80% → a new
finalized chunk every ~0.8 s, each sample averaged ~5×). Warm inference on WebGPU for a
small model is milliseconds. Jitter is low — fixed shapes, no data-dependent iteration —
plus the MQTT transport hop. This is a **quality** layer, not a sub-100 ms control path;
for that, use classical block filtering alongside it.

## Tests

```bash
npm test    # runs all four suites
```

- `test/math.test.mjs` — linear algebra (inverse, Cholesky, Jacobi eigen) and db4 wavelet
  perfect reconstruction, all to machine precision.
- `test/core.test.mjs` — config, codec, ring buffer, and overlap-add reconstructing a
  passthrough signal to <1e-4 at 4/8/16 channels.
- `test/backends.test.mjs` — method invariants: CCA reconstructs exactly when nothing is
  removed and reduces power when it does; ATAR is identity at a huge threshold and
  attenuates a transient; LMS cancels a reference-correlated artifact; SSP nulls a
  projected topography; moving average and OSCAR reduce noise/transients.
- `test/integration.test.mjs` — full pipeline (synthetic → passthrough) preserves data
  end to end.

## Notes / limits

- The browser worker imports onnxruntime-web from a CDN (pinned in
  `src/worker/inference-worker.js`); self-host by changing `ORT_URL` and `wasmPaths`.
- WebGPU ships in Chrome/Edge and (recently) Firefox/Safari; the WASM fallback covers all
  ops everywhere, so the app degrades gracefully.
- The ring buffer and inference run on the main thread and a worker respectively, with
  only copied window data crossing the boundary — no `SharedArrayBuffer` cross-thread
  requirement, so no special COOP/COEP headers are needed for the default setup.
- `single-channel` export uses static batch 1, so the worker loops channels; if you
  export with a dynamic batch you can switch it to a single batched run.
