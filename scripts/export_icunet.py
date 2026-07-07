#!/usr/bin/env python3
"""
Export CNElab's pretrained IC-U-Net to ONNX for neurostream — one file per channel count.

COPY THIS FILE INTO THE ROOT of a cloned ArtifactRemovalTransformer repo
(so that `from model import cumbersome_model2` resolves), then run it there.

It reproduces exactly how utils.py builds and loads IC-U-Net:
    model = cumbersome_model2.UNet1(n_channels=30, n_classes=30)
    ckpt  = torch.load('./model/ICUNet/modelsave/checkpoint.pth.tar')
    model.load_state_dict(ckpt['state_dict'], strict=False)

IC-U-Net's weights ONLY exist at 30 channels. To get 4/8/16-channel models we wrap the
same 30-ch net in a zero-fill adapter: the exported graph takes [1, N, 1024], pads the
missing (30-N) channels with zeros, runs IC-U-Net, and slices the first N channels back
out. So each denoiser_{N}.onnx has N-channel I/O but shares the one pretrained checkpoint.
N=30 is native (no padding); N<30 is off-distribution (the model is starved of the spatial
context of the absent channels) — quality degrades as N shrinks. A real montage mapping
(placing your electrodes at their true positions among the 30) would beat naive first-N
zero-fill; swap `zero-fill` for that if you have channel locations.

Usage (from inside the ArtifactRemovalTransformer repo root):
    python export_icunet.py
    # -> writes denoiser_4.onnx denoiser_8.onnx denoiser_16.onnx denoiser_30.onnx (+ .meta.json)
    # then: cp denoiser_*.onnx ../models/     (if the repo is cloned under neurostream/)
"""
import json, os, sys

CKPT = './model/ICUNet/modelsave/checkpoint.pth.tar'  # adjust if the Drive layout differs
MODEL_CH = 30        # IC-U-Net's fixed trained channel count
WINDOW = 1024        # samples; IC-U-Net's trained segment (4 s @ 256 Hz)
CHANNEL_COUNTS = [4, 8, 16, 30]
OUT_DIR = '.'        # written here; copy into neurostream/models/
OPSET = 17

import torch
from model import cumbersome_model2

if not os.path.exists(CKPT):
    sys.exit(f'checkpoint not found: {CKPT}\n'
             f'Download the ICUNet folder from the repo Google Drive and place it so '
             f'this path exists (it should contain modelsave/checkpoint.pth.tar).')

base = cumbersome_model2.UNet1(n_channels=MODEL_CH, n_classes=MODEL_CH)
# weights_only=False: the checkpoint is a full dict {'state_dict': ...}, not a bare tensor bag
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
base.load_state_dict(state, strict=False)   # repo itself loads non-strict
base.eval()


class ZeroFillAdapter(torch.nn.Module):
    """Take [1, n_in, W]; zero-pad to [1, 30, W]; run IC-U-Net; return first n_in channels."""
    def __init__(self, net, n_in, n_model):
        super().__init__()
        self.net = net
        self.n_in = n_in
        self.pad = n_model - n_in

    def forward(self, x):
        if self.pad > 0:
            zeros = torch.zeros(x.shape[0], self.pad, x.shape[2], dtype=x.dtype, device=x.device)
            x = torch.cat([x, zeros], dim=1)
        y = self.net(x)
        return y[:, :self.n_in, :]


try:
    import onnxruntime as ort, numpy as np
    have_ort = True
except Exception as e:
    print(f'(onnxruntime not available — skipping parity checks: {e})', file=sys.stderr)
    have_ort = False

for n in CHANNEL_COUNTS:
    model = base if n == MODEL_CH else ZeroFillAdapter(base, n, MODEL_CH)
    model.eval()
    out = os.path.join(OUT_DIR, f'denoiser_{n}.onnx')
    dummy = torch.randn(1, n, WINDOW)
    torch.onnx.export(
        model, dummy, out,
        input_names=['eeg'], output_names=['clean'],
        opset_version=OPSET, dynamic_axes=None,   # static [1, n, 1024]
        dynamo=False,   # torch 2.9+ defaults to the dynamo exporter (needs onnxscript);
                        # IC-U-Net is a plain conv U-Net, so use the legacy TorchScript path
    )
    tag = 'native' if n == MODEL_CH else f'zero-fill {n}->{MODEL_CH}'
    line = f'exported -> {out}  input [1,{n},{WINDOW}]  ({tag})'

    if have_ort:
        with torch.no_grad():
            ref = model(dummy).cpu().numpy()
        sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
        got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
        err = float(np.max(np.abs(ref - got)))
        line += f'  parity {err:.2e} ' + ('OK' if err < 1e-3 else 'CHECK')
    print(line)

    meta = {'model_name': os.path.basename(out), 'kind': 'multichannel',
            'channels': n, 'model_channels': MODEL_CH, 'adapter': tag,
            'window_samples': WINDOW, 'sample_rate': 256}
    with open(os.path.splitext(out)[0] + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

print(f'done: {len(CHANNEL_COUNTS)} models in {os.path.abspath(OUT_DIR)}')
