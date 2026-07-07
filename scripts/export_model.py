#!/usr/bin/env python3
"""
Export a PyTorch EEG denoiser (ART / IC-U-Net / CLEEGN / your own) to ONNX for
neurostream's browser worker, with a numerical parity check and a metadata sidecar.

Two model kinds:
  single-channel : input  [1, 1, W]     -> output [1, 1, W]   (run per channel in the browser)
  multichannel   : input  [1, C, W]     -> output [1, C, W]   (run once per window)

Usage:
  python export_model.py --checkpoint model.pt --kind single-channel --window 1024
  python export_model.py --checkpoint model.pt --kind multichannel  --window 1024 --channels 8

Static shapes (batch fixed at 1) are intentional: they let ONNX Runtime Web optimize
the graph and keep the browser hot-path allocation-free.
"""
import argparse, json, os, sys

def build_model(checkpoint, kind, channels):
    """
    Replace this with your architecture's construction + weight loading.
    It must return an nn.Module in eval() mode whose forward takes the tensor shape
    described above. Example scaffold shown for a generic case.
    """
    import torch
    # from my_models import ICUNet, ART   # <-- import your class here
    # model = ICUNet(in_ch=1 if kind == 'single-channel' else channels)
    # state = torch.load(checkpoint, map_location='cpu')
    # model.load_state_dict(state['model'] if 'model' in state else state)
    raise NotImplementedError(
        "Wire build_model() to your checkpoint: construct the net, load weights, return model.eval()."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--kind', choices=['single-channel', 'multichannel'], required=True)
    ap.add_argument('--window', type=int, default=1024, help='samples; must equal the trained segment')
    ap.add_argument('--channels', type=int, default=8, help='only used for multichannel')
    ap.add_argument('--sample-rate', type=int, default=256)
    ap.add_argument('--out', default='denoiser.onnx')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    import torch
    model = build_model(args.checkpoint, args.kind, args.channels).eval()

    C = 1 if args.kind == 'single-channel' else args.channels
    dummy = torch.randn(1, C, args.window)

    torch.onnx.export(
        model, dummy, args.out,
        input_names=['eeg'], output_names=['clean'],
        opset_version=args.opset,
        dynamic_axes=None,           # static shapes
    )
    print(f'exported -> {args.out}  input [1,{C},{args.window}]')

    # ---- parity check: PyTorch vs onnxruntime ----
    try:
        import onnxruntime as ort, numpy as np
        with torch.no_grad():
            ref = model(dummy).cpu().numpy()
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        got = sess.run(['clean'], {'eeg': dummy.numpy()})[0]
        max_abs = float(np.max(np.abs(ref - got)))
        print(f'parity max|torch-onnx| = {max_abs:.3e}  ->', 'OK' if max_abs < 1e-3 else 'CHECK')
    except Exception as e:
        print(f'parity check skipped: {e}', file=sys.stderr)

    # ---- metadata sidecar the browser reads at load ----
    meta = {'model_name': os.path.basename(args.out), 'kind': args.kind,
            'channels': C, 'window_samples': args.window, 'sample_rate': args.sample_rate}
    meta_path = os.path.splitext(args.out)[0] + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'metadata -> {meta_path}')

    # ---- optional: warn on ops not covered by the WebGPU EP (informational) ----
    try:
        import onnx
        m = onnx.load(args.out)
        ops = sorted({n.op_type for n in m.graph.node})
        print('graph ops:', ', '.join(ops))
        print('note: WASM EP covers all ops; if WebGPU is slow, check these against the WebGPU op list.')
    except Exception:
        pass

if __name__ == '__main__':
    main()
