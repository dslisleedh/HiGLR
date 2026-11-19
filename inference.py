"""Restore one image using a HiGLR checkpoint, with no degradation labels."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from higlr import HiGLR
from higlr.checkpoint import model_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', choices=['raw', 'ema'], default='ema')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=['torch', 'triton'], default='triton')
    parser.add_argument('--routing-backend', choices=['recompute', 'store'], default='recompute',
                        help='Triton inference strategy for block scores')
    parser.add_argument('--remote-k', type=int, default=16)
    args = parser.parse_args()
    if args.backend == 'triton':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.weights == 'ema' and 'ema' not in checkpoint:
        raise ValueError('No EMA weights in this checkpoint; select --weights raw')
    state = checkpoint['ema'] if args.weights == 'ema' else model_weights(checkpoint)
    classes = state['classifier.weight'].shape[0]
    model = HiGLR(num_classes=classes, backend=args.backend, eval_remote_k=args.remote_k,
                  routing_backend=args.routing_backend)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    with Image.open(args.input) as image:
        pixels = np.array(image.convert('RGB'))
    x = torch.from_numpy(pixels).permute(2, 0, 1)[None].float().div(255).to(args.device)
    h, w = x.shape[-2:]
    # Full-resolution evaluation uses reflect32, falling back to replicate only
    # for images too small for PyTorch's reflect padding constraint.
    pad = (0, (-w) % 32, 0, (-h) % 32)
    mode = 'reflect' if pad[1] < w and pad[3] < h else 'replicate'
    with torch.inference_mode():
        restored = model(F.pad(x, pad, mode=mode))[0, :, :h, :w].clamp(0, 1)
    result = restored.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)


if __name__ == '__main__':
    main()
