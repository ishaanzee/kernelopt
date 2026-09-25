"""Latency of MLX model variants (model only, input already resident on GPU)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from harness import MODEL_PATH, accuracy_report, bench, fmt, load_reference, preprocess  # noqa: E402
from fast_rfdetr.model import RFDETR  # noqa: E402
from fast_rfdetr.weights import extract  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--head-dtype", default=None)
    ap.add_argument("--unfused", action="store_true")
    ap.add_argument("--precision", default=None, help="use a fast_rfdetr.detector.PRECISIONS preset")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--no-acc", action="store_true")
    args = ap.parse_args()

    if args.precision:
        from fast_rfdetr.detector import PRECISIONS
        m = RFDETR(extract(MODEL_PATH), fused=not args.unfused, **PRECISIONS[args.precision])
    else:
        m = RFDETR(extract(MODEL_PATH), dtype=getattr(mx, args.dtype),
                   head_dtype=getattr(mx, args.head_dtype or args.dtype), fused=not args.unfused)
    fn = mx.compile(m.__call__) if args.compile else m.__call__

    def run_batch(images):
        x = mx.array(np.concatenate([preprocess(im).transpose(0, 2, 3, 1) for im in images]))
        b, l = fn(x)
        mx.eval(b, l)
        return np.array(b), np.array(l)

    if not args.no_acc:
        accuracy_report(run_batch, batch_size=args.batch)
    ref = load_reference()
    x = mx.array(np.concatenate([preprocess(ref[i][1]).transpose(0, 2, 3, 1) for i in range(args.batch)]))
    mx.eval(x)
    s = bench(lambda a: mx.eval(*fn(a)), x, n=args.n)
    print(f"{args.precision or args.dtype} compile={args.compile} batch={args.batch}: {fmt(s)}  ({s['median'] / args.batch:.2f} ms/image)")


if __name__ == "__main__":
    main()
