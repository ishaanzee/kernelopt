"""Per-stage latency of the MLX model (guarded against competing GPU jobs)."""
import sys; sys.path[:0] = ["rfdetr", "."]
import numpy as np, mlx.core as mx
from harness import MODEL_PATH, preprocess, load_reference, bench, fmt
from fast_rfdetr.detector import load_weights, PRECISIONS
from fast_rfdetr.model import RFDETR, GRID, D_MODEL, BB
from fast_rfdetr.quiet import measured
PAT = ["perf_run.py"]
prec = sys.argv[1] if len(sys.argv) > 1 else "fp32"
m = RFDETR(load_weights(MODEL_PATH, "cache/mlx-cache"), **PRECISIONS[prec])
x = mx.array(preprocess(load_reference()[0][1]).transpose(0, 2, 3, 1)).astype(m.dtype); mx.eval(x)
full = mx.compile(m.__call__)
bbf = mx.compile(m.backbone)
feats = bbf(x); mx.eval(feats)
pjf = mx.compile(lambda *f: m.projector([t.astype(m.proj_dtype) for t in f]))
mem = pjf(*feats).reshape(1, GRID * GRID, D_MODEL).astype(m.head_dtype); mx.eval(mem)
sel = mx.compile(m.select_queries); ref = sel(mem); mx.eval(ref)
dec = mx.compile(m.decoder)
pw = m.p[BB + "embeddings.patch_embeddings.projection.weight"]
patch = mx.compile(lambda a: mx.conv2d(a, pw, stride=16))
rows = [("full model", lambda: fmt(bench(lambda a: mx.eval(*full(a)), x, n=60))),
        ("backbone", lambda: fmt(bench(lambda a: mx.eval(bbf(a)), x, n=60))),
        ("  patch conv", lambda: fmt(bench(lambda a: mx.eval(patch(a)), x, n=60))),
        ("projector", lambda: fmt(bench(lambda f: mx.eval(pjf(*f)), feats, n=60))),
        ("select", lambda: fmt(bench(lambda a: mx.eval(sel(a)), mem, n=60))),
        ("decoder", lambda: fmt(bench(lambda a: mx.eval(dec(a, ref)), mem, n=60)))]
print(f"precision={prec}")
for name, f in rows:
    print(f"  {name:12s}", measured(f, PAT))
