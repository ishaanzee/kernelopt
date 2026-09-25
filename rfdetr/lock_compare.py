"""Throughput of two instances on two threads under three locking strategies (same process, alternating)."""
import sys, threading, time, contextlib; sys.path[:0] = ["rfdetr", "."]
import numpy as np, mlx.core as mx
import fast_rfdetr.detector as det
from harness import MODEL_PATH, load_reference
imgs = [r[1] for r in load_reference()]
a = det.FastBasketballDetector(str(MODEL_PATH), cache_dir="cache/mlx-cache")
b = det.FastBasketballDetector(str(MODEL_PATH), cache_dir="cache/mlx-cache")

def full_lock(d, im):
    with det._GPU_LOCK:
        x = mx.stack([d._preprocess(im)]); bx, lg = d._forward(x); mx.eval(bx, lg)
        return np.array(bx), np.array(lg)
def no_lock(d, im):
    x = mx.stack([d._preprocess(im)]); bx, lg = d._forward(x); mx.eval(bx, lg)
    return np.array(bx), np.array(lg)
submit_lock = lambda d, im: d.raw(im)

def run(fn, n=150):
    lat = [[], []]
    def w(k, d):
        for i in range(n):
            t = time.perf_counter(); fn(d, imgs[(i + 5 * k) % len(imgs)]); lat[k].append((time.perf_counter() - t) * 1e3)
    th = [threading.Thread(target=w, args=(k, d)) for k, d in enumerate((a, b))]
    t = time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]; wall = time.perf_counter() - t
    all_ = sorted(lat[0] + lat[1])
    return 2 * n / wall, all_[len(all_) // 2], all_[int(.95 * (len(all_) - 1))]
single = []
for i in range(60):
    t = time.perf_counter(); a.raw(imgs[i % len(imgs)]); single.append((time.perf_counter() - t) * 1e3)
single.sort(); print(f"single thread end-to-end: median {single[30]:.2f} ms p95 {single[57]:.2f}")
for rep in range(2):
    for name, fn in (("full lock", full_lock), ("submit lock", submit_lock), ("no lock", no_lock)):
        cps, med, p95 = run(fn)
        print(f"rep{rep} {name:12s}: {cps:5.1f} calls/s  per-call median {med:6.2f} ms  p95 {p95:6.2f} ms")
