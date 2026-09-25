"""Shared harness: reference data, exact Ballform pre/post-processing, accuracy check, timing."""
from __future__ import annotations

import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import kernelopt_paths  # noqa: E402

MODEL_PATH = kernelopt_paths.model_path()
MODEL_SHA256 = "708789b50c42b5265cced64276a8beb1b7f294d324f954d359fd8a2d01f5a939"
REFERENCE_NPZ = kernelopt_paths.reference_npz()

KEPT_CLASSES = {1, 2, 4, 5, 6, 7, 8, 9, 10}
MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
STD = np.asarray([.229, .224, .225], dtype=np.float32)


def preprocess(frame):
    """Byte-for-byte copy of ballform app.basketball.preprocess."""
    rgb = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])


from fast_rfdetr import evaluation as _evaluation  # noqa: E402
from fast_rfdetr.evaluation import decode, match_objects  # noqa: E402,F401


def load_reference():
    return _evaluation.load_reference(REFERENCE_NPZ)


def accuracy_report(run_batch, batch_size=1, verbose=True):
    return _evaluation.accuracy_report(run_batch, REFERENCE_NPZ, batch_size=batch_size, verbose=verbose)


def _stats(times_ms):
    s = sorted(times_ms)
    return dict(median=statistics.median(s), p95=s[min(len(s) - 1, int(round(.95 * (len(s) - 1))))],
                mean=statistics.fmean(s), min=s[0], n=len(s))


def bench(fn, arg, n=60, warmup=10):
    for _ in range(warmup):
        fn(arg)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn(arg)
        times.append((time.perf_counter() - t) * 1e3)
    return _stats(times)


def bench_concurrent(fns, args, n=60, warmup=10):
    """Runs each fn(arg) in its own thread simultaneously; returns per-thread stats and wall throughput."""
    for fn, a in zip(fns, args):
        for _ in range(warmup):
            fn(a)
    barrier = threading.Barrier(len(fns) + 1)
    results = [None] * len(fns)

    def worker(k):
        times = []
        barrier.wait()
        for _ in range(n):
            t = time.perf_counter()
            fns[k](args[k])
            times.append((time.perf_counter() - t) * 1e3)
        results[k] = times

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(fns))]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return dict(per_thread=[_stats(r) for r in results], calls_per_s=len(fns) * n / wall,
                all=_stats([x for r in results for x in r]))


def fmt(s):
    return f"median {s['median']:.2f} ms  p95 {s['p95']:.2f} ms  (min {s['min']:.2f}, n={s['n']})"
