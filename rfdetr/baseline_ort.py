"""Baseline: ONNX Runtime + Core ML EP exactly as Ballform configures it."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from harness import (MODEL_PATH, accuracy_report, bench, bench_concurrent, fmt, load_reference,  # noqa: E402
                     preprocess)

CACHE = Path(__file__).parent.parent / "cache" / "ort-coreml"


def make_session(model=MODEL_PATH, provider="coreml", cache=CACHE, profile=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = profile
    providers = ["CPUExecutionProvider"]
    if provider == "coreml":
        cache.mkdir(parents=True, exist_ok=True)
        providers.insert(0, ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
            "RequireStaticInputShapes": "1", "ModelCacheDirectory": str(cache)}))
    return ort.InferenceSession(str(model), sess_options=options, providers=providers)


def runner(session):
    name = session.get_inputs()[0].name
    outs = [o.name for o in session.get_outputs()]

    def run_x(x):
        res = dict(zip(outs, session.run(None, {name: x})))
        boxes = next(v for v in res.values() if v.shape[-1] == 4)
        logits = next(v for v in res.values() if v.shape[-1] == 11)
        return boxes, logits

    def run_batch(images):
        pairs = [run_x(preprocess(im)) for im in images]
        return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
    return run_x, run_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--provider", default="coreml", choices=["coreml", "cpu"])
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--no-concurrent", action="store_true")
    args = ap.parse_args()

    t = time.perf_counter()
    s1 = make_session(args.model, args.provider)
    print(f"load: {time.perf_counter() - t:.2f} s  providers={s1.get_providers()}")
    run_x, run_batch = runner(s1)
    accuracy_report(run_batch)

    ref = load_reference()
    frame = ref[0][1]
    x = preprocess(frame)
    print("run only (preprocessed input):   ", fmt(bench(run_x, x, n=args.n)))
    print("end to end (1080p BGR -> outputs):", fmt(bench(lambda f: run_x(preprocess(f)), frame, n=args.n)))
    print("preprocess only (CPU):           ", fmt(bench(preprocess, frame, n=args.n)))
    if args.no_concurrent:
        return
    t = time.perf_counter()
    s2 = make_session(args.model, args.provider)
    print(f"second session load: {time.perf_counter() - t:.2f} s")
    run_x2, _ = runner(s2)
    r = bench_concurrent([run_x, run_x2], [x, preprocess(ref[7][1])], n=args.n)
    print("two sessions, two threads:        ", fmt(r["all"]), f" throughput {r['calls_per_s']:.1f} calls/s")


if __name__ == "__main__":
    main()
