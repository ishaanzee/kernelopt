"""Benchmark + accuracy check for fast_rfdetr.FastBasketballDetector (optionally side by side with ONNX Runtime + Core ML).

    python bench.py                          # MLX detector: load time, latency alone, two threads, accuracy
    python bench.py --baseline               # also the ORT Core ML EP path exactly as Ballform configures it
    python bench.py --precision fp16         # the faster fp16 variant (see README for its accuracy caveat)

Latencies are end to end: uint8 BGR frame in, (boxes, logits) numpy out, preprocessing included for both paths.
Each measurement waits for competing jobs matching --quiet-pattern to finish and is retried if one starts meanwhile.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kernelopt_paths  # noqa: E402
from fast_rfdetr.evaluation import accuracy_report, load_reference  # noqa: E402
from fast_rfdetr.quiet import measured  # noqa: E402



def stats(ms):
    s = sorted(ms)
    return dict(median=statistics.median(s), p95=s[min(len(s) - 1, round(.95 * (len(s) - 1)))], min=s[0], n=len(s))


def fmt(s):
    return f"median {s['median']:6.2f} ms   p95 {s['p95']:6.2f} ms   (n={s['n']})"


def time_calls(fn, arg, n, warmup=10):
    for _ in range(warmup):
        fn(arg)
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn(arg)
        out.append((time.perf_counter() - t) * 1e3)
    return stats(out)


def two_threads(fn_a, arg_a, fn_b, arg_b, n, warmup=5):
    for _ in range(warmup):
        fn_a(arg_a)
        fn_b(arg_b)
    lat = [[], []]
    barrier = threading.Barrier(3)

    def work(k, fn, arg):
        barrier.wait()
        for _ in range(n):
            t = time.perf_counter()
            fn(arg)
            lat[k].append((time.perf_counter() - t) * 1e3)

    th = [threading.Thread(target=work, args=(0, fn_a, arg_a)), threading.Thread(target=work, args=(1, fn_b, arg_b))]
    for t in th:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    return dict(a=stats(lat[0]), b=stats(lat[1]), all=stats(lat[0] + lat[1]), calls_per_s=2 * n / wall)


def subprocess_seconds(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    return float(out.stdout.strip().splitlines()[-1])


# ---------------- MLX detector ----------------
def bench_mlx(args, frame, crop, crop2, log):
    from fast_rfdetr import FastBasketballDetector
    res = {}
    cache = Path(args.cache_dir) if args.cache_dir else None
    load_code = ("import time; t = time.perf_counter(); from fast_rfdetr import FastBasketballDetector; "
                 "d = FastBasketballDetector({model!r}, precision={prec!r}, cache_dir={cache!r}); "
                 "print(time.perf_counter() - t)")
    with tempfile.TemporaryDirectory() as tmp:
        res["load_cold_s"] = measured(lambda: subprocess_seconds(load_code.format(model=args.model, prec=args.precision,
                                                                                   cache=tmp)), args.quiet, log=log)
    res["load_warm_s"] = measured(lambda: subprocess_seconds(load_code.format(
        model=args.model, prec=args.precision, cache=str(cache) if cache else None)), args.quiet, log=log)
    log(f"  load (fresh process, incl. import + warm-up call): first ever {res['load_cold_s']:.2f} s "
        f"(converts ONNX weights to the cache), cached {res['load_warm_s']:.2f} s")

    a = FastBasketballDetector(args.model, precision=args.precision, warmup_batch_sizes=(1, 2, 3), cache_dir=cache)
    b = FastBasketballDetector(args.model, precision=args.precision, cache_dir=cache)

    rep1 = accuracy_report(a.raw_batch, args.npz, batch_size=1, verbose=False)
    rep3 = accuracy_report(a.raw_batch, args.npz, batch_size=3, verbose=False)
    res["accuracy_batch1"], res["accuracy_batch3"] = _acc(rep1), _acc(rep3)
    for name, rep in (("batch 1", rep1), ("batch 3", rep3)):
        log(f"  accuracy ({name}): {_acc_line(rep)}")

    n = args.n
    res["frame"] = measured(lambda: time_calls(a.raw, frame, n), args.quiet, log=log)
    res["crop"] = measured(lambda: time_calls(a.raw, crop, n), args.quiet, log=log)
    res["batch3"] = measured(lambda: time_calls(a.raw_batch, [frame, crop, crop2], n), args.quiet, log=log)
    log(f"  alone, raw(1920x1080 frame):          {fmt(res['frame'])}")
    log(f"  alone, raw(1152x1080 crop):           {fmt(res['crop'])}")
    log(f"  alone, raw_batch([frame, crop, crop]): {fmt(res['batch3'])}  -> {res['batch3']['median'] / 3:.2f} ms/image")
    res["two_threads"] = measured(lambda: two_threads(a.raw, frame, b.raw, crop, n), args.quiet, log=log)
    tt = res["two_threads"]
    log(f"  two instances, two threads (frames | crops): per call {fmt(tt['all'])}   "
        f"throughput {tt['calls_per_s']:.1f} calls/s")
    return res


# ---------------- ONNX Runtime + Core ML baseline ----------------
def ort_session(model, cache):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    cache.mkdir(parents=True, exist_ok=True)
    return ort.InferenceSession(model, sess_options=options, providers=[
        ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
                                     "RequireStaticInputShapes": "1", "ModelCacheDirectory": str(cache)}),
        "CPUExecutionProvider"])


def bench_ort(args, frame, crop, crop2, log):
    import cv2
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    mean = np.asarray([.485, .456, .406], dtype=np.float32)
    std = np.asarray([.229, .224, .225], dtype=np.float32)

    def preprocess(f):
        rgb = cv2.cvtColor(cv2.resize(f, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None])

    cache = Path(args.cache_dir or tempfile.gettempdir()) / "ort-coreml-bench"
    load_code = ("import time, sys; sys.path.insert(0, '.'); t = time.perf_counter(); import bench, numpy as np; "
                 "s = bench.ort_session({model!r}, bench.Path({cache!r})); "
                 "s.run(None, {{s.get_inputs()[0].name: np.zeros((1, 3, 640, 640), np.float32)}}); "
                 "print(time.perf_counter() - t)")
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        res["load_cold_s"] = measured(lambda: subprocess_seconds(load_code.format(model=args.model, cache=tmp)),
                                      args.quiet, log=log)
    subprocess_seconds(load_code.format(model=args.model, cache=str(cache)))   # populate the warm cache
    res["load_warm_s"] = measured(lambda: subprocess_seconds(load_code.format(model=args.model, cache=str(cache))),
                                  args.quiet, log=log)
    log(f"  load (fresh process, incl. first inference): uncached {res['load_cold_s']:.2f} s, "
        f"with Core ML model cache {res['load_warm_s']:.2f} s")
    s1, s2 = ort_session(args.model, cache), ort_session(args.model, cache)

    def runner(s):
        name = s.get_inputs()[0].name

        def run(f):
            outs = s.run(None, {name: preprocess(f)})
            return next(o for o in outs if o.shape[-1] == 4), next(o for o in outs if o.shape[-1] == 11)
        return run

    r1, r2 = runner(s1), runner(s2)
    rep = accuracy_report(lambda ims: tuple(np.concatenate(x) for x in zip(*[r1(i) for i in ims])), args.npz, verbose=False)
    res["accuracy_batch1"] = _acc(rep)
    log(f"  accuracy: {_acc_line(rep)}")
    res["frame"] = measured(lambda: time_calls(r1, frame, args.n), args.quiet, log=log)
    res["crop"] = measured(lambda: time_calls(r1, crop, args.n), args.quiet, log=log)
    res["batch3"] = measured(lambda: time_calls(lambda fs: [r1(f) for f in fs], [frame, crop, crop2], args.n),
                             args.quiet, log=log)
    log(f"  alone, frame:                 {fmt(res['frame'])}")
    log(f"  alone, crop:                  {fmt(res['crop'])}")
    log(f"  alone, frame + 2 crops (3 calls): {fmt(res['batch3'])}")
    res["two_threads"] = measured(lambda: two_threads(r1, frame, r2, crop, args.n), args.quiet, log=log)
    tt = res["two_threads"]
    log(f"  two sessions, two threads (frames | crops): per call {fmt(tt['all'])}   "
        f"throughput {tt['calls_per_s']:.1f} calls/s")
    return res


def _acc(rep):
    return {k: v for k, v in rep.items() if k != "misses"} | {"misses": [f"{n}: {m}" for n, m in rep["misses"]]}


def _acc_line(rep):
    return (f"{rep['matched']}/{rep['objects']} golden objects (conf >= 0.4) matched within 0.03 conf / 0.01 box -> "
            f"{'PASS' if rep['passed'] else 'FAIL'};  max |dlogit| {rep['max_logit_diff']:.5f}, max |dbox| "
            f"{rep['max_box_diff']:.5f}; worst matched dconf {rep['worst_matched_conf_diff']:.5f}, "
            f"dbox {rep['worst_matched_box_diff']:.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model, npz = kernelopt_paths.model_path(required=False), kernelopt_paths.reference_npz(required=False)
    ap.add_argument("--model", default=model and str(model), required=model is None,
                    help="RF-DETR ONNX file (default: KERNELOPT_MODEL / paths.local.json)")
    ap.add_argument("--npz", default=npz and str(npz), required=npz is None,
                    help="golden reference .npz (default: KERNELOPT_REFERENCE_NPZ / paths.local.json)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("-n", type=int, default=60, help="timed calls per measurement (>= 50 recommended)")
    ap.add_argument("--baseline", action="store_true", help="also benchmark ONNX Runtime + Core ML EP")
    ap.add_argument("--no-mlx", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="MLX weight cache dir (default: <model dir>/mlx-cache/<sha>)")
    ap.add_argument("--quiet-pattern", dest="quiet", action="append", default=None,
                    help="process pattern to wait out before measuring (default: perf_run.py); pass '' to disable")
    ap.add_argument("--json", default=None, help="write all results to this file")
    args = ap.parse_args()
    args.quiet = [p for p in (args.quiet if args.quiet is not None else ["perf_run.py"]) if p]
    ref = load_reference(args.npz)
    frame, crop, crop2 = ref[0][1], ref[7][1], ref[8][1]
    results = {}
    log = print
    if not args.no_mlx:
        log(f"== fast_rfdetr (MLX, {args.precision}) ==")
        results["mlx"] = bench_mlx(args, frame, crop, crop2, log)
    if args.baseline:
        log("== ONNX Runtime + Core ML EP (Ballform's current path) ==")
        results["ort_coreml"] = bench_ort(args, frame, crop, crop2, log)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
