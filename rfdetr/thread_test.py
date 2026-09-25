"""Two detector instances on two threads: outputs must equal single-threaded outputs, no crashes."""
import sys, threading, time, contextlib; sys.path[:0] = ["rfdetr", "."]
import numpy as np
import fast_rfdetr.detector as det
from harness import MODEL_PATH, load_reference

CACHE = "cache/mlx-cache"
use_lock = "--no-lock" not in sys.argv
if not use_lock:
    det._GPU_LOCK = contextlib.nullcontext()
t = time.perf_counter()
a = det.FastBasketballDetector(str(MODEL_PATH), cache_dir=CACHE)
print(f"instance A load {time.perf_counter() - t:.2f} s")
t = time.perf_counter()
b = det.FastBasketballDetector(str(MODEL_PATH), cache_dir=CACHE)
print(f"instance B load {time.perf_counter() - t:.2f} s")
ref = load_reference()
imgs = [r[1] for r in ref]
expected = {i: a.raw(im) for i, im in enumerate(imgs)}
errors, iters = [], int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 40

def worker(d, offset):
    try:
        for k in range(iters):
            i = (k + offset) % len(imgs)
            bx, lg = d.raw(imgs[i])
            if not (np.array_equal(bx, expected[i][0]) and np.array_equal(lg, expected[i][1])):
                errors.append((offset, k, i, float(np.abs(bx - expected[i][0]).max())))
    except Exception as e:  # noqa: BLE001
        errors.append(repr(e))

th = [threading.Thread(target=worker, args=(d, o)) for d, o in ((a, 0), (b, 5))]
t = time.perf_counter()
[x.start() for x in th]; [x.join() for x in th]
print(f"lock={use_lock}: {2 * iters} calls in {time.perf_counter() - t:.2f} s, mismatches/errors: {len(errors)} {errors[:3]}")
