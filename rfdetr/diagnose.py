"""For a model config, show nearest same-class candidate for every missed golden object, and TopK slot agreement."""
import sys; sys.path[:0] = ["rfdetr", "."]
import numpy as np, mlx.core as mx
from fast_rfdetr.weights import extract
from harness import MODEL_PATH, preprocess, load_reference, decode, match_objects

def diagnose(m, names=None):
    fn = mx.compile(m.__call__)
    for name, img, gb, gl in load_reference():
        if names and name not in names: continue
        b, l = fn(mx.array(preprocess(img).transpose(0, 2, 3, 1))); mx.eval(b, l)
        b, l = np.array(b)[0], np.array(l)[0]
        g, c = decode(gb, gl), decode(b, l)
        n, misses, _, _ = match_objects(g, c)
        # slot agreement: same slot has nearly the same box as golden?
        same = np.abs(b - gb).max(-1) < 0.01
        print(f"{name}: {n - len(misses)}/{n} matched, slots with same box as golden: {same.sum()}/300")
        for cls, conf, box in misses:
            best = min((cc for cc in c if cc[0] == cls), key=lambda cc: np.abs(np.subtract(cc[2], box)).max(), default=None)
            print(f"   miss cls={cls} conf={conf:.4f} box={np.round(box, 4)}")
            if best: print(f"   nearest cand conf={best[1]:.4f} box={np.round(best[2], 4)} dbox={np.abs(np.subtract(best[2], box)).max():.4f}")

if __name__ == "__main__":
    from fast_rfdetr.model import RFDETR
    w = extract(MODEL_PATH)
    for label, kw in [("fp32 fused", dict(dtype=mx.float32, fused=True)),
                      ("fp16 fused / fp32 head", dict(dtype=mx.float16, head_dtype=mx.float32, fused=True))]:
        print("=====", label)
        diagnose(RFDETR(w, **kw), names=sys.argv[1:] or None)
