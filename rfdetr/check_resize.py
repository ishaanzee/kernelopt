"""Bit-exactness check of fast_rfdetr.resize_tables against cv2.resize on reference + random images."""
import sys
import numpy as np
from pathlib import Path
import cv2
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
from harness import load_reference
from fast_rfdetr.resize_tables import resize_reference, uses_kleidicv
rng = np.random.default_rng(0)
cases = [(name, img) for name, img, _, _ in load_reference()]
sizes = [(1080, 1920), (1080, 1152), (720, 1280), (500, 333), (480, 640), (640, 640), (2160, 3840), (300, 1000),
         (700, 700), (641, 1279), (1000, 300), (1079, 1919), (1080, 1100), (650, 1900), (1440, 2560), (1200, 1600),
         (900, 1000), (1081, 1921), (720, 960), (645, 645)]
cases += [(f"rand{h}x{w}", rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in sizes]
bad = 0
for name, img in cases:
    ref = cv2.resize(img, (640, 640))
    d = np.abs(ref.astype(int) - resize_reference(img).astype(int))
    bad += d.max() > 0
    h, w = img.shape[:2]
    print(f"{name:14s} {'kleidicv' if uses_kleidicv(w, h, 640, 640) else 'opencv  '} max diff {d.max()}  "
          f"pixels differing {np.count_nonzero(d)}")
print("ALL EXACT" if not bad else f"{bad} cases differ")
