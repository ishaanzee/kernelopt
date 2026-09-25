"""FastBasketballDetector: RF-DETR-Medium (basketball fine-tune) on the Apple GPU via MLX, drop-in for the ONNX outputs."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np

from .kernels import preprocess_gpu
from .model import RFDETR
from .resize_tables import tables

MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
STD = np.asarray([.229, .224, .225], dtype=np.float32)
INPUT = 640
MAX_BATCH = 3

# fp32: meets the golden-output bar robustly (TopK slot order identical to the ONNX fp32 reference).
# fp16: backbone + projector in fp16 (query selection / decoder stay fp32). ~15% faster, but fp16 noise reorders
#       near-tied encoder scores, so some objects land in different query slots and their confidence can move by
#       ~0.1 versus the fp32 reference. See README.
PRECISIONS = {
    "fp32": dict(dtype=mx.float32),
    "fp16": dict(dtype=mx.float16, proj_dtype=mx.float16, head_dtype=mx.float32, residual_dtype=mx.float16),
}

# Graph construction, mx.compile tracing and Metal command encoding are serialized across instances/threads with
# this lock (MLX does not document them as thread-safe). Waiting for the GPU happens outside the lock, so two
# threads' calls still overlap on the GPU.
_GPU_LOCK = threading.Lock()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_weights(model_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    """Weights from the RF-DETR ONNX file, cached as safetensors keyed by the file's sha256.

    Only the first load of a given ONNX file needs the `onnx` package; later loads read the cache."""
    model_path = Path(model_path)
    digest = _file_sha256(model_path)[:20]
    cache = Path(cache_dir) if cache_dir else model_path.parent / "mlx-cache" / digest
    cached = cache / "weights.safetensors"
    if cached.exists():
        return {k: np.array(v) for k, v in mx.load(str(cached)).items()}
    from .weights import extract  # needs `onnx`
    w = extract(model_path)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / "weights.tmp.safetensors"
    mx.save_safetensors(str(tmp), {k: mx.array(v) for k, v in w.items()})
    tmp.replace(cached)
    return w


class FastBasketballDetector:
    def __init__(self, model_path: str, precision: str = "fp32", warmup_batch_sizes=(1,), cache_dir=None):
        """model_path: the RF-DETR ONNX file (basketball-rfdetr.onnx). Its weights are extracted once and cached.

        warmup_batch_sizes: batch sizes to trace and run once now, so the first real call is warm."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
        self.precision = precision
        with _GPU_LOCK:
            self.model = RFDETR(load_weights(model_path, cache_dir), **PRECISIONS[precision])
            self._forward = mx.compile(self.model.__call__)
            lut = (np.arange(256, dtype=np.float32)[:, None] / 255. - MEAN) / STD  # same float32 ops as preprocess
            self._lut = mx.array(np.ascontiguousarray(lut.T))
        self._tables: dict[tuple[int, int], tuple[int, mx.array, mx.array]] = {}
        dummy = np.zeros((1080, 1920, 3), np.uint8)
        for b in warmup_batch_sizes:
            self.raw_batch([dummy] * b)

    def _taps(self, w: int, h: int):
        key = (w, h)
        if key not in self._tables:
            mode, xt, yt = tables(w, h, INPUT, INPUT)
            self._tables[key] = (mode, mx.array(xt), mx.array(yt))
        return self._tables[key]

    def _preprocess(self, image_bgr: np.ndarray) -> mx.array:
        if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected a uint8 BGR image of shape (H, W, 3)")
        h, w, _ = image_bgr.shape
        mode, xt, yt = self._taps(w, h)
        return preprocess_gpu(mx.array(np.ascontiguousarray(image_bgr)), xt, yt, self._lut, mode, INPUT)

    def raw(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (boxes [1,300,4] f32, logits [1,300,11] f32), the same as the ONNX outputs."""
        return self.raw_batch([image_bgr])

    def raw_batch(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Images of any sizes (each resized to 640x640). Returns boxes [N,300,4], logits [N,300,11]."""
        if not 1 <= len(images) <= MAX_BATCH:
            raise ValueError(f"batch size must be 1..{MAX_BATCH}")
        with _GPU_LOCK:  # graph building + command encoding are serialized; GPU execution overlaps
            x = mx.stack([self._preprocess(im) for im in images])
            boxes, logits = self._forward(x)
            mx.async_eval(boxes, logits)
        mx.eval(boxes, logits)
        return np.array(boxes), np.array(logits)
