# kernelopt

A faster version of the ball detector in [Ballform](https://github.com/ishaanzee/ballform), my basketball shot
analysis app. The detector is RF-DETR-Medium (fine-tuned on basketball footage). It used to run through ONNX Runtime
with the Core ML execution provider. I ported it to [MLX](https://github.com/ml-explore/mlx) and replaced the slow
parts with custom Metal kernels.

On an M3 Pro, a 1080p frame now takes **28.2 ms instead of 50.9 ms (1.8× faster)**. The detections match the original
model on every test image. Ballform uses this package as its detector.

| per 1920×1080 frame, end to end | ONNX Runtime + Core ML | **kernelopt (fp32, default)** | kernelopt (fp16, opt-in) |
|---|---|---|---|
| latency, median (p95) | 50.9 ms (55.7) | **28.2 ms (28.6)** | 24.9 ms (25.0) |
| preprocessing | 4.7 ms on CPU | **0.42 ms on GPU** | 0.42 ms |
| accuracy check (147 objects, 11 images) | pass | **pass** | fail (see [why](#why-not-fp16)) |

## How it works

I profiled first, then only wrote kernels where the profile said time was going. All four kernels are in
[`fast_rfdetr/kernels.py`](fast_rfdetr/kernels.py). They're written with `mx.fast.metal_kernel`, so MLX compiles them
at runtime and you don't need Xcode.

1. **Residual + LayerNorm.** Every transformer layer adds its output back to the residual stream, then normalizes it.
   This kernel does both in one pass over memory (one simdgroup per row).
2. **Deformable attention sampling.** The decoder samples 2 points per head with bilinear interpolation and takes a
   weighted sum. That was about 20 small gather and elementwise ops per layer. Now it's one kernel.
3. **fp32 matrix multiply with bias + GELU built in.** The first MLP matmul in each backbone layer writes its output
   already activated, which skips a separate 20 MB read and write per layer.
4. **Preprocessing.** One kernel takes the raw uint8 BGR frame and resizes it to 640×640, converts it to RGB and
   normalizes it. Its output is bitwise identical to Ballform's `cv2` preprocessing (see
   [below](#reproducing-cv2resize-exactly)).

The rest is plain MLX: `mx.compile` for the whole forward pass, MLX's fused attention (SDPA), and a few layout changes
so that tensors never get copied between ops.

### What each step bought

Model-only latency for one 640×640 input at batch 1, fp32:

| step | latency | what changed |
|---|---|---|
| ONNX Runtime + Core ML | 42.6 ms | the starting point (12 Core ML partitions) |
| straight MLX port | 42.5 ms | matched the original outputs on the first try (max logit diff 0.0015) |
| + `mx.compile` | 32.8 ms | fuses elementwise ops and removes per-call graph building |
| + residual + LayerNorm kernel, biases folded into matmuls | 30.9 ms | LayerNorm took 60 µs on a 1.2 MB tensor; the fused kernel runs at memory bandwidth |
| + deformable attention kernel, one matmul for all 4 decoder value projections | ~30 ms | decoder 3.3 → 2.6 ms |
| + head-major QKV projection | 29.1 ms | see the note below |
| + matmul with fused GELU | 28.1 ms | |
| + GPU preprocessing | 28.2 ms end to end | preprocessing used to add 4.7 ms on the CPU |

**The QKV step.** The old code split the attention projection into q, k and v with integer indexing (`qkv[0]`). In MLX's
Python API that's a gather, so it copied the data on every call, costing 0.15 ms per layer. Slices are free views. So
the projection now writes straight into a `[n, 18 heads, L, 64]` layout, and `qkv[:, :6]`, `qkv[:, 6:12]` and
`qkv[:, 12:]` hand SDPA its inputs without copying anything.

## Using it

```bash
uv pip install -e /path/to/kernelopt     # installs mlx 0.32.2, numpy, onnx
```

```python
from fast_rfdetr import FastBasketballDetector

det = FastBasketballDetector("models/basketball-rfdetr.onnx")
boxes, logits = det.raw(frame_bgr)                   # [1, 300, 4], [1, 300, 11], same as the ONNX outputs
boxes, logits = det.raw_batch([frame, left, right])  # up to 3 images, sizes can differ
```

- **Weights:** the first time you load a given `.onnx` file, its weights are extracted and cached as safetensors next
  to it (`mlx-cache/<hash>/`). After that, loading takes about 0.25 s and doesn't import `onnx`.
- **Threads:** two detectors on two threads are safe. Graph building is serialized by a lock, but the GPU work
  overlaps, so you get 40 calls/s instead of 35.
- **Options:** `precision="fp16"` turns on the faster variant. `warmup_batch_sizes=(1, 3)` traces those batch sizes up
  front so the first real call isn't slow.

The model file isn't in this repo; it's Ballform's fine-tuned RF-DETR-Medium export. The model code mirrors that ONNX
graph op for op, so other RF-DETR exports would need changes.

## Accuracy

The reference is the original model running in ONNX Runtime on the CPU in fp32, on 11 frames and crops from real
games. To pass, every object it detects with confidence ≥ 0.4 (147 in total) has to show up again with the same
class, confidence within 0.03, and box within 0.01. The fp32 version passes at batch 1 and batch 3. Its largest logit
difference anywhere is 0.0012.

### Why not fp16

fp16 is about 12% faster, and its detections look just as good, but it fails the check, and the reason is structural.

RF-DETR ranks 1,600 candidate boxes and sends the top 300 to the decoder in rank order. Each of the 300 query slots has
its own learned embedding, so a detection's confidence depends on which slot it lands in. fp16 rounding moves the
scores by about 0.001, and lots of candidates are tied that closely. So about a third of the slots get reshuffled, and
a real object's confidence can move by around 0.1 (one went from 0.838 to 0.731 with the same box).

| fp16 used in… | slots matching the reference (of 300) |
|---|---|
| backbone | ~145 |
| backbone, with an fp32 residual stream | ~160 |
| projector only | ~190 |
| decoder only | ~190 |
| nowhere (fp32) | 300 |

fp32 noise is around 1e-6, too small to flip any ranks. That's why fp32 stays the default. bf16 has fewer mantissa
bits, so it's worse, and it isn't faster here.

### Reproducing `cv2.resize` exactly

The same ranking problem means preprocessing has to be bit-exact: changing one input pixel by 1 can reshuffle slots
just like fp16 does. That turned out to be harder than expected, because `cv2.resize` in the OpenCV wheel Ballform uses
(opencv-contrib-python 4.14, macOS arm64) isn't OpenCV's own resize.

- For both of Ballform's input sizes it hands the work to Arm's **KleidiCV** library. KleidiCV uses 16.16 fixed-point
  coordinates stepped once per 16-lane vector, 8-bit fractions, and NEON `vraddhn` rounding.
- Other sizes use OpenCV's own NEON path. That path has a quirk of its own: at the bottom edge it clips the row index
  but keeps the fraction, so both taps read the same row.

[`resize_tables.py`](fast_rfdetr/resize_tables.py) reproduces both as lookup tables, written against the KleidiCV
source. The GPU kernel reads those tables. The tests check the result against `cv2.resize` on the 11 reference images
and 20 other sizes.

This was validated on an M3 Pro with cv2 4.14.0. An M4 might take a different KleidiCV code path (untested).

### MLX's conv2d changes with batch size

Batched calls originally didn't match single-image calls. I traced it to MLX's `conv2d`. Once
`batch × height × width ≥ 4096`, MLX switches 3×3 convolutions to the Winograd algorithm, which is faster but has
about 7× more error (1.1e-4 vs 1.6e-5 against fp64). The projector's feature map is 40×40, so batch 3 crosses the
threshold, and that error was enough to reshuffle slots.

The workaround here is to run those six convolutions one image at a time when batched. I also sent the fix upstream:
[ml-explore/mlx#4639](https://github.com/ml-explore/mlx/pull/4639) adds an `MLX_CONV_WINOGRAD=0` switch to turn
Winograd off. It's merged but not in a release yet, so this repo keeps the workaround and stays pinned to MLX 0.32.2.

## What didn't work

- **A custom matmul for everything.** My first version staged tiles through threadgroup memory and ran at 0.8 TFLOPS.
  The accumulators were spilling to memory until I force-unrolled the loops, which brought it to 4.1. Loading straight
  from device memory got it to 4.3, but MLX's own matmul does 4.8. So the custom one is only used where the fused GELU
  makes up the difference.
- **Batching Ballform's frames with their side crops.** Batching saves only about 5% per image (28.2 → 26.7 ms)
  because the GPU is already saturated at batch 1. I tried it in Ballform anyway, and every test clip got 5–12%
  slower. Ballform already overlaps detection with other work. Only 23–56% of frames need the crops at all, and
  running them on every frame slowed down pose estimation, which shares the GPU.

I skipped two things on purpose:

- **A flash-attention kernel.** MLX's SDPA already runs at about 80% of peak on these shapes. Its real cost was the q/k/v
  copies, which are fixed.
- **MPSGraph or a Core ML re-export.** Neither can run custom kernels, and Core ML's GPU path was at about 30% of peak.

## How much faster it could get

Timed stage by stage, the backbone takes 23.7 ms, the projector 2.0 ms, query selection 0.8 ms and the decoder
2.6 ms. The backbone is about 90 GFLOP (68 in linear layers, 21 in attention). On large matmuls, MLX reaches about
4.9 TFLOPS in fp32 on this GPU, and the backbone's matmuls already run at 4.8–5.5.

- **The fp32 floor** is about 18.4 ms for the backbone's matmuls and attention alone, even with zero overhead.
- **About 2 ms is left:**
  - fuse residual + LayerNorm into the end of the matmuls before them (~1 ms);
  - merge the decoder's small matmuls (~0.3 ms);
  - fuse LayerNorm + SiLU in the projector (~0.2 ms).

  That would get it to roughly 26 ms. Anything beyond that needs fp16, which the accuracy check rules out.

## Repo layout

```
fast_rfdetr/          the package Ballform imports
  detector.py         public API: FastBasketballDetector, weight cache, thread lock
  model.py            RF-DETR-Medium forward pass in MLX
  kernels.py          the four Metal kernels
  resize_tables.py    bit-exact model of cv2.resize
  weights.py          pulls weights out of the ONNX file (first load only)
  evaluation.py       the accuracy check
  quiet.py            waits for other GPU jobs to finish before benchmarking
bench.py              latency + accuracy report, optionally next to ONNX Runtime
tests/                pytest: bit-exact preprocessing, accuracy, batch = single, thread safety
rfdetr/               scratch scripts from the optimization work (profiling, probes, ONNX graph dumps)
```

## Development

```bash
uv venv --python 3.12 .venv
uv pip install -e . pytest onnxruntime==1.30.0 opencv-contrib-python==4.14.0.94
```

`bench.py` and the tests need the ONNX model and the reference outputs (`.npz`), which aren't in the repo. Point to
them with the `KERNELOPT_MODEL` and `KERNELOPT_REFERENCE_NPZ` environment variables, or put both paths in a gitignored
`paths.local.json` (format in [`kernelopt_paths.py`](kernelopt_paths.py)).

```bash
pytest tests -q              # correctness checks
python bench.py --baseline   # the numbers in this README, next to ONNX Runtime + Core ML
```

All numbers were measured on an M3 Pro (18-core GPU, 36 GB) with nothing else running. They're medians over 60 warm
calls, and end-to-end latency includes preprocessing on both sides. The scripts in `rfdetr/` expect to be run from the
repo root.
