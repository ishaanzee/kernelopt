# kernelopt: Apple Silicon kernel optimization

Workspace for making ML inference fast on Apple GPUs (M3 Pro, 18 GPU cores, 36 GB). The approach is to port the model
to **MLX**, then replace only the parts that profiling shows are slow with **custom Metal kernels**
(`mx.fast.metal_kernel`). Those kernels are compiled at runtime, so no Xcode is needed; this Mac has Command Line
Tools only.

```
fast_rfdetr/         the deliverable package (import FastBasketballDetector)
  detector.py        public API, weight cache, GPU lock
  model.py           RF-DETR-Medium forward pass in MLX (mirrors the ONNX graph op for op)
  kernels.py         Metal kernels: residual+LayerNorm, fp32 GEMM+bias+GELU, deformable-attention sampling,
                     fused BGR->640x640 resize+normalize
  resize_tables.py   bit-exact model of cv2.resize (KleidiCV + OpenCV paths) as per-column/row tap tables
  weights.py         ONNX -> weight dict extraction (first load only)
  evaluation.py      Ballform decode + golden-object matching (the accuracy bar)
  quiet.py           benchmark hygiene: wait out / detect competing GPU jobs
bench.py             benchmark + accuracy report (optionally side by side with ORT + Core ML)
tests/               pytest: bit-exactness, accuracy bar, batch consistency, two-thread safety
rfdetr/              research scripts (graph dumps, probes, profiling, ORT baseline)
```

Dev setup: `uv venv --python 3.12 .venv && uv pip install -e . pytest onnxruntime==1.30.0 opencv-contrib-python==4.14.0.94`.
The model and golden-output files are not in the repo. Point `bench.py`, `tests/` and `rfdetr/` at them with
`KERNELOPT_MODEL` / `KERNELOPT_REFERENCE_NPZ`, or a gitignored `paths.local.json` (see `kernelopt_paths.py`).

---

## RF-DETR-Medium detector for Ballform

### Result

All numbers were measured with `bench.py --baseline` on an otherwise idle M3 Pro. Latency is end to end: a uint8 BGR
frame goes in and numpy `(boxes, logits)` come out, with preprocessing included for both paths. Medians and p95s are
over 60 warm calls.

| | ORT 1.30 + Core ML EP (today) | **fast_rfdetr fp32 (default)** | fast_rfdetr fp16 (opt-in) |
|---|---|---|---|
| 1920×1080 frame, alone | 50.9 ms (p95 55.7) | **28.2 ms (p95 28.6)** | 24.9 ms (p95 25.0) |
| 1152×1080 side crop, alone | 53.1 ms (p95 59.1) | **28.2 ms (p95 30.7)** | 24.8 ms |
| frame + 2 crops | 144 ms (3 calls) | **80.1 ms (1 `raw_batch` call, 26.7 ms/image)** | 69.1 ms |
| two instances on two threads | 87.0 ms/call, 22.8 calls/s | **49.9 ms/call, 40.0 calls/s** | 45.3 ms/call, 43.6 calls/s |
| load, fresh process | 31.2 s uncached / 0.46 s with Core ML cache | **0.34 s first ever / 0.24 s cached** | 0.30 s |
| accuracy bar (11 images) | 147/147 pass | **147/147 pass** | 144/147 **fail** |
| max \|Δlogit\| / \|Δbox\| vs golden | 0.00139 / 0.00010 | **0.00124 / 0.00014** (batch 3: 0.00130 / 0.00014) | slots reordered (see below) |

- ORT's run-only latency was 42.6 ms median (p95 43.4) this morning, matching your 43.7 ms. Its end-to-end number
  varied between 47 and 51 ms across runs. The MLX numbers were stable within ±0.3 ms.
- **Targets:** 20 ms is **not met** in fp32 (28.2 ms, 1.8× faster than today's path). The fp16 variant gets 24.9 ms
  but fails your accuracy bar. The 15 ms stretch goal is below the hardware floor at this FLOP count (see "Where
  the floor is").
- **Preprocessing:** now one GPU kernel, **0.42 ms instead of 4.7 ms**, and **bitwise identical** to Ballform's
  `preprocess()` on all 11 reference images and 20 other input sizes.

### Using it from Ballform

```bash
uv pip install --python <ballform>/.venv/bin/python -e <kernelopt>
# adds only: fast-rfdetr, mlx 0.32.2, mlx-metal, onnx, ml-dtypes (nothing upgraded or removed)
```
```python
from fast_rfdetr import FastBasketballDetector
det = FastBasketballDetector("models/basketball-rfdetr.onnx")   # ~0.25 s; first ever load writes
                                                                 # models/mlx-cache/<sha20>/weights.safetensors
boxes, logits = det.raw(frame_bgr)                     # [1,300,4], [1,300,11] float32, same as the ONNX outputs
boxes, logits = det.raw_batch([frame, left, right])    # [N,300,4], [N,300,11]; images may have different sizes
```
- **Outputs:** they feed straight into `app.basketball.decode`.
- **Runtime environment:** no network and no torch. `onnx` is only imported on the first load of a given `.onnx`
  file, keyed by sha256.
- **Threads:** two instances are safe on two threads. Graph building and Metal command encoding are serialized by a
  module lock, but GPU execution overlaps, which is where the 40 calls/s comes from. Verified with 600 concurrent
  calls, all bit-identical to single-threaded output (`tests/`).
- **Options:** `precision="fp16"` opts into the faster variant. `warmup_batch_sizes=(1, 3)` pre-traces batch sizes
  so the first real call is warm (each is traced on first use otherwise, ~0.1 s).

Run `python bench.py --baseline` for the numbers above, and `pytest tests -q` for the checks.

### What I tried and what each step bought

These are model-only timings for one 640² input at batch 1 in fp32, unless noted.

| Step | Latency | Notes |
|---|---|---|
| ORT + Core ML EP baseline | 42.6 ms | 12 Core ML partitions, fp32 |
| MLX port, eager | 42.5 ms | Matches the golden outputs on the first try (max Δlogit 0.0015) |
| + `mx.compile` | 32.8 ms | Elementwise fusion and graph-build overhead removed |
| + fused residual + LayerScale + LayerNorm Metal kernel, biases folded into GEMMs (`addmm`) | 30.9 ms | LayerNorm was 60 µs for a 1.2 MB tensor; the fused kernel runs at memory bandwidth (42 µs fp16) |
| + deformable-attention sampling kernel, one GEMM for all 4 decoder value projections | ~30 ms | Decoder 3.3 → 2.6 ms. GridSample plus the weighted sum was ~0.7 ms as ~20 gather/elementwise ops per layer |
| + head-major QKV projection | 29.1 ms | SDPA was silently copying strided q/k/v views (0.15 ms per layer). Projecting straight into `[n, 18, L, 64]` makes the per-head slices contiguous. SDPA's output is already `[B,L,H,D]` in memory, so merging heads is free |
| + custom fp32 GEMM with fused bias + GELU epilogue (MLP fc1) | 28.1 ms | Uses 8×8 simdgroup matrices loaded straight from device memory (4.3 TFLOPS vs MLX's 4.8), but it drops a 20 MB GELU pass per layer. Output is bit-identical to MLX matmul followed by `mx.erf` GELU |
| + fused GPU preprocessing | 28.2 ms end to end | Was 4.7 ms on the CPU |

**The main fp32 target, the backbone, is now close to the hardware.** It's about 90 GFLOP: 68 in the linears
and 21 in attention. The measured stage breakdown is backbone 23.7 ms, projector 2.0 ms, query selection 0.8 ms and
decoder 2.6 ms. On this GPU MLX reaches 4.9 TFLOPS in fp32 and 5.9 in fp16 on large GEMMs, and the backbone's GEMMs
already run at 4.8–5.5.

### What didn't work, and the two things that make this model unusual

1. **fp16 fails your bar, and the cause is structural rather than something to tune.**
   - RF-DETR's two-stage decoder ranks 1600 encoder proposals with TopK and puts rank *r* into query slot *r*.
     Every slot has its own learned content embedding and reference-point delta, so a detection's confidence
     depends on its proposal's rank.
   - Any fp16 upstream of TopK perturbs the scores by ~1e-3, and many proposals are near-tied at that level. About a
     third of the 300 slots get reassigned:

     | fp16 in… | slots matching golden |
     |---|---|
     | backbone (fp16 residual) | ~145 |
     | backbone (fp32 residual) | ~160 |
     | projector only | ~190 |
     | decoder only | ~190 |
     | nothing (all fp32) | 300 |

   - When a real object's proposal moves slot, its confidence can shift by about 0.1 (frame3: 0.838 → 0.731, same
     box). Whether an image passes the 0.03 tolerance is luck: some fp16 mixes happened to pass all 11 images, most
     didn't.
   - The detections are the same quality, just not reproducible to 0.03. Core ML fp32, ORT CPU and this port all
     agree slot for slot because fp32 noise (~1e-6) is too small to flip ranks.
   - The same sensitivity is why the naive fp16 ONNX conversion wouldn't have helped even if it had loaded. bf16 is
     worse (fewer mantissa bits) and no faster here.
2. **`cv2.resize` in Ballform's OpenCV wheel isn't OpenCV's own resize.**
   - The opencv-contrib-python 4.14 macOS arm64 wheel sends 3-channel uint8 downscales with ratio ≤ 3 (both Ballform
     sizes) through Arm's **KleidiCV 26.03** library.
   - KleidiCV uses 16.16 fixed-point coordinates that are stepped incrementally per 16-lane vector, 8-bit
     fractions, and NEON `vraddhn` rounding.
   - Other sizes use OpenCV's own NEON path, which has its own quirk: at the y border it keeps the fraction and
     clips the row index, so both taps read the same row.
   - I reproduce both exactly as precomputed tap tables (`resize_tables.py`, checked against the KleidiCV source).
     Bit-exactness matters because a 1-LSB input change reorders TopK just like fp16 does.
   - Caveats: this is validated on this M3 Pro with cv2 4.14.0. An M4 may take KleidiCV's SME path (untested).
     Ballform's venv also has `opencv-python` 5.0 installed alongside the 4.14 contrib wheel; `import cv2`
     currently loads 4.14. If that ever flips, the golden preprocessing changes, not this kernel.
3. **MLX's `conv2d` picks a less accurate 3×3 algorithm at batch 3.** Its error vs fp64 is 1.1e-4 instead of 1.6e-5,
   which reshuffled slots in batched calls. The projector now runs its six 3×3 convs per image when batched, so
   batch outputs match single-image outputs slot for slot.
4. **A threadgroup-staged custom GEMM** was 0.8 TFLOPS until the fragment loops were force-unrolled (the
   accumulators spilled), then 4.1 TFLOPS. Loading fragments straight from device memory reached 4.3. That's still
   below MLX's 4.8, so the custom GEMM is used only where its fused epilogue more than pays for the gap (fc1 + GELU).
5. **Batching** frame + 2 crops saves only ~5% per image (28.2 → 26.7 ms), because the GPU is already compute-bound
   at batch 1. Its value is one call instead of three. Two threads overlap GPU work for +13% throughput (40.0 vs 35.5 calls/s).
6. **Not attempted, on purpose:**
   - A flash-attention kernel: MLX's SDPA already runs at ~80% of peak on these shapes. Its real cost was the layout
     copies, which are fixed.
   - An MPSGraph or Core ML re-export: they can't express the custom kernels, and Core ML's GPU path was at about
     30% of peak.

### Where the floor is, and what's left

- **fp32 backbone floor:** the backbone's matmuls and attention alone need ≈18.4 ms at MLX's best fp32 throughput.
  So even with zero overhead the fp32 model can't reach 20 ms on this GPU.
- **fp16 floor:** ≈15.2 ms for the backbone plus ~4 ms for the rest. That puts the 15 ms stretch goal out of reach
  without lossy changes, and 20 ms is only plausible in fp16, which your accuracy bar rules out.
- **Remaining fp32 headroom, about 2 ms:**
  - Fuse the residual+LN into the fc2 and attention-projection GEMM epilogues. This needs row-complete tiles, about
    1 ms.
  - Merge the decoder's small GEMMs (q+k, offsets+weights), about 0.3 ms.
  - Fuse LayerNorm+SiLU in the projector, about 0.2 ms.
  - That would bring the model to ~26 ms.
- **Pipeline:** the biggest win for Ballform is probably structural. Batch each frame with its side crops (one 80 ms
  call instead of three 50 ms calls) and keep the two-thread layout. Per-call GPU time drops from ~48 to ~27 ms,
  which should cut the 4.7 s/clip spent waiting on side-crop calls roughly in half. I haven't measured that inside
  Ballform.
