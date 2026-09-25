"""Micro-benchmark one ViT layer variant-by-variant (fp16, batch 1, windowed and full attention)."""
import sys, math, time; sys.path[:0] = ["rfdetr", "."]
import numpy as np, mlx.core as mx
from fast_rfdetr.weights import extract
from harness import MODEL_PATH
from fast_rfdetr.model import RFDETR

def t(fn, *a, n=100, reps=12):
    f = lambda *a: [fn(*a) for _ in range(reps)]
    for _ in range(5): mx.eval(f(*a))
    ts = []
    for _ in range(n):
        s = time.perf_counter(); mx.eval(f(*a)); ts.append(time.perf_counter() - s)
    ts.sort(); return ts[len(ts)//2] / reps * 1e3

m = RFDETR(extract(MODEL_PATH), mx.float16, head_dtype=mx.float32)
h = mx.random.normal((4, 401, 384)).astype(mx.float16); mx.eval(h)
for i in (0, 3):
    full = mx.compile(lambda h: m.vit_layer(i, h, 1))
    print(f"layer {i} ({'full' if i == 3 else 'window'}) compiled: {t(full, h):.3f} ms")
p = m.p; pre = "backbone.0.encoder.encoder.encoder.layer.0."
W = [p["_qkv0.w"], p[pre+"attention.output.dense.weight"], p[pre+"mlp.fc1.weight"], p[pre+"mlp.fc2.weight"]]
x384 = h.reshape(1604, 384); x1536 = mx.random.normal((1604, 1536)).astype(mx.float16); mx.eval(x1536)
mm = mx.compile(lambda a, b: [a @ W[0], a @ W[1], a @ W[2], b @ W[3]])
print(f"4 matmuls only: {t(mm, x384, x1536):.3f} ms")
q = mx.random.normal((4, 6, 401, 64)).astype(mx.float16); mx.eval(q)
print(f"sdpa window only: {t(lambda q: mx.fast.scaled_dot_product_attention(q, q, q, scale=.125), q):.3f} ms")
ln = lambda x: mx.fast.layer_norm(x, p[pre+"norm1.weight"], p[pre+"norm1.bias"], 1e-6)
print(f"layernorm [1604x384]: {t(ln, h):.3f} ms")
g = mx.compile(lambda x: 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2)))))
print(f"gelu [1604x1536]: {t(g, x1536):.3f} ms")
tr = lambda x: mx.contiguous(x.reshape(4, 401, 6, 64).transpose(0, 2, 1, 3))
print(f"head transpose copy [1604x384]: {t(tr, h):.3f} ms")
res = mx.compile(lambda h, o, b, s: h + (o + b) * s)
print(f"bias+scale+residual [1604x384]: {t(res, h, h, p[pre+'mlp.fc2.bias'], p[pre+'layer_scale2.lambda1']):.3f} ms")

print("--- variants")
qw, qb = p["_qkv0.w"], p["_qkv0.b"]
y = h.reshape(1604, 384)
print(f"qkv matmul+bias (compiled add): {t(mx.compile(lambda y: y @ qw + qb), y):.3f} ms")
print(f"qkv addmm: {t(lambda y: mx.addmm(qb, y, qw), y):.3f} ms")
print(f"qkv matmul no bias: {t(lambda y: y @ qw, y):.3f} ms")
w1, b1 = p[pre+"mlp.fc1.weight"], p[pre+"mlp.fc1.bias"]
gelu = lambda x: 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))
print(f"fc1 matmul + (bias+gelu fused): {t(mx.compile(lambda y: gelu(y @ w1 + b1)), y):.3f} ms")
print(f"fc1 addmm + gelu: {t(mx.compile(lambda y: gelu(mx.addmm(b1, y, w1))), y):.3f} ms")
print(f"fc1 matmul only: {t(lambda y: y @ w1, y):.3f} ms")
# layernorm alternatives
lw, lb = p[pre+"norm1.weight"], p[pre+"norm1.bias"]
print(f"mx.fast.layer_norm fp16 [1604,384]: {t(lambda x: mx.fast.layer_norm(x, lw, lb, 1e-6), y):.3f} ms")
print(f"mx.fast.layer_norm fp16 [4,401,384]: {t(lambda x: mx.fast.layer_norm(x, lw, lb, 1e-6), h):.3f} ms")
print(f"mx.fast.layer_norm no affine: {t(lambda x: mx.fast.layer_norm(x, None, None, 1e-6), y):.3f} ms")
print(f"copy [1604,384] (x+1): {t(lambda x: x + 1, y):.3f} ms")
print(f"empty-ish (tiny add): {t(lambda x: x + 1, mx.ones((8,), mx.float16)):.3f} ms")
