"""Prototype: tiled simdgroup-matrix GEMM with fused bias + exact-GELU epilogue, vs MLX matmul + separate GELU."""
import math
import sys
import time

import mlx.core as mx

HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
// erf / expm1f copied from MLX (mlx/backend/metal/kernels/erf.h, expm1f.h; Copyright 2023 Apple Inc., MIT)
// so the fused GELU is bit-identical to mx.erf.
inline float k_expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y; int i;
  j = fma(1.442695f, a, 12582912.f); j = j - 12582912.0f; i = (int)j;
  f = fma(j, -6.93145752e-1f, a);
  s = f * f; if (a == 0.0f) s = a;
  r = 1.97350979e-4f; r = fma(r, f, 1.39309070e-3f); r = fma(r, f, 8.33343994e-3f);
  r = fma(r, f, 4.16668020e-2f); r = fma(r, f, 1.66666716e-1f); r = fma(r, f, 4.99999970e-1f);
  u = (j == 1) ? (f + 0.5f) : f; v = fma(r, s, u);
  s = 0.5f * b; t = ldexp(s, i); y = t - s; x = (t - y) - s;
  r = fma(v, t, x) + y; r = r + r;
  if (j == 0) r = v;
  if (j == 1) r = v + v;
  return r;
}
inline float k_expm1f(float a) {
  float r = k_expm1f_scaled_unchecked(a, 1.0f);
  if (abs(a - 1.0f) > 88.0f) { r = pow(2, a); r = fma(r, r, -1.0f); }
  return r;
}
inline float k_erf(float a) {
  float r, s, t, u;
  t = metal::abs(a); s = a * a;
  if (t > 0.927734375f) {
    r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
    u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f); r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f); r = metal::fma(r, t, -t);
    r = -k_expm1f(r); r = metal::copysign(r, a);
  } else {
    r = -5.96761703e-4f; r = metal::fma(r, s, 4.99119423e-3f); r = metal::fma(r, s, -2.67681349e-2f);
    r = metal::fma(r, s, 1.12819925e-1f); r = metal::fma(r, s, -3.76125336e-1f);
    r = metal::fma(r, s, 1.28379166e-1f); r = metal::fma(r, a, a);
  }
  return r;
}
"""

SRC = """
    // C[M,N] = act(A[M,K] @ B[K,N] + bias[N]); fp32 accumulate. Tiles BM x BN, BK deep; WM x WN simdgroups.
    constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
    constexpr int PA = BK + 4, PB = BN + 4;             // padded threadgroup strides
    constexpr int NT = 32 * WM * WN;
    threadgroup float As[BM * PA];
    threadgroup float Bs[BK * PB];
    const int M = m_rows[0];
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int m0 = threadgroup_position_in_grid.y * BM, n0 = threadgroup_position_in_grid.x * BN;
    const int sm = (sg / WN) * (TM * 8), sn = (sg % WN) * (TN * 8);
    simdgroup_matrix<float, 8, 8> acc[TM][TN];
    _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += BK) {
        // A tile: BM x BK as vec4 loads
        _Pragma("clang loop unroll(full)") for (int e = tid; e < BM * BK / 4; e += NT) {
            const int r = e / (BK / 4), c = (e % (BK / 4)) * 4;
            const int gm = m0 + r;
            float4 v = gm < M ? float4(*((const device vec<T, 4>*)(A + size_t(gm) * K + k0 + c))) : float4(0.0f);
            *((threadgroup float4*)(As + r * PA + c)) = v;
        }
        _Pragma("clang loop unroll(full)") for (int e = tid; e < BK * BN / 4; e += NT) {
            const int r = e / (BN / 4), c = (e % (BN / 4)) * 4;
            *((threadgroup float4*)(Bs + r * PB + c)) = float4(*((const device vec<T, 4>*)(W + size_t(k0 + r) * N + n0 + c)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)") for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> a[TM], b[TN];
            _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) simdgroup_load(a[i], As + (sm + i * 8) * PA + kk, PA);
            _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j) simdgroup_load(b[j], Bs + kk * PB + sn + j * 8, PB);
            _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i)
                _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue: lane owns row fm, cols fn, fn+1 of each 8x8 fragment
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) {
        const int row = m0 + sm + i * 8 + fm;
        if (row >= M) continue;
        _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j) {
            const int col = n0 + sn + j * 8 + fn;
            thread auto& el = acc[i][j].thread_elements();
            float v0 = el[0] + float(bias[col]), v1 = el[1] + float(bias[col + 1]);
            if (ACT == 1) {
                v0 = 0.5f * v0 * (1.0f + k_erf(v0 * M_SQRT1_2_F));
                v1 = 0.5f * v1 * (1.0f + k_erf(v1 * M_SQRT1_2_F));
            }
            *((device vec<T, 2>*)(out + size_t(row) * N + col)) = vec<T, 2>(T(v0), T(v1));
        }
    }
"""

_k = mx.fast.metal_kernel(name="gemm_bias_act", input_names=["A", "W", "bias", "m_rows"], output_names=["out"],
                          source=SRC, header=HEADER)


def gemm_bias_act(a, w, bias, act=1, bm=64, bn=64, bk=16, wm=2, wn=2):
    m, k = a.shape
    n = w.shape[1]
    return _k(inputs=[a, w, bias, mx.array([m], mx.int32)],
              template=[("T", a.dtype), ("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn), ("K", k), ("N", n),
                        ("ACT", act)],
              grid=((n // bn) * 32 * wm * wn, (m + bm - 1) // bm, 1), threadgroup=(32 * wm * wn, 1, 1),
              output_shapes=[(m, n)], output_dtypes=[a.dtype])[0]


def t(fn, *a, n=60, reps=10):
    f = lambda *a: [fn(*a) for _ in range(reps)]  # noqa: E731
    for _ in range(5):
        mx.eval(f(*a))
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        mx.eval(f(*a))
        ts.append(time.perf_counter() - s)
    ts.sort()
    return ts[len(ts) // 2] / reps * 1e3


if __name__ == "__main__":
    dt = getattr(mx, sys.argv[1]) if len(sys.argv) > 1 else mx.float32
    M, K, N = 1604, 384, 1536
    a = mx.random.normal((M, K)).astype(dt)
    w = (mx.random.normal((K, N)) * 0.05).astype(dt)
    b = mx.random.normal((N,)).astype(dt)
    mx.eval(a, w, b)
    gelu = lambda x: 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))  # noqa: E731
    ref_fn = mx.compile(lambda a: gelu(a @ w + b))
    ref = ref_fn(a)
    print(f"{dt}: MLX matmul only {t(lambda a: a @ w, a):.3f} ms, MLX matmul+bias+gelu {t(ref_fn, a):.3f} ms")
    for cfg in [dict(bm=64, bn=64, bk=16, wm=2, wn=2), dict(bm=32, bn=64, bk=16, wm=1, wn=2), dict(bm=64, bn=32, bk=16, wm=2, wn=1),
                dict(bm=64, bn=64, bk=32, wm=2, wn=2), dict(bm=128, bn=64, bk=16, wm=4, wn=2), dict(bm=64, bn=128, bk=16, wm=2, wn=4),
                dict(bm=32, bn=32, bk=16, wm=1, wn=1), dict(bm=64, bn=64, bk=8, wm=2, wn=2)]:
        out = gemm_bias_act(a, w, b, **cfg)
        err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        ms = t(lambda a: gemm_bias_act(a, w, b, **cfg), a)
        print(f"  custom {cfg}: {ms:.3f} ms  ({2 * M * K * N / ms / 1e9:.2f} TFLOPS)  max err {err:.2e}")
