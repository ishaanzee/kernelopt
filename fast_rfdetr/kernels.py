"""Custom Metal kernels (via mx.fast.metal_kernel, compiled at runtime; no Xcode needed)."""
from __future__ import annotations

import mlx.core as mx

# One simdgroup per row. Each lane owns C/128 half4 vectors; stats are reduced with simd_sum in fp32.
# h_out = h + (o + bias) * scale;  y = LayerNorm(h_out) * gamma + beta
# The residual stream h / h_out may be kept at higher precision (TH) than o / y / params (T).
_RESIDUAL_LN_SRC = """
    const uint lane = thread_index_in_simdgroup;
    const uint row = thread_position_in_grid.y;
    if (row >= uint(n_rows[0])) return;
    constexpr int V = C / 128;
    const device vec<TH, 4>* h4 = (const device vec<TH, 4>*)(h) + row * (C / 4);
    const device vec<T, 4>* o4 = (const device vec<T, 4>*)(o) + row * (C / 4);
    const device vec<T, 4>* b4 = (const device vec<T, 4>*)(bias);
    const device vec<T, 4>* s4 = (const device vec<T, 4>*)(scale);
    float4 v[V];
    float sum = 0.0f;
    for (int i = 0; i < V; ++i) {
        const uint c = lane + i * 32;
        v[i] = float4(h4[c]) + (float4(o4[c]) + float4(b4[c])) * float4(s4[c]);
        sum += v[i].x + v[i].y + v[i].z + v[i].w;
    }
    const float mean = simd_sum(sum) / C;
    float sq = 0.0f;
    for (int i = 0; i < V; ++i) {
        const float4 d = v[i] - mean;
        sq += dot(d, d);
    }
    const float rstd = metal::precise::rsqrt(simd_sum(sq) / C + eps[0]);
    device vec<TH, 4>* ho4 = (device vec<TH, 4>*)(h_out) + row * (C / 4);
    device vec<T, 4>* y4 = (device vec<T, 4>*)(y) + row * (C / 4);
    const device vec<T, 4>* g4 = (const device vec<T, 4>*)(gamma);
    const device vec<T, 4>* be4 = (const device vec<T, 4>*)(beta);
    for (int i = 0; i < V; ++i) {
        const uint c = lane + i * 32;
        ho4[c] = vec<TH, 4>(v[i]);
        y4[c] = vec<T, 4>((v[i] - mean) * rstd * float4(g4[c]) + float4(be4[c]));
    }
"""

_residual_ln = mx.fast.metal_kernel(
    name="residual_layernorm",
    input_names=["h", "o", "bias", "scale", "gamma", "beta", "eps", "n_rows"],
    output_names=["h_out", "y"],
    source=_RESIDUAL_LN_SRC,
)

_ROWS_PER_TG = 8


def residual_layernorm(h, o, bias, scale, gamma, beta, eps):
    """Returns (h + (o + bias) * scale, LayerNorm of that). h, o: [..., C] with C % 128 == 0.

    h_out has h's dtype; y has o's dtype (the matmul compute dtype)."""
    c = h.shape[-1]
    rows = h.size // c
    out = _residual_ln(
        inputs=[h, o, bias, scale, gamma, beta, mx.array([eps], mx.float32), mx.array([rows], mx.int32)],
        template=[("T", o.dtype), ("TH", h.dtype), ("C", c)],
        grid=(32, rows, 1),
        threadgroup=(32, _ROWS_PER_TG, 1),
        output_shapes=[h.shape, h.shape],
        output_dtypes=[h.dtype, o.dtype],
    )
    return out[0], out[1]


# ---- fused preprocessing: uint8 BGR (any size) -> resized, RGB, normalized float [640, 640, 3] ----
# Interpolation taps come from resize_tables (bit-exact model of the cv2 wheel's resize); normalization is a
# per-channel lookup table computed with the same float32 numpy ops as Ballform's preprocess, so the output is
# bitwise identical to preprocess(frame).
_PREPROCESS_SRC = """
    const uint e = thread_position_in_grid.x;     // x * 3 + c (BGR element within an output row)
    const uint y = thread_position_in_grid.y;
    if (e >= uint(DW * 3) || y >= uint(DH)) return;
    const int4 xt = ((const device int4*)(xtab))[e];
    const int4 yt = ((const device int4*)(ytab))[y];
    const device uint8_t* top = src + size_t(yt.x) * size_t(src_stride[0]);
    const device uint8_t* bot = src + size_t(yt.y) * size_t(src_stride[0]);
    const int A = top[xt.x], B = top[xt.y], C = bot[xt.x], D = bot[xt.y];
    int v;
    if (MODE == 0) {   // KleidiCV: vertical then horizontal lerp, 8-bit fractions, vraddhn rounding
        const int left = (A * 256 + (C - A) * yt.z + 128) >> 8;
        const int right = (B * 256 + (D - B) * yt.z + 128) >> 8;
        v = (left * 256 + (right - left) * xt.z + 128) >> 8;
    } else {           // OpenCV fixed point: 11-bit weights, NEON VResizeLinearVec_32s8u rounding
        const int h0 = A * xt.z + B * xt.w;
        const int h1 = C * xt.z + D * xt.w;
        v = ((((h0 >> 4) * yt.z) >> 16) + (((h1 >> 4) * yt.w) >> 16) + 2) >> 2;
    }
    v = clamp(v, 0, 255);
    const uint c_rgb = 2 - (e % 3);
    out[(y * DW + e / 3) * 3 + c_rgb] = lut[c_rgb * 256 + v];
"""

_preprocess = mx.fast.metal_kernel(
    name="bgr_resize_normalize",
    input_names=["src", "xtab", "ytab", "lut", "src_stride"],
    output_names=["out"],
    source=_PREPROCESS_SRC,
)


def preprocess_gpu(img_u8, xtab, ytab, lut, mode, dst=640):
    """img_u8: mx.array uint8 [H, W, 3] (row-contiguous). Returns float32 [dst, dst, 3] normalized RGB."""
    h, w, _ = img_u8.shape
    return _preprocess(
        inputs=[img_u8.reshape(-1), xtab, ytab, lut, mx.array([w * 3], mx.int32)],
        template=[("MODE", mode), ("DW", dst), ("DH", dst)],
        grid=(dst * 3, dst, 1),
        threadgroup=(64, 4, 1),
        output_shapes=[(dst, dst, 3)],
        output_dtypes=[mx.float32],
    )[0]


# ---- deformable cross-attention sampling (single level) ----
# out[b, q, h*HD + d] = sum_p aw[b,q,h,p] * bilinear(value[b, :, h, d] as HxW, loc[b,q,h,p])
# Bilinear with zero padding and align_corners=False, i.e. ONNX GridSample on grid = 2*loc - 1.
# One thread per (b, q, h, d); the HD threads of a head share the tap computation (cheap) and read
# contiguous channels of each tap row.
_DEFORM_SRC = """
    const uint d = thread_position_in_grid.x;          // channel within head
    const uint h = thread_position_in_grid.y;          // head
    const uint bq = thread_position_in_grid.z;         // b * NQ + q
    if (d >= uint(HD) || h >= uint(NH) || bq >= uint(n_bq[0])) return;
    const uint b = bq / NQ;
    const device T* vb = value + size_t(b) * (GH * GW * ROW) + v_off[0] + h * HD + d;
    float acc = 0.0f;
    for (int p = 0; p < NP; ++p) {
        const uint li = ((bq * NH + h) * NP + p);
        const float lx = loc[li * 2 + 0] * GW - 0.5f;
        const float ly = loc[li * 2 + 1] * GH - 0.5f;
        const float x0f = metal::floor(lx), y0f = metal::floor(ly);
        const float fx = lx - x0f, fy = ly - y0f;
        const int x0 = int(x0f), y0 = int(y0f);
        const float w = aw[li];
        float s = 0.0f;
        for (int dy = 0; dy < 2; ++dy) {
            const int yy = y0 + dy;
            if (yy < 0 || yy >= GH) continue;
            const float wy = dy ? fy : 1.0f - fy;
            for (int dx = 0; dx < 2; ++dx) {
                const int xx = x0 + dx;
                if (xx < 0 || xx >= GW) continue;
                const float wx = dx ? fx : 1.0f - fx;
                s += wx * wy * float(vb[size_t(yy * GW + xx) * ROW]);
            }
        }
        acc += w * s;
    }
    out[(bq * NH + h) * HD + d] = T(acc);
"""

_deform = mx.fast.metal_kernel(
    name="deform_attn_sample",
    input_names=["value", "loc", "aw", "n_bq", "v_off"],
    output_names=["out"],
    source=_DEFORM_SRC,
)


def deform_attn(value, loc, aw, grid_hw, nh, hd, offset=0):
    """value [B, H*W, ROW] with this layer's heads at columns [offset, offset + nh*hd) (lets all decoder layers
    share one value-projection GEMM); loc [B, NQ, NH, NP, 2] normalized x,y; aw [B, NQ, NH, NP].
    Returns [B, NQ, NH*HD] in value's dtype."""
    b, _, row = value.shape
    nq, npts = loc.shape[1], loc.shape[3]
    gh, gw = grid_hw
    return _deform(
        inputs=[value, loc.astype(mx.float32), aw.astype(mx.float32), mx.array([b * nq], mx.int32),
                mx.array([offset], mx.int32)],
        template=[("T", value.dtype), ("HD", hd), ("NH", nh), ("NQ", nq), ("NP", npts), ("GH", gh), ("GW", gw),
                  ("ROW", row)],
        grid=(hd, nh, b * nq),
        threadgroup=(hd, nh, 1) if hd * nh <= 1024 else (hd, 1, 1),
        output_shapes=[(b, nq, nh * hd)],
        output_dtypes=[value.dtype],
    )[0]


# ---- fp32 GEMM with fused bias + exact GELU epilogue (MLP fc1) ----
# simdgroup_matrix 8x8 fragments loaded straight from device memory; each simdgroup owns a (TM*8) x (TN*8) block.
# Rows need not be a multiple of 8: a fragment whose rows would pass M is anchored at M-8 instead, recomputing
# (bit-identically) rows another fragment also writes. GELU uses MLX's erf so results match mx.erf.
_GEMM_HEADER = """
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

_GEMM_GELU_SRC = """
    constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
    const int M = m_rows[0];
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int m0 = threadgroup_position_in_grid.y * BM, n0 = threadgroup_position_in_grid.x * BN;
    const int sm = m0 + (sg / WN) * (TM * 8), sn = n0 + (sg % WN) * (TN * 8);
    int rbase[TM];
    _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) rbase[i] = min(sm + i * 8, M - 8);
    simdgroup_matrix<float, 8, 8> acc[TM][TN];
    _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < K; k += 8) {
        simdgroup_matrix<float, 8, 8> a[TM], b[TN];
        _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) simdgroup_load(a[i], A + size_t(rbase[i]) * K + k, K);
        _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j) simdgroup_load(b[j], W + size_t(k) * N + sn + j * 8, N);
        _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j)
            simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    _Pragma("clang loop unroll(full)") for (int i = 0; i < TM; ++i) {
        const size_t row = size_t(rbase[i] + fm);
        _Pragma("clang loop unroll(full)") for (int j = 0; j < TN; ++j) {
            const int col = sn + j * 8 + fn;
            thread auto& el = acc[i][j].thread_elements();
            float v0 = el[0] + bias[col], v1 = el[1] + bias[col + 1];
            v0 = 0.5f * v0 * (1.0f + k_erf(v0 * M_SQRT1_2_F));
            v1 = 0.5f * v1 * (1.0f + k_erf(v1 * M_SQRT1_2_F));
            *((device float2*)(out + row * N + col)) = float2(v0, v1);
        }
    }
"""

_gemm_gelu = mx.fast.metal_kernel(name="gemm_bias_gelu_f32", input_names=["A", "W", "bias", "m_rows"],
                                  output_names=["out"], source=_GEMM_GELU_SRC, header=_GEMM_HEADER)

_GEMM_TILE = dict(BM=32, BN=64, WM=1, WN=2)


def linear_gelu_f32(x, w, bias):
    """GELU(x @ w + bias) for float32 x [..., K] (rows >= 8), w [K, N] with K % 8 == 0 and N % 64 == 0."""
    k, n = w.shape
    a = x.reshape(-1, k)
    m = a.shape[0]
    t = _GEMM_TILE
    out = _gemm_gelu(
        inputs=[a, w, bias, mx.array([m], mx.int32)],
        template=[("BM", t["BM"]), ("BN", t["BN"]), ("WM", t["WM"]), ("WN", t["WN"]), ("K", k), ("N", n)],
        grid=((n // t["BN"]) * 32 * t["WM"] * t["WN"], (m + t["BM"] - 1) // t["BM"], 1),
        threadgroup=(32 * t["WM"] * t["WN"], 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(*x.shape[:-1], n)
