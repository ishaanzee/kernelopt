"""Variant: fragments loaded directly from device memory (no threadgroup staging), M multiple of 8 only."""
import sys; sys.path.insert(0, "rfdetr")
import math, mlx.core as mx
from gemm_probe import HEADER, t
U = '_Pragma("clang loop unroll(full)") '
SRC = f"""
    constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int m0 = threadgroup_position_in_grid.y * BM, n0 = threadgroup_position_in_grid.x * BN;
    const int sm = m0 + (sg / WN) * (TM * 8), sn = n0 + (sg % WN) * (TN * 8);
    simdgroup_matrix<float, 8, 8> acc[TM][TN];
    {U}for (int i = 0; i < TM; ++i) {U}for (int j = 0; j < TN; ++j) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < K; k += 8) {{
        simdgroup_matrix<float, 8, 8> a[TM], b[TN];
        {U}for (int i = 0; i < TM; ++i) simdgroup_load(a[i], A + size_t(sm + i * 8) * K + k, K);
        {U}for (int j = 0; j < TN; ++j) simdgroup_load(b[j], W + size_t(k) * N + sn + j * 8, N);
        {U}for (int i = 0; i < TM; ++i) {U}for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }}
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    {U}for (int i = 0; i < TM; ++i) {{
        const int row = sm + i * 8 + fm;
        {U}for (int j = 0; j < TN; ++j) {{
            const int col = sn + j * 8 + fn;
            thread auto& el = acc[i][j].thread_elements();
            float v0 = el[0] + float(bias[col]), v1 = el[1] + float(bias[col + 1]);
            v0 = 0.5f * v0 * (1.0f + k_erf(v0 * M_SQRT1_2_F));
            v1 = 0.5f * v1 * (1.0f + k_erf(v1 * M_SQRT1_2_F));
            *((device vec<float, 2>*)(out + size_t(row) * N + col)) = vec<float, 2>(v0, v1);
        }}
    }}
"""
k = mx.fast.metal_kernel(name="gemm_direct", input_names=["A", "W", "bias"], output_names=["out"], source=SRC, header=HEADER)
M, K, N = 1600, 384, 1536
a = mx.random.normal((M, K)); w = mx.random.normal((K, N)) * 0.05; b = mx.random.normal((N,)); mx.eval(a, w, b)
gelu = lambda x: 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))
ref = mx.compile(lambda a: gelu(a @ w + b))(a)
print(f"MLX matmul only {t(lambda a: a @ w, a):.3f} ms")
for bm, bn, wm, wn in [(64, 64, 2, 2), (32, 64, 1, 2), (64, 32, 2, 1), (32, 32, 1, 1), (64, 128, 2, 2), (128, 64, 2, 2), (32, 128, 1, 2), (16, 64, 1, 2)]:
    f = lambda a: k(inputs=[a, w, b], template=[("BM", bm), ("BN", bn), ("WM", wm), ("WN", wn), ("K", K), ("N", N)],
                    grid=((N // bn) * 32 * wm * wn, M // bm, 1), threadgroup=(32 * wm * wn, 1, 1),
                    output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]
    ms = t(f, a)
    print(f"direct bm={bm} bn={bn} wm={wm} wn={wn}: {ms:.3f} ms ({2*M*K*N/ms/1e9:.2f} TFLOPS) err {mx.abs(f(a) - ref).max().item():.1e}")
