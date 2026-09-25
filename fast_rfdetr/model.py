"""RF-DETR-Medium (basketball fine-tune) forward pass in MLX, mirroring the ONNX graph op for op."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .kernels import deform_attn, linear_gelu_f32, residual_layernorm

BB = "backbone.0.encoder.encoder."
LAYER = BB + "encoder.layer.{}."
PROJ = "backbone.0.projector.stages.0."
DEC = "transformer.decoder."
FULL_ATTN = (3, 6, 9)
TAPS = (2, 5, 8, 11)
GRID = 40          # 640 / 16 patches per side
WIN = 20           # 2x2 windows of 20x20 patches
C = 384
HEADS = 6
D_MODEL = 256
CA_HEADS, CA_POINTS = 16, 2
SA_HEADS = 8


class RFDETR:
    def __init__(self, weights: dict[str, np.ndarray], dtype=mx.float32, head_dtype=None, proj_dtype=None,
                 fused=True, residual_dtype=None):
        """dtype: backbone compute dtype; proj_dtype / head_dtype: projector and (query selection + decoder).
        fused: use the custom residual+LayerNorm kernel and bias-in-GEMM (addmm) path in the backbone."""
        self.fused = fused
        self.residual_dtype = residual_dtype or dtype
        self.dtype = dtype
        self.proj_dtype = proj_dtype or dtype
        self.head_dtype = head_dtype or dtype
        p = {}
        for k, v in weights.items():
            if k.startswith("_eps."):
                continue
            if k.endswith("self_attn.out_proj.weight"):  # Gemm transB=1: stored [out, in]
                v = v.T
            if k in ("proposals", "proposal_valid", "refpoint_embed", "dim_t"):
                dt = mx.float32
            elif k.startswith(PROJ):
                dt = self.proj_dtype
            elif k.startswith("backbone."):
                dt = dtype
            else:
                dt = self.head_dtype
            p[k] = mx.array(np.ascontiguousarray(v), dtype=dt)
        self.eps = {k[5:]: float(np.asarray(v).reshape(())) for k, v in weights.items() if k.startswith("_eps.")}
        self.p = p
        self._prepare()
        mx.eval(self.p)

    def _prepare(self):
        p = self.p
        pos = p[BB + "embeddings.position_embeddings"][0]
        p["_cls"] = (p[BB + "embeddings.cls_token"][0, 0] + pos[0]).reshape(1, 1, C)
        p["_pos"] = pos[1:].reshape(1, GRID, GRID, C)
        for i in range(12):
            pre = LAYER.format(i) + "attention.attention."
            p[f"_qkv{i}.w"] = mx.concatenate([p[pre + n + ".weight"] for n in ("query", "key", "value")], axis=1)
            p[f"_qkv{i}.b"] = mx.concatenate([p[pre + n + ".bias"] for n in ("query", "key", "value")])
            # head-major copies: y[:, None] @ W -> [n, 3*HEADS, L, 64], so SDPA gets contiguous per-head slices
            p[f"_qkvh{i}.w"] = mx.contiguous(p[f"_qkv{i}.w"].reshape(C, 3 * HEADS, C // HEADS).transpose(1, 0, 2))
            p[f"_qkvh{i}.b"] = p[f"_qkv{i}.b"].reshape(3 * HEADS, 1, C // HEADS)
        vp = [DEC + f"layers.{i}.cross_attn.value_proj" for i in range(4)]
        p["_vproj.w"] = mx.concatenate([p[k + ".weight"] for k in vp], axis=1)                     # [256, 1024]
        p["_vproj.b"] = mx.concatenate([p[k + ".bias"] for k in vp])
        rp = p["refpoint_embed"]
        p["_rp_xy"], p["_rp_ewh"] = rp[:, :2], mx.exp(rp[:, 2:])

    # ---- helpers ----
    def lin(self, x, key):
        y = x @ self.p[key + ".weight"]
        b = self.p.get(key + ".bias")
        return y if b is None else y + b

    def ln(self, x, key):
        return mx.fast.layer_norm(x, self.p[key + ".weight"], self.p[key + ".bias"], self.eps[key + ".weight"])

    # ---- backbone ----
    def backbone(self, x):
        p = self.p
        b = x.shape[0]
        t = mx.conv2d(x, p[BB + "embeddings.patch_embeddings.projection.weight"], stride=16)
        t = t + p[BB + "embeddings.patch_embeddings.projection.bias"] + p["_pos"]            # [B,40,40,C]
        t = t.reshape(b, 2, WIN, 2, WIN, C).transpose(0, 1, 3, 2, 4, 5).reshape(b * 4, WIN * WIN, C)
        h = mx.concatenate([mx.broadcast_to(p["_cls"], (b * 4, 1, C)), t], axis=1)          # [B*4,401,C]
        feats = []
        if self.fused:
            h = h.astype(self.residual_dtype)
            y = self.ln(h, LAYER.format(0) + "norm1").astype(self.dtype)
            for i in range(12):
                h, y = self.vit_layer_fused(i, h, y, b)
                if i in TAPS:
                    feats.append(y if i == 11 else self.ln(h, BB + "layernorm").astype(self.dtype))
        else:
            for i in range(12):
                h = self.vit_layer(i, h, b)
                if i in TAPS:
                    feats.append(self.ln(h, BB + "layernorm"))
        return [f[:, 1:].reshape(b, 2, 2, WIN, WIN, C).transpose(0, 1, 3, 2, 4, 5).reshape(b, GRID, GRID, C)
                for f in feats]

    def _attention(self, i, qkv, n, L, b):
        qkv = qkv.reshape(b, 4 * L, 3, HEADS, C // HEADS) if i in FULL_ATTN else qkv.reshape(n, L, 3, HEADS, C // HEADS)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        o = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=1 / math.sqrt(C // HEADS))
        return o.transpose(0, 2, 1, 3).reshape(n, L, C)

    def vit_layer_fused(self, i, h, y, b):
        """h: residual stream, y: norm1(h). Returns (h_next, y_next) where y_next is the next LayerNorm of h_next."""
        p, pre = self.p, LAYER.format(i)
        n, L, _ = h.shape
        yy = y.reshape(b, 4 * L, C) if i in FULL_ATTN else y                       # global layers: merge windows
        qkv = mx.addmm(p[f"_qkvh{i}.b"], yy[:, None], p[f"_qkvh{i}.w"])           # [n', 18, L', 64]
        o = mx.fast.scaled_dot_product_attention(qkv[:, :HEADS], qkv[:, HEADS:2 * HEADS], qkv[:, 2 * HEADS:],
                                                 scale=1 / math.sqrt(C // HEADS))
        o = o.transpose(0, 2, 1, 3).reshape(n, L, C) @ p[pre + "attention.output.dense.weight"]
        h, y = residual_layernorm(h, o, p[pre + "attention.output.dense.bias"], p[pre + "layer_scale1.lambda1"],
                                  p[pre + "norm2.weight"], p[pre + "norm2.bias"], self.eps[pre + "norm2.weight"])
        if y.dtype == mx.float32:
            m = linear_gelu_f32(y, p[pre + "mlp.fc1.weight"], p[pre + "mlp.fc1.bias"])
        else:
            m = self.lin(y, pre + "mlp.fc1")
            m = 0.5 * m * (1 + mx.erf(m * (1 / math.sqrt(2))))
        o = m @ p[pre + "mlp.fc2.weight"]
        nxt = LAYER.format(i + 1) + "norm1" if i < 11 else BB + "layernorm"
        return residual_layernorm(h, o, p[pre + "mlp.fc2.bias"], p[pre + "layer_scale2.lambda1"],
                                  p[nxt + ".weight"], p[nxt + ".bias"], self.eps[nxt + ".weight"])

    def vit_layer(self, i, h, b):
        pre = LAYER.format(i)
        n, L, _ = h.shape
        y = self.ln(h, pre + "norm1")
        o = self._attention(i, y @ self.p[f"_qkv{i}.w"] + self.p[f"_qkv{i}.b"], n, L, b)
        o = self.lin(o, pre + "attention.output.dense")
        h = h + o * self.p[pre + "layer_scale1.lambda1"]
        y = self.ln(h, pre + "norm2")
        y = self.lin(y, pre + "mlp.fc1")
        y = 0.5 * y * (1 + mx.erf(y / math.sqrt(2)))
        y = self.lin(y, pre + "mlp.fc2")
        return h + y * self.p[pre + "layer_scale2.lambda1"]

    # ---- projector (C2f with channel LayerNorm + SiLU) ----
    def convx(self, x, key):
        w = self.p[key + ".conv.weight"]
        if w.shape[1] > 1 and x.shape[0] > 1:
            # MLX picks a less accurate 3x3 algorithm (Winograd-like) for larger batches; that noise reorders the
            # TopK query slots. Convolve image by image so batched results match single-image results exactly.
            y = mx.concatenate([mx.conv2d(x[j:j + 1], w, padding=w.shape[1] // 2) for j in range(x.shape[0])])
        else:
            y = mx.conv2d(x, w, padding=w.shape[1] // 2)
        y = self.ln(y, key + ".bn")
        return y * mx.sigmoid(y)

    def projector(self, feats):
        pre = PROJ + "0."
        y = self.convx(mx.concatenate(feats, axis=-1), pre + "cv1")
        parts = [y[..., :128], y[..., 128:]]
        for j in range(3):
            parts.append(self.convx(self.convx(parts[-1], pre + f"m.{j}.cv1"), pre + f"m.{j}.cv2"))
        y = self.convx(mx.concatenate(parts, axis=-1), pre + "cv2")
        return self.ln(y, PROJ + "1")

    # ---- two-stage query selection ----
    def mlp3(self, x, key):
        x = mx.maximum(self.lin(x, key + ".layers.0"), 0)
        x = mx.maximum(self.lin(x, key + ".layers.1"), 0)
        return self.lin(x, key + ".layers.2")

    def select_queries(self, memory):
        p = self.p
        om = memory * p["proposal_valid"].astype(memory.dtype)
        om = self.ln(self.lin(om, "transformer.enc_output.0"), "transformer.enc_output_norm.0")
        scores = self.lin(om, "transformer.enc_out_class_embed.0").astype(mx.float32).max(axis=-1)   # [B,1600]
        delta = self.mlp3(om, "transformer.enc_out_bbox_embed.0").astype(mx.float32)
        prop = p["proposals"]
        coord = mx.concatenate([delta[..., :2] * prop[:, 2:] + prop[:, :2], mx.exp(delta[..., 2:]) * prop[:, 2:]], -1)
        idx = mx.argsort(-scores, axis=-1)[:, :300]
        ts = mx.take_along_axis(coord, idx[..., None], axis=1)                                          # [B,300,4]
        return mx.concatenate([p["_rp_xy"] * ts[..., 2:] + ts[..., :2], p["_rp_ewh"] * ts[..., 2:]], -1)

    # ---- decoder ----
    def sine_embed(self, ref):
        dim_t = self.p["dim_t"]
        outs = []
        for c in (1, 0, 2, 3):  # y, x, w, h
            v = (ref[..., c] * (2 * math.pi))[..., None] / dim_t                                        # [B,300,128]
            outs.append(mx.stack([mx.sin(v[..., 0::2]), mx.cos(v[..., 1::2])], axis=-1).reshape(*v.shape[:2], 128))
        return mx.concatenate(outs, axis=-1)

    def grid_sample(self, value, loc):
        """value [B,1600,16,16] (tokens, heads, dim); loc [B,300,16,P,2] in [0,1] -> [B,300,16,P,16].

        Bilinear, zero padding, align_corners=False (ONNX GridSample semantics)."""
        b = value.shape[0]
        v = value.transpose(0, 2, 1, 3).reshape(b * CA_HEADS * GRID * GRID, D_MODEL // CA_HEADS)
        loc = loc.transpose(0, 2, 1, 3, 4)                                                               # [B,16,300,P,2]
        x = loc[..., 0] * GRID - 0.5
        y = loc[..., 1] * GRID - 0.5
        x0, y0 = mx.floor(x), mx.floor(y)
        fx, fy = x - x0, y - y0
        base = (mx.arange(b * CA_HEADS) * (GRID * GRID)).reshape(b, CA_HEADS, 1, 1)
        out = 0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                xi, yi = x0 + dx, y0 + dy
                ok = (xi >= 0) & (xi < GRID) & (yi >= 0) & (yi < GRID)
                idx = base + (mx.clip(yi, 0, GRID - 1) * GRID + mx.clip(xi, 0, GRID - 1)).astype(mx.int32)
                g = mx.take(v, idx.reshape(-1), axis=0).reshape(*idx.shape, -1)                          # [B,16,300,P,16]
                out = out + g * mx.where(ok, wx * wy, 0).astype(g.dtype)[..., None]
        return out.transpose(0, 2, 1, 3, 4)

    def cross_attn(self, q, ref, value, key, layer=0):
        """value: [B,1600,16,16] per layer (reference path) or [B,1600,4*256] for all layers (fused path)."""
        b, nq, _ = q.shape
        off = self.lin(q, key + ".sampling_offsets").astype(mx.float32).reshape(b, nq, CA_HEADS, CA_POINTS, 2)
        aw = mx.softmax(self.lin(q, key + ".attention_weights").astype(mx.float32).reshape(b, nq, CA_HEADS, CA_POINTS), axis=-1)
        r = ref[:, :, None, None, :]
        loc = r[..., :2] + off / CA_POINTS * r[..., 2:] * 0.5
        if self.fused:
            o = deform_attn(value, loc, aw, (GRID, GRID), CA_HEADS, D_MODEL // CA_HEADS, offset=layer * D_MODEL)
        else:
            s = self.grid_sample(value, loc)                                                             # [B,300,16,P,16]
            o = (s * aw[..., None].astype(s.dtype)).sum(axis=3).reshape(b, nq, D_MODEL)
        return self.lin(o, key + ".output_proj")

    def self_attn(self, qk, v, key):
        b, n, _ = qk.shape
        hd = D_MODEL // SA_HEADS
        q = self.lin(qk, key + ".q").reshape(b, n, SA_HEADS, hd).transpose(0, 2, 1, 3)
        k = self.lin(qk, key + ".k").reshape(b, n, SA_HEADS, hd).transpose(0, 2, 1, 3)
        v = self.lin(v, key + ".v").reshape(b, n, SA_HEADS, hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(hd))
        return self.lin(o.transpose(0, 2, 1, 3).reshape(b, n, D_MODEL), key + ".out_proj")

    def decoder(self, memory, ref):
        p = self.p
        b = memory.shape[0]
        emb = self.sine_embed(ref).astype(self.head_dtype)
        qpos = self.lin(mx.maximum(self.lin(emb, DEC + "ref_point_head.layers.0"), 0), DEC + "ref_point_head.layers.1")
        tgt = mx.broadcast_to(p["query_feat"], (b, 300, D_MODEL))
        if self.fused:
            values = mx.addmm(p["_vproj.b"], memory, p["_vproj.w"])                                    # [B,1600,1024]
        for i in range(4):
            pre = DEC + f"layers.{i}."
            tgt = self.ln(tgt + self.self_attn(tgt + qpos, tgt, pre + "self_attn"), pre + "norm1")
            if self.fused:
                value = values
            else:
                value = self.lin(memory, pre + "cross_attn.value_proj").reshape(b, GRID * GRID, CA_HEADS, D_MODEL // CA_HEADS)
            tgt = self.ln(tgt + self.cross_attn(tgt + qpos, ref, value, pre + "cross_attn", i), pre + "norm2")
            ffn = self.lin(mx.maximum(self.lin(tgt, pre + "linear1"), 0), pre + "linear2")
            tgt = self.ln(tgt + ffn, pre + "norm3")
        return self.ln(tgt, DEC + "norm")

    def __call__(self, x):
        """x: [B,640,640,3] normalized RGB (NHWC). Returns (boxes [B,300,4] cxcywh, logits [B,300,11]) float32."""
        x = x.astype(self.dtype)
        b = x.shape[0]
        feats = [f.astype(self.proj_dtype) for f in self.backbone(x)]
        memory = self.projector(feats).reshape(b, GRID * GRID, D_MODEL).astype(self.head_dtype)
        ref = self.select_queries(memory)
        hs = self.decoder(memory, ref)
        logits = self.lin(hs, "class_embed").astype(mx.float32)
        delta = self.mlp3(hs, "bbox_embed").astype(mx.float32)
        boxes = mx.concatenate([delta[..., :2] * ref[..., 2:] + ref[..., :2], mx.exp(delta[..., 2:]) * ref[..., 2:]], -1)
        return boxes, logits
