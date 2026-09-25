"""Extracts RF-DETR weights and folded constants from the ONNX file into a flat {name: np.ndarray} dict.

Linear weights are stored as [in, out] (x @ W), convs as MLX NHWC [out, kh, kw, in].
"""
from __future__ import annotations

import numpy as np


def _node_key(node_name: str) -> str:
    """'/backbone/backbone.0/.../query/MatMul' -> 'backbone.0....query'."""
    parts = node_name.strip("/").split("/")
    # ONNX names repeat the module path per level (e.g. /backbone/backbone.0/...): keep the deepest dotted path.
    out = []
    for p in parts[:-1]:
        if out and p.startswith(out[-1] + "."):
            out[-1] = p
        else:
            out.append(p)
    return ".".join(out)


def extract(model_path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    g = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    w: dict[str, np.ndarray] = {}
    for name, arr in inits.items():
        if not name.startswith("onnx::"):
            w[name] = arr
    eps = {}
    for n in g.node:
        if n.op_type == "MatMul" and n.input[1] in inits:
            key = _node_key(n.name)
            if key.endswith("self_attn"):  # nn.MultiheadAttention in_proj split into MatMul / MatMul_1 / MatMul_2
                key += {"MatMul": ".q", "MatMul_1": ".k", "MatMul_2": ".v"}[n.name.split("/")[-1]]
            w[key + ".weight"] = inits[n.input[1]]
            for c in consumers.get(n.output[0], []):
                if c.op_type == "Add":
                    other = c.input[0] if c.input[1] == n.output[0] else c.input[1]
                    if other in inits or other in consts:
                        w[key + ".bias"] = inits[other] if other in inits else consts[other]
        elif n.op_type == "LayerNormalization":
            eps[n.input[1]] = next(a.f for a in n.attribute if a.name == "epsilon")
    for k, e in eps.items():
        w["_eps." + k] = np.asarray(e, dtype=np.float32)

    # Conv weights: ONNX [out, in, kh, kw] -> MLX [out, kh, kw, in]
    for k in list(w):
        if k.endswith("conv.weight") or k.endswith("projection.weight"):
            w[k] = np.ascontiguousarray(w[k].transpose(0, 2, 3, 1))

    # Learned decoder queries / reference-point embedding (anonymous initializers feeding Expand)
    for n in g.node:
        if n.op_type == "Expand" and n.input[0].startswith("onnx::Expand"):
            arr = inits[n.input[0]]
            w["query_feat" if arr.shape[-1] == 256 else "refpoint_embed"] = arr[0]
    # Sine-embedding frequency divisor: first Div constant of length 128 in /transformer/decoder
    for n in g.node:
        if n.op_type == "Div" and n.name.startswith("/transformer/decoder/Div"):
            src = n.input[1]
            if src in consts and consts[src].shape == (128,):
                w["dim_t"] = consts[src].astype(np.float32)
                break
    w.update(_proposals())
    return {k: np.ascontiguousarray(v) for k, v in w.items()}


def _proposals(h=40, wd=40):
    """LW-DETR gen_encoder_output_proposals (bbox_reparam, one level): fixed cxcywh anchors + validity mask."""
    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(wd, dtype=np.float32), indexing="ij")
    xy = (np.stack([gx, gy], -1).reshape(-1, 2) + .5) / np.asarray([wd, h], np.float32)
    wh = np.full_like(xy, .05)
    props = np.concatenate([xy, wh], -1)
    valid = ((props > .01) & (props < .99)).all(-1, keepdims=True)
    return {"proposals": props.astype(np.float32), "proposal_valid": valid.astype(np.float32)}

