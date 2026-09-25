"""Readable dump of ONNX nodes (with inferred shapes, inline small constants) filtered by name prefix."""
import sys, onnx, numpy as np
from onnx import numpy_helper, shape_inference
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kernelopt_paths  # noqa: E402
m = shape_inference.infer_shapes(onnx.load(os.environ.get("GRAPH") or str(kernelopt_paths.model_path())))
g = m.graph
shapes = {vi.name: [d.dim_value if d.HasField("dim_value") else d.dim_param for d in vi.type.tensor_type.shape.dim]
          for vi in list(g.value_info) + list(g.input) + list(g.output)}
inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
consts = {}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                consts[n.output[0]] = numpy_helper.to_array(a.t)
def desc(name):
    if name in inits:
        v = inits[name]; return f"W[{name}]{list(v.shape)}" if v.size > 8 else f"W{v.tolist()}"
    if name in consts:
        v = consts[name]; return f"C{v.tolist()}" if v.size <= 8 else f"C{list(v.shape)}"
    return f"{name.split('/')[-1] if '/' in name else name}{shapes.get(name, '?')}"
prefixes = sys.argv[1:]
for n in g.node:
    if n.op_type == "Constant": continue
    if prefixes and not any(n.name.startswith(p) for p in prefixes): continue
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in attrs.items()}
    print(f"{n.name:60s} {n.op_type:18s} {', '.join(desc(i) for i in n.input)} -> {', '.join(desc(o) for o in n.output)} {attrs if attrs else ''}")
