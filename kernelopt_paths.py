"""Where the RF-DETR ONNX model and the golden reference .npz live on this machine.

Set KERNELOPT_MODEL / KERNELOPT_REFERENCE_NPZ in the environment, or put them in paths.local.json next to this file
(gitignored), e.g.
    {"KERNELOPT_MODEL": "/path/to/basketball-rfdetr.onnx",
     "KERNELOPT_REFERENCE_NPZ": "/path/to/rfdetr_reference.npz"}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

LOCAL_FILE = Path(__file__).with_name("paths.local.json")


def get(key: str, required: bool = True) -> Path | None:
    value = os.environ.get(key)
    if not value and LOCAL_FILE.exists():
        value = json.loads(LOCAL_FILE.read_text()).get(key)
    if value:
        return Path(value)
    if required:
        raise RuntimeError(f"{key} is not set: export it or add it to {LOCAL_FILE}")
    return None


def model_path(required: bool = True) -> Path | None:
    return get("KERNELOPT_MODEL", required)


def reference_npz(required: bool = True) -> Path | None:
    return get("KERNELOPT_REFERENCE_NPZ", required)
