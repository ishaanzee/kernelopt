"""Fast RF-DETR-Medium basketball detector for Apple Silicon GPUs (MLX + custom Metal kernels)."""
from .detector import FastBasketballDetector, load_weights

__all__ = ["FastBasketballDetector", "load_weights"]
