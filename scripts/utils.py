"""
Shared utilities for the biohack filament pipeline.
Centralises device selection so all scripts benefit from
CUDA → MPS → CPU auto-detection.
"""
import torch


def best_device() -> torch.device:
    """Return the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
