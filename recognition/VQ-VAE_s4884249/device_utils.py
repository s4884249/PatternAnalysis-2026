"""Device helpers for CUDA, Apple Silicon MPS, and CPU."""
from __future__ import annotations

import torch

def get_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve a requested device; CUDA requests fall back to MPS on Apple Silicon.

    `auto` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit `mps`
    request fails clearly if the installed PyTorch build cannot use MPS.
    """
    name = str(requested).lower()
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(name)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("CUDA is unavailable; using Apple Silicon MPS instead.")
            return torch.device("mps")
        raise RuntimeError("CUDA was requested, but neither CUDA nor Apple Silicon MPS is available.")
    if name in {"auto", "default"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "MPS was requested but is unavailable. Install a recent PyTorch build "
                "and verify torch.backends.mps.is_available()."
            )
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {requested!r}; use auto, mps, cuda, or cpu.")

def sync(device: torch.device) -> None:
    """Synchronize asynchronous accelerator work before timing it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # MPS does not expose a CUDA-style resettable peak allocator statistic.

def peak_memory_mb(device: torch.device) -> float | None:
    """Return CUDA peak or current MPS allocated memory where supported."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1024**2
    return None
