"""Helpers shared by train.py, predict.py, compare.py, experiments.py and explore.py."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is preferred over maximum throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def sync(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def to_device(batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    return batch["image"].to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))


def ssim_2d(real: np.ndarray, recon: np.ndarray) -> float:
    """SSIM of two [H, W] images in [0, 1] (Gaussian-weighted, as in the original paper)."""
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            np.asarray(real, dtype=np.float32),
            np.asarray(recon, dtype=np.float32),
            data_range=1.0,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
    )


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False makes a NaN metric fail loudly instead of writing invalid JSON.
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required experiment artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Union of keys (in first-seen order) so rows with extra columns are not dropped.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_image_grid(
    images: torch.Tensor,
    path: Path,
    titles: Sequence[str] | None = None,
    ncols: int = 8,
    cmap: str = "gray",
) -> None:
    """Save [N, 1, H, W] (or [N, H, W]) images in [0, 1] as one grid figure."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend: runs on clusters without a display
    import matplotlib.pyplot as plt

    arr = images.detach().float().cpu().clamp(0, 1).numpy()
    if arr.ndim == 4:
        arr = arr[:, 0]
    count = len(arr)
    ncols = min(ncols, max(1, count))
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 2.2), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < count:
            ax.imshow(arr[i], cmap=cmap, vmin=0.0, vmax=1.0)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=7)
    fig.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
