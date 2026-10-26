"""Helpers shared by train.py, predict.py, compare.py, experiments.py and explore.py."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# One implementation of device selection / syncing (CUDA, Apple MPS, CPU). Re-exported so every
# script that imports them from utils gets MPS support too.
from device_utils import get_device, sync  # noqa: F401


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is preferred over maximum throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def ssim_torch(real: torch.Tensor, recon: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Differentiable mean SSIM of [N, 1, H, W] images in [0, 1] (Gaussian window, valid padding).

    Same constants and window as `ssim_2d`, so it tracks the evaluation metric closely.
    """
    coords = torch.arange(window, dtype=real.dtype, device=real.device) - (window - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel)

    mu_x, mu_y = blur(real), blur(recon)
    var_x = blur(real * real) - mu_x * mu_x
    var_y = blur(recon * recon) - mu_y * mu_y
    cov = blur(real * recon) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean()


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
