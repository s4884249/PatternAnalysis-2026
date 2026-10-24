"""Train and evaluate the VAE or the VQ-VAE on 2D HipMRI slices.

The best epoch is chosen on validation SSIM and the test split is evaluated once,
with that checkpoint. The VQ-VAE then fits a PixelCNN-style prior on training-split
code maps so it can generate new images. All settings live in params.py; the run
directory is derived from them (see params.run_dir_name).

Progress images (at most 10 columns, evenly spaced epochs):
    reconstructions_by_epoch.png  fixed validation image: input + reconstruction per snapshot
    prior_samples_by_epoch.png    VQ-VAE only: one prior sample per snapshot
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import NiftiSliceDataset, load_manifest, split_from_manifest
from modules import CodePrior, VQVAE, build_model, count_parameters
from params import MANIFEST, MODELS, RUNS_DIR, TrainParams, add_overrides, apply_overrides, run_dir_name
from device_utils import get_device, peak_memory_mb, reset_peak_memory_stats
from utils import save_image_grid, save_json, set_seed, ssim_2d, to_device, write_csv

SSIM_TARGET = 0.6  # assignment target, drawn on the SSIM curve
MAX_SNAPSHOTS = 10  # upper bound on columns in the progress images


def snapshot_epochs(total: int) -> set:
    """Evenly spaced epochs for progress images: every ceil(total/10), plus the last.

    100 epochs -> 10, 20, ..., 100; 5 epochs -> 1..5; 15 epochs -> 2, 4, ..., 14, 15.
    The count never exceeds MAX_SNAPSHOTS.
    """
    step = math.ceil(total / MAX_SNAPSHOTS)
    return set(range(step, total + 1, step)) | {total}


def compute_loss(model, images: torch.Tensor, model_type: str) -> Tuple[torch.Tensor, Dict[str, float], dict]:
    """Return the total loss, a dict of float metrics and the raw model output.

    VAE:    MSE + kl_beta * KL
    VQ-VAE: MSE + codebook loss + commitment loss (the weight is applied in the quantiser)
    MSE on [0, 1] intensities is the reconstruction term for both, so scores are comparable.
    """
    output = model(images)
    mse = F.mse_loss(output["reconstruction"], images)
    if model_type == "vae":
        loss = mse + model.kl_beta * output["kl_loss"]
        extra = {"kl_loss": output["kl_loss"]}
    else:
        loss = mse + output["codebook_loss"] + output["commitment_loss"]
        extra = {k: output[k] for k in ("codebook_loss", "commitment_loss", "perplexity", "active_codes")}
    metrics = {"loss": loss, "reconstruction_mse": mse, **extra}
    return loss, {name: float(value.detach()) for name, value in metrics.items()}, output


def run_epoch(
    model,
    loader: DataLoader,
    model_type: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    limit_batches: int = 0,
    with_ssim: bool = False,
) -> Dict[str, float]:
    """One pass over `loader`: trains if an optimizer is given, otherwise evaluates.

    Metrics are averaged per image (not per batch) so a short last batch is not
    over-weighted. SSIM is only computed on request because it runs on the CPU.
    """
    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = defaultdict(float)
    n_images = 0
    with torch.inference_mode(not training):
        for index, batch in enumerate(loader):
            if limit_batches and index >= limit_batches:
                break
            images = to_device(batch, device)
            loss, metrics, output = compute_loss(model, images, model_type)
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at batch {index}: {loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # guards against rare gradient spikes
                optimizer.step()
            if with_ssim:
                real = images.clamp(0, 1)[:, 0].cpu().numpy()
                recon = output["reconstruction"].clamp(0, 1)[:, 0].cpu().numpy()
                metrics["ssim"] = float(np.mean([ssim_2d(a, b) for a, b in zip(real, recon)]))
            count = int(images.shape[0])
            n_images += count
            for name, value in metrics.items():
                sums[name] += value * count
    if n_images == 0:
        raise RuntimeError("Loader produced no images; check the manifest, split and limit_batches.")
    return {name: value / n_images for name, value in sums.items()} | {"n_images": n_images}


@torch.inference_mode()
def reconstruct_one(model, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reconstruct a single [1, H, W] image (eval mode, so the VAE uses its mean latent)."""
    model.eval()
    return model(image[None].to(device))["reconstruction"].clamp(0, 1)[0].cpu()


def plot_history(run_dir: Path, history: List[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, (loss_ax, ssim_ax) = plt.subplots(1, 2, figsize=(11, 4))
    loss_ax.plot(epochs, [row["train_loss"] for row in history], label="train total loss")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], label="validation total loss")
    loss_ax.set(xlabel="Epoch", ylabel="Loss")
    ssim_ax.plot(epochs, [row["val_ssim"] for row in history], label="validation SSIM")
    ssim_ax.axhline(SSIM_TARGET, linestyle="--", color="gray", label=f"assignment target {SSIM_TARGET}")
    ssim_ax.set(xlabel="Epoch", ylabel="SSIM")
    for ax in (loss_ax, ssim_ax):
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def make_loaders(splits: Dict[str, List[dict]], data_root: Path, p: TrainParams) -> Dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(p.seed)  # reproducible training shuffle order
    return {
        name: DataLoader(
            NiftiSliceDataset(rows, data_root, p.image_height, p.image_width),
            batch_size=p.batch_size,
            shuffle=(name == "train"),
            num_workers=p.workers,
            pin_memory=torch.cuda.is_available(),  # CUDA only; MPS does not use pinned-memory transfers
            persistent_workers=p.workers > 0,
            generator=generator if name == "train" else None,
        )
        for name, rows in splits.items()
    }


@torch.inference_mode()
def collect_code_maps(model: VQVAE, loader: DataLoader, device: torch.device, limit_batches: int = 0) -> torch.Tensor:
    """Encode training images into [N, H, W] code-index maps, the prior's training data."""
    model.eval()
    maps = []
    for index, batch in enumerate(loader):
        if limit_batches and index >= limit_batches:
            break
        maps.append(model.encode_indices(to_device(batch, device)).cpu())
    return torch.cat(maps, dim=0)


def train_code_prior(model: VQVAE, train_loader: DataLoader, p: TrainParams, device: torch.device, run_dir: Path) -> dict:
    """Fit the autoregressive prior on training-split code maps only (no test leakage)."""
    code_maps = collect_code_maps(model, train_loader, device, p.limit_batches)
    height, width = code_maps.shape[1:]
    loader = DataLoader(TensorDataset(code_maps), batch_size=p.batch_size, shuffle=True)
    prior = CodePrior(codebook_size=p.codebook_size, hidden_channels=p.prior_hidden).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=p.prior_lr)

    snapshots = snapshot_epochs(p.prior_epochs)
    frames, titles = [], []
    history = []
    start = time.perf_counter()
    for epoch in range(1, p.prior_epochs + 1):
        prior.train()
        total, count = 0.0, 0
        for (indices,) in loader:
            indices = indices.to(device=device, dtype=torch.long)
            # Next-code prediction: cross-entropy of every position given earlier positions.
            logits = prior(indices)
            loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, p.codebook_size), indices.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite code-prior loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach()) * indices.shape[0]
            count += indices.shape[0]
        history.append({"epoch": epoch, "train_cross_entropy": total / count})
        print(f"[prior] epoch {epoch:03d}/{p.prior_epochs:03d} | train CE={total / count:.5f}")

        if epoch in snapshots:
            # Same generator seed every time, so columns differ only because the prior improved.
            prior.eval()
            generator = torch.Generator(device=device).manual_seed(p.seed)
            with torch.inference_mode():
                codes = prior.sample(1, height, width, device, generator=generator)
                sample = model.decode_indices(codes).clamp(0, 1)[0].cpu()
            frames.append(sample)
            titles.append(f"Prior epoch {epoch}")
            save_image_grid(torch.stack(frames), run_dir / "prior_samples_by_epoch.png", titles, ncols=len(frames))
    elapsed = time.perf_counter() - start

    torch.save(
        {
            "state_dict": prior.state_dict(),
            "codebook_size": p.codebook_size,
            "hidden_channels": p.prior_hidden,
            "latent_height": int(height),
            "latent_width": int(width),
            "history": history,
        },
        run_dir / "prior.pt",
    )
    write_csv(run_dir / "prior_history.csv", history)
    return {
        "prior": "PixelCNN-like autoregressive prior trained on training-split code maps",
        "train_code_maps": int(code_maps.shape[0]),
        "latent_shape": [int(height), int(width)],
        "prior_parameters": count_parameters(prior),
        "prior_training_seconds": elapsed,
        "final_train_cross_entropy": history[-1]["train_cross_entropy"],
    }


def train_model(
    model_type: str, p: TrainParams, data_root: Path, manifest: Path, run_dir: Path, train_prior: bool = True
) -> dict:
    """Train one model end to end and write all artifacts into run_dir; returns the summary dict.

    train_prior=False skips the VQ-VAE prior (used by experiments.py, which only
    compares reconstruction quality and so does not need to generate images).
    """
    data_root, manifest, run_dir = data_root.expanduser().resolve(), manifest.expanduser().resolve(), run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(p.seed)
    device = get_device(p.device)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else (" (Apple Silicon GPU)" if device.type == "mps" else "")))

    # Use the dataset's own train/val/test folders; masks and unreadable files are excluded.
    splits = split_from_manifest(load_manifest(manifest, data_root))
    image_counts = {name: len(rows) for name, rows in splits.items()}
    save_json(
        run_dir / "split.json",
        {
            "seed": p.seed,
            "data_root": str(data_root),
            "manifest": str(manifest),
            "split_group_counts": {name: len({r["group_id"] for r in rows}) for name, rows in splits.items()},
            "split_image_counts": image_counts,
            "splits": splits,  # compare.py checks these lists match across models
        },
    )
    loaders = make_loaders(splits, data_root, p)
    # Fixed validation image for the progress strip (test data is never shown during training).
    val_set = loaders["val"].dataset
    fixed_image = val_set[len(val_set) // 2]["image"]

    config = asdict(p) | {"data_root": str(data_root), "manifest": str(manifest)}
    model = build_model(model_type, config).to(device)
    n_parameters = count_parameters(model)
    print(f"Model parameters: {n_parameters:,}")
    reset_peak_memory_stats(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    snapshots = snapshot_epochs(p.epochs)
    frames, titles = [fixed_image], ["Input"]
    best_ssim, best_epoch, history = -float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, p.epochs + 1):
        epoch_start = time.perf_counter()
        train_m = run_epoch(model, loaders["train"], model_type, device, optimizer, p.limit_batches)
        val_m = run_epoch(model, loaders["val"], model_type, device, None, p.limit_batches, with_ssim=True)
        seconds = time.perf_counter() - epoch_start
        history.append(
            {"epoch": epoch}
            | {f"train_{k}": v for k, v in train_m.items() if k != "n_images"}
            | {f"val_{k}": v for k, v in val_m.items() if k != "n_images"}
            | {"epoch_seconds": seconds}
        )
        write_csv(run_dir / "history.csv", history)
        print(
            f"epoch {epoch:03d}/{p.epochs:03d} | train loss={train_m['loss']:.5f} "
            f"val loss={val_m['loss']:.5f} val SSIM={val_m['ssim']:.4f} | {seconds:.1f}s"
        )

        if val_m["ssim"] > best_ssim:  # checkpoint selection uses validation SSIM only
            best_ssim, best_epoch = val_m["ssim"], epoch
            torch.save(
                {
                    "model_type": model_type,
                    "model_config": config,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_m,
                },
                run_dir / "best.pt",
            )
            save_json(run_dir / "best_validation_metrics.json", val_m)

        if epoch in snapshots:
            recon = reconstruct_one(model, fixed_image, device)
            ssim = ssim_2d(fixed_image[0].numpy(), recon[0].numpy())
            frames.append(recon)
            titles.append(f"Epoch {epoch}\nSSIM {ssim:.3f}")
            # Rewritten at every snapshot so an interrupted run still has a usable image.
            save_image_grid(torch.stack(frames), run_dir / "reconstructions_by_epoch.png", titles, ncols=len(frames))
    training_seconds = time.perf_counter() - start
    plot_history(run_dir, history)

    # Final, one-off test evaluation with the best validation checkpoint.
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_m = run_epoch(model, loaders["test"], model_type, device, None, p.limit_batches, with_ssim=True)
    save_json(run_dir / "test_metrics.json", test_m)

    if model_type == "vqvae" and train_prior:
        generation = train_code_prior(model, loaders["train"], p, device, run_dir)
    elif model_type == "vae":
        generation = {"prior": "standard-normal latent maps (the KL term trains the VAE towards N(0, I))"}
    else:
        generation = {"prior": "not trained"}

    summary = {
        "model": model_type,
        "params": asdict(p),
        "model_parameters": n_parameters,
        "device": str(device),
        "split_image_counts": image_counts,
        "best_epoch": best_epoch,
        "training_wall_seconds": training_seconds,
        "total_training_plus_prior_wall_seconds": training_seconds + generation.get("prior_training_seconds", 0.0),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "peak_accelerator_memory_mb": peak_memory_mb(device),
        "best_validation_metrics": checkpoint["validation_metrics"],
        "held_out_test_metrics": test_m,
        "generation": generation,
        "notes": [
            "Splits follow the dataset's own train/validate/test folders; case IDs were checked for overlap.",
            "SSIM measures paired input/reconstruction similarity, not the fidelity of generated images.",
        ],
    }
    save_json(run_dir / "run_summary.json", summary)
    print(f"Best validation SSIM: {best_ssim:.4f} (epoch {best_epoch}); held-out test SSIM: {test_m['ssim']:.4f}")
    print(f"Run artifacts: {run_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VAE or VQ-VAE on 2D HipMRI NIfTI slices")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing the keras_slices_* directories")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="CSV created by `dataset.py audit`")
    add_overrides(parser, TrainParams())  # any params.py default can be overridden, e.g. --epochs 5
    args = parser.parse_args()
    try:
        params = apply_overrides(TrainParams(), args)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = RUNS_DIR / run_dir_name(args.model, params)
    print(f"Run directory: {run_dir.resolve()}")
    train_model(args.model, params, args.data_root, args.manifest, run_dir)


if __name__ == "__main__":
    main()
