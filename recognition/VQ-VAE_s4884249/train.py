"""Train/evaluate either the AE reconstruction baseline or the VQ-VAE.

Examples are in README.md. The test split is evaluated once, after checkpoint
selection using validation SSIM. VQ-VAE synthetic sampling additionally trains
a small autoregressive prior on training-split code maps only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    NiftiSliceDataset,
    all_groups_verified,
    load_manifest,
    split_by_group,
    split_from_manifest,
)
from modules import CodePrior, VQVAE, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is prioritized for reproducibility over maximum throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_batch(batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    return batch["image"].to(
        device=device, dtype=torch.float32, non_blocking=(device.type == "cuda")
    )


def compute_loss(
    model, images: torch.Tensor, model_type: str
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    output = model(images)
    reconstruction_loss = F.mse_loss(output["reconstruction"], images)
    if model_type == "ae":
        total_loss = reconstruction_loss
        codebook_loss = torch.zeros((), device=images.device)
        commitment_loss = torch.zeros((), device=images.device)
    else:
        codebook_loss = output["codebook_loss"]
        commitment_loss = output["commitment_loss"]
        total_loss = reconstruction_loss + codebook_loss + commitment_loss
    stats = {
        "loss": float(total_loss.detach().item()),
        "reconstruction_mse": float(reconstruction_loss.detach().item()),
        "codebook_loss": float(codebook_loss.detach().item()),
        "commitment_loss": float(commitment_loss.detach().item()),
        "perplexity": float(output["perplexity"].detach().item())
        if model_type == "vqvae"
        else float("nan"),
        "active_codes": float(output["active_codes"].detach().item())
        if model_type == "vqvae"
        else float("nan"),
    }
    return total_loss, stats, output


def per_image_ssim(reconstructions: torch.Tensor, images: torch.Tensor) -> List[float]:
    from skimage.metrics import structural_similarity

    recon = reconstructions.detach().float().clamp(0, 1).cpu().numpy()[:, 0]
    real = images.detach().float().clamp(0, 1).cpu().numpy()[:, 0]
    return [
        float(
            structural_similarity(
                real[i],
                recon[i],
                data_range=1.0,
                gaussian_weights=True,
                use_sample_covariance=False,
            )
        )
        for i in range(len(real))
    ]


def evaluate_model(
    model,
    loader: DataLoader,
    model_type: str,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {
        "loss": 0.0,
        "reconstruction_mse": 0.0,
        "codebook_loss": 0.0,
        "commitment_loss": 0.0,
        "ssim": 0.0,
        "perplexity": 0.0,
        "active_codes": 0.0,
    }
    n_images = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images = load_batch(batch, device)
            _, stats, output = compute_loss(model, images, model_type)
            ssims = per_image_ssim(output["reconstruction"], images)
            batch_size = int(images.shape[0])
            n_images += batch_size
            for key in (
                "loss",
                "reconstruction_mse",
                "codebook_loss",
                "commitment_loss",
            ):
                sums[key] += stats[key] * batch_size
            sums["ssim"] += float(np.sum(ssims))
            if model_type == "vqvae":
                sums["perplexity"] += stats["perplexity"] * batch_size
                sums["active_codes"] += stats["active_codes"] * batch_size
    if n_images == 0:
        raise RuntimeError(
            "Evaluation loader produced no images; check split and limit settings."
        )
    result = {key: value / n_images for key, value in sums.items()}
    result["n_images"] = int(n_images)
    return result


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_type: str,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    keys = [
        "loss",
        "reconstruction_mse",
        "codebook_loss",
        "commitment_loss",
        "perplexity",
        "active_codes",
    ]
    sums = {key: 0.0 for key in keys}
    n_images = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = load_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, stats, _ = compute_loss(model, images, model_type)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}: {loss.item()}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        n = int(images.shape[0])
        n_images += n
        for key in keys:
            value = stats[key]
            if math.isfinite(value):
                sums[key] += value * n
    if n_images == 0:
        raise RuntimeError(
            "Training loader produced no images; check the manifest and split."
        )
    return {key: sums[key] / n_images for key in keys} | {"n_images": n_images}


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_history_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_history(run_dir: Path, history: Sequence[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        epochs, [float(row["train_loss"]) for row in history], label="train total loss"
    )
    axes[0].plot(
        epochs,
        [float(row["val_loss"]) for row in history],
        label="validation total loss",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        epochs, [float(row["val_ssim"]) for row in history], label="validation SSIM"
    )
    axes[1].axhline(0.6, linestyle="--", label="assignment target 0.6")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def make_loaders(
    splits: Dict[str, List[Dict[str, str]]],
    data_root: Path,
    image_height: int,
    image_width: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> Dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for name, rows in splits.items():
        dataset = NiftiSliceDataset(
            rows, data_root, image_height=image_height, image_width=image_width
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(workers > 0),
            generator=generator if name == "train" else None,
            drop_last=False,
        )
    return loaders


def save_checkpoint(
    path: Path,
    model,
    args,
    epoch: int,
    val_metrics: Dict[str, float],
    parameter_count: int,
) -> None:
    config = {
        "base_channels": args.base_channels,
        "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
        "commitment_beta": args.commitment_beta,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
    }
    torch.save(
        {
            "model_type": args.model,
            "model_config": config,
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "validation_metrics": val_metrics,
            "parameter_count": int(parameter_count),
        },
        path,
    )


@torch.inference_mode()
def fit_ae_latent_prior(
    model, loader, device: torch.device, output_path: Path, max_batches: int = 0
) -> None:
    """Fit an intentionally simple channel-wise diagonal Gaussian to AE latents.

    A deterministic AE does not provide a probabilistic sampling prior. This
    weak sampler is used only as an explicitly limited generation comparator;
    it is not part of the AE reconstruction training objective.
    """
    model.eval()
    total = None
    total_sq = None
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = load_batch(batch, device)
        z = model.encode(images).double()
        dims = (0, 2, 3)
        z_sum = z.sum(dim=dims)
        z_sq = z.square().sum(dim=dims)
        if total is None:
            total, total_sq = z_sum, z_sq
        else:
            total += z_sum
            total_sq += z_sq
        count += z.shape[0] * z.shape[2] * z.shape[3]
    if count == 0:
        raise RuntimeError(
            "Could not estimate AE latent statistics from an empty loader"
        )
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(1e-6)
    std = variance.sqrt().clamp_min(1e-3)
    torch.save(
        {
            "mean": mean.float().view(1, -1, 1, 1),
            "std": std.float().view(1, -1, 1, 1),
            "count": count,
        },
        output_path,
    )


@torch.inference_mode()
def collect_code_maps(
    model: VQVAE, loader: DataLoader, device: torch.device, max_batches: int = 0
) -> torch.Tensor:
    model.eval()
    maps = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = load_batch(batch, device)
        maps.append(model.encode_indices(images).cpu())
    if not maps:
        raise RuntimeError("No code maps produced for PixelCNN prior training")
    return torch.cat(maps, dim=0)


def train_code_prior(
    model: VQVAE,
    train_loader: DataLoader,
    args,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    code_maps = collect_code_maps(model, train_loader, device, args.limit_train_batches)
    dataset = TensorDataset(code_maps)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    prior = CodePrior(
        codebook_size=args.codebook_size, hidden_channels=args.prior_hidden
    ).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.prior_lr)
    prior_history = []
    start = time.perf_counter()
    for epoch in range(1, args.prior_epochs + 1):
        prior.train()
        total_loss = 0.0
        count = 0
        for (indices,) in loader:
            indices = indices.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = prior(indices)
            loss = F.cross_entropy(
                logits.permute(0, 2, 3, 1).reshape(-1, args.codebook_size),
                indices.reshape(-1),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite code-prior loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=5.0)
            optimizer.step()
            n = int(indices.shape[0])
            total_loss += float(loss.detach().item()) * n
            count += n
        epoch_loss = total_loss / max(1, count)
        prior_history.append({"epoch": epoch, "train_cross_entropy": epoch_loss})
        print(
            f"[prior] epoch {epoch:03d}/{args.prior_epochs:03d} | train CE={epoch_loss:.5f}"
        )
    elapsed = time.perf_counter() - start
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "codebook_size": args.codebook_size,
            "hidden_channels": args.prior_hidden,
            "latent_height": int(code_maps.shape[1]),
            "latent_width": int(code_maps.shape[2]),
            "train_examples": int(code_maps.shape[0]),
            "history": prior_history,
        },
        run_dir / "prior.pt",
    )
    write_history_csv(run_dir / "prior_history.csv", prior_history)
    return {
        "train_code_maps": int(code_maps.shape[0]),
        "latent_shape": list(code_maps.shape[1:]),
        "prior_parameters": count_parameters(prior),
        "prior_training_seconds": elapsed,
        "final_train_cross_entropy": prior_history[-1]["train_cross_entropy"]
        if prior_history
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a basic AE baseline or a VQ-VAE on 2D HipMRI NIfTI slices"
    )
    parser.add_argument("--model", choices=("ae", "vqvae"), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"),
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="CSV created by dataset.py audit"
    )
    parser.add_argument(
        "--split-mode",
        choices=("predefined", "grouped"),
        default="predefined",
        help="Use the existing keras_slices_{train,validate,test} folders (default), or create a new group split",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--commitment-beta", type=float, default=0.25)
    parser.add_argument("--prior-epochs", type=int, default=15)
    parser.add_argument("--prior-hidden", type=int, default=64)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include kind=unknown after manually auditing the manifest",
    )
    parser.add_argument(
        "--allow-unverified-groups",
        action="store_true",
        help="Unsafe override; use only for pipeline smoke tests",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=0,
        help="Debug only: cap train batches/epoch; 0 means all",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=0,
        help="Debug only: cap validation batches; 0 means all",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=int,
        default=0,
        help="Debug only: cap test batches; 0 means all",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers non-negative")
    if args.model == "vqvae" and args.prior_epochs < 1:
        parser.error(
            "VQ-VAE sampling requires --prior-epochs >= 1 to train the code-index prior"
        )
    args.data_root = args.data_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    rows = load_manifest(
        args.manifest, args.data_root, include_unknown=args.include_unknown
    )
    groups_verified = all_groups_verified(rows)
    if args.split_mode == "grouped":
        if not groups_verified and not args.allow_unverified_groups:
            raise SystemExit(
                "Refusing to create a new split because group_id_verified is not true for every selected image. "
                "For the supplied HipMRI layout, prefer the default --split-mode predefined; it uses the "
                "existing train/validate/test folders. For a custom grouped split, verify group IDs first."
            )
        if not groups_verified:
            print("WARNING: USING UNVERIFIED GROUP IDs for a newly generated split.")
        splits = split_by_group(rows, seed=args.seed)
        split_source = "new_group_split"
    else:
        splits = split_from_manifest(rows)
        split_source = "predefined_directory_split"
        print(
            "Split source: existing keras_slices_train / keras_slices_validate / keras_slices_test folders"
        )
        print(
            "Segmentation masks in keras_slices_seg_* folders are excluded from the MRI experiment."
        )

    split_json = {
        "seed": args.seed,
        "split_mode": args.split_mode,
        "split_source": split_source,
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "group_id_verified": groups_verified,
        "case_ids_disjoint_across_splits": True,
        "split_group_counts": {
            k: len({row["group_id"] for row in values}) for k, values in splits.items()
        },
        "split_image_counts": {k: len(values) for k, values in splits.items()},
        "splits": {k: values for k, values in splits.items()},
    }
    save_json(args.run_dir / "split.json", split_json)
    loaders = make_loaders(
        splits,
        args.data_root,
        args.image_height,
        args.image_width,
        args.batch_size,
        args.workers,
        args.seed,
    )
    if any(len(loaders[name].dataset) == 0 for name in ("train", "val", "test")):
        raise RuntimeError(
            f"At least one split is empty: {split_json['split_image_counts']}"
        )

    model_config = {
        "base_channels": args.base_channels,
        "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
        "commitment_beta": args.commitment_beta,
    }
    model = build_model(args.model, model_config).to(device)
    n_parameters = count_parameters(model)
    print(f"Model parameters: {n_parameters:,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_ssim = -float("inf")
    history: List[Dict[str, object]] = []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            args.model,
            device,
            args.limit_train_batches,
        )
        val_metrics = evaluate_model(
            model, loaders["val"], args.model, device, args.limit_val_batches
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reconstruction_mse": train_metrics["reconstruction_mse"],
            "train_codebook_loss": train_metrics["codebook_loss"],
            "train_commitment_loss": train_metrics["commitment_loss"],
            "train_perplexity": train_metrics["perplexity"],
            "train_active_codes": train_metrics["active_codes"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction_mse": val_metrics["reconstruction_mse"],
            "val_ssim": val_metrics["ssim"],
            "val_perplexity": val_metrics["perplexity"],
            "val_active_codes": val_metrics["active_codes"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | train loss={train_metrics['loss']:.5f} "
            f"val loss={val_metrics['loss']:.5f} val SSIM={val_metrics['ssim']:.4f} "
            f"| {epoch_seconds:.1f}s"
        )
        write_history_csv(args.run_dir / "history.csv", history)
        if val_metrics["ssim"] > best_val_ssim:
            best_val_ssim = val_metrics["ssim"]
            save_checkpoint(
                args.run_dir / "best.pt", model, args, epoch, val_metrics, n_parameters
            )
            save_json(args.run_dir / "best_validation_metrics.json", val_metrics)

    training_seconds = time.perf_counter() - train_start
    plot_history(args.run_dir, history)
    checkpoint = torch.load(
        args.run_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(
        model, loaders["test"], args.model, device, args.limit_test_batches
    )
    save_json(args.run_dir / "test_metrics.json", test_metrics)

    # The baseline gets a weak empirical diagonal-Gaussian latent sampler; the VQ-VAE gets a
    # learned autoregressive prior trained on training-split code maps only.
    extra: Dict[str, object] = {}
    if args.model == "ae":
        fit_ae_latent_prior(
            model,
            loaders["train"],
            device,
            args.run_dir / "ae_latent_prior.pt",
            args.limit_train_batches,
        )
        extra["generation_prior"] = (
            "channel-wise diagonal Gaussian fit to training latent activations (weak baseline only)"
        )
    else:
        extra.update(
            train_code_prior(model, loaders["train"], args, device, args.run_dir)
        )
        extra["generation_prior"] = (
            "PixelCNN-like autoregressive prior trained on training-split code maps"
        )

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    final_summary = {
        "model": args.model,
        "model_parameters": n_parameters,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "seed": args.seed,
        "group_id_verified": groups_verified,
        "split_group_counts": split_json["split_group_counts"],
        "split_image_counts": split_json["split_image_counts"],
        "epochs_requested": args.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "training_wall_seconds": training_seconds,
        "total_training_plus_prior_wall_seconds": training_seconds
        + float(extra.get("prior_training_seconds", 0.0) or 0.0),
        "peak_cuda_memory_mb": peak_memory_mb,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "held_out_test_metrics": test_metrics,
        "generation": extra,
        "notes": [
            (
                "The predefined directory split was preserved and case IDs were checked for overlap; verify the case-ID semantics against dataset documentation before making a formal patient-independent evaluation claim."
                if args.split_mode == "predefined"
                else "Test metrics are final only when the manifest and source-group mapping have been verified."
            ),
            "SSIM here measures paired input/reconstruction similarity, not unpaired synthetic-image fidelity.",
            "The AE latent sampler is not learned as part of the AE objective and is intentionally a weak comparator.",
        ],
    }
    save_json(args.run_dir / "run_summary.json", final_summary)
    print(
        f"Best validation SSIM: {best_val_ssim:.4f}; held-out test SSIM: {test_metrics['ssim']:.4f}"
    )
    print(f"Run artifacts: {args.run_dir}")
    if not groups_verified:
        print(
            "RESULT QUALIFICATION: group mapping is unverified; treat all scores as pipeline smoke-test results only."
        )


if __name__ == "__main__":
    main()
