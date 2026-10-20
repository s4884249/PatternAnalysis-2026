"""Inference, reconstruction diagnostics, sampling, and nearest-neighbour audit.

The synthetic sample audit is diagnostic rather than a privacy certification. A
close match triggers manual review; a distant nearest neighbour does not prove
that a trained medical generator is safe to share.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NiftiSliceDataset
from modules import CodePrior, VQVAE, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available")
    return device


def to_device(batch, device: torch.device) -> torch.Tensor:
    return batch["image"].to(
        device=device, dtype=torch.float32, non_blocking=(device.type == "cuda")
    )


def get_ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            np.asarray(a, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
            data_range=1.0,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
    )


def save_image_grid(
    images: torch.Tensor,
    path: Path,
    titles: Sequence[str] | None = None,
    ncols: int = 8,
    cmap: str = "gray",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = images.detach().float().cpu().clamp(0, 1).numpy()
    if arr.ndim == 4:
        arr = arr[:, 0]
    count = len(arr)
    ncols = min(ncols, max(1, count))
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0), squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < count:
            ax.imshow(arr[i], cmap=cmap, vmin=vmin, vmax=vmax)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=7)
    fig.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_failures(
    rows: Sequence[Dict[str, object]],
    originals: Dict[str, torch.Tensor],
    reconstructions: Dict[str, torch.Tensor],
    output_path: Path,
    max_cases: int = 5,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [
        row
        for row in rows
        if str(row["path"]) in originals and str(row["path"]) in reconstructions
    ]
    selected = sorted(available, key=lambda r: float(r["ssim"]))[:max_cases]
    if not selected:
        return []
    fig, axes = plt.subplots(
        len(selected), 3, figsize=(9, 2.6 * len(selected)), squeeze=False
    )
    for row_index, record in enumerate(selected):
        path = str(record["path"])
        original = originals[path][0].cpu().numpy()
        recon = reconstructions[path][0].cpu().numpy()
        diff = np.abs(original - recon)
        axes[row_index, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 0].set_title(f"Input\n{Path(path).name}", fontsize=8)
        axes[row_index, 1].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(
            f"Reconstruction\nSSIM={float(record['ssim']):.3f}", fontsize=8
        )
        axes[row_index, 2].imshow(
            diff, cmap="inferno", vmin=0, vmax=max(0.25, float(diff.max()))
        )
        axes[row_index, 2].set_title("Absolute error", fontsize=8)
        for col in range(3):
            axes[row_index, col].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [str(record["path"]) for record in selected]


def nearest_mse(
    query: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest reference by MSE after 4x area downsampling; returns distance and index."""
    q = F.adaptive_avg_pool2d(query.float(), (32, 20)).flatten(1).to(device)
    r = F.adaptive_avg_pool2d(reference.float(), (32, 20)).flatten(1).to(device)
    r_sq = r.square().sum(dim=1).unsqueeze(0)
    q_sq = q.square().sum(dim=1, keepdim=True)
    all_distances = []
    all_indices = []
    denom = float(q.shape[1])
    for start in range(0, q.shape[0], chunk_size):
        qb = q[start : start + chunk_size]
        squared = (
            (qb_sq := qb.square().sum(dim=1, keepdim=True)) + r_sq - 2.0 * (qb @ r.t())
        )
        squared = squared.clamp_min(0.0) / denom
        distances, indices = squared.min(dim=1)
        all_distances.append(distances.detach().cpu())
        all_indices.append(indices.detach().cpu())
    return torch.cat(all_distances), torch.cat(all_indices)


def latent_features(model, images: torch.Tensor) -> torch.Tensor:
    latent = model.encode(images)
    return F.adaptive_avg_pool2d(latent, (1, 1)).flatten(1)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def encode_ae_latent_pool(
    model,
    reference_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    max_latents: int = 2048,
) -> torch.Tensor:
    """Encode training-reference images to an empirical AE latent pool.

    A deterministic AE is not trained to map arbitrary Gaussian noise to valid
    anatomy. The pool keeps each encoded latent map's spatial correlations.
    """
    if reference_images.ndim != 4 or reference_images.shape[1] != 1:
        raise ValueError("reference_images must have shape [N, 1, H, W]")
    count = min(int(max_latents), int(reference_images.shape[0]))
    if count < 2:
        raise ValueError(
            "AE latent interpolation requires at least two training reference images"
        )
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            images = reference_images[start : start + batch_size].to(
                device=device, dtype=torch.float32
            )
            chunks.append(model.encode(images).detach().float().cpu())
    pool = torch.cat(chunks, dim=0)
    if not torch.isfinite(pool).all():
        raise FloatingPointError(
            "Non-finite values found in the encoded AE latent pool"
        )
    return pool


def sample_ae_latent_interpolations(
    latent_pool: torch.Tensor,
    num_samples: int,
    device: torch.device,
    seed: int = 42,
    neighbours: int = 8,
    alpha_min: float = 0.25,
    alpha_max: float = 0.75,
) -> torch.Tensor:
    """Create smooth AE latent samples by interpolating nearby training codes.

    This is a practical baseline sampler, not a learned probability model. It
    avoids independent per-pixel/channel latent noise, which is off-manifold for
    a deterministic AE. Since its source is the training latent pool, resulting
    samples must be checked for closeness to training examples.
    """
    if latent_pool.ndim != 4 or latent_pool.shape[0] < 2:
        raise ValueError("latent_pool must have shape [N, C, H, W] with N >= 2")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not (0.0 < alpha_min <= alpha_max < 1.0):
        raise ValueError(
            "Interpolation bounds must satisfy 0 < alpha_min <= alpha_max < 1"
        )

    pool = latent_pool.to(device=device, dtype=torch.float32)
    # Compare global latent descriptors with per-channel standardization so a
    # high-variance latent channel does not dominate neighbour selection.
    descriptors = pool.mean(dim=(2, 3))
    descriptor_mean = descriptors.mean(dim=0, keepdim=True)
    descriptor_std = descriptors.std(dim=0, unbiased=False, keepdim=True).clamp_min(
        1e-4
    )
    descriptors = (descriptors - descriptor_mean) / descriptor_std
    distances = torch.cdist(descriptors, descriptors)
    distances.fill_diagonal_(float("inf"))
    k = min(max(1, int(neighbours)), int(pool.shape[0]) - 1)
    nearest = distances.topk(k=k, dim=1, largest=False).indices

    generator = torch.Generator(device=device).manual_seed(int(seed))
    source_indices = torch.randint(
        pool.shape[0], (num_samples,), device=device, generator=generator
    )
    neighbour_slots = torch.randint(
        k, (num_samples,), device=device, generator=generator
    )
    target_indices = nearest[source_indices, neighbour_slots]
    alpha = torch.rand((num_samples, 1, 1, 1), device=device, generator=generator)
    alpha = alpha * (alpha_max - alpha_min) + alpha_min
    return (1.0 - alpha) * pool[source_indices] + alpha * pool[target_indices]


def generate_samples(
    model,
    model_type: str,
    run_dir: Path,
    num_samples: int,
    device: torch.device,
    latent_height: int,
    latent_width: int,
    temperature: float,
    ae_latent_pool: torch.Tensor | None = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, float, int | None]:
    if model_type == "vqvae":
        prior_file = run_dir / "prior.pt"
        if not prior_file.is_file():
            raise FileNotFoundError(
                f"VQ-VAE prior missing: {prior_file}; rerun train.py to fit the code prior"
            )
        prior_state = torch.load(prior_file, map_location=device, weights_only=True)
        prior = CodePrior(
            codebook_size=int(prior_state["codebook_size"]),
            hidden_channels=int(prior_state["hidden_channels"]),
        ).to(device)
        prior.load_state_dict(prior_state["state_dict"])
        prior.eval()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        indices = prior.sample(
            num_samples, latent_height, latent_width, device, temperature=temperature
        )
        samples = model.decode_indices(indices).clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        return samples, elapsed, count_parameters(prior)

    if ae_latent_pool is None:
        raise ValueError("AE sampling requires an empirical training latent pool")
    if tuple(ae_latent_pool.shape[1:]) != (
        int(model.latent_dim),
        latent_height,
        latent_width,
    ):
        raise ValueError(
            "AE latent pool shape does not match model/config: "
            f"pool={tuple(ae_latent_pool.shape[1:])}, "
            f"expected={(int(model.latent_dim), latent_height, latent_width)}"
        )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    latent = sample_ae_latent_interpolations(
        ae_latent_pool, num_samples, device, seed=seed
    )
    samples = model.decode(latent).clamp(0, 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return samples, elapsed, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images and run reconstruction / nearest-neighbour audits"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best.pt produced by train.py",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="PixelCNN sampling temperature (VQ-VAE only)",
    )
    parser.add_argument("--max-reference-images", type=int, default=5000)
    parser.add_argument("--audit-test-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.num_samples < 1
        or args.max_reference_images < 1
        or args.audit_test_count < 1
        or args.batch_size < 1
    ):
        parser.error(
            "num-samples, max-reference-images, audit-test-count, and batch-size must be positive"
        )
    if args.temperature <= 0:
        parser.error("temperature must be greater than zero")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    run_dir = checkpoint_path.parent
    args.output_dir = (
        (args.output_dir or (run_dir / "diagnostics")).expanduser().resolve()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = str(checkpoint["model_type"])
    config = dict(checkpoint["model_config"])
    data_root = Path(args.data_root or config["data_root"]).expanduser().resolve()
    manifest_path = Path(args.manifest or config["manifest"]).expanduser().resolve()
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    split_file = run_dir / "split.json"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split metadata missing: {split_file}")
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    if not split_data.get("group_id_verified", False):
        print(
            "WARNING: group IDs in split.json were unverified at training time; no leakage-safe or privacy claim is justified."
        )
    splits = split_data["splits"]
    train_rows = splits["train"]
    test_rows = splits["test"]
    if len(test_rows) == 0 or len(train_rows) == 0:
        raise RuntimeError("split.json has an empty train or test split")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    test_dataset = NiftiSliceDataset(
        test_rows,
        data_root,
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    from skimage.metrics import (
        structural_similarity,
    )  # fail early if metric dependency missing

    reconstruction_records: List[Dict[str, object]] = []
    original_examples: Dict[str, torch.Tensor] = {}
    reconstruction_examples: Dict[str, torch.Tensor] = {}
    total_start = time.perf_counter()
    reconstruction_inference_seconds = 0.0
    reconstruction_image_count = 0
    with torch.inference_mode():
        for batch in test_loader:
            images = to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            output = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reconstruction_inference_seconds += time.perf_counter() - inference_start
            reconstruction_image_count += int(images.shape[0])
            recon = output["reconstruction"].clamp(0, 1)
            originals_np = images.cpu().numpy()[:, 0]
            recon_np = recon.cpu().numpy()[:, 0]
            paths = list(batch["path"])
            for i, path in enumerate(paths):
                mse = float(np.mean((originals_np[i] - recon_np[i]) ** 2))
                ssim = get_ssim(originals_np[i], recon_np[i])
                reconstruction_records.append(
                    {
                        "path": path,
                        "group_id": batch["group_id"][i],
                        "mse": mse,
                        "ssim": ssim,
                    }
                )
                if len(original_examples) < 32:
                    original_examples[path] = images[i].detach().cpu()
                    reconstruction_examples[path] = recon[i].detach().cpu()
    write_csv(args.output_dir / "reconstruction_metrics.csv", reconstruction_records)

    # Choose a reproducible audit subset; full reconstruction metrics above use all test images.
    rng = random.Random(args.seed)
    audit_test_rows = list(test_rows)
    rng.shuffle(audit_test_rows)
    audit_test_rows = audit_test_rows[
        : min(args.audit_test_count, len(audit_test_rows))
    ]
    audit_test_dataset = NiftiSliceDataset(
        audit_test_rows,
        data_root,
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
    )
    audit_test_loader = DataLoader(
        audit_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    audit_test_images, audit_test_paths = [], []
    audit_test_features = []
    with torch.inference_mode():
        for batch in audit_test_loader:
            images = to_device(batch, device)
            audit_test_images.append(images.cpu())
            audit_test_paths.extend(batch["path"])
            audit_test_features.append(latent_features(model, images).cpu())
    audit_test_images = torch.cat(audit_test_images, dim=0)
    audit_test_features = torch.cat(audit_test_features, dim=0)

    # Use a random subset as the train reference pool; report its size so the audit is reproducible.
    ref_rows = list(train_rows)
    rng.shuffle(ref_rows)
    ref_rows = ref_rows[: min(args.max_reference_images, len(ref_rows))]
    ref_dataset = NiftiSliceDataset(
        ref_rows,
        data_root,
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
    )
    ref_loader = DataLoader(
        ref_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    reference_images, reference_paths, reference_features = [], [], []
    with torch.inference_mode():
        for batch in ref_loader:
            images = to_device(batch, device)
            reference_images.append(images.cpu())
            reference_paths.extend(batch["path"])
            reference_features.append(latent_features(model, images).cpu())
    reference_images = torch.cat(reference_images, dim=0)
    reference_features = torch.cat(reference_features, dim=0)
    if len(reference_images) == 0:
        raise RuntimeError(
            "No training reference images loaded for nearest-neighbour audit"
        )

    test_nn_mse, test_nn_idx = nearest_mse(audit_test_images, reference_images, device)
    height, width = int(config["image_height"]), int(config["image_width"])
    latent_height, latent_width = height // 8, width // 8
    ae_latent_pool = None
    ae_latent_pool_encoding_seconds = 0.0
    if model_type == "ae":
        pool_start = time.perf_counter()
        ae_latent_pool = encode_ae_latent_pool(
            model,
            reference_images,
            device,
            batch_size=args.batch_size,
            max_latents=2048,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ae_latent_pool_encoding_seconds = time.perf_counter() - pool_start
    samples, sampling_seconds, prior_parameters = generate_samples(
        model,
        model_type,
        run_dir,
        args.num_samples,
        device,
        latent_height,
        latent_width,
        args.temperature,
        ae_latent_pool=ae_latent_pool,
        seed=args.seed,
    )
    torch.save(samples.cpu(), args.output_dir / "generated_samples.pt")
    save_image_grid(samples, args.output_dir / "generated_samples.png", ncols=8)

    # Reload the true lowest-SSIM cases for an honest failure panel (they may be outside
    # the first few visualization examples cached above).
    worst_records = sorted(reconstruction_records, key=lambda r: float(r["ssim"]))[:5]
    row_by_path = {row["path"]: row for row in test_rows}
    failure_rows = [
        row_by_path[str(record["path"])]
        for record in worst_records
        if str(record["path"]) in row_by_path
    ]
    failure_dataset = NiftiSliceDataset(
        failure_rows,
        data_root,
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
    )
    failure_loader = DataLoader(
        failure_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    failure_originals, failure_reconstructions = {}, {}
    with torch.inference_mode():
        for batch in failure_loader:
            images = to_device(batch, device)
            recon = model(images)["reconstruction"].clamp(0, 1)
            for i, path in enumerate(batch["path"]):
                failure_originals[path] = images[i].detach().cpu()
                failure_reconstructions[path] = recon[i].detach().cpu()
    fail_paths = save_reconstruction_failures(
        reconstruction_records,
        failure_originals,
        failure_reconstructions,
        args.output_dir / "reconstruction_failures.png",
        max_cases=5,
    )
    sample_paths = list(original_examples.keys())
    if sample_paths:
        input_stack = torch.stack([original_examples[p] for p in sample_paths[:8]])
        recon_stack = torch.stack(
            [reconstruction_examples[p] for p in sample_paths[:8]]
        )
        comparison = torch.cat([input_stack, recon_stack], dim=0)
        comparison_titles = [f"Input {i + 1}" for i in range(len(input_stack))] + [
            f"Recon {i + 1}" for i in range(len(recon_stack))
        ]
        save_image_grid(
            comparison,
            args.output_dir / "reconstruction_grid.png",
            titles=comparison_titles,
            ncols=8,
        )

    # Latent interpolation between two distinct held-out inputs; this is not a calibrated probability path.
    if len(audit_test_images) >= 2:
        with torch.inference_mode():
            pair = audit_test_images[:2].to(device)
            z_a, z_b = model.encode(pair[:1]), model.encode(pair[1:])
            if model_type == "vqvae":
                z_a = model.quantizer(z_a)[0]
                z_b = model.quantizer(z_b)[0]
            interpolation = []
            for alpha in torch.linspace(0.0, 1.0, steps=9, device=device):
                z = (1.0 - alpha) * z_a + alpha * z_b
                interpolation.append(model.decode(z).clamp(0, 1).cpu())
            interpolation_stack = torch.cat(interpolation, dim=0)
        save_image_grid(
            interpolation_stack,
            args.output_dir / "latent_interpolation.png",
            ncols=9,
            titles=[f"alpha={i / 8:.2f}" for i in range(9)],
        )

    generated_nn_mse, generated_nn_idx = nearest_mse(
        samples.cpu(), reference_images, device
    )
    generated_np = samples.detach().cpu().numpy()[:, 0]
    reference_np = reference_images.numpy()[:, 0]
    nn_rows = []
    for i in range(len(samples)):
        ref_idx = int(generated_nn_idx[i])
        ref_ssim = get_ssim(generated_np[i], reference_np[ref_idx])
        nn_rows.append(
            {
                "generated_index": i,
                "nearest_train_path": reference_paths[ref_idx],
                "downsampled_pixel_mse": float(generated_nn_mse[i]),
                "nearest_neighbor_ssim": ref_ssim,
                "manual_review_flag": bool(
                    float(generated_nn_mse[i])
                    <= float(torch.quantile(test_nn_mse, 0.05))
                ),
            }
        )
    write_csv(args.output_dir / "generated_nearest_neighbors.csv", nn_rows)

    # Montage the closest generated samples and their closest training references.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_matches = sorted(nn_rows, key=lambda row: float(row["downsampled_pixel_mse"]))[
        :5
    ]
    fig, axes = plt.subplots(
        len(top_matches), 2, figsize=(6, 2.5 * max(1, len(top_matches))), squeeze=False
    )
    for i, match in enumerate(top_matches):
        gen_idx = int(match["generated_index"])
        ref_idx = reference_paths.index(str(match["nearest_train_path"]))
        axes[i, 0].imshow(generated_np[gen_idx], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(
            f"Generated {gen_idx}\ntrain MSE={float(match['downsampled_pixel_mse']):.5f}",
            fontsize=8,
        )
        axes[i, 1].imshow(reference_np[ref_idx], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Nearest train\nSSIM={float(match['nearest_neighbor_ssim']):.3f}",
            fontsize=8,
        )
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "nearest_neighbor_montage.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)

    # Encoder-space coverage proxy. Epsilon is the 95th percentile of held-out-real -> train distances.
    generated_features = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            generated_features.append(
                latent_features(model, samples[start : start + args.batch_size]).cpu()
            )
    generated_features = torch.cat(generated_features, dim=0)
    ref_feat_dev = reference_features.to(device)
    test_feat_dev = audit_test_features.to(device)
    gen_feat_dev = generated_features.to(device)
    test_train_latent = torch.cdist(test_feat_dev, ref_feat_dev).min(dim=1).values.cpu()
    test_synth_latent = torch.cdist(test_feat_dev, gen_feat_dev).min(dim=1).values.cpu()
    gen_train_latent = torch.cdist(gen_feat_dev, ref_feat_dev).min(dim=1).values.cpu()
    epsilon = float(torch.quantile(test_train_latent, 0.95))
    feature_coverage = float((test_synth_latent <= epsilon).float().mean())

    elapsed = time.perf_counter() - total_start
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else None
    )
    ssims = [float(row["ssim"]) for row in reconstruction_records]
    mses = [float(row["mse"]) for row in reconstruction_records]
    summary = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "group_id_verified": bool(split_data.get("group_id_verified", False)),
        "test_reconstruction": {
            "n_images": len(reconstruction_records),
            "inference_seconds": reconstruction_inference_seconds,
            "milliseconds_per_image": 1000.0
            * reconstruction_inference_seconds
            / max(1, reconstruction_image_count),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_median": float(np.median(ssims)),
            "ssim_min": float(np.min(ssims)),
            "mse_mean": float(np.mean(mses)),
            "lowest_ssim_cases": fail_paths,
        },
        "sampling": {
            "n_samples": int(len(samples)),
            "sampling_and_decode_seconds": sampling_seconds,
            "latency_seconds_per_sample": sampling_seconds / max(1, len(samples)),
            "prior_parameters": prior_parameters,
            "temperature": args.temperature if model_type == "vqvae" else None,
            "ae_sampler_method": "interpolation between nearby encoded training images"
            if model_type == "ae"
            else None,
            "ae_latent_pool_size": int(ae_latent_pool.shape[0])
            if ae_latent_pool is not None
            else None,
            "ae_latent_pool_encoding_seconds": ae_latent_pool_encoding_seconds
            if model_type == "ae"
            else None,
            "ae_sampler_warning": "The deterministic AE has no learned probabilistic prior. These are local latent interpolations, not independent draws from a learned image distribution; inspect nearest-training neighbours for similarity."
            if model_type == "ae"
            else None,
        },
        "nearest_neighbor_audit": {
            "train_reference_count": int(len(reference_images)),
            "test_audit_count": int(len(audit_test_images)),
            "generated_nn_mse_mean": float(generated_nn_mse.mean()),
            "heldout_test_nn_mse_mean": float(test_nn_mse.mean()),
            "heldout_test_nn_mse_5th_percentile": float(
                torch.quantile(test_nn_mse, 0.05)
            ),
            "generated_samples_flagged_for_manual_review": int(
                sum(bool(row["manual_review_flag"]) for row in nn_rows)
            ),
            "interpretation": "Pixel MSE is measured after 4x area downsampling. A small nearest-neighbour distance is a review trigger, not proof of memorization; a large distance is not proof of privacy.",
        },
        "latent_feature_coverage_proxy": {
            "embedding": "spatial mean of the model's continuous encoder output",
            "epsilon": epsilon,
            "heldout_test_coverage_by_generated_set": feature_coverage,
            "test_to_train_nearest_latent_distance_mean": float(
                test_train_latent.mean()
            ),
            "test_to_generated_nearest_latent_distance_mean": float(
                test_synth_latent.mean()
            ),
            "generated_to_train_nearest_latent_distance_mean": float(
                gen_train_latent.mean()
            ),
            "limitation": "This model-specific encoder-space proxy is not standard FID and is not validated as a clinical fidelity metric.",
        },
        "resource_profile": {
            "model_parameters": count_parameters(model),
            "peak_cuda_memory_mb": peak_memory_mb,
            "total_predict_diagnostics_seconds": elapsed,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "artifacts": [
            "reconstruction_metrics.csv",
            "reconstruction_grid.png",
            "reconstruction_failures.png",
            "generated_samples.png",
            "generated_samples.pt",
            "latent_interpolation.png",
            "generated_nearest_neighbors.csv",
            "nearest_neighbor_montage.png",
        ],
    }
    (args.output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved diagnostics to: {args.output_dir}")
    print(
        f"Mean test reconstruction SSIM: {summary['test_reconstruction']['ssim_mean']:.4f}"
    )
    print(
        f"Mean generated-to-train nearest-neighbour MSE (downsampled): {summary['nearest_neighbor_audit']['generated_nn_mse_mean']:.6f}"
    )
    print(f"Encoder-space test feature coverage proxy: {feature_coverage:.3f}")
    if not summary["group_id_verified"]:
        print(
            "WARNING: no patient-independent or privacy-safety claim is justified while source-group IDs remain unverified."
        )


if __name__ == "__main__":
    main()
