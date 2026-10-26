"""Generate images and run reconstruction / nearest-neighbour diagnostics for a trained run.

The nearest-neighbour audit is a diagnostic, not a privacy certificate: a close match
flags a sample for inspection, and a distant one does not prove a medical generator
is safe to share. All settings come from params.PredictParams.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NiftiSliceDataset
from modules import CodePrior, build_model, count_parameters
from params import PredictParams, add_overrides, apply_overrides
from device_utils import get_device, peak_memory_mb, reset_peak_memory_stats, sync
from utils import read_json, save_image_grid, save_json, set_seed, ssim_2d, write_csv


def load_images(rows: Sequence[dict], data_root: Path, config: dict, batch_size: int) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Load rows into one [N, 1, H, W] tensor (images are small, so everything fits in memory)."""
    dataset = NiftiSliceDataset(rows, data_root, config["image_height"], config["image_width"])
    images, paths, groups = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        images.append(batch["image"])
        paths.extend(batch["path"])
        groups.extend(batch["group_id"])
    return torch.cat(images), paths, groups


@torch.inference_mode()
def reconstruct(model, images: torch.Tensor, device: torch.device, batch_size: int) -> Tuple[torch.Tensor, float]:
    """Reconstruct all images; returns (reconstructions, model-only seconds) for latency reporting."""
    outputs, seconds = [], 0.0
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        sync(device)
        tick = time.perf_counter()
        output = model(batch)["reconstruction"]
        sync(device)
        seconds += time.perf_counter() - tick
        outputs.append(output.clamp(0, 1).cpu())
    return torch.cat(outputs), seconds


@torch.inference_mode()
def encode_features(model, images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    """One feature vector per image: the VAE posterior mean, or the spatial mean of the VQ-VAE encoder map."""
    features = []
    for i in range(0, len(images), batch_size):
        z = model.encode(images[i : i + batch_size].to(device))
        features.append((z.mean(dim=(2, 3)) if z.ndim == 4 else z).cpu())
    return torch.cat(features)


def nearest_mse(query: torch.Tensor, reference: torch.Tensor, device: torch.device, chunk_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest reference image per query by MSE after 4x average-pool downsampling.

    Downsampling makes the match about coarse anatomy rather than pixel noise.
    Uses |a-b|^2 = |a|^2 + |b|^2 - 2ab and processes queries in chunks to bound memory.
    """
    q = F.avg_pool2d(query.float(), 4).flatten(1).to(device)
    r = F.avg_pool2d(reference.float(), 4).flatten(1).to(device)
    r_sq = r.square().sum(dim=1).unsqueeze(0)
    distances, indices = [], []
    for start in range(0, len(q), chunk_size):
        qb = q[start : start + chunk_size]
        squared = qb.square().sum(dim=1, keepdim=True) + r_sq - 2.0 * qb @ r.t()
        best, index = (squared.clamp_min(0.0) / q.shape[1]).min(dim=1)
        distances.append(best.cpu())
        indices.append(index.cpu())
    return torch.cat(distances), torch.cat(indices)


def generate_samples(
    model, model_type: str, run_dir: Path, num_samples: int, device: torch.device,
    latent_hw: Tuple[int, int], temperature: float, seed: int,
) -> Tuple[torch.Tensor, float, int | None]:
    """Draw new images; returns (samples, seconds for sampling + decoding, prior parameter count).

    VAE:    decode a global z drawn from the Gaussian fitted to the training latents
            (latent_prior.pt); falls back to N(0, I) if that file is missing.
    VQ-VAE: sample code grids from the learned prior, then decode the codes.
    """
    generator = torch.Generator(device=device).manual_seed(seed)  # reproducible, device-matched
    prior, prior_parameters = None, None
    if model_type == "vqvae":
        prior_file = run_dir / "prior.pt"
        if not prior_file.is_file():
            raise FileNotFoundError(f"VQ-VAE prior missing: {prior_file}; rerun train.py to fit the code prior")
        state = torch.load(prior_file, map_location=device, weights_only=True)
        prior = CodePrior(codebook_size=int(state["codebook_size"]), hidden_channels=int(state["hidden_channels"])).to(device)
        prior.load_state_dict(state["state_dict"])
        prior.eval()
        prior_parameters = count_parameters(prior)

    with torch.inference_mode():
        sync(device)
        tick = time.perf_counter()
        if prior is not None:
            codes = prior.sample(num_samples, *latent_hw, device, temperature=temperature, generator=generator)
            samples = model.decode_indices(codes)
        else:
            eps = torch.randn((num_samples, model.latent_dim), device=device, generator=generator)
            prior_file = run_dir / "latent_prior.pt"
            if prior_file.is_file():  # Gaussian fitted to the training latents by train.py
                state = torch.load(prior_file, map_location=device, weights_only=True)
                z = state["mean"].to(device) + eps @ state["cholesky"].to(device).T
            else:
                print("latent_prior.pt not found; sampling from N(0, I), which usually gives poor VAE samples.")
                z = eps
            samples = model.decode(z)
        sync(device)
        seconds = time.perf_counter() - tick
    return samples.clamp(0, 1).cpu(), seconds, prior_parameters


def save_reconstruction_failures(
    originals: torch.Tensor, recons: torch.Tensor, ssims: Sequence[float], names: Sequence[str], path: Path
) -> None:
    """Input / reconstruction / absolute-error rows for the lowest-SSIM test images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(names), 3, figsize=(9, 2.6 * len(names)), squeeze=False)
    for row, (original, recon, ssim, name) in enumerate(zip(originals, recons, ssims, names)):
        original, recon = original[0].numpy(), recon[0].numpy()
        error = np.abs(original - recon)
        axes[row, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Input\n{Path(name).name}", fontsize=8)
        axes[row, 1].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Reconstruction\nSSIM={ssim:.3f}", fontsize=8)
        axes[row, 2].imshow(error, cmap="inferno", vmin=0, vmax=max(0.25, float(error.max())))
        axes[row, 2].set_title("Absolute error", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_nearest_neighbor_montage(
    samples: torch.Tensor, references: torch.Tensor, distances: torch.Tensor, nn_index: torch.Tensor,
    nn_ssim: Sequence[float], path: Path, count: int = 5,
) -> None:
    """The generated samples closest to a training image, next to that training image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    closest = torch.argsort(distances)[:count].tolist()
    fig, axes = plt.subplots(len(closest), 2, figsize=(6, 2.5 * len(closest)), squeeze=False)
    for row, i in enumerate(closest):
        axes[row, 0].imshow(samples[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Generated {i}\ntrain MSE={float(distances[i]):.5f}", fontsize=8)
        axes[row, 1].imshow(references[int(nn_index[i]), 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Nearest train\nSSIM={nn_ssim[i]:.3f}", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images and run reconstruction / nearest-neighbour audits")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt written by train.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <run dir>/diagnostics")
    parser.add_argument("--data-root", type=Path, default=None, help="override the data root stored in the checkpoint")
    add_overrides(parser, PredictParams())
    args = parser.parse_args()
    try:
        p = apply_overrides(PredictParams(), args)
    except ValueError as exc:
        parser.error(str(exc))

    checkpoint_path = args.checkpoint.expanduser().resolve()
    run_dir = checkpoint_path.parent
    out = (args.output_dir or run_dir / "diagnostics").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(p.seed)
    device = get_device(p.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type, config = checkpoint["model_type"], checkpoint["model_config"]
    data_root = Path(args.data_root or config["data_root"]).expanduser().resolve()
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    splits = read_json(run_dir / "split.json")["splits"]
    reset_peak_memory_stats(device)
    total_start = time.perf_counter()

    # 1. Reconstruction metrics over the whole held-out test split.
    test_images, test_paths, test_groups = load_images(splits["test"], data_root, config, p.batch_size)
    recons, recon_seconds = reconstruct(model, test_images, device, p.batch_size)
    mses = (test_images - recons).square().mean(dim=(1, 2, 3)).tolist()
    ssims = [ssim_2d(a[0].numpy(), b[0].numpy()) for a, b in zip(test_images, recons)]
    write_csv(
        out / "reconstruction_metrics.csv",
        [{"path": n, "group_id": g, "mse": m, "ssim": s} for n, g, m, s in zip(test_paths, test_groups, mses, ssims)],
    )

    # 2. Fixed random subsets (seeded, so reruns audit the same images): a test subset
    # and a pool of training images used as the nearest-neighbour reference.
    rng = random.Random(p.seed)
    audit_idx = rng.sample(range(len(test_images)), min(p.audit_test_count, len(test_images)))
    audit_images = test_images[audit_idx]
    ref_rows = rng.sample(splits["train"], min(p.max_reference_images, len(splits["train"])))
    ref_images, ref_paths, _ = load_images(ref_rows, data_root, config, p.batch_size)
    audit_features = encode_features(model, audit_images, device, p.batch_size)
    ref_features = encode_features(model, ref_images, device, p.batch_size)
    test_nn_mse, _ = nearest_mse(audit_images, ref_images, device)

    # 3. New samples.
    factor = 2 ** int(config.get("downsample_stages", 3))  # the encoder downsamples 4x or 8x
    latent_hw = (config["image_height"] // factor, config["image_width"] // factor)  # VQ-VAE code grid
    samples, sampling_seconds, prior_parameters = generate_samples(
        model, model_type, run_dir, p.num_samples, device, latent_hw, p.temperature, p.seed
    )
    torch.save(samples, out / "generated_samples.pt")
    save_image_grid(samples, out / "generated_samples.png", ncols=8)

    # 4. Reconstruction figures: the 5 worst test cases and a side-by-side grid of the first 8.
    worst = np.argsort(ssims)[:5].tolist()
    save_reconstruction_failures(
        test_images[worst], recons[worst], [ssims[i] for i in worst], [test_paths[i] for i in worst],
        out / "reconstruction_failures.png",
    )
    first = min(8, len(test_images))
    save_image_grid(
        torch.cat([test_images[:first], recons[:first]]), out / "reconstruction_grid.png",
        titles=[f"Input {i + 1}" for i in range(first)] + [f"Recon {i + 1}" for i in range(first)], ncols=8,
    )

    # 5. Latent interpolation between two held-out images (qualitative smoothness check).
    if len(audit_images) >= 2:
        with torch.inference_mode():
            z = model.encode(audit_images[:2].to(device))
            if model_type == "vqvae":
                z = model.quantize(z)[0]  # interpolate between the codes the decoder actually sees
            alphas = [i / 8 for i in range(9)]
            frames = torch.cat([model.decode((1 - a) * z[:1] + a * z[1:]).clamp(0, 1).cpu() for a in alphas])
        save_image_grid(frames, out / "latent_interpolation.png", titles=[f"alpha={a:.2f}" for a in alphas], ncols=9)

    # 6. Nearest-neighbour audit of generated samples. A generated sample is flagged when it
    # is closer to a training image than the closest 5% of real held-out images are.
    gen_nn_mse, gen_nn_idx = nearest_mse(samples, ref_images, device)
    threshold = float(torch.quantile(test_nn_mse, 0.05))
    nn_ssim = [ssim_2d(samples[i, 0].numpy(), ref_images[int(gen_nn_idx[i]), 0].numpy()) for i in range(len(samples))]
    nn_rows = [
        {
            "generated_index": i,
            "nearest_train_path": ref_paths[int(gen_nn_idx[i])],
            "downsampled_pixel_mse": float(gen_nn_mse[i]),
            "nearest_neighbor_ssim": nn_ssim[i],
            "close_match_flag": bool(float(gen_nn_mse[i]) <= threshold),
        }
        for i in range(len(samples))
    ]
    write_csv(out / "generated_nearest_neighbors.csv", nn_rows)
    save_nearest_neighbor_montage(samples, ref_images, gen_nn_mse, gen_nn_idx, nn_ssim, out / "nearest_neighbor_montage.png")

    # 7. Encoder-space coverage proxy: the share of held-out images that have a generated
    # image within epsilon, where epsilon is the 95th percentile of held-out -> train distances.
    gen_features = encode_features(model, samples, device, p.batch_size)
    test_to_train = torch.cdist(audit_features, ref_features).min(dim=1).values
    test_to_gen = torch.cdist(audit_features, gen_features).min(dim=1).values
    gen_to_train = torch.cdist(gen_features, ref_features).min(dim=1).values
    epsilon = float(torch.quantile(test_to_train, 0.95))
    coverage = float((test_to_gen <= epsilon).float().mean())

    elapsed = time.perf_counter() - total_start
    summary = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "test_reconstruction": {
            "n_images": len(ssims),
            "inference_seconds": recon_seconds,
            "milliseconds_per_image": 1000.0 * recon_seconds / len(ssims),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_median": float(np.median(ssims)),
            "ssim_min": float(np.min(ssims)),
            "mse_mean": float(np.mean(mses)),
            "lowest_ssim_cases": [test_paths[i] for i in worst],
        },
        "sampling": {
            "n_samples": len(samples),
            "sampling_and_decode_seconds": sampling_seconds,
            "latency_seconds_per_sample": sampling_seconds / len(samples),
            "prior_parameters": prior_parameters,
            "temperature": p.temperature if model_type == "vqvae" else None,
            "method": (
                "learned PixelCNN-style code prior" if model_type == "vqvae"
                else "Gaussian fitted to training latents" if (run_dir / "latent_prior.pt").is_file()
                else "global standard-normal latent vector (no latent_prior.pt)"
            ),
            "warning": "Inspect visual plausibility, diversity and nearest training neighbours; a prior does not guarantee anatomical plausibility.",
        },
        "nearest_neighbor_audit": {
            "train_reference_count": len(ref_images),
            "test_audit_count": len(audit_images),
            "generated_nn_mse_mean": float(gen_nn_mse.mean()),
            "heldout_test_nn_mse_mean": float(test_nn_mse.mean()),
            "heldout_test_nn_mse_5th_percentile": threshold,
            "generated_samples_flagged_close_match": sum(row["close_match_flag"] for row in nn_rows),
            "interpretation": "Pixel MSE after 4x downsampling. A small distance is a review trigger, not proof of memorisation; a large one is not proof of privacy.",
        },
        "latent_feature_coverage_proxy": {
            "embedding": "VAE posterior mean, or spatial mean of the VQ-VAE continuous encoder output",
            "epsilon": epsilon,
            "heldout_test_coverage_by_generated_set": coverage,
            "test_to_train_nearest_latent_distance_mean": float(test_to_train.mean()),
            "test_to_generated_nearest_latent_distance_mean": float(test_to_gen.mean()),
            "generated_to_train_nearest_latent_distance_mean": float(gen_to_train.mean()),
            "limitation": "Model-specific proxy; not standard FID and not a validated clinical fidelity metric.",
        },
        "resource_profile": {
            "model_parameters": count_parameters(model),
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
            "peak_accelerator_memory_mb": peak_memory_mb(device),
            "total_predict_diagnostics_seconds": elapsed,
            "device": str(device),
        },
    }
    save_json(out / "diagnostics_summary.json", summary)
    print(f"Saved diagnostics to: {out}")
    print(f"Mean test reconstruction SSIM: {summary['test_reconstruction']['ssim_mean']:.4f}")
    print(f"Mean generated-to-train nearest-neighbour MSE (downsampled): {summary['nearest_neighbor_audit']['generated_nn_mse_mean']:.6f}")
    print(f"Encoder-space test feature coverage proxy: {coverage:.3f}")


if __name__ == "__main__":
    main()
