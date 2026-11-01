"""Train and evaluate the VAE or the VQ-VAE on 2D HipMRI slices.

The best epoch is chosen on validation SSIM and the test split is evaluated once,
with that checkpoint. The VQ-VAE then fits a causal-transformer prior on training-split
code maps so it can generate new images. All settings live in params.py; the run
directory is derived from them (see params.run_dir_name).

Progress images (at most 10 columns, evenly spaced epochs; the same epochs for the first two):
    reconstructions_by_epoch.png  fixed validation image: input + reconstruction per snapshot
    generated_samples_by_epoch.png  one NEW image per snapshot (fixed random seed, so columns differ only
                                  because the model changed). VAE: decoded from a Gaussian fitted to the
                                  current training latents. VQ-VAE: decoded from codes drawn independently
                                  per grid position from the current training-code frequencies, a cheap
                                  stand-in because the real transformer prior is only trained after the VQ-VAE.
    prior_samples_by_epoch.png    VQ-VAE only: one sample from the transformer prior per prior epoch
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
from utils import save_image_grid, save_json, set_seed, ssim_2d, ssim_torch, to_device, write_csv

SSIM_TARGET = 0.6  # assignment target, drawn on the SSIM curve
MAX_SNAPSHOTS = 10  # upper bound on columns in the progress images
GRAD_CLIP = 5.0  # global gradient-norm clip for the autoencoders (see run_epoch)


def snapshot_epochs(total: int) -> set:
    """Evenly spaced epochs for progress images: every ceil(total/10), plus the last.

    100 epochs -> 10, 20, ..., 100; 5 epochs -> 1..5; 15 epochs -> 2, 4, ..., 14, 15.
    The count never exceeds MAX_SNAPSHOTS.
    """
    step = math.ceil(total / MAX_SNAPSHOTS)
    return set(range(step, total + 1, step)) | {total}


def compute_loss(
    model, images: torch.Tensor, model_type: str, ssim_weight: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, float], dict]:
    """Return the total loss, a dict of float metrics and the raw model output.

    VAE:    reconstruction + kl_beta * KL
    VQ-VAE: reconstruction + codebook loss + commitment loss (the weight is applied in the quantiser)
    reconstruction = MSE on [0, 1] intensities (+ ssim_weight * (1 - SSIM) when ssim_weight > 0),
    identical for both models so scores stay comparable.
    """
    output = model(images)
    mse = F.mse_loss(output["reconstruction"], images)
    reconstruction_term, extra = mse, {}
    if ssim_weight > 0:
        ssim_loss = 1.0 - ssim_torch(images, output["reconstruction"])
        reconstruction_term = mse + ssim_weight * ssim_loss
        extra["ssim_loss"] = ssim_loss
    if model_type == "vae":
        loss = reconstruction_term + model.kl_beta * output["kl_loss"]
        extra["kl_loss"] = output["kl_loss"]
    else:
        loss = reconstruction_term + output["codebook_loss"] + output["commitment_loss"]
        # In-place dict union: also log the VQ diagnostics next to the loss.
        extra |= {k: output[k] for k in ("codebook_loss", "commitment_loss", "perplexity", "active_codes")}
    metrics = {"loss": loss, "reconstruction_mse": mse, **extra}
    # The returned metrics are plain Python floats for logging; `loss` stays a tensor so the caller can
    # call backward() on it.
    return loss, {name: float(value.detach()) for name, value in metrics.items()}, output


def run_epoch(
    model,
    loader: DataLoader,
    model_type: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    limit_batches: int = 0,
    with_ssim: bool = False,
    ssim_weight: float = 0.0,
) -> Dict[str, float]:
    """One pass over `loader`: trains if an optimizer is given, otherwise evaluates.

    Metrics are averaged per image (not per batch) so a short last batch is not
    over-weighted. SSIM is only computed on request because it runs on the CPU.
    """
    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = defaultdict(float)
    n_images = 0
    # inference_mode(True) turns autograd off during evaluation (faster, less memory);
    # inference_mode(False) changes nothing while training.
    with torch.inference_mode(not training):
        for index, batch in enumerate(loader):
            if limit_batches and index >= limit_batches:
                break
            images = to_device(batch, device)
            loss, metrics, output = compute_loss(model, images, model_type, ssim_weight)
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at batch {index}: {loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # clip_grad_norm_ returns the PRE-clip global norm: logging it (and how often it exceeds the
                # clip value) is the gradient-hygiene evidence used by evaluate.py.
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP))
                metrics["grad_norm"] = grad_norm
                metrics["grad_clipped"] = float(grad_norm > GRAD_CLIP)
                optimizer.step()
            if with_ssim:
                # SSIM is computed on the CPU with NumPy / scikit-image: take channel 0 to
                # get [B, H, W] arrays.
                real = images.clamp(0, 1)[:, 0].cpu().numpy()
                recon = output["reconstruction"].clamp(0, 1)[:, 0].cpu().numpy()
                metrics["ssim"] = float(np.mean([ssim_2d(a, b) for a, b in zip(real, recon)]))
            count = int(images.shape[0])
            n_images += count
            for name, value in metrics.items():
                # Weight each batch mean by its batch size, so dividing by n_images at the
                # end gives a true per-image mean.
                sums[name] += value * count
    if n_images == 0:
        raise RuntimeError("Loader produced no images; check the manifest, split and limit_batches.")
    return {name: value / n_images for name, value in sums.items()} | {"n_images": n_images}


@torch.inference_mode()
def reconstruct_one(model, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reconstruct a single [1, H, W] image (eval mode, so the VAE uses its mean latent)."""
    model.eval()
    return model(image[None].to(device))["reconstruction"].clamp(0, 1)[0].cpu()


def plot_history(run_dir: Path, history: List[dict], model_type: str = "") -> None:
    """Save training_curves.png: train / validation loss on the left, validation SSIM with the assignment
    target on the right.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, (loss_ax, ssim_ax) = plt.subplots(1, 2, figsize=(11, 4))
    # A VAE trains with sampled latents but is evaluated with the posterior mean, so its two
    # curves are not measured the same way; say so rather than imply a like-for-like gap.
    train_label, val_label = "train total loss", "validation total loss"
    if model_type == "vae":
        train_label, val_label = "train total loss (sampled z)", "validation total loss (mean z)"
    loss_ax.plot(epochs, [row["train_loss"] for row in history], label=train_label)
    loss_ax.plot(epochs, [row["val_loss"] for row in history], label=val_label)
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
    """train / val / test loaders plus `train_eval`: the training images in a fixed order that
    never touches the training shuffle, used for fitting priors and progress samples."""
    generator = torch.Generator().manual_seed(p.seed)  # reproducible training shuffle order
    cache = bool(p.cache_data)
    workers = 0 if cache else p.workers  # cached tensors need no worker processes
    loaders: Dict[str, DataLoader] = {}
    for name, rows in splits.items():
        dataset = NiftiSliceDataset(
            rows, data_root, p.image_height, p.image_width, cache=cache, threads=max(1, p.workers)
        )

        def build(shuffle: bool, gen=None) -> DataLoader:
            """Create a DataLoader over the current split's dataset (uses `dataset`, `workers` and
            `p` from the enclosing loop).
            """
            return DataLoader(
                dataset,
                batch_size=p.batch_size,
                shuffle=shuffle,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),  # CUDA only; MPS does not use pinned-memory transfers
                persistent_workers=workers > 0,
                generator=gen,
            )

        # build() is called right away in the same iteration, so using the loop variable `dataset`
        # inside it is safe.
        loaders[name] = build(name == "train", generator if name == "train" else None)
        if name == "train":
            # The same training images in a fixed, unshuffled order and without the shuffling
            # generator, so prior fitting and progress samples never disturb the training
            # shuffle.
            loaders["train_eval"] = build(False)
    return loaders


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


@torch.no_grad()
def fit_vae_latent_prior(model, loader: DataLoader, device: torch.device, limit_batches: int = 0) -> dict:
    """Fit a full-covariance Gaussian to the training-split latents (post-hoc density fitting).

    A VAE's aggregate posterior rarely matches N(0, I) exactly, so decoding z ~ N(0, I)
    often lands in latent regions the decoder never saw. Sampling from a Gaussian fitted
    to the encoded training images (mean of the posterior means, covariance of the means
    plus the average posterior variance) fixes that without retraining. Training split only.
    """
    model.eval()
    means, variances = [], []
    for index, batch in enumerate(loader):
        if limit_batches and index >= limit_batches:
            break
        mu, logvar = model.encode_distribution(to_device(batch, device))
        # MPS has no float64: move to the CPU first, then widen for a well-conditioned covariance.
        means.append(mu.cpu().double())
        variances.append(logvar.cpu().double().exp())
    mu, var = torch.cat(means), torch.cat(variances)
    dim = mu.shape[1]
    # Covariance of the aggregate posterior = covariance of the posterior means + average posterior
    # variance (law of total variance). The 1e-5 * I term keeps it positive-definite for the Cholesky
    # factorisation.
    cov = torch.cov(mu.T) + torch.diag(var.mean(0)) + 1e-5 * torch.eye(dim, dtype=torch.float64)
    # The Cholesky factor L (cov = L L^T) lets callers sample z = mean + L eps. .float().clone() gives
    # compact, standalone float32 tensors that are safe to save.
    return {"mean": mu.mean(0).float().clone(), "cholesky": torch.linalg.cholesky(cov).float().clone(), "n_images": int(mu.shape[0])}


@torch.no_grad()
def generate_snapshot_sample(model, model_type: str, loader: DataLoader, device: torch.device, p: TrainParams) -> torch.Tensor:
    """One new [1, H, W] image from the CURRENT weights for the generated_samples_by_epoch strip.

    The random numbers come from a fresh generator with the run's seed on every call, so
    successive snapshots use identical noise and differ only because the model improved.
    VAE:    z ~ Gaussian fitted to the current training latents (as in the final latent prior).
    VQ-VAE: each grid position's code is drawn independently from that position's code
            frequencies over the training split. This is only a stand-in for the transformer prior
            (which cannot exist until the VQ-VAE has finished training), so expect less
            coherent images than prior_samples_by_epoch.png.
    """
    model.eval()
    generator = torch.Generator().manual_seed(p.seed)  # CPU generator: identical on CUDA / MPS / CPU
    if model_type == "vae":
        prior = fit_vae_latent_prior(model, loader, device, p.limit_batches)
        eps = torch.randn((1, model.latent_dim), generator=generator)
        # With eps ~ N(0, I), z = mean + L eps has covariance L L^T. For row vectors this is eps @
        # L.T.
        z = prior["mean"] + eps @ prior["cholesky"].T
        return model.decode(z.to(device)).clamp(0, 1)[0].cpu()
    codes = collect_code_maps(model, loader, device, p.limit_batches)  # [N, H, W]
    n, height, width = codes.shape
    # counts[p, k] = how many training images use code k at grid position p.
    counts = torch.zeros(height * width, model.codebook_size)
    # codes.reshape(n, -1).t() is [positions, n]: for each position, add 1 at the column given by each
    # image's code.
    counts.scatter_add_(1, codes.reshape(n, -1).t(), torch.ones(height * width, n))
    # multinomial normalises each row itself, so raw counts act as probabilities; one independent draw
    # per grid position.
    sampled = torch.multinomial(counts, 1, generator=generator).reshape(1, height, width)
    return model.decode_indices(sampled.to(device)).clamp(0, 1)[0].cpu()


@torch.no_grad()
def code_cross_entropy(prior: CodePrior, code_maps: torch.Tensor, batch_size: int, device: torch.device) -> float:
    """Mean next-code cross-entropy (nats per position) of `prior` on a set of code maps."""
    prior.eval()
    total, count = 0.0, 0
    # TensorDataset yields 1-tuples, hence the (indices,) unpacking.
    for (indices,) in DataLoader(TensorDataset(code_maps), batch_size=batch_size):
        indices = indices.to(device=device, dtype=torch.long)
        logits = prior(indices)
        # [B, K, H, W] logits become [B*H*W, K] rows matching indices.reshape(-1). reduction='sum'
        # makes total / count an exact per-position mean even when the last batch is short.
        flat = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, prior.codebook_size), indices.reshape(-1), reduction="sum")
        total += float(flat)
        count += indices.numel()
    return total / count


def train_code_prior(
    model: VQVAE, train_loader: DataLoader, p: TrainParams, device: torch.device, run_dir: Path,
    val_loader: DataLoader | None = None,
) -> dict:
    """Fit the autoregressive prior on training-split code maps only (no test leakage).

    With a validation loader the prior is also scored on validation code maps every epoch,
    training stops after `prior_patience` epochs without improvement and the best weights
    are restored, so an over-fitted prior is not what gets saved.
    """
    code_maps = collect_code_maps(model, train_loader, device, p.limit_batches)
    val_maps = collect_code_maps(model, val_loader, device, p.limit_batches) if val_loader is not None else None
    height, width = code_maps.shape[1:]
    loader = DataLoader(TensorDataset(code_maps), batch_size=p.batch_size, shuffle=True)
    prior = CodePrior(
        codebook_size=p.codebook_size, hidden_channels=p.prior_hidden, height=int(height), width=int(width),
        layers=p.prior_layers, dropout=p.prior_dropout,
    ).to(device)
    print(f"[prior] transformer over a {height}x{width} code grid ({height * width} tokens), {count_parameters(prior):,} parameters")
    # AdamW + linear warmup + cosine decay: transformers are unstable at a full LR from step 0.
    def decays(name: str, tensor: torch.Tensor) -> bool:  # weight matrices only, not biases/norms/embeddings
        """True for tensors that should get weight decay: 2D weight matrices, but not embeddings
        (biases and norm parameters are 1D).
        """
        return tensor.ndim == 2 and "embedding" not in name

    # Weight decay only on weight matrices; biases, LayerNorm parameters, embeddings and the BOS vector
    # are excluded.
    decay = [q for n, q in prior.named_parameters() if decays(n, q)]
    no_decay = [q for n, q in prior.named_parameters() if not decays(n, q)]
    optimizer = torch.optim.AdamW(
        # beta2 = 0.95 is the usual GPT-style choice (reacts faster than the default 0.999).
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=p.prior_lr, betas=(0.9, 0.95)
    )
    total_steps = p.prior_epochs * len(loader)
    # Warm up over the first 10% of the steps, capped at 500.
    warmup_steps = max(1, min(500, total_steps // 10))

    def lr_factor(step: int) -> float:
        """Learning-rate multiplier for optimiser step `step`: linear warmup, then cosine decay from
        1.0 down to 0.1 of the peak rate.
        """
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Stepped once per optimiser step (not per epoch), matching total_steps and lr_factor above.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    snapshots = snapshot_epochs(p.prior_epochs)
    frames, titles = [], []
    history = []
    best_ce, best_epoch, best_state = float("inf"), 0, None
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
            # Same clip value (5.0) as GRAD_CLIP for the autoencoders.
            torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * indices.shape[0]
            count += indices.shape[0]
        row = {"epoch": epoch, "train_cross_entropy": total / count}
        message = f"[prior] epoch {epoch:03d}/{p.prior_epochs:03d} | train CE={total / count:.5f}"
        if val_maps is not None:
            row["val_cross_entropy"] = code_cross_entropy(prior, val_maps, p.batch_size, device)
            message += f" val CE={row['val_cross_entropy']:.5f}"
            if row["val_cross_entropy"] < best_ce:
                best_ce, best_epoch = row["val_cross_entropy"], epoch
                # Snapshot the weights at the best validation CE; clone() is needed because
                # the live tensors keep changing.
                best_state = {k: v.detach().clone() for k, v in prior.state_dict().items()}
        history.append(row)
        print(message)

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
        # Early stopping needs validation maps; prior_patience = 0 disables it.
        if val_maps is not None and p.prior_patience and epoch - best_epoch >= p.prior_patience:
            print(f"[prior] no validation improvement for {p.prior_patience} epochs; stopping (best epoch {best_epoch})")
            break
    elapsed = time.perf_counter() - start
    # Restore the best epoch, so the over-fitted final weights are not what gets saved.
    if best_state is not None:
        prior.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": prior.state_dict(),
            "prior_type": "transformer",
            "codebook_size": p.codebook_size,
            "hidden_channels": p.prior_hidden,
            "layers": p.prior_layers,
            "dropout": p.prior_dropout,
            "latent_height": int(height),
            "latent_width": int(width),
            "history": history,
        },
        run_dir / "prior.pt",
    )
    write_csv(run_dir / "prior_history.csv", history)
    summary = {
        "prior": "causal-transformer autoregressive prior trained on training-split code maps",
        "train_code_maps": int(code_maps.shape[0]),
        "latent_shape": [int(height), int(width)],
        "prior_parameters": count_parameters(prior),
        "prior_training_seconds": elapsed,
        "prior_epochs_run": len(history),
        "final_train_cross_entropy": history[-1]["train_cross_entropy"],
    }
    if best_state is not None:
        summary |= {"prior_best_epoch": best_epoch, "best_val_cross_entropy": best_ce}
    return summary


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
    # Device label: CUDA shows the GPU name, MPS says Apple Silicon, CPU shows nothing extra.
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else (" (Apple Silicon GPU)" if device.type == "mps" else "")))

    # Use the dataset's own train/val/test folders; masks and unreadable files are excluded.
    splits = split_from_manifest(load_manifest(manifest, data_root, min_nonzero_frac=p.min_nonzero_frac))
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

    # Everything build_model needs plus the paths. It is stored in the checkpoint so evaluation can
    # rebuild the model without params.py.
    config = asdict(p) | {"data_root": str(data_root), "manifest": str(manifest)}
    model = build_model(model_type, config).to(device)
    n_parameters = count_parameters(model)
    print(f"Model parameters: {n_parameters:,}")
    reset_peak_memory_stats(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    # Cosine decay of the learning rate to 1% of its starting value over `epochs`, stepped once per
    # epoch.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p.epochs, eta_min=p.lr * 0.01) if p.cosine_lr else None
    )
    snapshots = snapshot_epochs(p.epochs)
    frames, titles = [fixed_image], ["Input"]
    generated_frames, generated_titles = [], []
    best_ssim, best_epoch, history = -float("inf"), 0, []
    stopped_early = False
    start = time.perf_counter()
    for epoch in range(1, p.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_m = run_epoch(model, loaders["train"], model_type, device, optimizer, p.limit_batches, ssim_weight=p.ssim_weight)
        val_m = run_epoch(
            model, loaders["val"], model_type, device, None, p.limit_batches, with_ssim=True, ssim_weight=p.ssim_weight
        )
        if scheduler is not None:
            scheduler.step()
        seconds = time.perf_counter() - epoch_start
        # One CSV row per epoch: the learning rate used, then every train_* and val_* metric
        # (n_images is dropped).
        history.append(
            {"epoch": epoch, "lr": learning_rate}
            | {f"train_{k}": v for k, v in train_m.items() if k != "n_images"}
            | {f"val_{k}": v for k, v in val_m.items() if k != "n_images"}
            | {"epoch_seconds": seconds}
        )
        write_csv(run_dir / "history.csv", history)
        usage = f" | codes {val_m['active_codes']:.0f}/{p.codebook_size} ppl {val_m['perplexity']:.0f}" if "perplexity" in val_m else ""
        print(
            f"epoch {epoch:03d}/{p.epochs:03d} | train loss={train_m['loss']:.5f} "
            f"val loss={val_m['loss']:.5f} val SSIM={val_m['ssim']:.4f}{usage} | {seconds:.1f}s"
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

        # Early stopping on validation SSIM (the checkpoint-selection metric).
        stop = bool(p.patience) and epoch - best_epoch >= p.patience

        # Also snapshot on the early-stop epoch, so the progress strips end at the final state.
        if epoch in snapshots or stop:
            recon = reconstruct_one(model, fixed_image, device)
            ssim = ssim_2d(fixed_image[0].numpy(), recon[0].numpy())
            frames.append(recon)
            titles.append(f"Epoch {epoch}\nSSIM {ssim:.3f}")
            # Rewritten at every snapshot so an interrupted run still has a usable image.
            save_image_grid(torch.stack(frames), run_dir / "reconstructions_by_epoch.png", titles, ncols=len(frames))
            # Same epochs, but a newly generated image instead of a reconstruction.
            generated_frames.append(generate_snapshot_sample(model, model_type, loaders["train_eval"], device, p))
            generated_titles.append(f"Epoch {epoch}")
            save_image_grid(
                torch.stack(generated_frames), run_dir / "generated_samples_by_epoch.png",
                generated_titles, ncols=len(generated_frames),
            )

        if stop:
            stopped_early = True
            print(f"Early stopping: validation SSIM has not improved for {p.patience} epochs (best epoch {best_epoch}).")
            break
    training_seconds = time.perf_counter() - start
    plot_history(run_dir, history, model_type)

    # Final, one-off test evaluation with the best validation checkpoint.
    # Reload the best-validation weights (the live model holds the LAST epoch). weights_only=True
    # restricts loading to tensors and plain data, with no pickled code.
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_m = run_epoch(model, loaders["test"], model_type, device, None, p.limit_batches, with_ssim=True, ssim_weight=p.ssim_weight)
    save_json(run_dir / "test_metrics.json", test_m)

    if model_type == "vqvae" and train_prior:
        generation = train_code_prior(model, loaders["train_eval"], p, device, run_dir, val_loader=loaders["val"])
    elif model_type == "vae":
        latent_prior = fit_vae_latent_prior(model, loaders["train_eval"], device, p.limit_batches)
        torch.save(latent_prior, run_dir / "latent_prior.pt")
        generation = {
            "prior": "full-covariance Gaussian fitted to training-split latents (latent_prior.pt)",
            "latent_prior_train_images": latent_prior["n_images"],
        }
    else:
        generation = {"prior": "not trained"}

    summary = {
        "model": model_type,
        "params": asdict(p),
        "model_parameters": n_parameters,
        "device": str(device),
        "split_image_counts": image_counts,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
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
            "generated_samples_by_epoch.png uses the current fitted-Gaussian (VAE) or an independent per-position "
            "code prior (VQ-VAE); prior_samples_by_epoch.png shows the real transformer prior.",
        ],
    }
    save_json(run_dir / "run_summary.json", summary)
    print(f"Best validation SSIM: {best_ssim:.4f} (epoch {best_epoch}); held-out test SSIM: {test_m['ssim']:.4f}")
    print(f"Run artifacts: {run_dir}")
    return summary


def main() -> None:
    """CLI entry point: parse the model choice and params.py overrides, derive the run directory name, and
    train.
    """
    parser = argparse.ArgumentParser(description="Train a VAE or VQ-VAE on 2D HipMRI NIfTI slices")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing the keras_slices_* directories")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="CSV created by `dataset.py audit`")
    add_overrides(parser, TrainParams())  # any params.py default can be overridden, e.g. --epochs 5
    args = parser.parse_args()
    try:
        # TrainParams.__post_init__ raises ValueError for invalid combinations; report that as a
        # normal command-line usage error.
        params = apply_overrides(TrainParams(), args)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = RUNS_DIR / run_dir_name(args.model, params)
    print(f"Run directory: {run_dir.resolve()}")
    train_model(args.model, params, args.data_root, args.manifest, run_dir)


if __name__ == "__main__":
    main()
