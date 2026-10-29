"""Figure helpers for the synthetic-MRI evaluation (NumPy + Matplotlib only; images are [N, H, W] in [0, 1])."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: works on clusters
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"real_train": "#4c72b0", "real_test": "#55a868", "vae": "#dd8452", "vqvae": "#c44e52", "control": "#8172b3"}
LABELS = {
    "vae": "VAE (baseline)", "vqvae": "VQ-VAE", "real_train": "Real (train)", "real_test": "Real (held-out test)",
    "control": "Train reconstructions (memorisation control)",
}
MODELS = ("vae", "vqvae")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def read_history(path: Path) -> List[dict]:
    """history.csv / prior_history.csv -> list of dicts with floats where possible."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def _col(rows: Sequence[dict], key: str) -> Optional[np.ndarray]:
    if not rows or key not in rows[0]:
        return None
    return np.asarray([r.get(key, np.nan) for r in rows], dtype=float)


def _tile(images: np.ndarray, ncols: int, pad: int = 1) -> np.ndarray:
    n, h, w = images.shape
    nrows = int(np.ceil(n / ncols))
    canvas = np.ones((nrows * (h + pad) + pad, ncols * (w + pad) + pad), np.float32)
    for i in range(n):
        r, c = divmod(i, ncols)
        canvas[pad + r * (h + pad) : pad + r * (h + pad) + h, pad + c * (w + pad) : pad + c * (w + pad) + w] = images[i]
    return canvas


def _no_axes(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _na(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, wrap=True, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------------------
# 1. training: stability, loss formulation
# --------------------------------------------------------------------------------------
def plot_training_curves(hist: Dict[str, List[dict]], out: Path, grad_clip: float = 5.0, codebook_size: int | None = None) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    (a_loss, a_ssim, a_mse), (a_grad, a_codes, a_lr) = axes
    for name in MODELS:
        rows = hist.get(name) or []
        if not rows:
            continue
        ep, color, lab = _col(rows, "epoch"), COLORS[name], LABELS[name]
        for key, style, suffix in (("train_loss", "-", "train"), ("val_loss", "--", "val")):
            y = _col(rows, key)
            if y is not None:
                a_loss.plot(ep, y, style, color=color, label=f"{lab} {suffix}")
        ssim = _col(rows, "val_ssim")
        if ssim is not None:
            a_ssim.plot(ep, ssim, color=color, label=lab)
            best = int(np.nanargmax(ssim))
            a_ssim.scatter(ep[best], ssim[best], color=color, zorder=5, marker="*", s=120)
        for key, style, suffix in (("train_reconstruction_mse", "-", "train"), ("val_reconstruction_mse", "--", "val")):
            y = _col(rows, key)
            if y is not None:
                a_mse.plot(ep, y, style, color=color, label=f"{lab} {suffix}")
        gn = _col(rows, "train_grad_norm")
        if gn is not None:
            a_grad.plot(ep, gn, color=color, label=f"{lab} mean pre-clip norm")
            clipped = _col(rows, "train_grad_clipped")
            if clipped is not None:
                a_grad.plot(ep, clipped * grad_clip, ":", color=color, label=f"{lab} share of clipped steps (scaled x{grad_clip:g})")
        lr = _col(rows, "lr")
        if lr is not None:
            a_lr.plot(ep, lr, color=color, label=lab)
    vq = hist.get("vqvae") or []
    active, ppl = _col(vq, "val_active_codes"), _col(vq, "val_perplexity")
    if active is not None and ppl is not None:
        ep = _col(vq, "epoch")
        a_codes.plot(ep, active, color=COLORS["vqvae"], label="active codes (val)")
        a_codes.plot(ep, ppl, "--", color="black", label="perplexity (val)")
        if codebook_size:
            a_codes.axhline(codebook_size, color="gray", ls=":", label=f"codebook size ({codebook_size})")
        a_codes.legend(fontsize=8)
    else:
        _na(a_codes, "No codebook-usage columns in history.csv")
    a_ssim.axhline(0.6, color="gray", ls="--", label="target 0.6")
    a_loss.set(title="Total loss (VAE includes KL; VQ-VAE includes codebook + commitment)", yscale="log", xlabel="epoch")
    a_ssim.set(title="Validation SSIM (star = checkpoint kept)", xlabel="epoch", ylabel="SSIM")
    a_mse.set(title="Reconstruction MSE", yscale="log", xlabel="epoch")
    a_grad.set(title=f"Gradient hygiene (global norm, clip = {grad_clip:g})", xlabel="epoch")
    if not a_grad.lines:
        _na(a_grad, "grad_norm not logged: retrain with the updated train.py")
    a_codes.set(title="VQ-VAE codebook usage (collapse check)", xlabel="epoch")
    a_lr.set(title="Learning rate (cosine decay)", xlabel="epoch")
    for ax in (a_loss, a_ssim, a_mse, a_grad, a_lr):
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    a_codes.grid(alpha=0.25)
    _save(fig, out)


def plot_loss_components(hist: Dict[str, List[dict]], out: Path) -> None:
    fig, (a_vae, a_vq) = plt.subplots(1, 2, figsize=(13, 4.5))
    vae = hist.get("vae") or []
    if vae:
        ep = _col(vae, "epoch")
        for key, label, style in (
            ("train_reconstruction_mse", "recon MSE (train)", "-"), ("val_reconstruction_mse", "recon MSE (val)", "--"),
            ("train_kl_loss", "KL nats/pixel (train)", "-"), ("val_kl_loss", "KL nats/pixel (val)", "--"),
        ):
            y = _col(vae, key)
            if y is not None:
                a_vae.plot(ep, y, style, label=label)
    a_vae.set(title="VAE: reconstruction vs KL (loss = MSE + beta * KL)", yscale="log", xlabel="epoch")
    vq = hist.get("vqvae") or []
    if vq:
        ep = _col(vq, "epoch")
        for key, label, style in (
            ("train_reconstruction_mse", "recon MSE (train)", "-"), ("val_reconstruction_mse", "recon MSE (val)", "--"),
            ("train_codebook_loss", "quantisation error (train)", "-"), ("train_commitment_loss", "commitment loss (train)", "-"),
        ):
            y = _col(vq, key)
            if y is not None:
                a_vq.plot(ep, y, style, label=label)
    a_vq.set(title="VQ-VAE: reconstruction, quantisation error, commitment", yscale="log", xlabel="epoch")
    for ax in (a_vae, a_vq):
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    _save(fig, out)


def plot_prior_curves(prior_rows: List[dict], codebook_size: int, out: Path) -> None:
    if not prior_rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep = _col(prior_rows, "epoch")
    ax.plot(ep, _col(prior_rows, "train_cross_entropy"), label="train CE")
    val = _col(prior_rows, "val_cross_entropy")
    if val is not None:
        ax.plot(ep, val, "--", label="validation CE")
        best = int(np.nanargmin(val))
        ax.scatter(ep[best], val[best], marker="*", s=140, color="black", zorder=5, label=f"best epoch {int(ep[best])}")
    ax.axhline(np.log(codebook_size), color="gray", ls=":", label=f"uniform over {codebook_size} codes (ln K)")
    ax.set(title="Transformer code prior: next-code cross-entropy", xlabel="prior epoch", ylabel="nats / code")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    _save(fig, out)


def plot_resource_profile(profile: Dict[str, Dict[str, Optional[float]]], out: Path) -> None:
    panels = [
        ("params_m", "Parameters (M)"), ("train_minutes", "Training time (min)"), ("train_peak_mb", "Peak accelerator memory, training (MB)"),
        ("gen_ms_per_sample", "Generation latency (ms / sample, log)"), ("gen_peak_mb", "Peak accelerator memory, generation (MB)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8))
    for ax, (key, title) in zip(axes, panels):
        for i, name in enumerate(MODELS):
            value = (profile.get(name) or {}).get(key)
            if value is None:
                ax.text(i, 0, "n/a", ha="center", va="bottom", fontsize=9)
                continue
            ax.bar(i, value, color=COLORS[name])
            ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["VAE", "VQ-VAE"]); ax.set_title(title, fontsize=9)
        if key == "gen_ms_per_sample":
            ax.set_yscale("log")
    _save(fig, out)


# --------------------------------------------------------------------------------------
# 2. samples / reconstructions
# --------------------------------------------------------------------------------------
def plot_sample_grids(gen: np.ndarray, gen_seed: np.ndarray, name: str, out: Path, per_seed: int = 32, ncols: int = 8) -> None:
    """One panel per random seed, showing the FIRST `per_seed` images drawn (no selection of any kind)."""
    seeds = list(dict.fromkeys(int(s) for s in gen_seed))
    ncol_panels = 2 if len(seeds) > 1 else 1
    nrow_panels = int(np.ceil(len(seeds) / ncol_panels))
    fig, axes = plt.subplots(nrow_panels, ncol_panels, figsize=(6.2 * ncol_panels, 6.8 * nrow_panels), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, seed in zip(axes.flat, seeds):
        ax.imshow(_tile(gen[gen_seed == seed][:per_seed], ncols), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{LABELS[name]} | seed {seed} | first {per_seed} samples, uncurated", fontsize=9)
    _save(fig, out)


def plot_real_vs_synth(rows: Dict[str, np.ndarray], out: Path, n: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(rows), 1, figsize=(1.7 * n, 2.9 * len(rows)), squeeze=False)
    for ax, (name, images) in zip(axes[:, 0], rows.items()):
        pick = np.sort(rng.choice(len(images), min(n, len(images)), replace=False))
        ax.imshow(_tile(images[pick], n), cmap="gray", vmin=0, vmax=1)
        ax.set_title(LABELS.get(name, name), fontsize=9, loc="left")
        _no_axes(ax)
    _save(fig, out)


def plot_reconstructions(test: np.ndarray, recons: Dict[str, np.ndarray], ssims: Dict[str, np.ndarray], idx: Sequence[int], out: Path) -> None:
    n = len(idx)
    labels = ["Input"] + [x for name in MODELS if name in recons for x in (LABELS[name], f"{LABELS[name]} |error|")]
    fig, axes = plt.subplots(len(labels), n, figsize=(1.7 * n, 3.0 * len(labels) / 1.5), squeeze=False)
    for c, i in enumerate(idx):
        axes[0, c].imshow(test[i], cmap="gray", vmin=0, vmax=1)
        r = 1
        for name in MODELS:
            if name not in recons:
                continue
            axes[r, c].imshow(recons[name][i], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"SSIM {ssims[name][i]:.2f}", fontsize=7)
            axes[r + 1, c].imshow(np.abs(test[i] - recons[name][i]), cmap="inferno", vmin=0, vmax=0.3)
            r += 2
    for r, label in enumerate(labels):
        axes[r, 0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    for ax in axes.flat:
        _no_axes(ax)
    fig.suptitle("Test reconstructions (evenly spaced test slices, not selected)", fontsize=10)
    _save(fig, out)


# --------------------------------------------------------------------------------------
# 3. clarity / metrics / diversity / embedding
# --------------------------------------------------------------------------------------
def plot_clarity(spectra: Dict[str, tuple], sharp: Dict[str, np.ndarray], checker: Dict[str, np.ndarray], out: Path) -> None:
    fig, (a_spec, a_sharp, a_check) = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, (freq, power) in spectra.items():
        a_spec.plot(freq[1:], power[1:], color=COLORS[name], label=LABELS[name])
    a_spec.set(xscale="log", yscale="log", xlabel="spatial frequency (cycles/pixel)", ylabel="mean power", title="Radial power spectrum: blur = missing high frequencies")
    for ax, data, title, xlabel in (
        (a_sharp, sharp, "Sharpness (variance of Laplacian)", "log10 Laplacian variance"),
        (a_check, checker, "Checkerboard energy (Nyquist corner)", "log10 fraction of power in corner"),
    ):
        allv = np.log10(np.concatenate([np.maximum(v, 1e-12) for v in data.values()]))
        bins = np.linspace(allv.min(), allv.max(), 40)
        for name, v in data.items():
            ax.hist(np.log10(np.maximum(v, 1e-12)), bins=bins, alpha=0.5, density=True, color=COLORS[name], label=LABELS[name])
        ax.set(title=title, xlabel=xlabel)
    for ax in (a_spec, a_sharp, a_check):
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    _save(fig, out)


def plot_metric_bars(m: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    names = [("vae", "VAE"), ("vqvae", "VQ-VAE"), ("real_test", "Real test\n(reference)")]

    def bars(ax, entries, title, ylabel, target=None):
        for i, (key, label) in enumerate(names):
            if key not in entries:
                continue
            v = entries[key]
            value, lo, hi = (v if isinstance(v, (tuple, list)) else (v, None, None))
            ax.bar(i, value, color=COLORS[key], yerr=None if lo is None else [[value - lo], [hi - value]], capsize=4)
            ax.text(i, value, f"{value:.3f}" if value < 100 else f"{value:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n for _, n in names], fontsize=8)
        ax.set(title=title, ylabel=ylabel)
        if target is not None:
            ax.axhline(target, color="gray", ls="--"); ax.text(len(names) - 0.5, target, " target 0.6", fontsize=8, va="bottom", ha="right")
        ax.grid(alpha=0.25, axis="y")

    bars(axes[0], m["recon_ssim"], "Reconstruction SSIM (test, paired)", "SSIM", 0.6)
    bars(axes[1], m["match_ssim"], "Best-match SSIM to real train slices", "SSIM", 0.6)
    bars(axes[2], m["fd"], f"Frechet distance to real train\n({m['fd_name']}; lower = better)", "FD")
    ax = axes[3]
    keys = ["precision", "recall", "density", "coverage"]
    width = 0.26
    for j, (key, label) in enumerate(names):
        if key in m["prdc"]:
            ax.bar(np.arange(4) + (j - 1) * width, [m["prdc"][key][k] for k in keys], width, color=COLORS[key], label=label.replace("\n", " "))
    ax.set_xticks(range(4)); ax.set_xticklabels(keys); ax.set_title("Precision / recall / density / coverage (k=5)"); ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, out)


def plot_diversity(occ_real: np.ndarray, occ: Dict[str, np.ndarray], pair_ssim: Dict[str, np.ndarray], code_usage: Optional[dict], out: Path) -> None:
    fig, (a_occ, a_pair, a_code) = plt.subplots(1, 3, figsize=(17, 4.4))
    k = len(occ_real)
    width = 0.27
    a_occ.bar(np.arange(k) - width, occ_real, width, color=COLORS["real_train"], label="Real (train)")
    for j, name in enumerate(MODELS):
        if name in occ:
            a_occ.bar(np.arange(k) + j * width, occ[name], width, color=COLORS[name], label=LABELS[name])
    a_occ.set(title="Mode occupancy: share of images per real-data cluster", xlabel="k-means cluster of real images", ylabel="share")
    bins = np.linspace(0, 1, 40)
    for name, v in pair_ssim.items():
        a_pair.hist(v, bins=bins, alpha=0.5, density=True, color=COLORS[name], label=LABELS[name])
    a_pair.set(title="Pairwise SSIM between random image pairs\n(shifted right = less diverse)", xlabel="SSIM")
    if code_usage:
        for key, color, label in (("train", COLORS["real_train"], "codes used by training images"), ("sampled", COLORS["vqvae"], "codes in prior samples")):
            counts = np.sort(np.asarray(code_usage[key], float))[::-1]
            a_code.plot(np.arange(1, len(counts) + 1), np.maximum(counts / counts.sum(), 1e-7), color=color, label=label)
        a_code.set(title="VQ-VAE code usage (rank-frequency)", xlabel="code rank", ylabel="frequency", yscale="log")
    else:
        _na(a_code, "no code statistics")
    for ax in (a_occ, a_pair, a_code):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    _save(fig, out)


def embed_2d(features: np.ndarray, seed: int = 0):
    """t-SNE (falls back to PCA if scikit-learn is missing). Returns (embedding [N,2], method name)."""
    x = np.asarray(features, np.float64)
    x = x - x.mean(0)
    if x.shape[1] > 50:  # standard practice: PCA to 50 dims first
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        x = x @ vt[:50].T
    try:
        from sklearn.manifold import TSNE

        perplexity = float(min(30, max(5, (len(x) - 1) // 4)))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(x), "t-SNE"
    except ImportError:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return x @ vt[:2].T, "PCA (scikit-learn not installed)"


def plot_embedding(emb: np.ndarray, labels: np.ndarray, method: str, feature_name: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharex=True, sharey=True)
    order = ["real_train", "real_test", "vae", "vqvae"]
    panels = [("All", order), ("Real vs VAE", ["real_train", "real_test", "vae"]), ("Real vs VQ-VAE", ["real_train", "real_test", "vqvae"])]
    for ax, (title, keep) in zip(axes, panels):
        for name in keep:
            sel = labels == name
            ax.scatter(emb[sel, 0], emb[sel, 1], s=9, alpha=0.55, color=COLORS[name], label=f"{LABELS[name]} (n={sel.sum()})", edgecolors="none")
        ax.set_title(f"{method} of {feature_name}: {title}", fontsize=9); ax.legend(fontsize=7, markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, out)


# --------------------------------------------------------------------------------------
# 4. autopsy / memorisation / interpolation
# --------------------------------------------------------------------------------------
def plot_triplets(rows: List[dict], out: Path, suptitle: str) -> None:
    """rows: {"image", "neighbor", "title"}. Columns: generated | nearest training slice | |difference|."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(8.6, 2.9 * n), squeeze=False)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(row["neighbor"], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(np.abs(row["image"] - row["neighbor"]), cmap="inferno", vmin=0, vmax=0.5)
        axes[r, 0].set_title("\n".join(textwrap.wrap(row["title"], 46)), fontsize=7, loc="left")
        axes[r, 1].set_title("nearest training slice", fontsize=7)
        axes[r, 2].set_title("|difference|", fontsize=7)
        for ax in axes[r]:
            _no_axes(ax)
    fig.suptitle(suptitle, fontsize=10)
    _save(fig, out)


def plot_nn_audit(mse: Dict[str, np.ndarray], threshold: float, ssim: Dict[str, np.ndarray], near_dup: float,
                  copy_pairs: Dict[str, tuple], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    a_mse, a_ssim, a_copy = axes
    allv = np.log10(np.maximum(np.concatenate(list(mse.values())), 1e-9))
    bins = np.linspace(allv.min(), allv.max(), 45)
    for name, v in mse.items():
        a_mse.hist(np.log10(np.maximum(v, 1e-9)), bins=bins, alpha=0.5, density=True, color=COLORS[name], label=LABELS[name])
    a_mse.axvline(np.log10(max(threshold, 1e-9)), color="black", ls="--", label="close-match threshold\n(5th pct of real test)")
    a_mse.set(title="Distance to nearest TRAINING slice (pixel MSE)", xlabel="log10 MSE")
    bins = np.linspace(0, 1, 45)
    for name, v in ssim.items():
        a_ssim.hist(v, bins=bins, alpha=0.5, density=True, color=COLORS[name], label=LABELS[name])
    a_ssim.axvline(near_dup, color="black", ls="--", label=f"near-duplicate SSIM {near_dup:g}")
    a_ssim.axvline(0.6, color="gray", ls=":", label="SSIM 0.6")
    a_ssim.set(title="Best-match SSIM to a training slice", xlabel="SSIM")
    lim = 0
    for name, (d_train, d_test) in copy_pairs.items():
        a_copy.scatter(d_test, d_train, s=8, alpha=0.5, color=COLORS[name], label=LABELS[name])
        lim = max(lim, float(max(d_test.max(), d_train.max())))
    a_copy.plot([0, lim], [0, lim], "k--", lw=1)
    a_copy.set(title="Data-copying test: below the line = closer to a TRAIN slice\nthan to an (equally many) unseen TEST slice",
               xlabel="distance to nearest test slice", ylabel="distance to nearest train slice (equal-size subset)")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    _save(fig, out)


def plot_interpolation(paths: Dict[str, List[dict]], out: Path) -> None:
    rows = [(name, p) for name in MODELS for p in paths.get(name, [])]
    if not rows:
        return
    n_frames = len(rows[0][1]["frames"])
    fig, axes = plt.subplots(len(rows), n_frames, figsize=(1.45 * n_frames, 2.6 * len(rows)), squeeze=False)
    for r, (name, path) in enumerate(rows):
        for c, frame in enumerate(path["frames"]):
            axes[r, c].imshow(frame, cmap="gray", vmin=0, vmax=1)
            _no_axes(axes[r, c])
            if r == 0:
                axes[r, c].set_title(f"a={c / (n_frames - 1):.2f}", fontsize=7)
        axes[r, 0].set_ylabel(f"{LABELS[name]}\n{path['label']}", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("Latent interpolation: synthetic sample (left) -> real training slice (right)", fontsize=10)
    _save(fig, out)
