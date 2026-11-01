"""Sweep one hyper-parameter for the VAE and the VQ-VAE and evaluate its effect on both.

    # fast: reconstruction quality + cost only (no prior, no sampling)
    python3 experiments.py --param ssim-weight --values 0 0.1 0.25 0.5 --epochs 30 --data-root <keras_slices_data>

    # full: also trains the VQ-VAE code prior, samples images and scores them (needs --with-generation)
    python3 experiments.py --param codebook-size --values 64 128 256 512 1024 \
        --with-generation --repeats 2 --epochs 30 --prior-epochs 20 --data-root <keras_slices_data>

Every (model, value, repeat) is trained with the params.py defaults plus any other overrides given here
(e.g. --epochs 30), so only the swept parameter differs. Repeats use seeds seed, seed+1, ...; with
--repeats > 1 every plot shows mean +/- std and the report says whether a difference is larger than seed noise.
Parameters that exist for one model only (kl-beta, vae-latent-dim, downsample-stages for the VAE;
codebook-size, latent-dim, vq-downsample-stages, res-blocks, commitment-beta, ... for the VQ-VAE) are swept
for that model alone. Finished runs are reused unless --rerun.

Why --with-generation matters: reconstruction SSIM is a paired autoencoder score. A setting can improve
it and still hurt generation (a finer code grid reconstructs better but is harder for the prior to model), so
the generation metrics are what decide the project question. They are computed in a fixed pixel-PCA feature
space shared by every run (no Inception download, identical across sweeps) against the same real reference
slices, and every metric is also computed for held-out real slices as a "real floor".

Outputs in experiments/<param>.../ (or --output-dir):
    results.csv, summary.csv, real_floor.json, sweep_config.json, <param>_report.md
    <param>_effect.png         reconstruction SSIM / MSE, training time, memory, parameters, best epoch
    <param>_learning_curves.png validation SSIM and MSE per epoch for every value
    <param>_codebook.png       VQ-VAE codebook usage (active codes, perplexity)   [VQ-VAE sweeps]
    <param>_generation.png     realism, diversity, memorisation, sharpness, artefacts [--with-generation]
    <param>_tradeoff.png       reconstruction vs generation, and precision vs recall   [--with-generation]
    <param>_samples.png        uncurated samples per value                              [--with-generation]
"""

from __future__ import annotations

import argparse
import numbers
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import eval_metrics as M
from eval_plots import _tile, read_history
from params import (
    EXPERIMENTS_DIR, MANIFEST, MODEL_SPECIFIC, MODELS, TrainParams, add_overrides, apply_overrides, run_dir_name,
)
from train import train_model
from utils import read_json, save_json, write_csv

LABELS = {"vae": "VAE", "vqvae": "VQ-VAE"}
COLORS = {"vae": "#dd8452", "vqvae": "#c44e52"}
SSIM_TARGET = 0.6
NOT_SWEEPABLE = {"limit_batches", "workers", "cache_data", "image_height", "image_width"}  # debug / data-shape settings

# (title, results key, reference-line value or None, reference label)
EFFECT_PANELS = [
    ("Held-out test SSIM (higher is better)", "test_ssim", SSIM_TARGET, "target 0.6"),
    ("Held-out test MSE (lower is better)", "test_mse", None, ""),
    ("Autoencoder training time (s)", "training_seconds", None, ""),
    ("Peak accelerator memory, training (MB)", "peak_mb", None, ""),
    ("Autoencoder parameters", "parameters", None, ""),
    ("Epoch of best validation SSIM", "best_epoch", None, ""),
]
GEN_PANELS = [
    ("Frechet distance, pixel-PCA (lower is better)", "gen_fd", "floor", "real-test floor"),
    ("Precision: realism (higher is better)", "gen_precision", "floor", "real-test floor"),
    ("Recall: diversity (higher is better)", "gen_recall", "floor", "real-test floor"),
    ("Coverage (higher is better)", "gen_coverage", "floor", "real-test floor"),
    ("Best-match SSIM to real slices (target > 0.6)", "gen_match_ssim", SSIM_TARGET, "target 0.6"),
    ("Close-match rate to training slices (5% expected)", "gen_close_match_rate", 0.05, "expected for unseen real data"),
    ("Sharpness ratio to real (1 = real)", "gen_sharpness_ratio", 1.0, "real"),
    ("Checkerboard-artefact rate (5% expected)", "gen_checker_rate", 0.05, "expected for real data"),
]
# +1 higher is better, -1 lower is better, 0 closer to 1 is better.
DIRECTION = {
    "test_ssim": 1, "test_mse": -1, "val_ssim": 1, "gen_fd": -1, "gen_precision": 1, "gen_recall": 1, "gen_coverage": 1, "gen_density": 1,
    "gen_match_ssim": 1, "gen_close_match_rate": -1, "gen_near_dup_rate": -1, "gen_checker_rate": -1, "gen_sharpness_ratio": 0,
    "training_seconds": -1, "gen_ms_per_sample": -1, "prior_best_val_ce": -1,
}
NON_METRIC = {"model", "value", "repeat", "seed", "run_dir"}


# --------------------------------------------------------------------------------------
# sweep setup
# --------------------------------------------------------------------------------------
def sweepable() -> Dict[str, type]:
    """Numeric TrainParams fields (flag name -> int/float type), minus debug and data-shape settings."""
    defaults = TrainParams()
    return {
        f.name.replace("_", "-"): type(getattr(defaults, f.name))
        for f in fields(defaults)
        if type(getattr(defaults, f.name)) in (int, float) and f.name not in NOT_SWEEPABLE
    }


# --------------------------------------------------------------------------------------
# generation metrics (NumPy only; shared reference context so every run is scored identically)
# --------------------------------------------------------------------------------------
def _features(images: np.ndarray, ctx: dict) -> np.ndarray:
    """Project images into the shared pixel-PCA space stored in ctx (4x average pooling, then the PCA
    fitted on the reference slices).
    """
    mean, comps = ctx["pca"]
    return M.pca_apply(M.avg_pool(images, 4).reshape(len(images), -1), mean, comps)


def make_context(ref: np.ndarray, test: np.ndarray, n_gen: int, seed: int = 0) -> dict:
    """Reference statistics computed once: pixel-PCA space, real-test floors, close-match threshold."""
    rng = np.random.default_rng(seed)
    # n = the common subset size, so every precision / recall computation compares equal-size sets.
    n = int(min(n_gen, len(test), len(ref)))
    if n <= 6:
        raise ValueError("Need more than 6 reference and test images")
    pooled = M.avg_pool(ref, 4).reshape(len(ref), -1)
    ctx = {"n": n, "ref": ref}
    # At most (samples - 1) components exist for that many samples.
    ctx["pca"] = M.pca_fit(pooled, min(64, len(ref) - 1))
    ref_f, test_f = _features(ref, ctx), _features(test, ctx)
    ref_sub = np.sort(rng.choice(len(ref), n, replace=False))
    test_sub = np.sort(rng.choice(len(test), n, replace=False))
    # Real test slices against the reference pool: gives the close-match threshold and the real floor
    # for best-match SSIM.
    d_test, ix_test = M.nn_topk(test, ref, k=5)
    best_test, _ = M.best_match_ssim(test, ref, ix_test)
    ctx.update(
        ref_f=ref_f, ref_sub_f=ref_f[ref_sub], close_threshold=float(np.quantile(d_test[:, 0], 0.05)),
        sharp_median=float(np.median(M.sharpness(test))), checker_p95=float(np.quantile(M.checkerboard_score(test), 0.95)),
    )
    prdc = M.prdc(ref_f[ref_sub], test_f[test_sub])
    # 'Real floors': the same metrics computed for held-out REAL slices, i.e. the best a perfect
    # generator could expect.
    ctx["floors"] = {
        "gen_fd": M.frechet_distance(test_f, ref_f), "gen_precision": prdc["precision"], "gen_recall": prdc["recall"],
        "gen_coverage": prdc["coverage"], "gen_density": prdc["density"], "gen_match_ssim": float(best_test.mean()),
        "gen_pairwise_ssim": float(M.pairwise_ssim(test, 200, seed).mean()),
    }
    return ctx


def generation_metrics(gen: np.ndarray, codes: Optional[np.ndarray], ctx: dict, seed: int = 0) -> Dict[str, float]:
    """Score a synthetic cohort ([N,H,W] in [0,1]) against the shared real reference."""
    feats = _features(gen, ctx)
    d, ix = M.nn_topk(gen, ctx["ref"], k=5)
    best, _ = M.best_match_ssim(gen, ctx["ref"], ix)
    # Equal-size sets: the reference subset (n images) against the first n generated images.
    prdc = M.prdc(ctx["ref_sub_f"], feats[: ctx["n"]])
    out = {
        "gen_fd": M.frechet_distance(feats, ctx["ref_f"]),
        "gen_precision": prdc["precision"], "gen_recall": prdc["recall"], "gen_density": prdc["density"], "gen_coverage": prdc["coverage"],
        "gen_match_ssim": float(best.mean()), "gen_match_ssim_above_target": float((best > SSIM_TARGET).mean()),
        # 0.95 is the near-duplicate SSIM threshold (same as NEAR_DUP_SSIM in eval_analysis.py).
        "gen_close_match_rate": float((d[:, 0] <= ctx["close_threshold"]).mean()), "gen_near_dup_rate": float((best >= 0.95).mean()),
        "gen_sharpness_ratio": float(np.median(M.sharpness(gen)) / max(ctx["sharp_median"], 1e-12)),
        "gen_checker_rate": float((M.checkerboard_score(gen) > ctx["checker_p95"]).mean()),
        "gen_pairwise_ssim": float(M.pairwise_ssim(gen, 200, seed).mean()),
    }
    if codes is not None:
        out["gen_code_perplexity"] = M.entropy_perplexity(np.bincount(np.asarray(codes).ravel()))
    return out


def build_context(args, base: TrainParams) -> dict:
    """Load the real reference/test slices once (the split is identical for every run in the sweep)."""
    from dataset import load_manifest, split_from_manifest
    from evaluate import load_images

    data_root, manifest = args.data_root.expanduser().resolve(), args.manifest.expanduser().resolve()
    splits = split_from_manifest(load_manifest(manifest, data_root, min_nonzero_frac=base.min_nonzero_frac))
    rng = np.random.default_rng(0)
    rows = splits["train"]
    if args.ref_images and len(rows) > args.ref_images:
        # Optional random subset of the training slices (sorted to keep file order) to bound memory
        # and time.
        rows = [rows[i] for i in np.sort(rng.choice(len(rows), args.ref_images, replace=False))]
    print(f"Loading {len(rows)} reference and {len(splits['test'])} test slices for the generation metrics ...")
    hw = (base.image_height, base.image_width)
    ref, _, _ = load_images(rows, data_root, *hw, base.workers)
    test, _, _ = load_images(splits["test"], data_root, *hw, base.workers)
    return make_context(ref, test, args.gen_samples)


def _free_memory() -> None:
    """Release memory between runs: run Python garbage collection and, when PyTorch with CUDA is available,
    empty the CUDA cache.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # PyTorch is not installed: nothing more to free.
    except ImportError:
        pass


def sample_and_score(run_dir: Path, params: TrainParams, ctx: dict, args) -> Dict[str, float]:
    """Sample from a trained run (uncurated, fixed seed) and score it. Cached in generation_metrics.json."""
    cache = run_dir / "generation_metrics.json"
    # Everything that influences the cached numbers; the cache is only reused when all of these match.
    settings = {"gen_samples": args.gen_samples, "gen_seed": args.gen_seed, "temperature": args.temperature, "top_k": args.top_k, "ref_n": ctx["n"]}
    if cache.is_file() and not args.rerun:
        stored = read_json(cache)
        # Same settings: reuse the cache and return the metrics without the settings keys.
        if {k: stored.get(k) for k in settings} == settings:
            return {k: v for k, v in stored.items() if k not in settings}
    from device_utils import get_device
    from evaluate import generate_cohort, load_run

    device = get_device(params.device)
    run = load_run(run_dir, device)
    cohort = generate_cohort(run, (params.image_height, params.image_width), [args.gen_seed], args.gen_samples, device, args.temperature, args.top_k)
    codes = cohort["latents"].numpy() if run["type"] == "vqvae" else None
    metrics = generation_metrics(cohort["gen"], codes, ctx)
    metrics["gen_ms_per_sample"] = 1000.0 * cohort["gen_seconds_per_sample"]
    metrics["gen_peak_mb"] = cohort["gen_peak_mb"]
    # Keep 16 samples in half precision for the later samples figure.
    np.save(run_dir / "samples_preview.npy", cohort["gen"][:16].astype(np.float16))
    # Stored together with the settings so the cache can be validated next time.
    save_json(cache, metrics | settings)
    # Drop the model before the next training run starts, then free memory.
    del run
    _free_memory()
    return metrics


# --------------------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------------------
def run_one(model: str, params: TrainParams, args, root: Path, value: float, repeat: int, ctx: Optional[dict]) -> Dict[str, object]:
    """Train (or reuse) one run and return every number the figures and the report need."""
    run_dir = root / "runs" / run_dir_name(model, params)
    summary_path = run_dir / "run_summary.json"
    need_prior = args.with_generation and model == "vqvae"
    # A finished run is reused unless --rerun is given; with generation enabled, a VQ-VAE run must also
    # have its prior (prior.pt).
    reusable = summary_path.is_file() and not args.rerun and (not need_prior or (run_dir / "prior.pt").is_file())
    if reusable:
        print(f"Reusing finished run: {run_dir}")
        summary = read_json(summary_path)
    else:
        summary = train_model(model, params, args.data_root, args.manifest, run_dir, train_prior=need_prior)
    test, val = summary["held_out_test_metrics"], summary["best_validation_metrics"]
    row: Dict[str, object] = {
        "model": model, "value": value, "repeat": repeat, "seed": params.seed,
        "test_ssim": test["ssim"], "test_mse": test["reconstruction_mse"], "val_ssim": val["ssim"],
        "best_epoch": summary["best_epoch"], "epochs_run": summary.get("epochs_run"),
        "training_seconds": summary["training_wall_seconds"], "parameters": summary["model_parameters"],
        "peak_mb": summary.get("peak_accelerator_memory_mb"), "run_dir": str(run_dir),
    }
    if "perplexity" in test:  # VQ-VAE codebook usage on the held-out test split
        row |= {"test_perplexity": test["perplexity"], "test_active_codes": test["active_codes"]}
    history = read_history(run_dir / "history.csv")
    if history and "train_grad_norm" in history[0]:
        row["grad_norm_mean"] = float(np.mean([h["train_grad_norm"] for h in history]))
        row["clip_rate"] = float(np.mean([h.get("train_grad_clipped", 0.0) for h in history]))
    generation = summary.get("generation", {})
    if need_prior and "prior_parameters" in generation:
        row |= {
            "prior_parameters": generation["prior_parameters"], "prior_seconds": generation.get("prior_training_seconds"),
            "prior_best_val_ce": generation.get("best_val_cross_entropy"), "prior_epochs_run": generation.get("prior_epochs_run"),
            "total_seconds": summary.get("total_training_plus_prior_wall_seconds"),
        }
    if ctx is not None:
        row |= sample_and_score(run_dir, params, ctx, args)
    return row


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------
def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[Tuple[str, float], Dict[str, Tuple[float, float, int]]]:
    """(model, value) -> metric -> (mean, std, n) over repeats (None / NaN ignored; std uses ddof=1)."""
    groups: Dict[Tuple[str, float], List[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["value"]), []).append(r)
    out = {}
    for key, members in groups.items():
        stats = {}
        # Union of the metric names over the repeats (rows can differ, e.g. the VAE has no codebook
        # statistics).
        for metric in {k for r in members for k in r if k not in NON_METRIC}:
            # Keep only finite real numbers: bool is a Real subclass so it is excluded
            # explicitly, and None / NaN are ignored.
            vals = np.asarray([r[metric] for r in members if isinstance(r.get(metric), numbers.Real) and not isinstance(r.get(metric), bool) and np.isfinite(r[metric])], float)
            if len(vals):
                # Sample standard deviation (ddof=1); 0.0 when there is a single repeat.
                stats[metric] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, int(len(vals)))
        out[key] = stats
    return out


def _series(agg, model: str, key: str, values: Sequence[float]):
    """Collect (x values, means, stds) of one metric for one model across the swept values; three empty
    arrays when nothing exists.
    """
    pts = [(v, *agg[(model, v)][key][:2]) for v in values if (model, v) in agg and key in agg[(model, v)]]
    # zip(*pts) transposes [(value, mean, std), ...] into (values, means, stds).
    return tuple(map(np.asarray, zip(*pts))) if pts else (np.array([]),) * 3


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def _plt():
    """Import matplotlib lazily with the headless Agg backend and return pyplot."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend for clusters
    import matplotlib.pyplot as plt

    return plt


def _style_axis(ax, values, log_x: bool, param: str) -> None:
    """Common x-axis styling: optional log scale, ticks exactly at the swept values, a label and a light
    grid.
    """
    if log_x:
        ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:g}" for v in values], fontsize=8)
    # No minor ticks: on a log axis they would clutter the labelled swept values.
    ax.minorticks_off()
    ax.set_xlabel(param)
    ax.grid(alpha=0.2)


def _panel(ax, agg, models, values, key: str, title: str, ref=None, ref_label: str = "", floors: Optional[dict] = None, log_y: bool = False) -> bool:
    """Plot mean +/- std of one metric for every model; add a dashed reference line (a fixed value, or the
    real-data floor when ref == 'floor'). Returns True if any data was drawn, otherwise the panel shows
    'n/a'.
    """
    drawn = False
    for model in models:
        x, y, s = _series(agg, model, key, values)
        if len(x):
            # yerr is None when every std is 0 (single repeat), which draws no error bars.
            ax.errorbar(x, y, yerr=s if s.any() else None, marker="o", capsize=3, color=COLORS[model], label=LABELS[model])
            drawn = True
    if not drawn:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
    else:
        # ref == 'floor' means: use the real-data floor for this metric; otherwise ref is the line
        # value itself (or None).
        line = (floors or {}).get(key) if ref == "floor" else ref
        if line is not None:
            ax.axhline(line, color="gray", ls="--", lw=1, label=ref_label)
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    return drawn


def plot_effect(agg, param, models, values, log_x, path: Path) -> None:
    """2x3 figure of reconstruction quality and cost (test SSIM / MSE, training time, memory, parameters,
    best epoch) versus the swept value.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, key, ref, label) in zip(axes.flat, EFFECT_PANELS):
        _panel(ax, agg, models, values, key, title, ref, label, log_y=key in ("test_mse", "parameters"))
        _style_axis(ax, values, log_x, param)
    fig.suptitle(f"Effect of {param}: reconstruction quality and cost (mean +/- std over repeats)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_codebook(agg, param, values, log_x, path: Path) -> bool:
    """Figure of VQ-VAE codebook usage (active codes, perplexity, perplexity of sampled codes) versus the
    swept value; returns False when no such data exists.
    """
    keys = [("test_active_codes", "Active codes (test split)"), ("test_perplexity", "Codebook perplexity (test split)"), ("gen_code_perplexity", "Perplexity of codes sampled by the prior")]
    if not any(len(_series(agg, "vqvae", k, values)[0]) for k, _ in keys):
        return False
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, keys):
        _panel(ax, agg, ["vqvae"], values, key, title)
        if param == "codebook-size":
            # The y = x line: perfect codebook usage means every code of the swept size is
            # active.
            ax.plot(values, values, ":", color="gray", label="codebook size")
            ax.legend(fontsize=7)
        _style_axis(ax, values, log_x, param)
    fig.suptitle(f"VQ-VAE codebook usage vs {param} (collapse = far below the codebook size)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_generation(agg, param, models, values, log_x, floors, path: Path) -> None:
    """2x4 figure of the generation metrics (realism, diversity, memorisation, sharpness, artefacts) versus
    the swept value, with real-data references.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, (title, key, ref, label) in zip(axes.flat, GEN_PANELS):
        _panel(ax, agg, models, values, key, title, ref, label, floors)
        _style_axis(ax, values, log_x, param)
    fig.suptitle(f"Effect of {param} on generated images (pixel-PCA features; dashed = real-data reference)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_curves(rows, param, models, values, path: Path) -> None:
    """Validation SSIM and validation MSE per epoch for every swept value (first repeat only), one column
    per model.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, len(models), figsize=(7.5 * len(models), 8), squeeze=False)
    # One colour per swept value; stopping at 0.9 avoids the pale yellow end of the colour map.
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(values)))
    for c, model in enumerate(models):
        for color, value in zip(colors, values):
            # Curves come from the first repeat only (repeat == 0).
            run = next((r for r in rows if r["model"] == model and r["value"] == value and r["repeat"] == 0), None)
            hist = read_history(Path(run["run_dir"]) / "history.csv") if run else []
            if not hist:
                continue
            ep = [h["epoch"] for h in hist]
            axes[0, c].plot(ep, [h["val_ssim"] for h in hist], color=color, label=f"{param}={value:g}")
            axes[1, c].plot(ep, [h["val_reconstruction_mse"] for h in hist], color=color)
        axes[0, c].axhline(SSIM_TARGET, color="gray", ls="--", lw=1)
        axes[0, c].set(title=f"{LABELS[model]}: validation SSIM", xlabel="epoch")
        axes[1, c].set(title=f"{LABELS[model]}: validation reconstruction MSE", xlabel="epoch", yscale="log")
        axes[0, c].legend(fontsize=7)
        for ax in axes[:, c]:
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tradeoff(agg, param, models, values, path: Path, floors: dict) -> None:
    """Reconstruction vs generation, and the precision/recall plane; every point is annotated with its value."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    for model in models:
        pts = [(v, agg[(model, v)]) for v in values if (model, v) in agg and "gen_fd" in agg[(model, v)] and "test_ssim" in agg[(model, v)]]
        if not pts:
            continue
        for ax, kx, ky in ((a, "test_ssim", "gen_fd"), (b, "gen_recall", "gen_precision")):
            xs, ys = [p[1][kx][0] for p in pts], [p[1][ky][0] for p in pts]
            ax.plot(xs, ys, "-o", color=COLORS[model], label=LABELS[model], alpha=0.8)
            for (v, _), x, y in zip(pts, xs, ys):
                ax.annotate(f"{v:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    a.set(xlabel="test reconstruction SSIM (higher is better)", ylabel="Frechet distance of samples (lower is better)", title=f"Reconstruction vs generation ({param} annotated)")
    a.axhline(floors["gen_fd"], color="gray", ls="--", lw=1, label="real-test floor")
    a.axvline(SSIM_TARGET, color="gray", ls=":", lw=1)
    b.set(xlabel="recall (diversity)", ylabel="precision (realism)", title="Precision / recall plane (up and right is better)")
    b.plot([floors["gen_recall"]], [floors["gen_precision"]], "k*", ms=12, label="real-test floor")
    for ax in (a, b):
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_samples(rows, param, models, values, path: Path) -> bool:
    """One strip per (model, value) with the first 8 stored samples of the first repeat; returns False when
    no sample previews exist.
    """
    entries = []
    for model in models:
        for value in values:
            run = next((r for r in rows if r["model"] == model and r["value"] == value and r["repeat"] == 0), None)
            preview = Path(run["run_dir"]) / "samples_preview.npy" if run else None
            if preview is not None and preview.is_file():
                entries.append((f"{LABELS[model]}\n{param}={value:g}", np.load(preview).astype(np.float32)[:8]))
    if not entries:
        return False
    plt = _plt()
    fig, axes = plt.subplots(len(entries), 1, figsize=(13, 3.3 * len(entries)), squeeze=False)
    for ax, (label, imgs) in zip(axes[:, 0], entries):
        ax.imshow(_tile(imgs, 8), cmap="gray", vmin=0, vmax=1)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"First 8 samples of the first repeat for every {param} (uncurated)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def _fmt(stat, digits: int = 4) -> str:
    """Format a (mean, std, n) statistic as 'mean +/- std' (std only when n > 1); None becomes 'n/a'."""
    if stat is None:
        return "n/a"
    mean, std, n = stat
    text = f"{mean:.{digits}g}"
    return f"{text} ± {std:.2g}" if n > 1 else text


def _score(key: str, mean: float) -> float:
    """Sort key where larger is always better: direction * mean, or the negative distance from 1 for
    metrics whose best value is 1 (direction 0).
    """
    direction = DIRECTION.get(key, 1)
    return -abs(mean - 1.0) if direction == 0 else direction * mean


def _trend(values: Sequence[float], means: List[float]) -> str:
    """Describe in words how a metric changes across the swept values (flat, monotonic, interior maximum /
    minimum or non-monotonic).
    """
    if len(means) < 3:
        return "too few values for a trend"
    means = list(means)
    # Peak-to-peak range essentially zero relative to the scale of the values: the metric is flat.
    if np.ptp(means) <= 1e-6 * max(abs(m) for m in means) + 1e-12:
        return "is flat"
    d = np.diff(means)
    if (d >= 0).all():
        return "increases monotonically"
    if (d <= 0).all():
        return "decreases monotonically"
    best, worst = int(np.argmax(means)), int(np.argmin(means))
    if 0 < best < len(means) - 1:
        return f"peaks at {values[best]:g} (interior maximum)"
    if 0 < worst < len(means) - 1:
        return f"is lowest at {values[worst]:g} (interior minimum)"
    return "is non-monotonic"


def write_report(path: Path, param: str, rows, agg, models, values, floors: Optional[dict], info: dict) -> None:
    """Write the markdown report: per-model result tables, best value per criterion (with a seed-noise
    check), trends, a suggested value under the project criteria, and caveats.
    """
    w: List[str] = []
    cols = [("test_ssim", "test SSIM"), ("test_mse", "test MSE"), ("best_epoch", "best ep"), ("training_seconds", "train s"), ("parameters", "params")]
    if any("test_active_codes" in a for a in agg.values()):
        cols += [("test_active_codes", "active codes"), ("test_perplexity", "perplexity")]
    if floors is not None:
        cols += [("gen_match_ssim", "match SSIM"), ("gen_fd", "FD"), ("gen_precision", "precision"), ("gen_recall", "recall"), ("gen_coverage", "coverage"),
                 ("gen_close_match_rate", "close-match"), ("gen_sharpness_ratio", "sharpness ratio"), ("gen_checker_rate", "checkerboard"), ("prior_best_val_ce", "prior val CE")]
    w.append(f"# Hyper-parameter sweep: `{param}`\n")
    w.append(f"*Generated by `experiments.py`. Values: {', '.join(f'{v:g}' for v in values)}; repeats per value: {info['repeats']} "
             f"(seeds {info['seeds']}); epochs: {info['epochs']}; generation metrics: {'yes' if floors is not None else 'no (reconstruction only)'}. "
             "Entries are mean ± std over repeats.*\n")
    for model in models:
        w.append(f"## {LABELS[model]}\n")
        w.append("| " + " | ".join([param] + [c for _, c in cols]) + " |")
        w.append("|" + "|".join("---" for _ in range(len(cols) + 1)) + "|")
        for v in values:
            a = agg.get((model, v), {})
            w.append("| " + " | ".join([f"{v:g}"] + [_fmt(a.get(k)) for k, _ in cols]) + " |")
        w.append("")
    if floors is not None:
        w.append("Real-data floor (held-out real slices scored the same way): " + ", ".join(f"{k.replace('gen_', '')} {v:.3g}" for k, v in floors.items()) + ".\n")

    w.append("## Best value per criterion\n")
    w.append("| Criterion | " + " | ".join(LABELS[m] for m in models) + " |")
    w.append("|---|" + "|".join("---" for _ in models) + "|")
    for key, name in [("test_ssim", "Test SSIM (higher)"), ("test_mse", "Test MSE (lower)")] + (
        [("gen_fd", "Sample FD (lower)"), ("gen_precision", "Precision (higher)"), ("gen_recall", "Recall (higher)"), ("gen_match_ssim", "Best-match SSIM (higher)"),
         ("gen_close_match_rate", "Close-match rate (lower)"), ("gen_sharpness_ratio", "Sharpness closest to real")] if floors is not None else []):
        cells = []
        for m in models:
            cand = [(v, agg[(m, v)][key]) for v in values if (m, v) in agg and key in agg[(m, v)]]
            if not cand:
                cells.append("n/a")
                continue
            # Best candidate first, according to each metric's direction (higher / lower /
            # closest to 1).
            cand.sort(key=lambda c: _score(key, c[1][0]), reverse=True)
            note = ""
            if len(cand) > 1:
                (_, best), (_, second) = cand[0], cand[1]
                if best[2] < 2:
                    note = "single seed: noise unknown"
                else:
                    # Rule of thumb: a gap larger than twice the bigger seed std counts
                    # as a real difference.
                    gap, noise = abs(best[0] - second[0]), max(best[1], second[1])
                    note = "distinguishable from runner-up" if gap > 2 * noise else "within seed noise of runner-up"
            cells.append(f"**{cand[0][0]:g}** ({_fmt(cand[0][1])}; {note})" if note else f"**{cand[0][0]:g}**")
        w.append(f"| {name} | " + " | ".join(cells) + " |")
    w.append("")

    w.append("## Trends\n")
    for m in models:
        for key, name in [("test_ssim", "test SSIM"), ("gen_fd", "sample FD"), ("gen_recall", "recall"), ("gen_close_match_rate", "close-match rate"), ("test_perplexity", "codebook perplexity"), ("training_seconds", "training time")]:
            have = [v for v in values if (m, v) in agg and key in agg[(m, v)]]
            means = [agg[(m, v)][key][0] for v in have]
            if len(means) >= 2:
                w.append(f"- {LABELS[m]} {name} {_trend(have, means)} ({means[0]:.4g} at {have[0]:g} to {means[-1]:.4g} at {have[-1]:g}).")
    w.append("")

    w.append("## Suggested value under the project criteria\n")
    w.append(f"Rule (fixed in advance): test SSIM > {SSIM_TARGET}" + ("; best-match SSIM > 0.6; close-match rate <= 10%; then the lowest sample FD wins." if floors is not None else "; then the highest test SSIM wins (no generation metrics were computed)."))
    for m in models:
        ok, failures = [], []
        for v in values:
            a = agg.get((m, v), {})
            why = []
            if "test_ssim" not in a or a["test_ssim"][0] <= SSIM_TARGET:
                why.append("test SSIM <= 0.6")
            if floors is not None:
                if "gen_match_ssim" not in a or a["gen_match_ssim"][0] <= SSIM_TARGET:
                    why.append("best-match SSIM <= 0.6")
                if a.get("gen_close_match_rate", (1.0,))[0] > 0.10:
                    why.append("close-match rate > 10%")
            # Expression used as a statement: record the value as failed (with its reasons) or
            # as qualifying.
            (failures.append(f"{v:g} ({', '.join(why)})") if why else ok.append(v))
        if not ok:
            w.append(f"- {LABELS[m]}: no value meets the criteria. Failures: {'; '.join(failures)}.")
        else:
            # With generation metrics the lowest sample FD wins; without them the highest test
            # SSIM wins.
            best = min(ok, key=lambda v: agg[(m, v)]["gen_fd"][0]) if floors is not None else max(ok, key=lambda v: agg[(m, v)]["test_ssim"][0])
            w.append(f"- {LABELS[m]}: **{best:g}** (qualifying values: {', '.join(f'{v:g}' for v in ok)}" + (f"; rejected: {'; '.join(failures)}" if failures else "") + ").")
    w.append("\n## Caveats\n")
    w.append("- Differences smaller than the seed standard deviation are not evidence; use `--repeats 3` or more for conclusions.\n"
             "- Generation metrics use pixel-PCA features and small cohorts, so absolute values are not comparable with published FID; compare values with each other and with the real floor.\n"
             "- A low close-match rate does not prove privacy; run `evaluate.py` on the chosen configuration for the full memorisation audit.")
    path.write_text("\n".join(w) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: validate the sweep, train / reuse a run for every (repeat, value, model), score
    generations if requested, then aggregate, plot and write the report and a console summary.
    """
    options = sweepable()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--param", required=True, choices=sorted(options), help="parameter to sweep")
    parser.add_argument("--values", nargs="+", required=True, help="two or more values to try")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: experiments/<param>_seed<S>x<E>")
    parser.add_argument("--repeats", type=int, default=1, help="independent seeds per value (seed, seed+1, ...); gives error bars")
    parser.add_argument("--with-generation", action="store_true", help="also train the VQ-VAE prior, sample images and score them")
    parser.add_argument("--gen-samples", type=int, default=256, help="images sampled per run for the generation metrics")
    parser.add_argument("--gen-seed", type=int, default=1, help="sampling seed (same for every run)")
    parser.add_argument("--temperature", type=float, default=1.0, help="VQ-VAE prior temperature (default: untuned)")
    parser.add_argument("--top-k", type=int, default=0, help="VQ-VAE prior top-k (default: off)")
    parser.add_argument("--ref-images", type=int, default=4000, help="training slices used as the real reference pool (0 = all)")
    parser.add_argument("--rerun", action="store_true", help="retrain / re-score even if finished results exist")
    add_overrides(parser, TrainParams())  # settings shared by every run in the sweep
    args = parser.parse_args()

    name = args.param.replace("-", "_")
    # int or float: the parsed values are cast to this type.
    kind = options[args.param]
    try:
        base = apply_overrides(TrainParams(), args)
        values = [float(v) for v in args.values]
        if kind is int and not all(v.is_integer() for v in values):
            raise ValueError(f"--param {args.param} needs integer values")
        # De-duplicate and sort the swept values.
        values = sorted({kind(v) for v in values})
        if len(values) < 2:
            raise ValueError("--values needs at least two different values")
        if args.repeats < 1 or (name == "seed" and args.repeats > 1):
            raise ValueError("--repeats must be >= 1 and cannot be combined with --param seed")
        if name.startswith("prior_") and not args.with_generation:
            raise ValueError("prior-* parameters only affect generation: add --with-generation")
        if args.gen_samples < 16 or (args.ref_images and args.ref_images < 16) or args.top_k < 0 or args.temperature <= 0:
            raise ValueError("--gen-samples and --ref-images (unless 0) must be >= 16, --top-k >= 0 and --temperature > 0")
        for value in values:  # fail before any training if a value is invalid
            replace(base, **{name: value})
    except ValueError as exc:
        parser.error(str(exc))

    # The swept parameter is already in the folder name, so seed/epochs are only added when not swept.
    suffix = "" if name in ("seed", "epochs") else f"_seed{base.seed}x{base.epochs}"
    out_dir = (args.output_dir or EXPERIMENTS_DIR / f"{name.replace('_', '-')}{suffix}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # A parameter owned by one model is swept for that model only; shared parameters are not in
    # MODEL_SPECIFIC, so the default (m) selects both models.
    models = [m for m in MODELS if MODEL_SPECIFIC.get(name, m) == m]
    seeds = [base.seed + r for r in range(args.repeats)]
    print(f"Sweeping {args.param} over {values} for: {', '.join(LABELS[m] for m in models)} | seeds {seeds}"
          + (" | with generation metrics" if args.with_generation else " | reconstruction only"))
    if name.startswith("prior_"):
        print("NOTE: prior-* values do not change the autoencoder, but each value retrains it (separate run folders).")
    save_json(out_dir / "sweep_config.json", {"param": args.param, "values": values, "models": models, "seeds": seeds,
                                              "with_generation": args.with_generation, "base_params": {f.name: getattr(base, f.name) for f in fields(base)},
                                              "gen_samples": args.gen_samples, "gen_seed": args.gen_seed, "temperature": args.temperature, "top_k": args.top_k})

    ctx = build_context(args, base) if args.with_generation else None
    if ctx is not None:
        save_json(out_dir / "real_floor.json", ctx["floors"] | {"close_match_threshold_mse": ctx["close_threshold"], "n": ctx["n"]})

    rows: List[Dict[str, object]] = []
    for repeat, seed in enumerate(seeds):
        for value in values:
            # Set the swept value and this repeat's seed (unless the seed itself is the swept
            # parameter).
            params = replace(base, **{name: value}, **({} if name == "seed" else {"seed": seed}))
            for model in models:
                print(f"\n=== {LABELS[model]} | {args.param}={value:g} | seed {params.seed} ===")
                rows.append(run_one(model, params, args, out_dir, value, repeat, ctx))
                write_csv(out_dir / "results.csv", rows)  # written after every run: an interrupted sweep keeps its results

    agg = aggregate(rows)
    # Flatten the aggregates into one row per (model, value) with <metric>_mean, <metric>_std and
    # <metric>_n columns.
    summary_rows = [{"model": m, "value": v, **{f"{k}_{s}": stat[i] for k, stat in a.items() for i, s in enumerate(("mean", "std", "n"))}} for (m, v), a in sorted(agg.items())]
    write_csv(out_dir / "summary.csv", summary_rows)
    log_x = min(values) > 0 and max(values) / min(values) >= 10  # log axis when values span 10x or more (typical for learning rates)
    floors = ctx["floors"] if ctx is not None else None
    plot_effect(agg, args.param, models, values, log_x, out_dir / f"{args.param}_effect.png")
    plot_learning_curves(rows, args.param, models, values, out_dir / f"{args.param}_learning_curves.png")
    if "vqvae" in models:
        plot_codebook(agg, args.param, values, log_x, out_dir / f"{args.param}_codebook.png")
    if floors is not None:
        plot_generation(agg, args.param, models, values, log_x, floors, out_dir / f"{args.param}_generation.png")
        plot_tradeoff(agg, args.param, models, values, out_dir / f"{args.param}_tradeoff.png", floors)
        plot_samples(rows, args.param, models, values, out_dir / f"{args.param}_samples.png")
    write_report(out_dir / f"{args.param}_report.md", args.param, rows, agg, models, values, floors,
                 {"repeats": args.repeats, "seeds": seeds, "epochs": base.epochs})

    header = f"{'model':8} {args.param:>16} {'test SSIM':>16} {'test MSE':>10} {'train s':>9}" + (f" {'FD':>9} {'recall':>7} {'close%':>7}" if floors is not None else "")
    print("\n" + header)
    for (m, v), a in sorted(agg.items()):
        line = f"{LABELS[m]:8} {v:>16g} {_fmt(a['test_ssim']):>16} {a['test_mse'][0]:>10.6f} {a['training_seconds'][0]:>9.1f}"
        if floors is not None and "gen_fd" in a:
            line += f" {a['gen_fd'][0]:>9.3f} {a['gen_recall'][0]:>7.3f} {100 * a['gen_close_match_rate'][0]:>6.1f}%"
        print(line)
    print(f"\nResults: {out_dir / 'results.csv'}\nReport:  {out_dir / (args.param + '_report.md')}\nFigures: {out_dir}")


if __name__ == "__main__":
    main()
