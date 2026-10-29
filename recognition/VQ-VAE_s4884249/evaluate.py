"""Evaluate a trained VAE (baseline) and VQ-VAE: generate uncurated cohorts, then produce every figure,
metric and the EVALUATION_RESULTS.md report.

    python3 evaluate.py \
        --vae-run   runs/vae_seedx42-epochsx50 \
        --vqvae-run runs/vqvae_seedx42-epochsx50-ssim-weightx0.1 \
        --data-root /path/to/keras_slices_data --out evaluation

Stages
  collect()  (this file, needs PyTorch): load checkpoints + split.json, reconstruct the test split, sample
             `--samples-per-seed` images for each of `--seeds` (NOTHING is filtered or selected), time the
             generation, extract features, build latent-interpolation paths.
  analyse()  (eval_analysis.py, NumPy only): metrics, figures, rule-based verdict, markdown.

Run it after train.py has finished for both models (best.pt, split.json, history.csv, run_summary.json and,
for the VQ-VAE, prior.pt; for the VAE, latent_prior.pt).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import eval_metrics as M
from dataset import NiftiSliceDataset
from device_utils import get_device, peak_memory_mb, reset_peak_memory_stats, sync
from eval_analysis import analyse
from eval_plots import read_history
from modules import CodePrior, build_model, count_parameters, latent_stages
from utils import read_json, set_seed

GRAD_CLIP = 5.0  # must match train.GRAD_CLIP


# --------------------------------------------------------------------------------------
# data / model loading
# --------------------------------------------------------------------------------------
def load_images(rows: Sequence[dict], data_root: Path, height: int, width: int, workers: int):
    dataset = NiftiSliceDataset(rows, data_root, height, width, cache=True, threads=max(1, workers))
    images = torch.stack([dataset[i]["image"] for i in range(len(dataset))])[:, 0].numpy().astype(np.float32)
    return images, [r["path"] for r in rows], [r["group_id"] for r in rows]


def load_run(run_dir: Path, device: torch.device) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model_type, config = checkpoint["model_type"], checkpoint["model_config"]
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "run_dir": run_dir, "type": model_type, "config": config, "model": model,
        "run_summary": read_json(run_dir / "run_summary.json"), "split": read_json(run_dir / "split.json")["splits"],
        "history": read_history(run_dir / "history.csv"), "prior_history": read_history(run_dir / "prior_history.csv") or None,
    }


@torch.inference_mode()
def reconstruct_np(model, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i : i + batch_size])[:, None].to(device)
        out.append(model(x)["reconstruction"].clamp(0, 1)[:, 0].cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def train_code_maps(model, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    maps = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i : i + batch_size])[:, None].to(device)
        maps.append(model.encode_indices(x).cpu().numpy())
    return np.concatenate(maps)


# --------------------------------------------------------------------------------------
# generation (uncurated) + timing
# --------------------------------------------------------------------------------------
def load_prior(run_dir: Path, device: torch.device):
    state = torch.load(run_dir / "prior.pt", map_location=device, weights_only=True)
    if state.get("prior_type") != "transformer":
        raise RuntimeError(f"{run_dir / 'prior.pt'} is not a transformer prior; retrain with the updated train.py")
    prior = CodePrior(
        codebook_size=int(state["codebook_size"]), hidden_channels=int(state["hidden_channels"]),
        height=int(state["latent_height"]), width=int(state["latent_width"]),
        layers=int(state["layers"]), dropout=float(state["dropout"]),
    ).to(device)
    prior.load_state_dict(state["state_dict"])
    prior.eval()
    return prior


@torch.inference_mode()
def generate_cohort(run: dict, hw: Tuple[int, int], seeds: List[int], per_seed: int, device: torch.device,
                    temperature: float, top_k: int) -> dict:
    """Draw `per_seed` samples for each seed. Returns images, per-sample seed ids, latents/codes and timing."""
    model, run_dir = run["model"], run["run_dir"]
    images, seed_ids, latents = [], [], []
    prior, vae_prior = None, None
    if run["type"] == "vqvae":
        prior = load_prior(run_dir, device)
        grid = tuple(s // 2 ** latent_stages("vqvae", run["config"]) for s in hw)
    elif (run_dir / "latent_prior.pt").is_file():
        vae_prior = torch.load(run_dir / "latent_prior.pt", map_location=device, weights_only=True)
    else:
        print("WARNING: latent_prior.pt missing; VAE sampled from N(0, I) (expect poor samples)")
    reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        if prior is not None:
            codes = prior.sample(per_seed, *grid, device, temperature=temperature, generator=generator, top_k=top_k)
            batch = model.decode_indices(codes)
            latents.append(codes.cpu())
        else:
            eps = torch.randn((per_seed, model.latent_dim), device=device, generator=generator)
            z = vae_prior["mean"].to(device) + eps @ vae_prior["cholesky"].to(device).T if vae_prior is not None else eps
            batch = model.decode(z)
            latents.append(z.cpu())
        images.append(batch.clamp(0, 1)[:, 0].cpu())
        seed_ids += [seed] * per_seed
    sync(device)
    seconds = time.perf_counter() - start
    return {
        "gen": torch.cat(images).numpy().astype(np.float32), "gen_seed": np.asarray(seed_ids), "latents": torch.cat(latents),
        "gen_seconds_per_sample": seconds / len(seed_ids), "gen_peak_mb": peak_memory_mb(device),
        "prior_params": count_parameters(prior) if prior is not None else None,
    }


# --------------------------------------------------------------------------------------
# interpolation between synthetic samples and real training slices
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def interpolation_paths(run: dict, cohort: dict, train: np.ndarray, device: torch.device, rng: np.random.Generator,
                        n_steps: int = 9, n_nearest: int = 2, n_random: int = 2) -> List[dict]:
    """First `n_nearest + n_random` samples of the first seed (fixed positions, so not hand-picked) are
    interpolated linearly in the decoder's input space towards (a) their nearest training slice and
    (b) a random training slice. VQ-VAE: between quantised code maps; VAE: between latent vectors."""
    model = run["model"]
    first = np.flatnonzero(cohort["gen_seed"] == cohort["gen_seed"][0])[: n_nearest + n_random]
    nearest = M.nn_topk(cohort["gen"][first], train, 1)[1][:, 0]
    random_idx = rng.choice(len(train), n_random, replace=False)
    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1, 1, 1)
    paths = []
    for row, i in enumerate(first):
        j = int(nearest[row]) if row < n_nearest else int(random_idx[row - n_nearest])
        x = torch.from_numpy(train[j])[None, None].to(device)
        if run["type"] == "vqvae":
            z_s = model.quantizer.embedding(cohort["latents"][i : i + 1].to(device)).permute(0, 3, 1, 2)
            z_t = model.quantize(model.encode(x))[0]
            frames = model.decode((1 - alphas) * z_s + alphas * z_t)
        else:
            z_s, z_t = cohort["latents"][i : i + 1].to(device), model.encode(x)
            frames = model.decode((1 - alphas.view(-1, 1)) * z_s + alphas.view(-1, 1) * z_t)
        paths.append({"frames": frames.clamp(0, 1)[:, 0].cpu().numpy(),
                      "label": f"sample {int(i)} -> " + ("its nearest train slice" if row < n_nearest else "a random train slice")})
    return paths


# --------------------------------------------------------------------------------------
# feature extractor (Inception-v3 if available, otherwise model-agnostic pixel PCA)
# --------------------------------------------------------------------------------------
def build_inception(device: torch.device):
    import torchvision

    weights = torchvision.models.Inception_V3_Weights.DEFAULT
    net = torchvision.models.inception_v3(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract(images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i : i + batch_size])[:, None].to(device).float()
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            feats.append(net((x - mean) / std).float().cpu().numpy())
        return np.concatenate(feats)

    return extract


# --------------------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------------------
def collect(args, device: torch.device) -> dict:
    rng = np.random.default_rng(args.seed)
    runs = {"vae": load_run(args.vae_run, device), "vqvae": load_run(args.vqvae_run, device)}
    for name, run in runs.items():
        if run["type"] != name:
            raise ValueError(f"--{name}-run points at a '{run['type']}' run")
    cfg = runs["vqvae"]["config"]
    hw = (int(cfg["image_height"]), int(cfg["image_width"]))
    if (int(runs["vae"]["config"]["image_height"]), int(runs["vae"]["config"]["image_width"])) != hw:
        raise ValueError("The two runs were trained at different image sizes")
    data_root = Path(args.data_root or cfg["data_root"]).expanduser().resolve()
    test_rows, train_rows = runs["vqvae"]["split"]["test"], runs["vqvae"]["split"]["train"]
    if [r["path"] for r in runs["vae"]["split"]["test"]] != [r["path"] for r in test_rows]:
        print("WARNING: the two runs used different test splits; metrics use the VQ-VAE run's split")
    n_train_total = len(train_rows)
    if args.max_reference_images and len(train_rows) > args.max_reference_images:
        keep = np.sort(rng.choice(len(train_rows), args.max_reference_images, replace=False))
        train_rows = [train_rows[i] for i in keep]
    print(f"Loading {len(train_rows)} training and {len(test_rows)} test slices ...")
    train, train_paths, train_groups = load_images(train_rows, data_root, *hw, args.workers)
    test, test_paths, test_groups = load_images(test_rows, data_root, *hw, args.workers)

    control_idx = np.sort(rng.choice(len(train), min(args.control_count, len(train)), replace=False))
    models: Dict[str, dict] = {}
    for name, run in runs.items():
        print(f"[{name}] reconstructing, sampling {len(args.seeds)} seeds x {args.samples_per_seed} ...")
        model = run["model"]
        cohort = generate_cohort(run, hw, args.seeds, args.samples_per_seed, device, args.temperature, args.top_k)
        entry = {
            "history": run["history"], "prior_history": run["prior_history"], "run_summary": run["run_summary"], "config": run["config"],
            "n_params": count_parameters(model), "prior_params": cohort["prior_params"],
            "recon_test": reconstruct_np(model, test, device, args.batch_size),
            "recon_control": reconstruct_np(model, train[control_idx], device, args.batch_size),
            "gen": cohort["gen"], "gen_seed": cohort["gen_seed"],
            "gen_seconds_per_sample": cohort["gen_seconds_per_sample"], "gen_peak_mb": cohort["gen_peak_mb"],
            "interp": interpolation_paths(run, cohort, train, device, rng),
        }
        if name == "vqvae":
            entry.update(codes=cohort["latents"].numpy(), train_codes=train_code_maps(model, train, device, args.batch_size),
                         codebook_size=int(cfg["codebook_size"]))
        models[name] = entry

    # Feature space for FD / precision / recall / coverage / embedding.
    feat_idx = np.sort(rng.choice(len(train), min(args.max_feature_images, len(train)), replace=False))
    sets = [train[feat_idx], test, models["vae"]["gen"], models["vqvae"]["gen"]]
    extractor, feature_name = None, ""
    if args.features in ("auto", "inception"):
        try:
            extractor = build_inception(device)
            feature_name = "Inception-v3 pool features"
        except Exception as exc:  # noqa: BLE001 - weights download / torchvision missing
            if args.features == "inception":
                raise
            print(f"Inception unavailable ({type(exc).__name__}: {exc}); using pixel-PCA features instead")
    if extractor is not None:
        print("Extracting Inception features ...")
        feats = [extractor(s) for s in sets]
    else:
        feats = M.pixel_pca_features(sets[0], *sets[1:], pool=4, n_components=64)
        feature_name = "pixel PCA (64 components of 4x-downsampled pixels, fitted on real training slices)"
    return {
        "hw": hw, "train": train, "train_paths": train_paths, "train_groups": train_groups, "test": test, "test_paths": test_paths,
        "test_groups": test_groups, "n_train_total": n_train_total, "models": models, "feature_name": feature_name,
        "features": {"train": feats[0], "train_idx": feat_idx, "test": feats[1], "vae": feats[2], "vqvae": feats[3]},
        "seeds": list(args.seeds), "samples_per_seed": args.samples_per_seed, "temperature": args.temperature,
        "top_k": args.top_k, "grad_clip": GRAD_CLIP,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate VAE vs VQ-VAE: figures, metrics, memorisation audit, report")
    p.add_argument("--vae-run", type=Path, required=True, help="run directory of the baseline VAE")
    p.add_argument("--vqvae-run", type=Path, required=True, help="run directory of the VQ-VAE")
    p.add_argument("--data-root", type=Path, default=None, help="override the data root stored in the checkpoint")
    p.add_argument("--out", type=Path, default=Path("evaluation"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4], help="sampling seeds (one uncurated grid each)")
    p.add_argument("--samples-per-seed", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0, help="VQ-VAE prior temperature (default: untuned 1.0)")
    p.add_argument("--top-k", type=int, default=0, help="VQ-VAE prior top-k (default: 0 = off)")
    p.add_argument("--max-reference-images", type=int, default=0, help="cap on training slices used as the reference pool (0 = all)")
    p.add_argument("--max-feature-images", type=int, default=5000)
    p.add_argument("--control-count", type=int, default=256, help="training images reconstructed as the memorisation positive control")
    p.add_argument("--features", choices=["auto", "inception", "pixel"], default="auto")
    p.add_argument("--tsne-points", type=int, default=400, help="points per group in the embedding plot")
    p.add_argument("--modes", type=int, default=10, help="k-means clusters for the mode-occupancy analysis")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")
    data = collect(args, device)
    print("Analysing ...")
    results = analyse(data, args.out, seed=args.seed, tsne_points=args.tsne_points, k_modes=args.modes)
    for name, verdict in results["verdict"].items():
        print(f"{name}: {verdict['label']}")
    print(f"Report: {args.out.resolve() / 'EVALUATION_RESULTS.md'}")


if __name__ == "__main__":
    main()
