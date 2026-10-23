"""Sweep one hyper-parameter for the VAE and the VQ-VAE and plot its effect on both.

    python3 experiments.py --param latent-dim --values 16 32 64 --data-root <keras_slices_data>
    python3 experiments.py --param lr --values 1e-4 2e-4 1e-3 --epochs 20 --data-root <...>

Every (model, value) pair is trained with the params.py defaults plus any other
overrides given here (e.g. --epochs 20), so only the swept parameter differs.
Parameters that exist for one model only (kl-beta; codebook-size, commitment-beta,
prior-*) are swept for that model alone. Finished runs are reused unless --rerun.

Only reconstruction quality and cost are compared, so the VQ-VAE prior is not
trained here; use train.py + predict.py for generation diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, List

from params import (
    EXPERIMENTS_DIR, MANIFEST, MODEL_SPECIFIC, MODELS, TrainParams, add_overrides, apply_overrides, run_dir_name,
)
from train import train_model
from utils import read_json, write_csv

LABELS = {"vae": "VAE", "vqvae": "VQ-VAE"}
# (panel title, results key) for the four panels of the figure.
PANELS = [
    ("Held-out test SSIM (higher is better)", "test_ssim"),
    ("Held-out test MSE (lower is better)", "test_mse"),
    ("Training time (seconds)", "training_seconds"),
    ("Model parameters", "parameters"),
]


def sweepable() -> Dict[str, type]:
    """Numeric TrainParams fields (flag name -> int/float type)."""
    defaults = TrainParams()
    return {
        f.name.replace("_", "-"): type(getattr(defaults, f.name))
        for f in fields(defaults)
        if type(getattr(defaults, f.name)) in (int, float)
    }


def run_one(model: str, params: TrainParams, args, root: Path, value: float) -> Dict[str, object]:
    """Train (or reuse) one run and return the metrics the figure needs."""
    run_dir = root / "runs" / run_dir_name(model, params)
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file() and not args.rerun:
        print(f"Reusing finished run: {run_dir}")
        summary = read_json(summary_path)
    else:
        summary = train_model(model, params, args.data_root, args.manifest, run_dir, train_prior=False)
    test = summary["held_out_test_metrics"]
    return {
        "model": model,
        "value": value,
        "test_ssim": test["ssim"],
        "test_mse": test["reconstruction_mse"],
        "val_ssim": summary["best_validation_metrics"]["ssim"],
        "best_epoch": summary["best_epoch"],
        "training_seconds": summary["training_wall_seconds"],
        "parameters": summary["model_parameters"],
        "run_dir": str(run_dir),
    }


def plot_effect(rows: List[Dict[str, object]], param: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless backend for clusters
    import matplotlib.pyplot as plt

    values = sorted({r["value"] for r in rows})
    # Log axis when the values span 10x or more (typical for learning rates).
    log_x = min(values) > 0 and max(values) / min(values) >= 10
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (title, key) in zip(axes.flat, PANELS):
        for model in MODELS:
            points = sorted((r["value"], r[key]) for r in rows if r["model"] == model)
            if points:
                ax.plot(*zip(*points), marker="o", label=LABELS[model])
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(values)
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.minorticks_off()
        ax.set(title=title, xlabel=param)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"Effect of {param} on VAE and VQ-VAE")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    options = sweepable()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--param", required=True, choices=sorted(options), help="parameter to sweep")
    parser.add_argument("--values", nargs="+", required=True, help="two or more values to try")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: experiments/<param>_seed<S>x<E>")
    parser.add_argument("--rerun", action="store_true", help="retrain even if a finished run exists")
    add_overrides(parser, TrainParams())  # settings shared by every run in the sweep
    args = parser.parse_args()

    name = args.param.replace("-", "_")
    kind = options[args.param]
    try:
        base = apply_overrides(TrainParams(), args)
        values = [float(v) for v in args.values]
        if kind is int and not all(v.is_integer() for v in values):
            raise ValueError(f"--param {args.param} needs integer values")
        values = [kind(v) for v in values]
        if len(set(values)) < 2:
            raise ValueError("--values needs at least two different values")
        for value in values:  # fail before any training if a value is invalid
            replace(base, **{name: value})
    except ValueError as exc:
        parser.error(str(exc))

    # The swept parameter is already in the folder name, so seed/epochs are only added when not swept.
    suffix = "" if name in ("seed", "epochs") else f"_seed{base.seed}x{base.epochs}"
    out_dir = (args.output_dir or EXPERIMENTS_DIR / f"{name.replace('_', '-')}{suffix}").resolve()
    models = [m for m in MODELS if MODEL_SPECIFIC.get(name, m) == m]
    print(f"Sweeping {args.param} over {values} for: {', '.join(LABELS[m] for m in models)}")

    rows = []
    for value in values:
        params = replace(base, **{name: value})
        for model in models:
            print(f"\n=== {LABELS[model]} | {args.param}={value:g} ===")
            rows.append(run_one(model, params, args, out_dir, value))

    write_csv(out_dir / "results.csv", rows)
    plot_effect(rows, args.param, out_dir / f"{args.param}_effect.png")
    print(f"\n{'model':8} {args.param:>16} {'test SSIM':>10} {'test MSE':>10} {'train s':>9}")
    for r in rows:
        print(f"{LABELS[r['model']]:8} {r['value']:>16g} {r['test_ssim']:>10.4f} {r['test_mse']:>10.6f} {r['training_seconds']:>9.1f}")
    print(f"\nResults: {out_dir / 'results.csv'}\nFigure:  {out_dir / (args.param + '_effect.png')}")


if __name__ == "__main__":
    main()
