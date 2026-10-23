"""Create a matched-split comparison report from VAE and VQ-VAE run directories.

Both runs must have been trained and evaluated on identical train/val/test file lists;
otherwise their scores are not comparable and the script refuses to write a report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from utils import read_json, write_csv


def split_paths(path: Path) -> dict:
    """Sorted file paths per split, so list order cannot cause a false mismatch."""
    return {name: sorted(row["path"] for row in rows) for name, rows in read_json(path)["splits"].items()}


def make_row(model_name: str, run_dir: Path) -> Dict[str, object]:
    """Flatten one run's training summary and predict.py diagnostics into a table row."""
    summary = read_json(run_dir / "run_summary.json")
    diagnostics = read_json(run_dir / "diagnostics" / "diagnostics_summary.json")
    test = summary["held_out_test_metrics"]
    sampling = diagnostics["sampling"]
    nn = diagnostics["nearest_neighbor_audit"]
    return {
        "model": model_name,
        "best_epoch": summary["best_epoch"],
        "test_images": test["n_images"],
        "test_reconstruction_ssim": test["ssim"],
        "test_reconstruction_mse": test["reconstruction_mse"],
        "test_loss": test["loss"],
        "model_parameters": summary["model_parameters"],
        "prior_parameters": sampling["prior_parameters"],
        "training_wall_seconds": summary["training_wall_seconds"],
        "prior_training_seconds": summary["generation"].get("prior_training_seconds", 0.0),
        "total_training_plus_prior_seconds": summary["total_training_plus_prior_wall_seconds"],
        "peak_cuda_memory_mb_train": summary["peak_cuda_memory_mb"],
        "peak_cuda_memory_mb_predict": diagnostics["resource_profile"]["peak_cuda_memory_mb"],
        "reconstruction_ms_per_image": diagnostics["test_reconstruction"]["milliseconds_per_image"],
        "generation_ms_per_image": 1000.0 * sampling["latency_seconds_per_sample"],
        "generated_nearest_train_pixel_mse": nn["generated_nn_mse_mean"],
        "heldout_nearest_train_pixel_mse": nn["heldout_test_nn_mse_mean"],
        "generated_encoder_feature_coverage": diagnostics["latent_feature_coverage_proxy"]["heldout_test_coverage_by_generated_set"],
        "diagnostics_dir": str((run_dir / "diagnostics").resolve()),
    }


# (label, row key, format spec) for the markdown table.
METRICS = [
    ("Held-out reconstruction SSIM (higher is better)", "test_reconstruction_ssim", ".4f"),
    ("Held-out reconstruction MSE (lower is better)", "test_reconstruction_mse", ".6f"),
    ("Model parameter count", "model_parameters", ","),
    ("Main model training wall time (seconds)", "training_wall_seconds", ".1f"),
    ("Prior training wall time (seconds)", "prior_training_seconds", ".1f"),
    ("Model plus prior training time (seconds)", "total_training_plus_prior_seconds", ".1f"),
    ("Peak CUDA memory during training (MB)", "peak_cuda_memory_mb_train", ".1f"),
    ("Reconstruction latency (ms/image)", "reconstruction_ms_per_image", ".3f"),
    ("Generation latency (ms/image)", "generation_ms_per_image", ".2f"),
    ("Generated-to-training nearest-neighbour pixel MSE", "generated_nearest_train_pixel_mse", ".6f"),
    ("Held-out-to-training nearest-neighbour pixel MSE", "heldout_nearest_train_pixel_mse", ".6f"),
    ("Generated encoder-space feature coverage proxy", "generated_encoder_feature_coverage", ".3f"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-run-dir", type=Path, required=True)
    parser.add_argument("--vqvae-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    args = parser.parse_args()
    vae_dir, vq_dir = args.vae_run_dir.resolve(), args.vqvae_run_dir.resolve()

    if split_paths(vae_dir / "split.json") != split_paths(vq_dir / "split.json"):
        raise SystemExit(
            "The VAE and VQ-VAE did not use identical file-level splits, so their test scores are not "
            "comparable. Retrain both from the same manifest."
        )
    rows = [make_row("VAE", vae_dir), make_row("VQ-VAE", vq_dir)]
    if rows[0]["test_images"] != rows[1]["test_images"]:
        raise SystemExit("Test image counts differ; do not treat these results as a paired comparison.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_comparison.csv", rows)

    lines = [
        "# VAE vs VQ-VAE Matched-Split Comparison",
        "",
        "Both run directories were checked for identical train/validation/test file lists.",
        "Nearest-neighbour and feature-coverage values are diagnostics, not privacy certification or validated clinical metrics.",
        "",
        "| Metric | VAE | VQ-VAE |",
        "|---|---:|---:|",
    ]
    for label, key, fmt in METRICS:
        # None (e.g. no GPU, or the VAE has no prior) is shown as N/A rather than 0.
        values = ["N/A" if row[key] is None else format(row[key], fmt) for row in rows]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += [
        "",
        "## Interpretation checks",
        "",
        "- Reconstruction SSIM is paired input/reconstruction fidelity, not the fidelity of newly generated scans.",
        "- Compare each model's generated-to-training nearest-neighbour distance with the held-out-to-training distance; very close generated matches need a manual look.",
        "- Use the diagnostic grids to document at least 3-5 concrete failures and plausible root causes.",
        "- Do not recommend release or clinical use from this comparison alone; check anatomy, diversity and memorisation manually.",
    ]
    (args.output_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output_dir / 'model_comparison.csv'}")
    print(f"Wrote: {args.output_dir / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
