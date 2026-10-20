"""Create a matched-split comparison report from AE and VQ-VAE run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required experiment artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_paths(path: Path) -> dict:
    obj = read_json(path)
    return {
        name: sorted(row["path"] for row in rows)
        for name, rows in obj["splits"].items()
    }


def make_row(model_name: str, run_dir: Path) -> Dict[str, object]:
    summary = read_json(run_dir / "run_summary.json")
    diagnostics = read_json(run_dir / "diagnostics" / "diagnostics_summary.json")
    test = summary["held_out_test_metrics"]
    recon = diagnostics["test_reconstruction"]
    sampling = diagnostics["sampling"]
    nn = diagnostics["nearest_neighbor_audit"]
    feature = diagnostics["latent_feature_coverage_proxy"]
    profile = diagnostics["resource_profile"]
    return {
        "model": model_name,
        "group_id_verified": summary["group_id_verified"],
        "best_epoch": summary["best_epoch"],
        "test_images": test["n_images"],
        "test_reconstruction_ssim": test["ssim"],
        "test_reconstruction_mse": test["reconstruction_mse"],
        "test_loss": test["loss"],
        "model_parameters": summary["model_parameters"],
        "prior_parameters": sampling["prior_parameters"],
        "training_wall_seconds": summary["training_wall_seconds"],
        "prior_training_seconds": summary["generation"].get(
            "prior_training_seconds", 0.0
        )
        or 0.0,
        "total_training_plus_prior_seconds": summary.get(
            "total_training_plus_prior_wall_seconds", summary["training_wall_seconds"]
        ),
        "peak_cuda_memory_mb_train": summary["peak_cuda_memory_mb"],
        "peak_cuda_memory_mb_predict": profile["peak_cuda_memory_mb"],
        "reconstruction_ms_per_image": recon["milliseconds_per_image"],
        "generation_ms_per_image": 1000.0 * sampling["latency_seconds_per_sample"],
        "generated_nearest_train_pixel_mse": nn["generated_nn_mse_mean"],
        "heldout_nearest_train_pixel_mse": nn["heldout_test_nn_mse_mean"],
        "generated_encoder_feature_coverage": feature[
            "heldout_test_coverage_by_generated_set"
        ],
        "diagnostics_dir": str((run_dir / "diagnostics").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-run-dir", type=Path, required=True)
    parser.add_argument("--vqvae-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    args = parser.parse_args()
    ae_dir, vq_dir = args.ae_run_dir.resolve(), args.vqvae_run_dir.resolve()
    ae_splits = split_paths(ae_dir / "split.json")
    vq_splits = split_paths(vq_dir / "split.json")
    if ae_splits != vq_splits:
        raise SystemExit(
            "The AE and VQ-VAE did not use identical file-level splits. Do not compare their test scores; "
            "regenerate both runs from the same reviewed manifest, seed, and grouping rules."
        )
    ae_summary = read_json(ae_dir / "run_summary.json")
    vq_summary = read_json(vq_dir / "run_summary.json")
    if not (ae_summary["group_id_verified"] and vq_summary["group_id_verified"]):
        print(
            "WARNING: one or both group mappings were unverified. This report is a pipeline comparison only, not a leakage-safe result."
        )
    if (
        ae_summary["held_out_test_metrics"]["n_images"]
        != vq_summary["held_out_test_metrics"]["n_images"]
    ):
        raise SystemExit(
            "Test image counts differ; do not aggregate these results as a paired comparison."
        )

    rows = [make_row("Basic autoencoder", ae_dir), make_row("VQ-VAE", vq_dir)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        (
            "Held-out reconstruction SSIM (higher is better)",
            "test_reconstruction_ssim",
            ".4f",
        ),
        (
            "Held-out reconstruction MSE (lower is better)",
            "test_reconstruction_mse",
            ".6f",
        ),
        ("Model parameter count", "model_parameters", ","),
        ("Main model training wall time (seconds)", "training_wall_seconds", ".1f"),
        ("Prior training wall time (seconds)", "prior_training_seconds", ".1f"),
        (
            "Model plus prior training time (seconds)",
            "total_training_plus_prior_seconds",
            ".1f",
        ),
        ("Peak CUDA memory during training (MB)", "peak_cuda_memory_mb_train", ".1f"),
        ("Reconstruction latency (ms/image)", "reconstruction_ms_per_image", ".3f"),
        ("Generation latency (ms/image)", "generation_ms_per_image", ".2f"),
        (
            "Generated-to-training nearest-neighbour pixel MSE",
            "generated_nearest_train_pixel_mse",
            ".6f",
        ),
        (
            "Held-out-to-training nearest-neighbour pixel MSE",
            "heldout_nearest_train_pixel_mse",
            ".6f",
        ),
        (
            "Generated encoder-space feature coverage proxy",
            "generated_encoder_feature_coverage",
            ".3f",
        ),
    ]
    markdown = [
        "# AE vs VQ-VAE Matched-Split Comparison",
        "",
        "Both run directories were checked for identical train/validation/test file lists.",
        "Nearest-neighbour and feature-coverage values are diagnostics, not privacy certification or validated clinical metrics.",
        "",
        "| Metric | Basic autoencoder | VQ-VAE |",
        "|---|---:|---:|",
    ]
    for label, key, fmt in metrics:
        vals = []
        for row in rows:
            value = row[key]
            vals.append("N/A" if value is None else format(value, fmt))
        markdown.append(f"| {label} | {vals[0]} | {vals[1]} |")
    markdown += [
        "",
        "## Interpretation checks",
        "",
        "- Reconstruction SSIM is paired input/reconstruction fidelity. It is not the same as fidelity of unpaired newly generated scans.",
        "- Compare each model's generated-to-training nearest-neighbour distances against held-out-test-to-training distances; very close generated matches require manual audit.",
        "- Use the diagnostic grids to document at least 3-5 concrete failures and plausible root causes.",
        "- Do not recommend release or clinical use from this comparison alone. Check anatomy, diversity, group leakage, and memorization manually.",
    ]
    md_path = args.output_dir / "model_comparison.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
