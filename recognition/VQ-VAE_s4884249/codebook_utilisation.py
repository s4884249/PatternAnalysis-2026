"""Figure: codebook utilisation distribution of a trained VQ-VAE.

    python3 codebook_utilisation.py --run runs/vqvae_seedx42-epochsx50 --data-root <keras_slices_data>

Counts how often each of the K codes is used when the encoder quantises (a) training images, (b) held-out test
images and (c) when the transformer prior generates new images, then plots:
  1. usage per code, sorted by training frequency (the tail of zeros = dead codes)
  2. cumulative share of all code assignments vs number of codes (how many codes carry 50 / 90 / 99%)
  3. training vs generated per-code frequency (points far below the diagonal = codes the prior under-uses)
Writes images/codebook_utilisation.png (override with --out). Use --samples 0 to skip the prior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

import eval_metrics as M

COLORS = {"train": "#4c72b0", "test": "#55a868", "generated": "#c44e52"}
LABELS = {"train": "Training images", "test": "Test images", "generated": "Prior samples"}


def plot_utilisation(counts: Dict[str, np.ndarray], out: Path) -> Dict[str, dict]:
    """counts: name -> usage count per code (all the same length K). Returns the statistics printed in the legend."""
    K = len(next(iter(counts.values())))
    freq = {k: v / v.sum() for k, v in counts.items()}
    order = np.argsort(-freq["train"])  # codes sorted by training usage, shared by every curve
    stats = {}
    for k, c in counts.items():
        cum = np.cumsum(np.sort(freq[k])[::-1])
        stats[k] = {"active": int((c > 0).sum()), "perplexity": M.entropy_perplexity(c),
                    **{f"codes_for_{int(q * 100)}%": int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.9, 0.99)}}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.8))
    floor = 0.5 / max(sum(v.sum() for v in counts.values()), 1)  # draw unused codes on a log axis
    for k in counts:
        s = stats[k]
        label = f"{LABELS[k]}: {s['active']}/{K} active, perplexity {s['perplexity']:.0f}"
        style = "--" if k == "test" else "-"
        a.plot(np.arange(K), np.maximum(freq[k][order], floor), style, color=COLORS[k], lw=1.2, label=label)
        cum = np.cumsum(np.sort(freq[k])[::-1])
        b.plot(np.arange(1, K + 1), cum, style, color=COLORS[k], label=f"{LABELS[k]}: 90% of use in {s['codes_for_90%']} codes")
    a.axhline(1 / K, color="gray", ls="--", lw=1, label=f"uniform use (1/{K})")
    a.set(yscale="log", xlabel="code (sorted by training frequency)", ylabel="share of assignments", title="Usage per code")
    b.plot([0, K], [0, 1], color="gray", ls="--", lw=1, label="uniform use")
    b.set(xlabel="number of codes (most used first)", ylabel="cumulative share of assignments", title="Cumulative utilisation")
    if "generated" in counts:
        lim = [max(floor, 1e-6), max(freq["train"].max(), freq["generated"].max()) * 1.5]
        c.scatter(np.maximum(freq["train"], floor), np.maximum(freq["generated"], floor), s=8, alpha=0.6, color=COLORS["generated"])
        c.plot(lim, lim, "k--", lw=1)
        c.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="frequency in training images", ylabel="frequency in prior samples",
              title="Does the prior reproduce code usage?")
    else:
        c.text(0.5, 0.5, "no prior samples (--samples 0 or prior.pt missing)", ha="center", va="center", transform=c.transAxes)
        c.set_xticks([]); c.set_yticks([])
    for ax in (a, b):
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    c.grid(alpha=0.25)
    fig.suptitle("VQ-VAE codebook utilisation distribution (dead codes and over-used codes indicate collapse)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="VQ-VAE run directory (needs best.pt, split.json, run_summary.json)")
    p.add_argument("--data-root", type=Path, default=None, help="default: the data root stored in the checkpoint")
    p.add_argument("--out", type=Path, default=Path("images/codebook_utilisation.png"))
    p.add_argument("--max-images", type=int, default=3000, help="cap on images per split used for the counts")
    p.add_argument("--samples", type=int, default=256, help="images sampled from the prior (0 = skip)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from device_utils import get_device  # heavy imports only when actually running on a model
    from evaluate import generate_cohort, load_images, load_run, train_code_maps

    device = get_device(args.device)
    run = load_run(args.run, device)
    if run["type"] != "vqvae":
        raise SystemExit("--run must be a VQ-VAE run")
    cfg = run["config"]
    hw = (int(cfg["image_height"]), int(cfg["image_width"]))
    root = Path(args.data_root or cfg["data_root"]).expanduser().resolve()
    rng, K = np.random.default_rng(0), int(cfg["codebook_size"])
    counts = {}
    for name in ("train", "test"):
        rows = run["split"][name]
        if len(rows) > args.max_images:
            rows = [rows[i] for i in np.sort(rng.choice(len(rows), args.max_images, replace=False))]
        images, _, _ = load_images(rows, root, *hw, args.workers)
        counts[name] = np.bincount(train_code_maps(run["model"], images, device, args.batch_size).ravel(), minlength=K)
    if args.samples > 0:
        try:
            cohort = generate_cohort(run, hw, [args.seed], args.samples, device, 1.0, 0)
            counts["generated"] = np.bincount(cohort["latents"].numpy().ravel(), minlength=K)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Skipping prior samples: {exc}")
    stats = plot_utilisation(counts, args.out)
    for name, s in stats.items():
        print(f"{LABELS[name]:16} {s}")
    print(f"Figure: {args.out.resolve()}")


if __name__ == "__main__":
    main()
