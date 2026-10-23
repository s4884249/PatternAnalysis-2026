"""Describe and visualise the dataset listed in manifest.csv.

    python3 explore.py --data-root <keras_slices_data> [--manifest manifest.csv] [--output-dir explore]

Writes report.txt (the numbers printed to the terminal) and three figures:
    manifest_overview.png  files per split, slices per case, weeks, shapes, raw intensity ranges
    sample_slices.png      random slices per split exactly as the models see them (normalised, padded)
    pixel_statistics.png   mean/std image and intensity histogram of preprocessed training slices
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from dataset import FILENAME_PATTERN, NiftiSliceDataset, nifti_stem, read_manifest
from params import MANIFEST, TrainParams

SPLITS = ("train", "val", "test")


def summarise(values) -> str:
    """min / median / mean / max (and std) of a list of numbers on one line."""
    v = np.asarray(values, dtype=float)
    return f"min {v.min():.4g} | median {np.median(v):.4g} | mean {v.mean():.4g} | max {v.max():.4g} | std {v.std():.4g}"


def parse_name(row: Dict[str, str]):
    """(week, slice) parsed from the filename, or None if it does not follow case_XXX_week_W_slice_S."""
    match = FILENAME_PATTERN.match(nifti_stem(Path(row["path"])))
    return (int(match["week"]), int(match["slice"])) if match else None


def load_stack(rows: List[Dict[str, str]], data_root: Path, params: TrainParams) -> np.ndarray:
    """Preprocessed slices ([N, H, W] in [0, 1]) through the same dataset class training uses."""
    dataset = NiftiSliceDataset(rows, data_root, params.image_height, params.image_width)
    return np.stack([dataset[i]["image"][0].numpy() for i in range(len(dataset))])


def build_report(rows: List[Dict[str, str]]) -> tuple[List[str], dict]:
    """Numeric description of the manifest. Returns (report lines, arrays reused by the plots)."""
    lines: List[str] = []
    out = lines.append
    ok = [r for r in rows if not r["error"]]
    images = [r for r in ok if r["kind"] == "image"]
    masks = [r for r in ok if r["kind"] == "mask"]
    by_split = {s: [r for r in images if r["split"] == s] for s in SPLITS}

    out("== Files ==")
    out(f"manifest rows: {len(rows)} | readable: {len(ok)} | unreadable: {len(rows) - len(ok)}")
    for r in [r for r in rows if r["error"]][:5]:
        out(f"  error: {r['path']}: {r['error']}")
    out(f"images: {len(images)} | masks: {len(masks)}")
    out("")
    out("== Images and masks per split ==")
    for s in SPLITS:
        out(f"{s:5s} images {len(by_split[s]):6d} | masks {sum(m['split'] == s for m in masks):6d}")

    out("")
    out("== Cases (patients) ==")
    case_slices = {s: Counter(r["group_id"] for r in by_split[s]) for s in SPLITS}
    for s in SPLITS:
        if case_slices[s]:
            out(f"{s:5s} {len(case_slices[s]):4d} cases | slices per case: {summarise(list(case_slices[s].values()))}")
    overlaps = {
        f"{a}/{b}": sorted(set(case_slices[a]) & set(case_slices[b]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    leaks = {k: v for k, v in overlaps.items() if v}
    out(f"cases shared between splits: {leaks if leaks else 'none (no leakage)'}")

    out("")
    out("== Time points (weeks) ==")
    parsed = [(r, parse_name(r)) for r in images]
    week_counts = Counter(p[0] for _, p in parsed if p)
    weeks_per_case = defaultdict(set)
    for r, p in parsed:
        if p:
            weeks_per_case[r["group_id"]].add(p[0])
    out(f"slices per week: {dict(sorted(week_counts.items()))}")
    out(f"weeks per case: {summarise([len(w) for w in weeks_per_case.values()])}")
    unparsed = sum(p is None for _, p in parsed)
    if unparsed:
        out(f"WARNING: {unparsed} image names do not follow case_XXX_week_W_slice_S")

    out("")
    out("== Image / mask pairing ==")
    img_keys = {(r["group_id"], *p) for r, p in parsed if p}
    mask_keys = {(r["group_id"], *p) for r in masks if (p := parse_name(r))}
    out(f"images with a mask: {len(img_keys & mask_keys)} / {len(img_keys)} | masks without an image: {len(mask_keys - img_keys)}")

    out("")
    out("== Raw geometry ==")
    shapes = Counter(r["shape"] for r in images)
    for shape, n in shapes.most_common():
        out(f"shape {shape}: {n} ({100 * n / len(images):.1f}%)")

    out("")
    out("== Raw intensities (per slice, before normalisation) ==")
    stats = {k: np.array([float(r[k]) for r in images]) for k in ("min", "max", "mean", "std", "nonzero_frac")}
    out(f"global min {stats['min'].min():.4g} | global max {stats['max'].max():.4g}")
    for k in ("max", "mean", "std", "nonzero_frac"):
        out(f"slice {k:12s}: {summarise(stats[k])}")
    out("(normalisation clips each slice to its 1st-99.5th percentile and rescales to [0, 1], so the raw scale varies freely between slices)")

    arrays = {"images": images, "masks": masks, "by_split": by_split, "case_slices": case_slices,
              "week_counts": week_counts, "shapes": shapes, "stats": stats}
    return lines, arrays


def plot_overview(a: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    x = np.arange(len(SPLITS))
    ax.bar(x - 0.2, [len(a["by_split"][s]) for s in SPLITS], 0.4, label="images")
    ax.bar(x + 0.2, [sum(m["split"] == s for m in a["masks"]) for s in SPLITS], 0.4, label="masks")
    ax.set_xticks(x, SPLITS)
    ax.set(title="Files per split", ylabel="files")
    ax.legend()

    ax = axes[0, 1]
    counts = [n for s in SPLITS for n in a["case_slices"][s].values()]
    ax.hist(counts, bins=min(30, max(5, len(set(counts)))))
    ax.set(title=f"Slices per case ({len(counts)} cases)", xlabel="slices", ylabel="cases")

    ax = axes[0, 2]
    weeks = sorted(a["week_counts"])
    ax.bar([str(w) for w in weeks], [a["week_counts"][w] for w in weeks])
    ax.set(title="Slices per week", xlabel="week", ylabel="slices")

    ax = axes[1, 0]
    shapes = a["shapes"].most_common()
    ax.bar([s for s, _ in shapes], [n for _, n in shapes])
    ax.set(title="Raw image shapes", ylabel="slices")

    for ax, key, title in ((axes[1, 1], "max", "Raw per-slice maximum intensity"),
                           (axes[1, 2], "nonzero_frac", "Fraction of non-zero pixels per slice")):
        split_of = np.array([r["split"] for r in a["images"]])
        for s in SPLITS:
            if (split_of == s).any():
                ax.hist(a["stats"][key][split_of == s], bins=40, histtype="step", density=True, label=s)
        ax.set(title=title, ylabel="density")
        ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_samples(a: dict, data_root: Path, params: TrainParams, per_split: int, rng, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(SPLITS), per_split, figsize=(per_split * 1.8, len(SPLITS) * 2.5), squeeze=False)
    for r, s in enumerate(SPLITS):
        rows = a["by_split"][s]
        picked = [rows[i] for i in rng.choice(len(rows), min(per_split, len(rows)), replace=False)]
        stack = load_stack(picked, data_root, params)
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c < len(picked):
                ax.imshow(stack[c], cmap="gray", vmin=0, vmax=1)
                week, sl = parse_name(picked[c]) or ("?", "?")
                ax.set_title(f"{s}\n{picked[c]['group_id']} w{week} s{sl}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("explore"))
    parser.add_argument("--samples-per-split", type=int, default=8, help="slices shown per split")
    parser.add_argument("--pixel-sample", type=int, default=500, help="training slices loaded for pixel statistics")
    parser.add_argument("--seed", type=int, default=TrainParams().seed)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")  # headless backend for clusters
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.expanduser().resolve()
    params = TrainParams()  # image size matches what training uses
    rng = np.random.default_rng(args.seed)

    lines, a = build_report(read_manifest(args.manifest))
    plot_overview(a, args.output_dir / "manifest_overview.png")
    plot_samples(a, data_root, params, args.samples_per_split, rng, args.output_dir / "sample_slices.png")

    # Pixel statistics come from a random training subset: loading every slice would be slow
    # and a few hundred are enough for stable means and histograms.
    train = a["by_split"]["train"]
    sample = [train[i] for i in rng.choice(len(train), min(args.pixel_sample, len(train)), replace=False)]
    stack = load_stack(sample, data_root, params)
    lines += [
        "",
        f"== Preprocessed pixels ({len(sample)} random training slices, {params.image_height}x{params.image_width}) ==",
        f"pixel mean {stack.mean():.4f} | pixel std {stack.std():.4f}",
        f"background (pixels < 0.05): {100 * (stack < 0.05).mean():.1f}%",
        f"per-slice mean: {summarise(stack.mean(axis=(1, 2)))}",
    ]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, image, title in ((axes[0], stack.mean(0), "Mean slice"), (axes[1], stack.std(0), "Std per pixel")):
        shown = ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(shown, ax=ax, fraction=0.046)
    axes[2].hist(stack.ravel(), bins=100, log=True)
    axes[2].set(title="Pixel intensity histogram", xlabel="intensity", ylabel="pixels (log)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "pixel_statistics.png", dpi=140)
    plt.close(fig)

    (args.output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote report and figures to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
