"""HipMRI NIfTI manifest creation, split loading and image preprocessing.

`audit` scans the dataset once and writes manifest.csv (one row per NIfTI file with
its split, kind, case ID and intensity statistics). The manifest is assumed to be
pre-reviewed, so folder and file names alone decide each row's meaning:

    keras_slices_{train,validate,test}      -> kind=image, split=train/val/test
    keras_slices_seg_{train,validate,test}  -> kind=mask (excluded from training)
    case_040_week_5_slice_18.nii.gz         -> group_id=case_040
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MASK_TOKENS = ("mask", "label", "seg", "annotation", "groundtruth", "ground_truth", "segmentation")
# Images are case_XXX_week_W_slice_S and masks seg_XXX_week_W_slice_S.
FILENAME_PATTERN = re.compile(r"(?i)^(?:case|seg)[_-]?(?P<case>\d+)_week_(?P<week>\d+)_slice_(?P<slice>\d+)$")
SPLIT_PATTERN = re.compile(r"keras_slices_(?:seg_)?(train|validate|test)")
MANIFEST_FIELDS = ["path", "split", "kind", "group_id", "shape", "min", "max", "mean", "std", "nonzero_frac", "error"]


def nifti_stem(path: Path) -> str:
    name = path.name
    for extension in (".nii.gz", ".nii"):
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return path.stem


def infer_kind(relative: Path) -> str:
    """Masks live in keras_slices_seg_* folders; anything mask-like is excluded, the rest is an image."""
    text = relative.as_posix().lower().replace("-", "_")
    return "mask" if any(token in text for token in MASK_TOKENS) else "image"


def infer_split(relative: Path) -> str:
    """Use the supplied top-level folder (train/validate/test) as the split; validate -> val."""
    match = SPLIT_PATTERN.fullmatch(relative.parts[0].lower().replace("-", "_"))
    if not match:
        return ""
    return "val" if match.group(1) == "validate" else match.group(1)


def infer_group_id(stem: str) -> str:
    """Case ID from the filename. Grouping by case (not case+week) means a patient's
    follow-up scans can never straddle two splits."""
    match = FILENAME_PATTERN.match(stem)
    if match:
        return f"case_{match['case']}"
    # Fallback for unexpected names: drop a trailing slice index.
    return re.sub(r"(?i)[_-](?:slice|sl|z)?[_-]?\d+$", "", stem) or stem


def load_array(path: Path) -> np.ndarray:
    """Load a NIfTI as a finite 2D float32 array, keeping its in-plane geometry."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("NIfTI I/O requires nibabel (pip install nibabel).") from exc
    array = np.squeeze(np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image (or singleton extra dimensions), got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite voxel values")
    return array


def audit_directory(data_root: Path, output_csv: Path) -> Dict[str, int]:
    """Read every NIfTI under data_root and write one manifest row per file.

    Unreadable files are kept in the manifest with an `error` message instead of
    aborting, so one corrupt slice cannot stop the audit; training skips them.
    """
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    paths = sorted(p for p in data_root.rglob("*") if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz")))
    if not paths:
        raise FileNotFoundError(f"No .nii or .nii.gz files found under {data_root}")

    counts = {"files_seen": len(paths), "read_ok": 0, "read_error": 0}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for path in paths:
            relative = path.relative_to(data_root)
            row = dict.fromkeys(MANIFEST_FIELDS, "")
            row.update(
                path=relative.as_posix(),
                split=infer_split(relative),
                kind=infer_kind(relative),
                group_id=infer_group_id(nifti_stem(path)),
            )
            try:
                array = load_array(path)
                row.update(
                    shape="x".join(map(str, array.shape)),
                    min=f"{array.min():.7g}",
                    max=f"{array.max():.7g}",
                    mean=f"{array.mean():.7g}",
                    std=f"{array.std():.7g}",
                    nonzero_frac=f"{np.count_nonzero(array) / array.size:.7g}",
                )
                counts["read_ok"] += 1
            except Exception as exc:  # noqa: BLE001 - record the reason and keep auditing
                row["error"] = f"{type(exc).__name__}: {exc}"
                counts["read_error"] += 1
            writer.writerow(row)
    return counts


def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    """All manifest rows as dicts of strings (used by load_manifest and explore.py)."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path} (run `python3 dataset.py audit` first)")
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_manifest(manifest_path: Path, data_root: Path) -> List[Dict[str, str]]:
    """Return the readable image rows (path, split, group_id); masks and error rows are dropped."""
    data_root = data_root.expanduser().resolve()
    rows = []
    for source in read_manifest(manifest_path):
        if source["kind"] != "image" or source["error"]:
            continue
        if not (data_root / source["path"]).is_file():
            raise FileNotFoundError(f"Manifest file missing under --data-root: {data_root / source['path']}")
        rows.append({k: source[k] for k in ("path", "split", "group_id")})
    if not rows:
        raise ValueError("The manifest contains no readable image rows")
    return rows


def split_from_manifest(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group rows by the dataset's own train/val/test folders (no re-shuffling).

    Case IDs must be disjoint between splits; we fail loudly rather than train on a
    layout that leaks a patient into the test set.
    """
    splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        if row["split"] not in splits:
            raise ValueError(f"Row has no train/val/test split (is --data-root the keras_slices_data folder?): {row['path']}")
        splits[row["split"]].append(row)
    empty = [name for name, values in splits.items() if not values]
    if empty:
        raise ValueError(f"No image rows for split(s) {empty}; check --data-root and the manifest.")

    cases = {name: {row["group_id"] for row in values} for name, values in splits.items()}
    overlaps = {
        f"{a}/{b}": sorted(cases[a] & cases[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    overlaps = {pair: ids for pair, ids in overlaps.items() if ids}
    if overlaps:
        detail = "; ".join(f"{pair}: {ids[:10]}" for pair, ids in overlaps.items())
        raise ValueError(f"Case IDs overlap across splits (leakage): {detail}")
    return splits


def normalize_mri(array: np.ndarray) -> np.ndarray:
    """Per-slice scaling to [0, 1] after clipping to the 1st-99.5th percentile.

    Clipping stops a few very bright voxels from compressing the useful range, and
    per-slice scaling removes the arbitrary absolute intensity scale between scans.
    """
    low, high = np.percentile(array, [1.0, 99.5])
    if high <= low:  # constant slice: nothing to scale
        return np.zeros_like(array, dtype=np.float32)
    return ((np.clip(array, low, high) - low) / (high - low)).astype(np.float32)


class NiftiSliceDataset(Dataset):
    """Load one NIfTI slice at a time, resize it to fit the canvas and zero-pad.

    The aspect ratio is preserved: a 256x128 slice becomes 128x64 and is centred on
    the default 128x80 canvas instead of being stretched.
    """

    def __init__(self, rows: Sequence[Dict[str, str]], data_root: Path, image_height: int = 128, image_width: int = 80):
        if image_height % 8 or image_width % 8:
            raise ValueError("image_height and image_width must both be divisible by 8")
        if min(image_height, image_width) < 32:
            raise ValueError("Image canvas must be at least 32 pixels on each side")
        self.rows = list(rows)
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        array = normalize_mri(load_array(self.data_root / row["path"]))
        scale = min(self.image_height / array.shape[0], self.image_width / array.shape[1])
        new_h = max(1, round(array.shape[0] * scale))
        new_w = max(1, round(array.shape[1] * scale))
        resized = F.interpolate(
            torch.from_numpy(array)[None, None], size=(new_h, new_w), mode="bilinear", align_corners=False
        )[0]
        canvas = torch.zeros((1, self.image_height, self.image_width), dtype=resized.dtype)
        top, left = (self.image_height - new_h) // 2, (self.image_width - new_w) // 2
        canvas[:, top : top + new_h, left : left + new_w] = resized
        return {"image": canvas, "path": row["path"], "group_id": row["group_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manifest.csv from the HipMRI NIfTI folders")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="scan .nii/.nii.gz files and write the manifest")
    audit.add_argument("--data-root", type=Path, required=True, help="Folder containing keras_slices_* directories")
    audit.add_argument("--output", type=Path, default=Path("manifest.csv"), help="Output CSV path")
    args = parser.parse_args()
    try:
        counts = audit_directory(args.data_root, args.output)
    except Exception as exc:  # noqa: BLE001 - report cleanly instead of a traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(f"NIfTI audit complete: {counts}")
    print(f"Manifest: {args.output.resolve()}")


if __name__ == "__main__":
    main()
