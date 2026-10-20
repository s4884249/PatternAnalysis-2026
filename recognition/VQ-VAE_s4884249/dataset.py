"""NIfTI data audit, leakage-resistant grouping, and image preprocessing.

The audit command deliberately writes a reviewable manifest rather than assuming
that every NIfTI under keras_slices_data is an MRI image. Review the output and
verify patient/source-volume group IDs before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NIFTI_EXTENSIONS = (".nii", ".nii.gz")
MASK_TOKENS = (
    "mask",
    "label",
    "seg",
    "annotation",
    "groundtruth",
    "ground_truth",
    "segmentation",
)
SLICE_SUFFIX_PATTERNS = (
    re.compile(r"(?i)(?:[_-](?:slice|sl|z|img|image))[_-]?\d+$"),
    re.compile(r"(?i)(?:[_-](?:slice|sl|z))[_-]?\d+(?:[_-].*)?$"),
)


def _nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "NIfTI I/O requires nibabel. Install the project requirements in the Rangpur environment."
        ) from exc
    return nib


def is_nifti(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(ext) for ext in NIFTI_EXTENSIONS)


def nifti_stem(path: Path) -> str:
    name = path.name
    return (
        name[:-7]
        if name.lower().endswith(".nii.gz")
        else name[:-4]
        if name.lower().endswith(".nii")
        else path.stem
    )


def infer_kind(path: Path) -> str:
    """Identify the known HipMRI image/mask folders before using generic heuristics."""
    normalized_parts = [part.lower().replace("-", "_") for part in path.parts]
    for part in normalized_parts:
        if re.fullmatch(r"keras_slices_seg_(?:train|validate|test)", part):
            return "mask"
        if re.fullmatch(r"keras_slices_(?:train|validate|test)", part):
            return "image"

    combined = (str(path.parent) + "/" + nifti_stem(path)).lower().replace("-", "_")
    if "semantic_labels_only" in combined or any(
        token in combined for token in MASK_TOKENS
    ):
        return "mask"
    if "semantic_mrs" in combined:
        return "image"
    image_tokens = ("image", "img", "mri", "mr_scan", "scan", "intensity")
    if any(token in combined for token in image_tokens):
        return "image"
    return "unknown"


def infer_split(path: Path, root: Path | None = None) -> str:
    """Infer the supplied split from the top-level HipMRI directory name.

    The dataset's image directories are keras_slices_train, keras_slices_validate,
    and keras_slices_test. Segmentation directories are intentionally mapped to the
    same split name but remain kind=mask and are excluded from the image experiment.
    """
    try:
        relative = path.relative_to(root) if root is not None else path
    except ValueError:
        relative = path
    first_part = relative.parts[0].lower().replace("-", "_") if relative.parts else ""
    match = re.fullmatch(r"keras_slices_(?:seg_)?(train|validate|test)", first_part)
    if not match:
        return ""
    value = match.group(1)
    return "val" if value == "validate" else value


def infer_group_id(path: Path, root: Path) -> Tuple[str, str]:
    """Return a tentative subject/case ID and how it was inferred."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parent_parts = relative.parts[:-1]
    stem = nifti_stem(path)

    # HipMRI names encode a stable case identifier followed by time point and slice,
    # e.g. case_040_week_5_slice_18 or seg_040_week_5_slice_18. Use the case ID,
    # not case+week, to detect whether one case leaks across the supplied splits.
    case_match = re.match(r"(?i)^(?:case|seg)[_-]?(\d+)_week_\d+_slice_\d+$", stem)
    if case_match:
        return f"case_{case_match.group(1)}", "case_id_from_filename"

    # Prefer explicitly slice-like suffixes before using parent folders, since
    # datasets commonly put every slice in one generic directory (e.g. semantic_MRs).
    for pattern in SLICE_SUFFIX_PATTERNS:
        updated = pattern.sub("", stem)
        if updated != stem:
            return updated, "heuristic_slice_suffix"

    # Treat a parent as a case only when it looks identifier-like, not when it is
    # a generic collection name such as images, semantic_MRs, or train.
    if parent_parts:
        parent_name = parent_parts[-1]
        if re.search(
            r"(?i)(patient|subject|subj|case|volume|vol|study|scan|hip|\bsub\b)",
            parent_name,
        ) or re.search(r"[A-Za-z_-]+\d+", parent_name):
            return "/".join(parent_parts), "heuristic_parent_directory"

    # A numeric suffix can be a slice index, but can also belong to a subject ID.
    updated = re.sub(r"(?:[_-]\d{1,5})$", "", stem)
    if updated != stem and updated:
        return updated, "heuristic_numeric_suffix_REVIEW_REQUIRED"
    return stem, "unverified_filename_isolation"


def _canonical_array(path: Path) -> np.ndarray:
    """Load a NIfTI as a finite 2D float32 array; preserve its in-plane geometry."""
    nib = _nibabel()
    image = nib.load(str(path))
    array = np.asarray(
        image.get_fdata(dtype=np.float32, caching="unchanged"), dtype=np.float32
    )
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D image (or singleton extra dimensions), got shape={array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite voxel values")
    return array


def audit_directory(data_root: Path, output_csv: Path) -> Dict[str, int]:
    """Inspect candidate NIfTI files and write a manifest for human verification."""
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    paths = sorted(
        path for path in data_root.rglob("*") if path.is_file() and is_nifti(path)
    )
    if not paths:
        raise FileNotFoundError(f"No .nii or .nii.gz files found under {data_root}")

    fieldnames = [
        "path",
        "split",
        "kind",
        "group_id",
        "group_id_verified",
        "group_inference",
        "shape",
        "dtype",
        "finite",
        "min",
        "max",
        "mean",
        "std",
        "error",
    ]
    counts: Dict[str, int] = {"files_seen": len(paths), "read_ok": 0, "read_error": 0}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            relative_path = path.relative_to(data_root)
            kind = infer_kind(path)
            split = infer_split(relative_path)
            group_id, group_inference = infer_group_id(path, data_root)
            row = {
                "path": relative_path.as_posix(),
                "split": split,
                "kind": kind,
                "group_id": group_id,
                # Never auto-certify a heuristic group mapping.
                "group_id_verified": "false",
                "group_inference": group_inference,
                "shape": "",
                "dtype": "",
                "finite": "",
                "min": "",
                "max": "",
                "mean": "",
                "std": "",
                "error": "",
            }
            try:
                array = _canonical_array(path)
                row.update(
                    {
                        "shape": "x".join(str(n) for n in array.shape),
                        "dtype": str(array.dtype),
                        "finite": str(bool(np.isfinite(array).all())).lower(),
                        "min": f"{float(array.min()):.7g}",
                        "max": f"{float(array.max()):.7g}",
                        "mean": f"{float(array.mean()):.7g}",
                        "std": f"{float(array.std()):.7g}",
                    }
                )
                counts["read_ok"] += 1
            except (
                Exception
            ) as exc:  # audit all files and preserve the reason for manual triage
                row["error"] = f"{type(exc).__name__}: {exc}"
                counts["read_error"] += 1
            writer.writerow(row)

    # A machine-readable audit summary complements the row-level manifest.
    with output_csv.open(newline="", encoding="utf-8") as handle:
        audited_rows = list(csv.DictReader(handle))
    shape_counts = Counter(row["shape"] for row in audited_rows if row.get("shape"))
    kind_counts = Counter(row["kind"] for row in audited_rows)
    inference_counts = Counter(row["group_inference"] for row in audited_rows)
    numeric_mins = [float(row["min"]) for row in audited_rows if row.get("min")]
    numeric_maxs = [float(row["max"]) for row in audited_rows if row.get("max")]
    audit_summary = {
        "data_root": str(data_root),
        "manifest": str(output_csv.resolve()),
        "file_counts": counts,
        "shape_counts": dict(shape_counts),
        "kind_counts_from_filename_heuristic": dict(kind_counts),
        "group_id_inference_counts": dict(inference_counts),
        "split_counts": dict(Counter(row.get("split", "") for row in audited_rows)),
        "global_observed_min": min(numeric_mins) if numeric_mins else None,
        "global_observed_max": max(numeric_maxs) if numeric_maxs else None,
        "all_group_ids_require_manual_verification": True,
        "caveats": [
            "Known keras_slices_{train,validate,test} folders are classified as MRI images; keras_slices_seg_* folders are masks.",
            "Split labels come from the supplied top-level directory; image rows are checked for case-ID overlap across splits.",
            "Unknown folders still require manual review before training.",
        ],
    }
    summary_path = output_csv.with_name("audit_summary.json")
    summary_path.write_text(
        json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def load_manifest(
    manifest_path: Path, data_root: Path, include_unknown: bool = False
) -> List[Dict[str, str]]:
    """Load image rows and reject known masks or unreadable entries."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data_root = data_root.expanduser().resolve()
    rows: List[Dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "kind", "group_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        for line_no, source in enumerate(reader, start=2):
            relative_path = (source.get("path") or "").strip()
            if not relative_path:
                raise ValueError(f"Empty path on manifest line {line_no}")
            path = (data_root / relative_path).resolve()
            if data_root not in path.parents and path != data_root:
                raise ValueError(
                    f"Manifest path escapes data root on line {line_no}: {relative_path}"
                )
            if not path.is_file():
                raise FileNotFoundError(
                    f"File listed in manifest does not exist: {path}"
                )

            # Folder conventions are authoritative for this dataset. This also
            # lets an older manifest be used safely after upgrading this script.
            folder_kind = infer_kind(path)
            kind = (
                folder_kind
                if folder_kind in {"image", "mask"}
                else (source.get("kind") or "unknown").strip().lower()
            )
            if kind == "mask":
                continue
            if kind != "image" and not (include_unknown and kind == "unknown"):
                continue
            if (source.get("error") or "").strip():
                continue
            if (source.get("finite") or "").strip().lower() == "false":
                continue

            inferred_group, group_method = infer_group_id(path, data_root)
            group_id = (
                inferred_group
                if group_method == "case_id_from_filename"
                else (source.get("group_id") or "").strip()
            )
            if not group_id:
                raise ValueError(f"Empty group_id on manifest line {line_no}")
            directory_split = infer_split(Path(relative_path), root=None)
            manifest_split = (source.get("split") or "").strip().lower()
            if manifest_split == "validate":
                manifest_split = "val"
            if manifest_split and directory_split and manifest_split != directory_split:
                raise ValueError(
                    f"Split mismatch on manifest line {line_no}: CSV says {manifest_split!r}, "
                    f"but path belongs to {directory_split!r}: {relative_path}"
                )
            split = directory_split or manifest_split
            row = dict(source)
            row["path"] = relative_path
            row["group_id"] = group_id
            row["group_inference"] = (
                group_method
                if group_method == "case_id_from_filename"
                else (source.get("group_inference") or group_method)
            )
            row["kind"] = kind
            row["split"] = split
            row["group_id_verified"] = (
                (source.get("group_id_verified") or "false").strip().lower()
            )
            rows.append(row)
    if len(rows) < 3:
        raise ValueError(
            f"Only {len(rows)} usable image rows were found. Mark verified MRI files as kind=image in the manifest."
        )
    return rows


def all_groups_verified(rows: Sequence[Dict[str, str]]) -> bool:
    return all(
        row.get("group_id_verified", "false").strip().lower() in {"true", "1", "yes"}
        for row in rows
    )


def split_by_group(
    rows: Sequence[Dict[str, str]],
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Dict[str, List[Dict[str, str]]]:
    """Split groups, not individual slices, so one source group cannot cross splits."""
    if not (
        0.0 < train_fraction < 1.0
        and 0.0 < val_fraction < 1.0
        and train_fraction + val_fraction < 1.0
    ):
        raise ValueError("Invalid fractions; require train_fraction + val_fraction < 1")
    unique_groups = sorted({row["group_id"] for row in rows})
    if len(unique_groups) < 3:
        raise ValueError(
            f"Need at least 3 distinct patient/source-volume groups for train/val/test; found {len(unique_groups)}."
        )
    rng = np.random.default_rng(seed)
    permutation = list(
        np.asarray(unique_groups, dtype=object)[rng.permutation(len(unique_groups))]
    )
    n_groups = len(permutation)
    n_train = max(1, int(math.floor(n_groups * train_fraction)))
    n_val = max(1, int(math.floor(n_groups * val_fraction)))
    if n_train + n_val >= n_groups:
        n_train = n_groups - 2
        n_val = 1
    train_groups = set(permutation[:n_train])
    val_groups = set(permutation[n_train : n_train + n_val])
    test_groups = set(permutation[n_train + n_val :])

    splits = {
        "train": [row for row in rows if row["group_id"] in train_groups],
        "val": [row for row in rows if row["group_id"] in val_groups],
        "test": [row for row in rows if row["group_id"] in test_groups],
    }
    # Defensive assertion: source groups are disjoint by construction.
    group_sets = {
        name: {row["group_id"] for row in value} for name, value in splits.items()
    }
    if (
        group_sets["train"] & group_sets["val"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["val"] & group_sets["test"]
    ):
        raise RuntimeError("Leakage detected: group IDs overlap between splits")
    return splits


def split_from_manifest(
    rows: Sequence[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    """Use the dataset's existing train/validate/test directory assignments.

    No re-shuffling is performed. Case IDs must be disjoint between all three
    folders; fail loudly if the folder layout would leak a case across splits.
    """
    aliases = {"train": "train", "val": "val", "validate": "val", "test": "test"}
    splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    unknown_paths = []
    for row in rows:
        value = aliases.get((row.get("split") or "").strip().lower())
        if value is None:
            unknown_paths.append(row.get("path", "<unknown path>"))
            continue
        splits[value].append(row)
    if unknown_paths:
        sample = ", ".join(unknown_paths[:5])
        raise ValueError(
            "The predefined split could not be inferred for some image paths. "
            f"Expected keras_slices_train/validate/test directories. Examples: {sample}"
        )
    empty = [name for name, values in splits.items() if not values]
    if empty:
        raise ValueError(
            f"Predefined split is missing image rows for {empty}. Check that --data-root points to "
            "the directory containing keras_slices_train, keras_slices_validate and keras_slices_test."
        )

    group_sets = {
        name: {row["group_id"] for row in values} for name, values in splits.items()
    }
    overlaps = {
        "train/val": sorted(group_sets["train"] & group_sets["val"]),
        "train/test": sorted(group_sets["train"] & group_sets["test"]),
        "val/test": sorted(group_sets["val"] & group_sets["test"]),
    }
    overlaps = {name: groups for name, groups in overlaps.items() if groups}
    if overlaps:
        detail = "; ".join(
            f"{name}: {groups[:10]}" for name, groups in overlaps.items()
        )
        raise ValueError(
            "Case IDs overlap across the supplied predefined splits; refusing to train to avoid leakage. "
            + detail
        )
    return splits


def normalize_mri(array: np.ndarray) -> np.ndarray:
    """Per-slice percentile clipping and scaling to [0, 1]."""
    finite = np.asarray(array, dtype=np.float32)
    if not np.isfinite(finite).all():
        raise ValueError("Cannot normalize non-finite MRI values")
    low, high = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(finite, dtype=np.float32)
    finite = np.clip(finite, low, high)
    finite = (finite - low) / (high - low)
    return finite.astype(np.float32, copy=False)


class NiftiSliceDataset(Dataset):
    """Load one NIfTI slice at a time; resize while preserving aspect ratio and pad.

    The feasibility audit found 256x128 and 256x144 candidate dimensions. The
    default 128x80 canvas maps their long sides to 128 and pads widths 64/72 to
    80 without stretching anatomy. Change dimensions only after inspecting data.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        data_root: Path,
        image_height: int = 128,
        image_width: int = 80,
    ):
        if image_height % 8 != 0 or image_width % 8 != 0:
            raise ValueError("image_height and image_width must both be divisible by 8")
        if image_height < 32 or image_width < 32:
            raise ValueError("Image canvas must be at least 32 pixels on each side")
        self.rows = list(rows)
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        path = (self.data_root / row["path"]).resolve()
        array = normalize_mri(_canonical_array(path))
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        scale = min(
            self.image_height / array.shape[0], self.image_width / array.shape[1]
        )
        new_h = max(1, int(round(array.shape[0] * scale)))
        new_w = max(1, int(round(array.shape[1] * scale)))
        tensor = F.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        canvas = torch.zeros(
            (1, 1, self.image_height, self.image_width), dtype=tensor.dtype
        )
        top = (self.image_height - new_h) // 2
        left = (self.image_width - new_w) // 2
        canvas[:, :, top : top + new_h, left : left + new_w] = tensor
        return {
            "image": canvas.squeeze(0),
            "path": row["path"],
            "group_id": row["group_id"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit candidate HipMRI NIfTI files and create a reviewable manifest"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser(
        "audit", help="scan .nii/.nii.gz files and write manifest.csv"
    )
    audit_parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root folder containing candidate NIfTI files",
    )
    audit_parser.add_argument(
        "--output", type=Path, default=Path("manifest.csv"), help="Output CSV path"
    )
    args = parser.parse_args()
    try:
        counts = audit_directory(args.data_root, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(f"NIfTI audit complete: {counts}")
    print(f"Manifest: {args.output.resolve()}")
    print(
        "IMPORTANT: review kind and group_id values; set group_id_verified=true only after confirming grouping."
    )


if __name__ == "__main__":
    main()
