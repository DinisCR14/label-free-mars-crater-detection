"""Prepare the supplied 512x512 Mars crater archive for this repository."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path


LATITUDE_PATTERN = re.compile(r"(?:^|_)lat_(-?\d+(?:\.\d+)?)_long_")


def _latitude(stem: str) -> float:
    match = LATITUDE_PATTERN.search(stem)
    if match is None:
        raise ValueError(f"Could not parse latitude from image stem: {stem}")
    return float(match.group(1))


def _split_stems(
    stems: list[str], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[str]]:
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def prepare_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_abs_latitude: float = 50.0,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Filter, split, and copy the supplied image archive."""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a non-empty test split")

    labels_path = source_dir / "labels.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Could not find labels.json in {source_dir}")
    with labels_path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, dict):
        raise ValueError("labels.json must contain an object keyed by image stem")

    originals = {
        path.stem.removesuffix("_original"): path
        for path in source_dir.rglob("*_original.png")
    }
    if not originals:
        raise FileNotFoundError(f"No *_original.png images found in {source_dir}")
    marked = {
        path.stem.removesuffix("_marked"): path
        for path in source_dir.rglob("*_marked.png")
    }

    eligible = []
    for stem in sorted(originals):
        if abs(_latitude(stem)) <= max_abs_latitude:
            if stem not in labels:
                raise ValueError(f"Missing labels.json entry for image stem: {stem}")
            eligible.append(stem)
    if not eligible:
        raise ValueError("No images remain after latitude filtering")

    splits = _split_stems(eligible, train_ratio, val_ratio, seed)
    for split, stems in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            shutil.copy2(originals[stem], split_dir / originals[stem].name)

    marked_dir = output_dir / "marked"
    marked_dir.mkdir(parents=True, exist_ok=True)
    for stem in eligible:
        marked_path = marked.get(stem)
        if marked_path is not None:
            shutil.copy2(marked_path, marked_dir / marked_path.name)

    filtered_labels = {stem: labels[stem] for stem in eligible}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(filtered_labels, handle)

    image_list = [f"train/{originals[stem].name}" for stem in splits["train"]]
    (output_dir / "image-list.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
    manifest = {
        "source_dir": str(source_dir),
        "seed": seed,
        "max_abs_latitude": max_abs_latitude,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "counts": {split: len(stems) for split, stems in splits.items()},
        "splits": splits,
    }
    with (output_dir / "split-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-abs-latitude", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    splits = prepare_dataset(
        args.source_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_abs_latitude=args.max_abs_latitude,
        seed=args.seed,
    )
    print("Prepared " + ", ".join(f"{split}={len(stems)}" for split, stems in splits.items()))


if __name__ == "__main__":
    main()