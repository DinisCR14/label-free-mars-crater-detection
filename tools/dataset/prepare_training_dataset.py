"""Prepare deterministic CutLER train/validation/test data from COCO documents."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crater_detection.image_manifest import ImageManifest, ManifestImage

TILE_DEGREES = 2.0
FILENAME_PATTERN = re.compile(
    r"^lat_(?P<lat>-?\d+(?:\.\d+)?)_long_(?P<lon>-?\d+(?:\.\d+)?)_corrected\.png$"
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a COCO object: {path}")
    return payload


def _split_images(
    images: tuple[ManifestImage, ...],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[ManifestImage]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def _corrected_footprint(image: ManifestImage) -> dict[str, float]:
    match = FILENAME_PATTERN.match(image.file_name)
    if match is None:
        raise ValueError(
            "Spatial splitting requires corrected filenames matching "
            "lat_<latitude>_long_<longitude>_corrected.png: "
            f"{image.file_name}"
        )
    latitude = float(match.group("lat"))
    longitude = float(match.group("lon"))
    center_latitude = latitude + TILE_DEGREES / 2.0
    compression = math.cos(math.radians(abs(center_latitude)))
    if compression <= 0:
        raise ValueError(f"Invalid corrected tile latitude: {latitude}")
    width = TILE_DEGREES / compression
    center_longitude = (longitude + TILE_DEGREES / 2.0) % 360.0
    return {
        "latitude": latitude,
        "latitude_min": latitude,
        "latitude_max": latitude + TILE_DEGREES,
        "longitude_center": center_longitude,
        "longitude_width": width,
    }


def _longitude_parts(center: float, width: float) -> list[tuple[float, float]]:
    start = center - width / 2.0
    end = center + width / 2.0
    if start < 0:
        return [(start + 360.0, 360.0), (0.0, end)]
    if end > 360.0:
        return [(start, 360.0), (0.0, end - 360.0)]
    return [(start, end)]


def _overlap_groups(
    images: tuple[ManifestImage, ...],
) -> tuple[list[list[ManifestImage]], dict[str, dict[str, float]]]:
    footprints = {
        image.file_name: _corrected_footprint(image)
        for image in images
    }
    parent = list(range(len(images)))
    index_by_name = {image.file_name: index for index, image in enumerate(images)}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    rows: dict[float, list[tuple[ManifestImage, float, float]]] = {}
    for image in images:
        footprint = footprints[image.file_name]
        for start, end in _longitude_parts(
            footprint["longitude_center"], footprint["longitude_width"]
        ):
            rows.setdefault(footprint["latitude"], []).append((image, start, end))

    for intervals in rows.values():
        intervals.sort(key=lambda item: (item[1], item[2], item[0].file_name))
        active_end = -math.inf
        active_index = -1
        for image, start, end in intervals:
            if start < active_end - 1e-9:
                union(index_by_name[image.file_name], active_index)
            if end > active_end:
                active_end = end
                active_index = index_by_name[image.file_name]

    grouped: dict[int, list[ManifestImage]] = {}
    for image in images:
        grouped.setdefault(find(index_by_name[image.file_name]), []).append(image)
    groups = [
        sorted(group, key=lambda image: image.file_name)
        for group in grouped.values()
    ]
    groups.sort(
        key=lambda group: (
            _corrected_footprint(group[0])["latitude"],
            group[0].file_name,
        )
    )
    return groups, footprints


def _spatial_split_images(
    images: tuple[ManifestImage, ...],
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, list[ManifestImage]], list[list[ManifestImage]], dict[str, dict[str, float]]]:
    groups, footprints = _overlap_groups(images)
    if len(groups) < 3:
        raise ValueError(
            "Spatial splitting requires at least three non-overlapping groups"
        )
    cumulative_sizes = []
    total = 0
    for group in groups:
        total += len(group)
        cumulative_sizes.append(total)

    def closest_boundary(target: float, lower: int, upper: int) -> int:
        return min(
            range(lower, upper + 1),
            key=lambda index: abs(cumulative_sizes[index - 1] - target),
        )

    train_end = closest_boundary(total * train_ratio, 1, len(groups) - 2)
    val_end = closest_boundary(
        total * (train_ratio + val_ratio),
        train_end + 1,
        len(groups) - 1,
    )
    split_groups = {
        "train": groups[:train_end],
        "val": groups[train_end:val_end],
        "test": groups[val_end:],
    }
    split_images = {
        split: [image for group in assigned for image in group]
        for split, assigned in split_groups.items()
    }
    return split_images, groups, footprints


def _subset_document(
    source: dict,
    images: Iterable[ManifestImage],
    *,
    source_manifest: ImageManifest,
) -> dict:
    selected = list(images)
    selected_ids = {image.id for image in selected}
    annotations = [
        copy.deepcopy(annotation)
        for annotation in source.get("annotations", [])
        if annotation.get("image_id") in selected_ids
    ]
    output = {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key not in {"images", "annotations", "mapping"}
    }
    output["images"] = [
        {
            "id": index,
            "file_name": image.file_name,
            "width": image.width,
            "height": image.height,
        }
        for index, image in enumerate(selected)
    ]
    id_map = {image.id: index for index, image in enumerate(selected)}
    for annotation_id, annotation in enumerate(annotations, start=1):
        annotation["id"] = annotation_id
        annotation["image_id"] = id_map[annotation["image_id"]]
    output["annotations"] = annotations
    output["categories"] = copy.deepcopy(
        source.get("categories", [{"id": 1, "name": "crater"}])
    )
    output["source_mapping"] = source_manifest.metadata()
    return output


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _materialize_images(
    image_root: Path,
    output_root: Path,
    split_images: dict[str, list[ManifestImage]],
    *,
    link_mode: str,
) -> None:
    for split, images in split_images.items():
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        for image in tqdm(
            images,
            desc=f"Materializing {split} images",
            unit="image",
        ):
            source = image_root / image.file_name
            destination = split_root / image.file_name
            if not source.is_file():
                raise FileNotFoundError(f"Missing image for manifest record: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                continue
            if link_mode == "symlink":
                destination.symlink_to(source)
            elif link_mode == "hardlink":
                destination.hardlink_to(source)
            else:
                shutil.copy2(source, destination)


def prepare_training_dataset(
    manifest_path: str | Path,
    image_root: str | Path,
    pseudo_labels_path: str | Path,
    ground_truth_path: str | Path,
    output_root: str | Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
    link_mode: str = "symlink",
    split_strategy: str = "spatial",
) -> dict:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    if split_strategy not in {"random", "spatial"}:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    manifest = ImageManifest.load(manifest_path)
    image_root = Path(image_root).resolve()
    output_root = Path(output_root).resolve()
    pseudo_labels = _load_json(Path(pseudo_labels_path))
    ground_truth = _load_json(Path(ground_truth_path))
    groups = []
    footprints = {}
    if split_strategy == "spatial":
        split_images, groups, footprints = _spatial_split_images(
            manifest.images, train_ratio, val_ratio
        )
    else:
        split_images = _split_images(manifest.images, train_ratio, val_ratio, seed)
    _materialize_images(image_root, output_root, split_images, link_mode=link_mode)

    annotations_root = output_root / "annotations"
    _write_json(
        annotations_root / "train_initial.json",
        _subset_document(pseudo_labels, split_images["train"], source_manifest=manifest),
    )
    _write_json(
        annotations_root / "val_truegt.json",
        _subset_document(ground_truth, split_images["val"], source_manifest=manifest),
    )
    _write_json(
        annotations_root / "test_truegt_clipped.json",
        _subset_document(ground_truth, split_images["test"], source_manifest=manifest),
    )
    split_manifest = {
        "schema_version": 1,
        "split_strategy": (
            "spatial_overlap_aware" if split_strategy == "spatial" else "random"
        ),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
        "source_manifest": manifest.metadata(),
        "images": {
            split: [image.file_name for image in images]
            for split, images in split_images.items()
        },
    }
    if split_strategy == "spatial":
        split_manifest["overlap_policy"] = {
            "footprint_model": "corrected horizontal extent 2 degrees / cos(abs(center latitude))",
            "connected_overlap_groups_kept_together": True,
            "group_count": len(groups),
        }
        split_manifest["groups"] = [
            {
                "id": index,
                "images": [image.file_name for image in group],
                "footprints": {
                    image.file_name: footprints[image.file_name] for image in group
                },
            }
            for index, group in enumerate(groups)
        ]
    _write_json(output_root / "split-manifest.json", split_manifest)
    return split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-strategy",
        choices=("spatial", "random"),
        default="spatial",
        help="Use overlap-aware spatial groups or legacy random image splitting",
    )
    parser.add_argument(
        "--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    args = parser.parse_args()
    try:
        prepare_training_dataset(
            args.manifest,
            args.image_root,
            args.pseudo_labels,
            args.ground_truth,
            args.output_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            link_mode=args.link_mode,
            split_strategy=args.split_strategy,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
