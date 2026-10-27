"""Prepare deterministic CutLER train/validation/test data from COCO documents."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crater_detection.image_manifest import ImageManifest, ManifestImage


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
        for image in images:
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
) -> dict:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    manifest = ImageManifest.load(manifest_path)
    image_root = Path(image_root).resolve()
    output_root = Path(output_root).resolve()
    pseudo_labels = _load_json(Path(pseudo_labels_path))
    ground_truth = _load_json(Path(ground_truth_path))
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
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
        "source_manifest": manifest.metadata(),
        "images": {
            split: [image.file_name for image in images]
            for split, images in split_images.items()
        },
    }
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
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
