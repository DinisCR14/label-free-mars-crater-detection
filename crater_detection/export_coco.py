"""Export refined crater labels in the COCO instance-annotation schema."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


COCO_CATEGORIES = [{"id": 1, "name": "crater", "supercategory": "crater"}]


def _validate_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("each annotation bbox must be [x, y, width, height]")
    values = [float(value) for value in bbox]
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("each annotation bbox must have positive width and height")
    return values


def export_coco_annotations(
    source_json: str | Path,
    output_json: str | Path,
) -> None:
    """Normalize refined labels and write a Detectron2-compatible COCO file."""
    source_json = Path(source_json)
    output_json = Path(output_json)
    with source_json.open("r", encoding="utf-8") as handle:
        source = json.load(handle)

    images = copy.deepcopy(source.get("images", []))
    image_ids = {image["id"] for image in images}
    annotations = []
    for annotation_id, source_annotation in enumerate(source.get("annotations", []), start=1):
        if source_annotation["image_id"] not in image_ids:
            raise ValueError(
                f"annotation references unknown image_id {source_annotation['image_id']}"
            )
        annotation = {
            "id": annotation_id,
            "image_id": source_annotation["image_id"],
            "category_id": 1,
            "segmentation": copy.deepcopy(source_annotation["segmentation"]),
            "area": float(source_annotation["area"]),
            "bbox": _validate_bbox(source_annotation["bbox"]),
            "iscrowd": int(source_annotation.get("iscrowd", 0)),
        }
        annotations.append(annotation)

    output = {
        "info": copy.deepcopy(source.get("info", {})),
        "licenses": copy.deepcopy(source.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": copy.deepcopy(COCO_CATEGORIES),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    export_coco_annotations(args.source_json, args.output_json)


if __name__ == "__main__":
    main()
