"""Export refined crater labels in the COCO instance-annotation schema."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

COCO_CATEGORIES = [{"id": 1, "name": "crater", "supercategory": "crater"}]
ELLIPSE_POINT_COUNT = 72


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


def _ellipse_polygon(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> list[float]:
    points = []
    for index in range(ELLIPSE_POINT_COUNT):
        angle = 2.0 * math.pi * index / ELLIPSE_POINT_COUNT
        points.extend([
            center_x + radius_x * math.cos(angle),
            center_y + radius_y * math.sin(angle),
        ])
    return points


def _image_paths(image_root: Path, image_list: Path | None) -> list[Path]:
    if image_list is not None:
        paths = []
        for line in image_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                path = Path(line)
                paths.append(path if path.is_absolute() else image_root / path)
        return paths
    return sorted(image_root.glob("**/*_original.*"))


def convert_ellipse_labels_to_coco(
    labels_json: str | Path,
    image_root: str | Path,
    output_json: str | Path,
    *,
    image_list: str | Path | None = None,
    clip_bboxes: bool = False,
) -> None:
    """Convert legacy [cx, cy, rx, ry] labels for one split to COCO."""
    labels_json = Path(labels_json)
    image_root = Path(image_root)
    output_json = Path(output_json)
    image_list_path = Path(image_list) if image_list is not None else None
    with labels_json.open("r", encoding="utf-8") as handle:
        labels_by_image = json.load(handle)

    images = []
    annotations = []
    annotation_id = 1
    for image_id, image_path in enumerate(_image_paths(image_root, image_list_path)):
        if not image_path.is_file():
            raise FileNotFoundError(f"Could not read image: {image_path}")
        relative_name = image_path.relative_to(image_root).as_posix()
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({
            "id": image_id,
            "file_name": relative_name,
            "width": width,
            "height": height,
        })

        image_key = image_path.stem.removesuffix("_original")
        for label in labels_by_image.get(image_key, []):
            if not isinstance(label, (list, tuple)) or len(label) != 4:
                raise ValueError(f"Invalid ellipse label for {image_key}: {label!r}")
            center_x, center_y, radius_x, radius_y = (float(value) for value in label)
            if radius_x <= 0 or radius_y <= 0:
                raise ValueError(f"Ellipse radii must be positive for {image_key}: {label!r}")
            x = center_x - radius_x
            y = center_y - radius_y
            box_width = 2.0 * radius_x
            box_height = 2.0 * radius_y
            if clip_bboxes:
                x2 = min(x + box_width, width)
                y2 = min(y + box_height, height)
                x = max(x, 0.0)
                y = max(y, 0.0)
                box_width = x2 - x
                box_height = y2 - y
                if box_width <= 0 or box_height <= 0:
                    continue
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [_ellipse_polygon(center_x, center_y, radius_x, radius_y)],
                "area": math.pi * radius_x * radius_y,
                "bbox": [x, y, box_width, box_height],
                "iscrowd": 0,
            })
            annotation_id += 1

    output = {
        "info": {"description": "Mars crater ground-truth ellipses converted to COCO"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": copy.deepcopy(COCO_CATEGORIES),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_json", type=Path, nargs="?")
    parser.add_argument("output_json", type=Path, nargs="?")
    parser.add_argument("--ellipse-labels", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--output-json", dest="output_json_option", type=Path)
    parser.add_argument("--clip-bboxes", action="store_true")
    args = parser.parse_args()
    output_json = args.output_json_option or args.output_json
    if args.ellipse_labels is not None:
        if args.image_root is None or output_json is None:
            parser.error("ellipse conversion requires --image-root and --output-json")
        convert_ellipse_labels_to_coco(
            args.ellipse_labels,
            args.image_root,
            output_json,
            image_list=args.image_list,
            clip_bboxes=args.clip_bboxes,
        )
    elif args.source_json is not None and output_json is not None:
        export_coco_annotations(args.source_json, output_json)
    else:
        parser.error("provide source_json and output_json, or use --ellipse-labels")


if __name__ == "__main__":
    main()
