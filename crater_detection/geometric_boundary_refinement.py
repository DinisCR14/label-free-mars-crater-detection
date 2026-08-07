"""Correct boundary-touching COCO boxes and filter irregular crater masks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np


def _decode_segmentation(segmentation: dict, mask_utils) -> np.ndarray:
    encoded = dict(segmentation)
    counts = encoded.get("counts")
    if isinstance(counts, str):
        encoded["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(encoded)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def _boundary_sides(annotation: dict, image: dict) -> tuple[str, ...]:
    x, y, width, height = annotation["bbox"]
    image_width = image["width"]
    image_height = image["height"]
    sides = []
    if x == 0:
        sides.append("left")
    if x + width == image_width:
        sides.append("right")
    if y == 0:
        sides.append("top")
    if y + height == image_height:
        sides.append("bottom")
    return tuple(sides)


def _correct_boundary_box(annotation: dict, sides: tuple[str, ...]) -> bool:
    if len(sides) != 1:
        return False

    x, y, width, height = annotation["bbox"]
    side = sides[0]
    if side in {"left", "right"} and height > width:
        if side == "left":
            x -= height - width
        annotation["bbox"] = [x, y, height, height]
        return True
    if side in {"top", "bottom"} and width > height:
        if side == "top":
            y -= width - height
        annotation["bbox"] = [x, y, width, width]
        return True
    return False


def _ellipse_axis_ratio(mask: np.ndarray) -> float | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) >= 5]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    _, axes, _ = cv2.fitEllipse(contour)
    if axes[1] == 0:
        return None
    return float(axes[0] / axes[1])


def refine_coco_annotations(
    input_json: str | Path,
    output_json: str | Path,
    *,
    ellipse_min_ratio: float = 0.7,
    ellipse_max_ratio: float = 1.3,
) -> None:
    """Apply boundary correction and mask-shape filtering to a COCO file."""
    from pycocotools import mask as mask_utils

    input_json = Path(input_json)
    output_json = Path(output_json)
    with input_json.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    output = copy.deepcopy(coco)
    image_by_id = {image["id"]: image for image in coco.get("images", [])}
    refined = []
    for source_annotation in coco.get("annotations", []):
        image = image_by_id.get(source_annotation["image_id"])
        if image is None:
            continue
        annotation = copy.deepcopy(source_annotation)
        sides = _boundary_sides(annotation, image)
        if len(sides) >= 2:
            continue
        corrected = _correct_boundary_box(annotation, sides)
        if corrected:
            refined.append(annotation)
            continue

        try:
            mask = _decode_segmentation(annotation["segmentation"], mask_utils)
            ratio = _ellipse_axis_ratio(mask)
        except (KeyError, ValueError, cv2.error):
            ratio = None
        if ratio is None or not (ellipse_min_ratio <= ratio <= ellipse_max_ratio):
            if ratio is None:
                refined.append(annotation)
            continue
        refined.append(annotation)

    output["annotations"] = refined
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--ellipse-min-ratio", type=float, default=0.7)
    parser.add_argument("--ellipse-max-ratio", type=float, default=1.3)
    args = parser.parse_args()
    refine_coco_annotations(
        args.input_json,
        args.output_json,
        ellipse_min_ratio=args.ellipse_min_ratio,
        ellipse_max_ratio=args.ellipse_max_ratio,
    )


if __name__ == "__main__":
    main()
