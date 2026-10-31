"""Refine mask annotations by removing geometrically eccentric predictions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _decode_segmentation(segmentation: dict, mask_utils) -> np.ndarray:
    encoded = dict(segmentation)
    counts = encoded.get("counts")
    if isinstance(counts, str):
        encoded["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(encoded)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def _ellipse_axis_ratio(mask: np.ndarray) -> float | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) >= 5]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    try:
        _, axes, _ = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    major_axis, minor_axis = max(axes), min(axes)
    if major_axis <= 0:
        return None
    return float(minor_axis / major_axis)


def refine_geometric_annotations(
    input_json: str | Path,
    output_json: str | Path,
    *,
    ellipse_min_ratio: float = 0.7,
) -> None:
    """Keep original annotations whose masks are sufficiently circular."""
    from pycocotools import mask as mask_utils

    if not 0.0 < ellipse_min_ratio <= 1.0:
        raise ValueError("ellipse_min_ratio must be greater than 0 and at most 1")
    input_json = Path(input_json)
    output_json = Path(output_json)
    with input_json.open(encoding="utf-8") as handle:
        coco = json.load(handle)

    output = copy.deepcopy(coco)
    image_by_id = {image["id"]: image for image in coco.get("images", [])}
    refined = []
    for source_annotation in tqdm(
        coco.get("annotations", []),
        desc="Geometric refinement",
        unit="annotation",
    ):
        image = image_by_id.get(source_annotation.get("image_id"))
        segmentation = source_annotation.get("segmentation")
        if image is None or not isinstance(segmentation, dict):
            continue
        try:
            mask = _decode_segmentation(segmentation, mask_utils)
            ratio = _ellipse_axis_ratio(mask)
        except (KeyError, TypeError, ValueError, cv2.error):
            ratio = None
        if ratio is not None and ratio >= ellipse_min_ratio:
            refined.append(copy.deepcopy(source_annotation))

    output["annotations"] = refined
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--ellipse-min-ratio", type=float, default=0.7)
    args = parser.parse_args()
    try:
        refine_geometric_annotations(
            args.input_json,
            args.output_json,
            ellipse_min_ratio=args.ellipse_min_ratio,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
