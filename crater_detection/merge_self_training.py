"""Merge new crater predictions with annotations from a previous training round."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from export_coco import COCO_CATEGORIES


def _decode_segmentation(segmentation: Any, height: int, width: int, mask_utils) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return np.asarray(mask_utils.decode(rle), dtype=bool)


def _mask_iou_matrix(first: np.ndarray, second: np.ndarray, device: str) -> np.ndarray:
    if first.size == 0 or second.size == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)

    try:
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        first_tensor = torch.as_tensor(first, dtype=torch.bool, device=device)
        second_tensor = torch.as_tensor(second, dtype=torch.bool, device=device)
        intersection = (first_tensor[:, None] & second_tensor[None, :]).sum(dim=(-2, -1))
        union = (first_tensor[:, None] | second_tensor[None, :]).sum(dim=(-2, -1))
        iou = torch.zeros_like(intersection, dtype=torch.float32)
        torch.divide(intersection.float(), union.float(), out=iou, where=union > 0)
        return iou.cpu().numpy()
    except ImportError:
        first_bool = first.astype(bool)
        second_bool = second.astype(bool)
        intersection = np.logical_and(first_bool[:, None], second_bool[None, :]).sum(axis=(-2, -1))
        union = np.logical_or(first_bool[:, None], second_bool[None, :]).sum(axis=(-2, -1))
        return np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )


def _annotation_mask(annotation: dict, image: dict, mask_utils) -> np.ndarray:
    return _decode_segmentation(
        annotation["segmentation"], image["height"], image["width"], mask_utils
    )


def _prepare_annotation(annotation: dict, image: dict, mask_utils) -> dict:
    prepared = copy.deepcopy(annotation)
    prepared["category_id"] = 1
    prepared["iscrowd"] = int(prepared.get("iscrowd", 0))
    prepared["bbox"] = [float(value) for value in prepared["bbox"]]
    if "area" not in prepared:
        prepared["area"] = float(_annotation_mask(prepared, image, mask_utils).sum())
    else:
        prepared["area"] = float(prepared["area"])
    return prepared


def merge_self_training_annotations(
    previous_json: str | Path,
    new_predictions_json: str | Path,
    output_json: str | Path,
    *,
    confidence_threshold: float = 0.70,
    mask_iou_threshold: float = 0.50,
    device: str = "auto",
) -> None:
    """Merge confidence-filtered predictions with non-overlapping old labels."""
    from pycocotools import mask as mask_utils

    with Path(previous_json).open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    with Path(new_predictions_json).open("r", encoding="utf-8") as handle:
        new_predictions = json.load(handle)

    previous_images = {image["id"]: image for image in previous.get("images", [])}
    previous_annotations = previous.get("annotations", [])
    if isinstance(new_predictions, dict):
        new_images = {image["id"]: image for image in new_predictions.get("images", [])}
        prediction_list = new_predictions.get("annotations", [])
    else:
        new_images = {}
        prediction_list = new_predictions

    predictions_by_image: dict[int, list[dict]] = {}
    for annotation in prediction_list:
        if float(annotation.get("score", 0.0)) >= confidence_threshold:
            predictions_by_image.setdefault(annotation["image_id"], []).append(annotation)

    previous_by_image: dict[int, list[dict]] = {}
    for annotation in previous_annotations:
        previous_by_image.setdefault(annotation["image_id"], []).append(annotation)

    images = dict(previous_images)
    images.update(new_images)
    merged_annotations: list[dict] = []
    for image_id, image in images.items():
        old_annotations = previous_by_image.get(image_id, [])
        new_annotations = predictions_by_image.get(image_id, [])
        prepared_new = [
            _prepare_annotation(annotation, image, mask_utils)
            for annotation in new_annotations
        ]
        prepared_old = [
            _prepare_annotation(annotation, image, mask_utils)
            for annotation in old_annotations
        ]

        if prepared_old and prepared_new:
            old_masks = np.stack(
                [_annotation_mask(annotation, image, mask_utils) for annotation in prepared_old]
            )
            new_masks = np.stack(
                [_annotation_mask(annotation, image, mask_utils) for annotation in prepared_new]
            )
            max_iou = _mask_iou_matrix(old_masks, new_masks, device).max(axis=1)
            retained_old = [
                annotation
                for annotation, overlap in zip(prepared_old, max_iou)
                if overlap < mask_iou_threshold
            ]
        else:
            retained_old = prepared_old

        merged_annotations.extend(prepared_new)
        merged_annotations.extend(retained_old)

    for annotation_id, annotation in enumerate(merged_annotations, start=1):
        annotation["id"] = annotation_id
        annotation.pop("score", None)

    output = {
        "info": copy.deepcopy(previous.get("info", {})),
        "licenses": copy.deepcopy(previous.get("licenses", [])),
        "images": list(images.values()),
        "annotations": merged_annotations,
        "categories": copy.deepcopy(COCO_CATEGORIES),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-json", type=Path, required=True)
    parser.add_argument("--new-predictions-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--mask-iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    merge_self_training_annotations(
        args.previous_json,
        args.new_predictions_json,
        args.output_json,
        confidence_threshold=args.confidence_threshold,
        mask_iou_threshold=args.mask_iou_threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
